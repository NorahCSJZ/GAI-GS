import torch
import torch.nn as nn
import torch.nn.functional as F

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar
from performer_pytorch import CrossAttention
from utils.general_utils import normalize_dir, try_embed_pluecker_ray


class Embedder:
    """NeRF-style sinusoidal positional encoding."""
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, is_embeded=True, input_dims=3):
    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class GATrEncoder(nn.Module):
    """GATr-based encoder for wireless signal geometry."""
    def __init__(
        self,
        token_dim: int = 64,
        d_out: int = 1,
        use_film: bool = True,
        gatr_blocks: int = 1,
        term=True
    ):
        super().__init__()
        self.token_dim = token_dim
        self.d_out = d_out
        self.use_film = use_film
        self.term = term

        in_mv = 2 if term else 1
        self.gatr = GATr(
            in_mv_channels=in_mv,
            out_mv_channels=token_dim,
            hidden_mv_channels=token_dim,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=16,
            num_blocks=gatr_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )

    def forward(self, pts, view, tx):
        # Handle 2D inputs [N, 3] -> add batch dim [1, N, 3]
        squeeze_out = pts.dim() == 2
        if squeeze_out:
            pts = pts.unsqueeze(0)
            view = view.unsqueeze(0)
            tx = tx.unsqueeze(0)

        view = normalize_dir(view)
        mv_rays = try_embed_pluecker_ray(pts, view).unsqueeze(-2).contiguous()

        if self.term:
            mv_geom = embed_point(tx).unsqueeze(-2).contiguous()
            mv_seq = torch.cat([mv_rays, mv_geom], dim=1).unsqueeze(0)
        else:
            mv_seq = mv_rays.unsqueeze(0)

        try:
            mv_out, _ = self.gatr(mv_seq, scalars=None)
        except Exception as e:
            print(tx.shape, pts.shape, view.shape)
            raise Exception(f"Error in GATr: {e}")

        mv_out = extract_scalar(mv_out)
        out = mv_out.squeeze(0) if squeeze_out else mv_out
        return out.squeeze().clone()


class DirectionalCrossAttention(nn.Module):
    def __init__(self, input_dim1, input_dim2, dim=128, heads=2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.query_proj = nn.Linear(input_dim1, dim)
        self.key_proj = nn.Linear(input_dim2, dim)
        self.cross_attention = CrossAttention(dim=dim, heads=heads)

    def forward(self, pts, view, cls):
        direction = torch.cat([pts, view], dim=-1)
        q = self.query_proj(direction).unsqueeze(0)
        kv = self.key_proj(cls).unsqueeze(0)
        attn_out = self.cross_attention(q.contiguous().clone(), context=kv.contiguous().clone())
        return attn_out.squeeze(0)


class MappingNetwork(nn.Module):
    """
    Wireless Signal Mapping Network.

    Maps (Gaussian point positions, transmitter position) -> per-Gaussian signal features
    used as the color/signal channel in 3DGS rendering.

    Architecture:
      - Attenuation network: encodes Gaussian positions via positional encoding + skip-connection MLP
      - Signal network: fuses transmitter position encoding with attenuation features
      - Output: signal_rgb [N, 3] passed through abs() for non-negative amplitude

    When use_view=True, a GATr geometry encoder and directional cross-attention module
    are added to incorporate view direction (currently disabled for RFID/BLE).
    """
    def __init__(self, D=8, W=256, skips=[4],
                 input_dims={'pts': 3, 'view': 3, 'tx': 3},
                 multires={'pts': 10, 'view': 10, 'tx': 10},
                 is_embeded={'pts': True, 'view': True, 'tx': True},
                 token_dim=4,
                 use_view=True):
        super().__init__()
        self.skips = skips
        self.use_view = use_view

        # Positional encodings
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])

        if use_view:
            self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])
        else:
            input_view_dim = 0

        # Attenuation network: encodes Gaussian positions
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )

        # Signal network: combines attenuation features with transmitter encoding
        signal_input_dim = input_tx_dim + W + (input_view_dim if use_view else 0)
        self.signal_linears = nn.ModuleList(
            [nn.Linear(signal_input_dim, W)] +
            [nn.Linear(W, W)]
        )

        # Feature projection and 3-channel signal output for 3DGS rendering
        self.feature_layer = nn.Linear(W, W)
        self.signal_rgb = nn.Linear(W * 2, 3)

        if use_view:
            # View-conditioned path: GATr geometry encoder + directional cross-attention
            self.cls_encoder = GATrEncoder(token_dim=token_dim, gatr_blocks=1)
            input_dim1 = input_pts_dim + input_view_dim
            self.directional_cross_attention = DirectionalCrossAttention(
                input_dim1=input_dim1, input_dim2=token_dim, dim=W, heads=4)
            self.mlp = nn.Linear(W * 2, W)

        self.rotation_linear = nn.Linear(W * 2, 4)
        self.scaling_linear = nn.Linear(W * 2, 3)

    def forward(self, pts, tx, view=None):
        """
        Args:
            pts:  [N, 3]  Gaussian point positions
            tx:   [N, 3]  Transmitter (TX) position, broadcast to all Gaussians
            view: [N, 3]  View direction (only used when use_view=True)

        Returns:
            signal_final: [N, 3]  Non-negative signal amplitude features for 3DGS
        """
        pts_encoded = self.embed_pts_fn(pts).contiguous()
        tx_encoded = self.embed_tx_fn(tx).contiguous()

        pts_flat = pts_encoded.view(-1, pts_encoded.shape[-1])
        tx_flat = tx_encoded.view(-1, tx_encoded.shape[-1])

        # Attenuation network
        x = pts_flat
        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts_flat, x], -1)

        attn = x
        feature = self.feature_layer(x)  # [N, W]

        if self.use_view:
            cls = self.cls_encoder(pts, view, tx)
            view_encoded = self.embed_view_fn(view).contiguous()
            view_flat = view_encoded.view(-1, view_encoded.shape[-1])

            directional_attn = self.directional_cross_attention(pts_flat, view_flat, cls)
            feature = self.mlp(torch.cat([feature.clone(), directional_attn], -1)) + feature
            x = torch.cat([feature, view_flat, tx_flat], -1)
        else:
            x = torch.cat([feature, tx_flat], -1)

        # Signal network
        for i, layer in enumerate(self.signal_linears):
            x = F.relu(layer(x))

        signal = x
        outputs = torch.cat([attn, signal], -1).contiguous()  # [N, W*2]

        d_rotation = self.rotation_linear(outputs)
        d_scaling = self.scaling_linear(outputs)

        signal_rgb = self.signal_rgb(outputs)   # [N, 3]
        signal_final = torch.abs(signal_rgb)
        return d_rotation, d_scaling, signal_final
