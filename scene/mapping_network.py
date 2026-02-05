#
# MappingNetwork - Wireless Signal Field Network
# Separated from gaussian_model.py for cleaner architecture
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar
from performer_pytorch import CrossAttention
from utils.general_utils import normalize_dir, try_embed_pluecker_ray


class Embedder:
    """Positional encoding"""
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
    """
    GATr-based encoder for wireless signal field
    """
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
        
        if term:
            self.gatr = GATr(
                in_mv_channels=2,
                out_mv_channels=token_dim,
                hidden_mv_channels=token_dim,
                in_s_channels=None,
                out_s_channels=None,
                hidden_s_channels=16,
                num_blocks=gatr_blocks,
                attention=SelfAttentionConfig(),
                mlp=MLPConfig(),
            )
        else:
            self.gatr = GATr(
                in_mv_channels=1,
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
        return mv_out.squeeze().clone()


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
        q_in = q.contiguous().clone()
        kv_in = kv.contiguous().clone()
        attn_out = self.cross_attention(q_in, context=kv_in)
        return attn_out.squeeze(0)


class MappingNetwork(nn.Module):
    """
    Wireless Signal Mapping Network for RFID-Spectrum
    
    Maps (Gaussian points, position condition, view direction) -> signal features
    """
    def __init__(self, D=8, W=128, skips=[4],
                 input_dims={'pts': 3, 'view': 3, 'tx': 3},
                 multires={'pts': 10, 'view': 10, 'tx': 10},
                 is_embeded={'pts': True, 'view': True, 'tx': True},
                 attn_output_dims=2, sig_output_dims=2, token_dim=4,
                 use_view=True):
        super().__init__()
        self.skips = skips
        self.use_view = use_view
        
        # Positional encoding
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])
        self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])

        # Attenuation network
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )

        # Signal network
        signal_input_dim = input_tx_dim + W + input_view_dim
        self.signal_linears = nn.ModuleList(
            [nn.Linear(signal_input_dim, W)] +
            [nn.Linear(W, W // 2)]
        )

        # Output heads
        self.attenuation_output = nn.Linear(W, attn_output_dims)
        self.feature_layer = nn.Linear(W, W)
        self.signal_output = nn.Linear(W // 2, sig_output_dims)
        
        # RFID: Use view-based structure with GATr and cross attention
        cls_token_dim = token_dim
        self.cls_encoder = GATrEncoder(token_dim=cls_token_dim, gatr_blocks=1)
        attention_heads = 4
        input_dim1 = input_pts_dim + input_view_dim
        self.directional_cross_attention = DirectionalCrossAttention(
            input_dim1=input_dim1, input_dim2=cls_token_dim, dim=W, heads=attention_heads)
        self.mlp = nn.Linear(W * 2, W)

    def forward(self, pts, tx, view):
        """
        Args:
            pts: [N, 3] Gaussian point positions
            tx: [N, 3] Transmitter position condition
            view: [N, 3] View direction
        
        Returns:
            outputs: [N, 4] Signal features (att_amp, att_phase, sig_amp, sig_phase)
        """
        # Position encoding
        pts_encoded = self.embed_pts_fn(pts).contiguous()
        tx_encoded = self.embed_tx_fn(tx).contiguous()
        
        shape = pts_encoded.shape
        pts_flat = pts_encoded.view(-1, pts_encoded.shape[-1])
        tx_flat = tx_encoded.view(-1, tx_encoded.shape[-1])

        # Attenuation network
        x = pts_flat
        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts_flat, x], -1)

        attn = self.attenuation_output(x)  # [N, 2]
        feature = self.feature_layer(x)    # [N, W]

        # RFID path with view direction
        cls = self.cls_encoder(pts, view, tx)
        view_encoded = self.embed_view_fn(view).contiguous()
        view_flat = view_encoded.view(-1, view_encoded.shape[-1])
        
        directional_attn = self.directional_cross_attention(pts_flat, view_flat, cls)
        feature = self.mlp(torch.cat([feature.clone(), directional_attn], -1)) + feature
        x = torch.cat([feature, view_flat, tx_flat], -1)

        # Signal network
        for i, layer in enumerate(self.signal_linears):
            x = F.relu(layer(x))
    
        signal = self.signal_output(x)  # [N, 2]

        outputs = torch.cat([attn, signal], -1).contiguous()  # [N, 4]
        outputs = outputs.view(shape[:-1] + outputs.shape[-1:])
        return outputs


def raw_to_signal(raw, pts, r_o, data_type='rfid'):
    """
    Convert network raw output to signal features for rasterizer
    
    Args:
        raw: [N, 4] Network output (att_amp, att_phase, sig_amp, sig_phase)
        pts: [N, 3] Gaussian point positions
        r_o: [3] Receiver position
        data_type: 'rfid' (kept for compatibility)
    
    Returns:
        signal: [N, 3] Signal features for rasterizer RGB channels
    """
    # Ensure tensors are on the correct device
    if not pts.is_cuda:
        pts = pts.cuda()
    if not r_o.is_cuda:
        r_o = r_o.cuda()
    
    # Compute distance from each point to receiver [N]
    path = torch.norm(pts - r_o, dim=-1)  # [N] distance for each point
    path_loss = 0.025 / path  # [N]
    
    att_a, att_p, s_a, s_p = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]  # [N]
    
    # Phase: sigmoid -> [0, 2π]
    att_p = torch.sigmoid(att_p) * np.pi * 2
    s_p = torch.sigmoid(s_p) * np.pi * 2
    
    # Amplitude: use softplus instead of abs(leaky_relu) to maintain gradient flow
    # abs() has zero gradient at zero, which breaks gradient flow!
    att_a = F.softplus(att_a)  # [N] ensures positive values, continuous gradients
    s_a = F.softplus(s_a)      # [N]
    
    # Complex phase
    s_p = torch.exp(1j * s_p)   # [N] complex
    att_p = torch.exp(1j * att_p)  # [N] complex
    
    s_a = s_a.unsqueeze(1)  # [N, 1]
    att_a = att_a.unsqueeze(1)  # [N, 1]
    att_p = att_p.unsqueeze(1)  # [N, 1]
    
    # Signal computation: s_a * att_a / path_loss
    # path_loss is [N], will broadcast automatically
    signal = s_a * att_a / path_loss.unsqueeze(1)  # [N, 1]
    att = att_a * att_p  # [N, 1] complex
    
    # Expand to 3 channels
    signal = torch.cat((signal, signal, signal), -1)  # [N, 3]
    
    return signal


def create_mapping_network(data_type='rfid', device='cuda'):
    """
    Factory function to create MappingNetwork with RFID config
    
    Args:
        data_type: 'rfid' (kept for compatibility)
        device: 'cuda' or 'cpu'
    
    Returns:
        network: MappingNetwork instance
    """
    # RFID network configuration
    network = MappingNetwork(
        D=8,
        W=256,
        input_dims={'pts': 3, 'view': 3, 'tx': 3},
        multires={'pts': 10, 'view': 10, 'tx': 10},
        is_embeded={'pts': True, 'view': True, 'tx': True},
        token_dim=4,
        skips=[4],
        use_view=True,
    )
    
    return network.to(device)
