import torch
import torch.nn as nn
import torch.nn.functional as F

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar
from performer_pytorch import CrossAttention
from utils.general_utils import normalize_dir, try_embed_pluecker_ray
from utils.rigid_utils import exp_se3


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
    """GATr-based encoder for wireless signal geometry.

    When use_view=False, builds a compact token sequence from:
      - A learnable CLS token (no physical position)
      - The transmitter (TX) position
      - Top-K anchor Gaussians selected by opacity
    GATr performs equivariant self-attention over this sequence; the CLS
    output encodes the TX–scene geometric relationship.
    """
    def __init__(
        self,
        token_dim: int = 64,
        d_out: int = 1,
        use_film: bool = True,
        gatr_blocks: int = 1,
        term=True,
        use_view=True,
        num_anchors: int = 16,
        s_channels: int = 2,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.d_out = d_out
        self.use_film = use_film
        self.term = term
        self.use_view = use_view
        self.num_anchors = num_anchors

        if use_view:
            # Original view-conditioned path (unchanged)
            in_mv = 2 if term else 1
            self.gatr = GATr(
                in_mv_channels=in_mv,
                out_mv_channels=token_dim,
                hidden_mv_channels=token_dim,
                in_s_channels=0,
                out_s_channels=None,
                hidden_s_channels=4,
                num_blocks=gatr_blocks,
                attention=SelfAttentionConfig(),
                mlp=MLPConfig(),
            )
        else:
            # Anchor-based path: [CLS, TX, Anchor×num_anchors]
            self.cls_point = nn.Parameter(torch.zeros(1, 3))
            self.type_embedding = nn.Embedding(3, s_channels)  # 0=CLS, 1=TX, 2=Anchor
            self.gatr = GATr(
                in_mv_channels=1,
                out_mv_channels=1,
                hidden_mv_channels=token_dim,
                in_s_channels=s_channels,
                out_s_channels=s_channels,
                hidden_s_channels=4,
                num_blocks=gatr_blocks,
                attention=SelfAttentionConfig(),
                mlp=MLPConfig(),
            )
            self.cls_proj = nn.Linear(16 + s_channels, token_dim)

    def forward(self, pts, view, tx, opacities=None):
        if self.use_view:
            # Original view-conditioned path (unchanged)
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
            mv_out, _ = self.gatr(mv_seq, scalars=None)
            mv_out = extract_scalar(mv_out)
            out = mv_out.squeeze(0) if squeeze_out else mv_out
            return out.squeeze().clone()

        # ---- Anchor-based path (use_view=False) ----
        M = self.num_anchors
        device = pts.device

        # Select top-M anchors by opacity
        if opacities is not None and pts.shape[0] > M:
            _, top_idx = opacities.topk(M)
            anchor_pos = pts[top_idx]        # [M, 3]
        else:
            anchor_pos = pts[:M]             # [M, 3]

        tx_pos = tx[0:1]                     # [1, 3]

        # Embed as GA trivectors
        cls_ga    = embed_point(self.cls_point.to(device))  # [1, 16]
        tx_ga     = embed_point(tx_pos)                     # [1, 16]
        anchor_ga = embed_point(anchor_pos)                 # [M, 16]

        # Sequence: [CLS, TX, A1, ..., AM]
        ga_seq = torch.cat([cls_ga, tx_ga, anchor_ga], dim=0)  # [2+M, 16]

        # Type embeddings: 0=CLS, 1=TX, 2=Anchor
        type_ids = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            torch.ones(1, dtype=torch.long, device=device),
            torch.full((M,), 2, dtype=torch.long, device=device),
        ])
        type_emb = self.type_embedding(type_ids)  # [2+M, s_channels]

        # GATr input: (..., items, mv_channels, 16) and (..., items, s_channels)
        mv_in = ga_seq.unsqueeze(0).unsqueeze(-2)   # [1, 2+M, 1, 16]
        s_in  = type_emb.unsqueeze(0)                # [1, 2+M, s_channels]

        mv_out, s_out = self.gatr(mv_in, scalars=s_in)
        # mv_out: [1, 2+M, 1, 16],  s_out: [1, 2+M, s_channels]

        # Extract CLS token output (index 0)
        cls_mv = mv_out[0, 0, 0]   # [16]
        cls_s  = s_out[0, 0]       # [s_channels]

        cls_raw = torch.cat([cls_mv, cls_s])  # [16 + s_channels]
        return self.cls_proj(cls_raw)          # [token_dim]


class DirectionalCrossAttention(nn.Module):
    def __init__(self, input_dim1, input_dim2, dim=128, heads=2):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.query_proj = nn.Linear(input_dim1, dim)
        self.key_proj = nn.Linear(input_dim2, dim)
        self.cross_attention = CrossAttention(dim=dim, heads=heads)

    def forward(self, pts, view, cls):
        if view is not None:    
            direction = torch.cat([pts, view], dim=-1)
        else:
            direction = pts
        q = self.query_proj(direction).unsqueeze(0)
        kv = self.key_proj(cls).unsqueeze(0)
        attn_out = self.cross_attention(q.contiguous().clone(), context=kv.contiguous().clone())
        return attn_out.squeeze(0)


# class DeformNetwork(nn.Module):
#     def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, is_blender=True, is_6dof=False):
#         super(DeformNetwork, self).__init__()
#         self.D = D
#         self.W = W
#         self.input_ch = input_ch
#         self.t_multires = 6 if is_blender else 10
#         self.skips = [D // 2]

#         self.embed_pos_fn, pos_input_ch = get_embedder(self.t_multires, 3)
#         self.embed_fn, xyz_input_ch = get_embedder(multires, 3)
#         self.input_ch = xyz_input_ch + pos_input_ch

#         if is_blender:
           
#             self.pos_out = 90

#             self.posnet = nn.Sequential(
#                 nn.Linear(pos_input_ch, 256), nn.ReLU(inplace=True),
#                 nn.Linear(256, self.pos_out))

#             self.linear = nn.ModuleList(
#                 [nn.Linear(xyz_input_ch + self.pos_out, W)] + [
#                     nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.pos_out, W)
#                     for i in range(D - 1)]
#             )

#         else:
#             self.linear = nn.ModuleList(
#                 [nn.Linear(self.input_ch, W)] + [
#                     nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
#                     for i in range(D - 1)]
#             )

#         self.is_blender = is_blender
#         self.is_6dof = is_6dof

#         if is_6dof:
#             self.branch_w = nn.Linear(W, 3)
#             self.branch_v = nn.Linear(W, 3)
#         else:
#             self.gaussian_warp = nn.Linear(W, 3)
#         self.gaussian_rotation = nn.Linear(W, 4)
#         self.gaussian_scaling = nn.Linear(W, 3)
#         self.gaussian_signal = nn.Linear(W,3)
#         self.gaussian_phase = nn.Linear(W, 3)

#     def forward(self, x, t, opacities=None):
#         t_emb = self.embed_pos_fn(t)
#         if self.is_blender:
#             t_emb = self.posnet(t_emb)  # better for D-NeRF Dataset
#         x_emb = self.embed_fn(x)
        
        
#         h = torch.cat([x_emb, t_emb], dim=-1)
#         for i, l in enumerate(self.linear):
#             h = self.linear[i](h)
#             h = F.relu(h)
#             if i in self.skips:
#                 h = torch.cat([x_emb, t_emb, h], -1)

#         if self.is_6dof:
#             w = self.branch_w(h)
#             v = self.branch_v(h)
#             theta = torch.norm(w, dim=-1, keepdim=True)
#             w = w / theta + 1e-5
#             v = v / theta + 1e-5
#             screw_axis = torch.cat([w, v], dim=-1)
#             d_xyz = exp_se3(screw_axis, theta)
#         else:
#             d_xyz = self.gaussian_warp(h)
#         scaling = self.gaussian_scaling(h)
#         rotation = self.gaussian_rotation(h)
#         signal_real = self.gaussian_signal(h)
#         signal_img = self.gaussian_phase(h)
#         signal_complex = signal_real*torch.exp(1j*signal_img)
#         signal = torch.abs(signal_complex)

#         return rotation, scaling, signal









class MappingNetwork(nn.Module):
    """
    Wireless Signal Mapping Network.

    Maps (Gaussian point positions, transmitter position) -> per-Gaussian signal features
    used as the color/signal channel in 3DGS rendering.

    Paper-inspired architecture (adapted from arXiv:2505.20714):
      - Stage-1 attenuation module: F1(P(G), cls) -> (h_att, delta_f)
      - Stage-2 signal module: F2(P(G), P(TX), cls, h_att) -> (h_sig, Sig_f)
      - Here paper's frequency embedding is replaced by our cls embedding.
      - d_rotation/d_scaling outputs are preserved.

    When use_view=True, a GATr geometry encoder and directional cross-attention module
    are added to incorporate view direction (currently disabled in the current RFID setup).
    """
    def __init__(self, D=8, W=256, skips=[4],
                 input_dims={'pts': 3, 'view': 3, 'tx': 3},
                 multires={'pts': 10, 'view': 10, 'tx': 5},
                 is_embeded={'pts': True, 'view': True, 'tx': True},
                 token_dim=16,
                 use_view=False,
                 num_anchors=16,
                 gatr_blocks=1):
        super().__init__()
        self.skips = skips
        self.use_view = use_view
        self.token_dim = token_dim

        # Positional encodings
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])
        self.tx_feat_dim = 90
        self.tx_posnet = nn.Sequential(
            nn.Linear(input_tx_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.tx_feat_dim),
        )

        if use_view:
            self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])
        else:
            input_view_dim = 0

        # Stage-1 attenuation module: F1(P(G), cls)
        atten_input_dim = input_pts_dim + self.token_dim + self.tx_feat_dim
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(atten_input_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + atten_input_dim, W)
             for i in range(D - 1)]
        )
        self.attenuation_head = nn.Sequential(
            nn.Linear(W, W // 2),
            nn.GELU(),
            nn.Linear(W // 2, 1),
        )

        # Stage-2 signal module: F2(P(G), P(TX), cls, h_att)
        signal_input_dim = atten_input_dim + W + (input_view_dim if use_view else 0)
        self.signal_linears = nn.ModuleList(
            [nn.Linear(signal_input_dim, W)] +
            [nn.Linear(W, W) for _ in range(D // 2 - 1)]
        )

        # 3-channel unconstrained residual signal (Sig_f in Eq. 5)
        self.signal_head = nn.Linear(W * 2, 3)
        self.cls_encoder = GATrEncoder(token_dim=token_dim, gatr_blocks=gatr_blocks,
                                       use_view=use_view, num_anchors=num_anchors)

        if use_view:
            # View-conditioned path: GATr geometry encoder + directional cross-attention
            
            input_dim1 = input_pts_dim + input_view_dim
            self.directional_cross_attention = DirectionalCrossAttention(
                input_dim1=input_dim1, input_dim2=token_dim, dim=W, heads=4)
            self.mlp = nn.Linear(W * 2, W)

        self.rotation_linear = nn.Linear(W * 2, 4)
        self.scaling_linear = nn.Linear(W * 2, 3)

    def forward(self, pts, tx, view=None, opacities=None):
        """
        Args:
            pts:       [N, 3]  Gaussian point positions
            tx:        [N, 3]  Transmitter (TX) position, broadcast to all Gaussians
            view:      [N, 3]  View direction (only used when use_view=True)
            opacities: [N]     Gaussian opacities (used for anchor selection when use_view=False)

        Returns:
            signal_final: [N, 3]  Non-negative signal amplitude features for 3DGS
        """
        pts_encoded = self.embed_pts_fn(pts).contiguous()
        tx_encoded = self.embed_tx_fn(tx).contiguous()
        tx_encoded = self.tx_posnet(tx_encoded)

        pts_flat = pts_encoded.view(-1, pts_encoded.shape[-1])
        tx_flat = tx_encoded.view(-1, tx_encoded.shape[-1])
        if self.use_view:
            cls = self.cls_encoder(pts, view, tx)
            view_encoded = self.embed_view_fn(view).contiguous()
            view_flat = view_encoded.view(-1, view_encoded.shape[-1])

            directional_attn = self.directional_cross_attention(pts_flat, view_flat, cls)
            feature = self.mlp(torch.cat([feature.clone(), directional_attn], -1)) + feature
        else:
            cls = self.cls_encoder(pts, None, tx, opacities=opacities)

        # Stage-1 attenuation module: F1(P(G), cls)
        cls_expanded = cls.expand(pts_flat.shape[0], -1)
        atten_input = torch.cat([pts_flat, tx_flat, cls_expanded], -1)
        x = atten_input
        for i, layer in enumerate(self.attenuation_linears):
            x = layer(x)
            x = F.gelu(x)
            if i in self.skips:
                x = torch.cat([atten_input, x], -1)

        h_att = x
        delta_f = self.attenuation_head(h_att)  # [N, 1] raw residual; renderer clamps opacity+delta_f to [0,1]

        if self.use_view:
            view_encoded = self.embed_view_fn(view).contiguous()
            view_flat = view_encoded.view(-1, view_encoded.shape[-1])
            directional_attn = self.directional_cross_attention(pts_flat, view_flat, cls)
            h_att = self.mlp(torch.cat([h_att.clone(), directional_attn], -1)) + h_att
            signal_input = torch.cat([pts_flat, tx_flat, h_att, cls_expanded, view_flat], -1)
        else:
            signal_input = torch.cat([pts_flat, tx_flat, h_att, cls_expanded], -1)

        # Stage-2 signal module: F2(P(G), P(TX), cls, h_att)
        x = signal_input
        for i, layer in enumerate(self.signal_linears):
            x = layer(x)
            x = F.gelu(x)


        h_sig = x
        outputs = torch.cat([h_att, h_sig], -1).contiguous()  # [N, W*2]

        d_rotation = self.rotation_linear(outputs)
        d_scaling = self.scaling_linear(outputs) # 0.89

        d_signal = self.signal_head(outputs)          # [N, 3] unconstrained residual (Eq. 5 Sig_f)
        return d_rotation, d_scaling, d_signal, delta_f