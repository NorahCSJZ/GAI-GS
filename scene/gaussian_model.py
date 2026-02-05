#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import pi
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.optim as optim
import torch.nn.functional as F
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, embed_scalar, embed_pluecker_ray, extract_point, extract_pluecker_ray
from gatr.layers.linear import EquiLinear
from gatr.interface.translation import embed_translation
from performer_pytorch import CrossAttention, SelfAttention, FastAttention
from utils.general_utils import normalize_dir, try_embed_pluecker_ray, batched_gather_tokens
from einops import rearrange, repeat


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass


class ResidualMLP(nn.Module):
    """Residual MLP block with residual connection in intermediate layer"""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        
        # First transform: dim -> hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Second transform: hidden_dim -> dim (back to original dim for residual)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # Third transform: dim -> dim (after residual connection)
        self.fc3 = nn.Linear(dim, dim)
        self.ln3 = nn.LayerNorm(dim)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # First transform: expand
        out = self.dropout(self.act(self.ln1(self.fc1(x))))  # [B, hidden_dim]
        
        # Second transform: compress back + residual connection in intermediate layer
        out = self.ln2(self.fc2(out))  # [B, dim]
        out = self.act(out + x)  # ← Residual connection here (intermediate)
        
        # Third transform: further processing after residual
        out = self.dropout(self.act(self.ln3(self.fc3(out))))  # [B, dim]
        
        return out


class FastAttentionRenderer(nn.Module):
    def __init__(
        self,
        in_dim=4,
        hidden_dim=500,
        output_dim=4,
        heads=10,
        nb_features=256,   # Small head dim: 64/128, larger models: 128/256
        causal=False,
        use_input_ln=True,
        temp_init=1   # Initial value of τ; <1 sharper, >1 smoother
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.h = heads
        self.dh = hidden_dim // heads   # e.g., 500//10 = 50
        self.hidden_dim = hidden_dim

        # 1) Input normalization, prevent small variance from being absorbed
        self.in_norm = nn.LayerNorm(in_dim) if use_input_ln else nn.Identity()

        # 2) Using bias=True makes it easier to break symmetry (more stable in small variance scenarios)
        self.qkv = nn.Linear(in_dim, 3 * hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, output_dim, bias=True)

        # 3) FastAttention: note some versions use dim_head, others use dim_heads
        try:
            self.fast_attn = FastAttention(
                dim_head=self.dh,          # If your version uses dim_heads, keep next line, delete this
                nb_features=nb_features,
                causal=causal
            )
        except TypeError:
            self.fast_attn = FastAttention(
                dim_heads=self.dh,         # Compatible with other signature
                nb_features=nb_features,
                causal=causal
            )
        # 4) Learnable temperature τ: implement softmax(QK^T / τ) by scaling Q/K
        self.tau = nn.Parameter(torch.tensor(float(temp_init)))


    def forward(self, x):
        """
        x: (B, N, in_dim) or (N, in_dim)
        Returns: (B, N, output_dim)
        """
        if x.dim() == 2:  # (N, in_dim) → (1, N, in_dim)
            x = x.unsqueeze(0)

        # Input normalization
        x = self.in_norm(x)

        B, N, _ = x.shape

        # Linear projection to Q,K,V
        qkv = self.qkv(x)                             # (B, N, 3*hidden_dim)
        q, k, v = qkv.chunk(3, dim=-1)               # each (B, N, hidden_dim)

        # Split into multi-head
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.h)  # (B, H, N, dh)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.h)

        # Use τ to adjust attention "sharpness": softmax(QK^T / τ)
        # In FAVOR+, this is equivalent to dividing Q,K by sqrt(τ)
        # clamp ensures numerical stability and τ>0
        sqrt_tau = torch.clamp(self.tau, min=1e-4).sqrt()
        q = q / sqrt_tau
        k = k / sqrt_tau

        # FastAttention expects (B, H, N, dh) → (B, H, N, dh)
        out = self.fast_attn(q.contiguous(), k.contiguous(), v.contiguous())

        # Merge multi-head and output
        out = rearrange(out, 'b h n d -> b n (h d)')       # (B, N, hidden_dim)
        out = self.out(out)                          # (B, N, output_dim)
        return out

class FiLMBlock(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        # MLP projects cond_dim to gamma and beta, output dimension is feature_dim
        self.fc_gamma = nn.Linear(cond_dim, feature_dim)
        self.fc_beta  = nn.Linear(cond_dim, feature_dim)
    def forward(self, x, cond):
        """
        x: intermediate features, shape (B, feature_dim)
        cond: condition vector (global scene vector), shape (B, cond_dim)
        Output: features after FiLM transformation (B, feature_dim)
        """
        gamma = self.fc_gamma(cond)  # Generate scaling coefficient gamma
        beta  = self.fc_beta(cond)   # Generate shift coefficient beta
        return gamma * x + beta     # Element-wise feature scaling and shifting

class GATrEncoder(nn.Module):
    """
    Scheme A: Neighborhood packing (no mask needed)

    Inputs:
      - rx_pts:   (B, Np, 3)       Sampling points for each path (or receive points)
      - view_dir: (B, Np, 3)       Direction for each path, needs to be normalized
      - geom_pts: (B, Ng, 3)       Geometry library: e.g. Tx point cloud

      - sig_feats: (B, Np, D_sig)  (Optional) Signal branch features, FiLM modulated by z

    Outputs:
      - z:        (B, Np, d_mod)   Direction modulation code (MV→s readout)
      - y:        (B, Np, D_out)   (Optional) Modulated signal prediction; returns None if sig_feats not provided
    """

    def __init__(
        self,
        token_dim: int = 128,    # GATr MV channel count (matches checkpoint 20251104_214548)
        d_out: int = 1,          # Signal branch final output dimension
        use_film: bool = True,   # Whether to use FiLM modulation
        gatr_blocks: int = 1,    # Number of GATr stacked layers
        term=True                 # Whether to use term
    ):
        super().__init__()
        self.token_dim = token_dim
        self.d_out = d_out
        self.use_film = use_film
        self.term = term
        # GATr running on single small sequence (MV flow)
        # Note: in_mv_channels=1 (each token has only 1 MV channel)
        if term:
            self.gatr = GATr(
                in_mv_channels=2,
                out_mv_channels=token_dim,
                hidden_mv_channels=token_dim,
                in_s_channels=None,          # No scalar flow
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
                in_s_channels=None,          # No scalar flow
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


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        # RFID network configuration (matches checkpoint 20251104_214548: W=128, D=8, token_dim=4)
        tx_dim = 3        # RFID: 3D position
        tx_multires = 10  # Use positional encoding
        tx_is_embedded = True
        network_D = 8
        network_W = 128
        network_multires_pts = 10
        network_multires_view = 10
        skips = [4]
        token_dim = 4
        
        self.network_fn = MappingNetwork(
            D=network_D,
            W=network_W,
            input_dims={'pts':3, 'view':3, 'tx':tx_dim},
            multires={'pts':network_multires_pts, 'view':network_multires_view, 'tx':tx_multires},
            is_embeded={'pts':True, 'view':True, 'tx':tx_is_embedded},
            token_dim=token_dim,
            skips=skips,
            use_view=True,
        ).cuda()
        


    def __init__(self, sh_degree, optimizer_type="default", data_type='rfid'):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.data_type = data_type
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    '''The function of generating from point cloud '''        
    # def fetchPly(self):
    #     plydata = PlyData.read('./data/lab.ply')
    #     vertices = plydata['vertex']
    #     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    #     colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    #     # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    #     return BasicPointCloud(points=positions, colors=colors)
    
    def gaussian_init(self):
        # Randomly generate tensor following Gaussian distribution
        # RFID Spectrum: output is [360, 90] = 32400 pixels
        num_points = 20000
        fused_point_cloud = torch.randn((num_points,3)).float().cuda()
        fused_point_cloud = (fused_point_cloud*10)
        fused_color = RGB2SH(torch.rand((num_points,3)).float().cuda())

        # pcd = self.fetchPly()
        # fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() 
        features[:, :3, 0 ] = fused_color 
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)

        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.4 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(1, 1, 1)
        
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
    
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.optimizer_type == "default":
            self.optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.AdamW(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.AdamW([self._exposure], lr=training_args.exposure_lr_init)

        # Cosine scheduler with warmup for all optimizers (using LambdaLR to avoid deprecation warning)
        total_iters = training_args.iterations
        warmup_iters = int(total_iters * 0.05)  # 5% warmup
        
        import math
        def lr_lambda(step):
            if step < warmup_iters:
                # Linear warmup: 0.01 -> 1.0
                return 0.01 + 0.99 * step / warmup_iters
            else:
                # Cosine annealing: 1.0 -> 0.01
                progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
                return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Gaussian scheduler
        self.gaussian_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        # Exposure scheduler
        self.exposure_scheduler = optim.lr_scheduler.LambdaLR(self.exposure_optimizer, lr_lambda=lr_lambda)

        # Create independent optimizer for network_fn
        mlp_params = []
        if hasattr(self, 'network_fn') and self.network_fn is not None:
            network_lr = 5e-4  # Matches checkpoint 20251104_214548
            mlp_params.append({'params': list(self.network_fn.parameters()), 'lr': network_lr})
            print(f"Adding network_fn to mlp_optimizer (lr={network_lr})")
        
        if mlp_params:
            self.mlp_optimizer = torch.optim.AdamW(mlp_params, lr=1e-3, weight_decay=5e-5, betas=(0.9, 0.999))
            # MLP scheduler: LambdaLR with warmup + cosine
            self.mlp_scheduler = optim.lr_scheduler.LambdaLR(self.mlp_optimizer, lr_lambda=lr_lambda)
        else:
            self.mlp_optimizer = None
            self.mlp_scheduler = None

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        r_o = torch.tensor(r_o).cuda()
        pts = self._xyz
        view = self._xyz - tx
        tx = tx.repeat(pts.size(0), 1)
        raw = self.network_fn(pts, tx, view)
        path = torch.norm(self._xyz - r_o)
        path_loss = 0.025/path
        att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3] # [batchsize]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2, torch.sigmoid(s_p)*np.pi*2
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))    
        s_p = torch.exp(1j*s_p) 
        att_p = torch.exp(1j*att_p) 
        s_a = s_a.unsqueeze(1)
        att_a = att_a.unsqueeze(1)
        att_p = att_p.unsqueeze(1)
        
        '''Different from the paper, opacity is used here to replace the attenuation component of the network output.'''
        signal = s_a*att_a/path_loss
        att = att_a*att_p
        # signal = signal*att
        signal = torch.cat((signal, signal, signal),-1)
        self._features_dc = signal.unsqueeze(-2)
 

        
class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']    # input dimension of gamma
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L


        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
def get_embedder(multires, is_embeded=True, input_dims=3):

    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class DirectionalCrossAttention(nn.Module):
    def __init__(self, input_dim1, input_dim2, dim=128, heads=4):  # heads=4 matches checkpoint
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
        q_in  = q.contiguous().clone()
        kv_in = kv.contiguous().clone()
        attn_out = self.cross_attention(q_in, context=kv_in)
        return attn_out.squeeze(0)

    
class MappingNetwork(nn.Module):

    def __init__(self, D=8, W=128, skips=[4],
                 input_dims={'pts':3, 'view':3, 'tx':3},
                 multires = {'pts':10, 'view':10, 'tx':10},
                 is_embeded={'pts':True, 'view':True, 'tx':True},
                 attn_output_dims=2, sig_output_dims=2, token_dim=4,
                 use_view=True):
        super().__init__()
        self.skips = skips
        self.use_view = use_view
       
        # Set positional encoding function
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])
        self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])

        ## Attenuation network
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )

        ## Signal network (with view direction for RFID)
        signal_input_dim = input_tx_dim + W + input_view_dim
        self.signal_linears = nn.ModuleList(
            [nn.Linear(signal_input_dim, W)] +
            [nn.Linear(W, W//2)]
        )

        ## output head, 2 for amplitude and phase
        self.attenuation_output = nn.Linear(W, attn_output_dims)
        self.feature_layer = nn.Linear(W, W)
        self.signal_output = nn.Linear(W//2, sig_output_dims)
        
        # RFID: full structure using view
        cls_token_dim = token_dim
        self.cls_encoder = GATrEncoder(token_dim=cls_token_dim, gatr_blocks=1)
        attention_heads = 4
        input_dim1 = input_pts_dim + input_view_dim
        self.directional_cross_attention = DirectionalCrossAttention(
            input_dim1=input_dim1, input_dim2=cls_token_dim, dim=W, heads=attention_heads)
        self.mlp = nn.Linear(W*2, W)



    def forward(self, pts, tx, view=None):
        """
        Args:
            pts: [N, 3] Gaussian point positions
            tx: [N, D] Condition input (RFID: position [N, 3])
            view: [N, 3] View direction (used for RFID)
        """
        # position encoding
        pts_encoded = self.embed_pts_fn(pts).contiguous()
        tx_encoded = self.embed_tx_fn(tx).contiguous()
        
        shape = pts_encoded.shape
        pts_flat = pts_encoded.view(-1, pts_encoded.shape[-1])
        tx_flat = tx_encoded.view(-1, tx_encoded.shape[-1])

        # Attenuation network (only uses pts)
        x = pts_flat
        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts_flat, x], -1)

        attn = self.attenuation_output(x)  # [N, 2]
        feature = self.feature_layer(x)    # [N, W]

        # RFID: full structure using view
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
    
