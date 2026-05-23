"""Vanilla NeRF: 8-layer MLP with skip connection and view-dependent colour.

Architecture (from Mildenhall et al. 2020):
  - Position encoded with L=10 Fourier freqs (63-dim input)
  - 8 fully-connected layers (256 units, ReLU) with a skip at layer 5
  - σ head: Linear(256→1), ReLU activation
  - Feature vector: Linear(256→256) → concat with view encoding (27-dim)
  - 1 hidden colour layer (128 units) → sigmoid RGB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import PositionalEncoding
from .renderer import (
    get_rays,
    sample_stratified,
    sample_importance,
    volume_render,
)


class NeRFMLP(nn.Module):
    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden: int = 256,
        skip_at: int = 4,  # zero-indexed layer index at which to add skip
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_freqs)
        self.dir_enc = PositionalEncoding(dir_freqs)
        self.skip_at = skip_at

        pos_ch = self.pos_enc.out_dim   # 63
        dir_ch = self.dir_enc.out_dim   # 27

        # Position-only trunk (8 layers)
        self.pts_layers = nn.ModuleList()
        in_ch = pos_ch
        for i in range(8):
            self.pts_layers.append(nn.Linear(in_ch, hidden))
            # After the skip layer, the next layer's input is hidden + pos_ch
            in_ch = hidden + pos_ch if i == skip_at else hidden

        self.sigma_head   = nn.Linear(hidden, 1)
        self.feature_head = nn.Linear(hidden, hidden)

        # View-dependent colour head
        self.color_layer = nn.Linear(hidden + dir_ch, hidden // 2)
        self.rgb_head    = nn.Linear(hidden // 2, 3)

    def forward(self, pts: torch.Tensor, dirs: torch.Tensor):
        """
        pts  : (N, 3) world-space points
        dirs : (N, 3) unit viewing directions

        Returns
        -------
        rgb   : (N, 3) in [0, 1]
        sigma : (N,)   density ≥ 0
        """
        p = self.pos_enc(pts)
        d = self.dir_enc(F.normalize(dirs, dim=-1))

        h = p
        for i, layer in enumerate(self.pts_layers):
            h = F.relu(layer(h), inplace=True)
            if i == self.skip_at:
                h = torch.cat([h, p], dim=-1)

        sigma = F.softplus(self.sigma_head(h)).squeeze(-1)  # (N,)
        feat  = self.feature_head(h)

        h = F.relu(self.color_layer(torch.cat([feat, d], dim=-1)), inplace=True)
        rgb = torch.sigmoid(self.rgb_head(h))                # (N, 3)
        return rgb, sigma


class NeRF(nn.Module):
    """Hierarchical NeRF: coarse + fine network with importance resampling."""

    def __init__(
        self,
        n_coarse: int = 64,
        n_fine: int = 128,
        near: float = 2.0,
        far: float = 6.0,
        white_bg: bool = True,
        **mlp_kwargs,
    ):
        super().__init__()
        self.n_coarse  = n_coarse
        self.n_fine    = n_fine
        self.near      = near
        self.far       = far
        self.white_bg  = white_bg

        self.coarse = NeRFMLP(**mlp_kwargs)
        self.fine   = NeRFMLP(**mlp_kwargs)

    def render_rays(
        self,
        rays_o: torch.Tensor,  # (N_rays, 3)
        rays_d: torch.Tensor,  # (N_rays, 3)
        perturb: bool = True,
    ):
        N = rays_o.shape[0]
        device = rays_o.device

        # ── Coarse pass ──────────────────────────────────────────────────────
        z_coarse = sample_stratified(self.near, self.far, self.n_coarse, N, device, perturb)
        pts_c = rays_o[:, None] + rays_d[:, None] * z_coarse[..., None]  # (N, n_c, 3)
        pts_c_flat = pts_c.reshape(-1, 3)
        dirs_c_flat = rays_d[:, None].expand_as(pts_c).reshape(-1, 3)

        rgb_c, sigma_c = self.coarse(pts_c_flat, dirs_c_flat)
        rgb_c   = rgb_c.reshape(N, self.n_coarse, 3)
        sigma_c = sigma_c.reshape(N, self.n_coarse)

        rgb_coarse, depth_coarse, acc_coarse, weights_c = volume_render(
            rgb_c, sigma_c, z_coarse, rays_d, self.white_bg
        )

        # ── Fine pass (importance sampling) ──────────────────────────────────
        z_mid = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
        z_imp = sample_importance(z_mid, weights_c[..., 1:-1], self.n_fine, perturb)
        z_fine, _ = torch.sort(torch.cat([z_coarse, z_imp], dim=-1), dim=-1)

        pts_f = rays_o[:, None] + rays_d[:, None] * z_fine[..., None]
        pts_f_flat = pts_f.reshape(-1, 3)
        dirs_f_flat = rays_d[:, None].expand_as(pts_f).reshape(-1, 3)

        rgb_f, sigma_f = self.fine(pts_f_flat, dirs_f_flat)
        rgb_f   = rgb_f.reshape(N, -1, 3)
        sigma_f = sigma_f.reshape(N, -1)

        rgb_fine, depth_fine, acc_fine, _ = volume_render(
            rgb_f, sigma_f, z_fine, rays_d, self.white_bg
        )

        return {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }

    def render_image(
        self,
        H: int,
        W: int,
        focal: float,
        c2w: torch.Tensor,
        chunk: int = 1024,
    ) -> torch.Tensor:
        """Render a full image by chunking rays to fit in memory."""
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        rgbs = []
        for i in range(0, rays_o.shape[0], chunk):
            out = self.render_rays(rays_o[i:i+chunk], rays_d[i:i+chunk], perturb=False)
            rgbs.append(out["rgb_fine"])
        return torch.cat(rgbs).reshape(H, W, 3)
