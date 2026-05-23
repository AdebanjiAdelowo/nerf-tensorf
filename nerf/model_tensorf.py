"""TensoRF — Vector-Matrix (VM) decomposition of the radiance field.

Key idea  (Chen et al. 2022, https://apchenstu.github.io/TensoRF/)
----------
A 3-D scalar field G: R³ → R can be written exactly as a rank-1 tensor sum:
    G(x,y,z) ≈ Σᵣ M_XY^r(x,y)·v_Z^r(z)
              + Σᵣ M_XZ^r(x,z)·v_Y^r(y)
              + Σᵣ M_YZ^r(y,z)·v_X^r(x)

Each M (matrix) is a feature plane bilinearly sampled; each v (vector) is
a 1-D feature line linearly sampled.  For appearance, the scalar at each
rank is replaced by a C-dim feature vector that is decoded by a small MLP.

Why this is fast
----------------
Parameter count  : O(R·N²)   vs  O(N³)  for a full voxel grid.
Memory access    : 2-D plane interpolation is cache-friendly and can use
                   hardware bilinear sampling (F.grid_sample).
Training         : Gradients flow only through the sampled grid cells,
                   so sparse updates are natural; convergence in ~15 k iters
                   vs ~100 k iters for vanilla NeRF at similar resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import PositionalEncoding
from .renderer import get_rays, sample_stratified, volume_render


# ──────────────────────────────────────────────────────────────────────────────
# MPS-safe grid sampling
#
# F.grid_sample's backward (grid_sampler_2d_backward) is not yet implemented
# on MPS.  We replace it with manual bilinear / linear interpolation using
# torch.gather, whose backward IS supported on MPS, CUDA, and CPU.
# ──────────────────────────────────────────────────────────────────────────────

def _bilinear_2d(
    plane: torch.Tensor,    # (1, C, H, W)
    coords: torch.Tensor,   # (N, 2)  values in [-1, 1]  (x, y order)
) -> torch.Tensor:          # (N, C)
    """Differentiable bilinear interpolation via torch.gather — works on MPS."""
    _, C, H, W = plane.shape
    N = coords.shape[0]

    # [-1, 1] → pixel-space coordinates
    px = (coords[:, 0] + 1.0) * 0.5 * (W - 1)   # (N,)
    py = (coords[:, 1] + 1.0) * 0.5 * (H - 1)

    x0 = px.long().clamp(0, W - 2)
    y0 = py.long().clamp(0, H - 2)
    x1 = (x0 + 1).clamp(0, W - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    wx = (px - x0.float()).unsqueeze(-1)   # (N, 1)
    wy = (py - y0.float()).unsqueeze(-1)

    flat = plane.view(C, H * W)            # (C, H*W)

    def _fetch(xi, yi):
        idx = (yi * W + xi).unsqueeze(0).expand(C, N)   # (C, N)
        return flat.gather(1, idx).T                      # (N, C)

    return (
        _fetch(x0, y0) * (1 - wx) * (1 - wy)
      + _fetch(x1, y0) *      wx  * (1 - wy)
      + _fetch(x0, y1) * (1 - wx) *      wy
      + _fetch(x1, y1) *      wx  *      wy
    )


def _linear_1d(
    line: torch.Tensor,     # (1, C, L, 1)
    coord: torch.Tensor,    # (N,)  values in [-1, 1]
) -> torch.Tensor:          # (N, C)
    """Differentiable linear interpolation via torch.gather — works on MPS."""
    _, C, L, _ = line.shape
    N = coord.shape[0]

    t = (coord + 1.0) * 0.5 * (L - 1)    # (N,)
    t0 = t.long().clamp(0, L - 2)
    t1 = (t0 + 1).clamp(0, L - 1)
    wt = (t - t0.float()).unsqueeze(-1)    # (N, 1)

    flat = line.view(C, L)                 # (C, L)

    def _fetch(ti):
        idx = ti.unsqueeze(0).expand(C, N)
        return flat.gather(1, idx).T       # (N, C)

    return _fetch(t0) * (1 - wt) + _fetch(t1) * wt


# ──────────────────────────────────────────────────────────────────────────────
# VM-decomposed radiance field
# ──────────────────────────────────────────────────────────────────────────────

class TensoRF(nn.Module):
    """
    Parameters
    ----------
    aabb        : (2, 3) scene bounding box [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    grid_size   : voxel resolution on each axis (e.g. 128)
    R_sigma     : number of VM components for the density field
    R_feat      : number of VM components for the appearance field
    feat_ch     : appearance feature channels per component
    dir_freqs   : Fourier frequencies for view direction encoding
    near / far  : ray integration bounds
    white_bg    : composite onto white background
    """

    def __init__(
        self,
        aabb: torch.Tensor,
        grid_size: int = 128,
        R_sigma: int = 16,
        R_feat:  int = 8,
        feat_ch: int = 16,
        dir_freqs: int = 4,
        near: float = 2.0,
        far:  float = 6.0,
        white_bg: bool = True,
        n_samples: int = 128,
        pt_chunk: int = 16384,   # max points per forward call (memory guard)
    ):
        super().__init__()
        self.register_buffer("aabb", aabb)  # (2, 3)
        self.grid_size = grid_size
        self.R_sigma   = R_sigma
        self.R_feat    = R_feat
        self.feat_ch   = feat_ch
        self.near      = near
        self.far       = far
        self.white_bg  = white_bg
        self.n_samples = n_samples
        self.pt_chunk  = pt_chunk

        G = grid_size

        # ── Density VM components ────────────────────────────────────────────
        # One scalar feature per rank; shape: (1, R, G, G) for planes
        #                                      (1, R, G, 1) for lines (stored as 2D)
        self.sigma_plane_XY = nn.Parameter(torch.randn(1, R_sigma, G, G) * 0.1)
        self.sigma_plane_XZ = nn.Parameter(torch.randn(1, R_sigma, G, G) * 0.1)
        self.sigma_plane_YZ = nn.Parameter(torch.randn(1, R_sigma, G, G) * 0.1)
        self.sigma_line_Z   = nn.Parameter(torch.randn(1, R_sigma, G, 1) * 0.1)
        self.sigma_line_Y   = nn.Parameter(torch.randn(1, R_sigma, G, 1) * 0.1)
        self.sigma_line_X   = nn.Parameter(torch.randn(1, R_sigma, G, 1) * 0.1)

        # Density bias (scalar offset, like the original paper)
        self.sigma_bias = nn.Parameter(torch.zeros(1))

        # ── Appearance VM components ─────────────────────────────────────────
        # C feature channels per rank; shape: (1, R*C, G, G) for planes
        self.feat_plane_XY = nn.Parameter(torch.randn(1, R_feat * feat_ch, G, G) * 0.1)
        self.feat_plane_XZ = nn.Parameter(torch.randn(1, R_feat * feat_ch, G, G) * 0.1)
        self.feat_plane_YZ = nn.Parameter(torch.randn(1, R_feat * feat_ch, G, G) * 0.1)
        self.feat_line_Z   = nn.Parameter(torch.randn(1, R_feat, G, 1) * 0.1)
        self.feat_line_Y   = nn.Parameter(torch.randn(1, R_feat, G, 1) * 0.1)
        self.feat_line_X   = nn.Parameter(torch.randn(1, R_feat, G, 1) * 0.1)

        # ── Colour MLP: feature → RGB ────────────────────────────────────────
        self.dir_enc = PositionalEncoding(dir_freqs)
        total_feat   = 3 * feat_ch  # XY + XZ + YZ contributions summed per-mode
        dir_ch       = self.dir_enc.out_dim

        self.color_mlp = nn.Sequential(
            nn.Linear(total_feat + dir_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    # ── Coordinate helpers ───────────────────────────────────────────────────

    def _normalize(self, pts: torch.Tensor) -> torch.Tensor:
        """Map world coords in AABB to [-1, 1]³."""
        lo, hi = self.aabb[0], self.aabb[1]
        return 2.0 * (pts - lo) / (hi - lo) - 1.0

    # ── Field queries ────────────────────────────────────────────────────────

    def query_sigma(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts : (N, 3) in world space
        Returns density σ : (N,)
        """
        p = self._normalize(pts)               # (N, 3)  in [-1,1]
        x, y, z = p[..., 0], p[..., 1], p[..., 2]

        # XY plane × Z line
        xy = torch.stack([x, y], dim=-1)        # (N, 2)
        xz = torch.stack([x, z], dim=-1)
        yz = torch.stack([y, z], dim=-1)

        # plane: (N, R_sigma) — one scalar per rank after sampling
        pXY = _bilinear_2d(self.sigma_plane_XY, xy)  # (N, R_sigma)
        pXZ = _bilinear_2d(self.sigma_plane_XZ, xz)
        pYZ = _bilinear_2d(self.sigma_plane_YZ, yz)

        # lines: (N, R_sigma)
        lZ = _linear_1d(self.sigma_line_Z, z)
        lY = _linear_1d(self.sigma_line_Y, y)
        lX = _linear_1d(self.sigma_line_X, x)

        # VM sum: element-wise product of plane and matched line, summed over rank
        sigma = (pXY * lZ + pXZ * lY + pYZ * lX).sum(dim=-1) + self.sigma_bias
        return F.softplus(sigma)   # (N,)

    def query_color(self, pts: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        """
        pts  : (N, 3)
        dirs : (N, 3)
        Returns RGB : (N, 3)
        """
        p = self._normalize(pts)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]

        xy = torch.stack([x, y], dim=-1)
        xz = torch.stack([x, z], dim=-1)
        yz = torch.stack([y, z], dim=-1)

        R, C = self.R_feat, self.feat_ch

        # Each plane returns (N, R*C); reshape to (N, R, C)
        pXY = _bilinear_2d(self.feat_plane_XY, xy).view(-1, R, C)  # (N, R, C)
        pXZ = _bilinear_2d(self.feat_plane_XZ, xz).view(-1, R, C)
        pYZ = _bilinear_2d(self.feat_plane_YZ, yz).view(-1, R, C)

        # Lines return (N, R) scalars; unsqueeze to broadcast over channels.
        lZ = _linear_1d(self.feat_line_Z, z)  # (N, R)
        lY = _linear_1d(self.feat_line_Y, y)
        lX = _linear_1d(self.feat_line_X, x)

        # Outer product then sum over rank → (N, C) per mode pair
        fXY = (pXY * lZ.unsqueeze(-1)).sum(dim=1)   # (N, C)
        fXZ = (pXZ * lY.unsqueeze(-1)).sum(dim=1)
        fYZ = (pYZ * lX.unsqueeze(-1)).sum(dim=1)

        feat = torch.cat([fXY, fXZ, fYZ], dim=-1)   # (N, 3C)

        d_enc = self.dir_enc(F.normalize(dirs, dim=-1))
        return self.color_mlp(torch.cat([feat, d_enc], dim=-1))  # (N, 3)

    def render_rays(
        self,
        rays_o: torch.Tensor,  # (N_rays, 3)
        rays_d: torch.Tensor,  # (N_rays, 3)
        perturb: bool = True,
    ) -> dict:
        N = rays_o.shape[0]
        device = rays_o.device

        z_vals = sample_stratified(self.near, self.far, self.n_samples, N, device, perturb)
        pts = rays_o[:, None] + rays_d[:, None] * z_vals[..., None]  # (N, S, 3)
        S   = z_vals.shape[-1]

        pts_flat  = pts.reshape(-1, 3)
        dirs_flat = rays_d[:, None].expand(N, S, 3).reshape(-1, 3)

        # Chunk point queries to avoid OOM from large gather tensors.
        PC = self.pt_chunk
        sigma_parts, rgb_parts = [], []
        for j in range(0, pts_flat.shape[0], PC):
            sigma_parts.append(self.query_sigma(pts_flat[j:j+PC]))
            rgb_parts.append(self.query_color(pts_flat[j:j+PC], dirs_flat[j:j+PC]))
        sigma = torch.cat(sigma_parts).reshape(N, S)
        rgb   = torch.cat(rgb_parts).reshape(N, S, 3)

        rgb_map, depth_map, acc_map, _ = volume_render(
            rgb, sigma, z_vals, rays_d, self.white_bg
        )
        return {"rgb_fine": rgb_map, "depth_fine": depth_map}

    def render_image(
        self,
        H: int,
        W: int,
        focal: float,
        c2w: torch.Tensor,
        chunk: int = 512,
    ) -> torch.Tensor:
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        rgbs = []
        for i in range(0, rays_o.shape[0], chunk):
            out = self.render_rays(rays_o[i:i+chunk], rays_d[i:i+chunk], perturb=False)
            rgbs.append(out["rgb_fine"])
        return torch.cat(rgbs).reshape(H, W, 3)

    def tv_loss(self) -> torch.Tensor:
        """Total-variation regularisation on the density planes (sparsity prior)."""
        def tv(t):  # (1, C, H, W)
            return (
                (t[..., 1:, :] - t[..., :-1, :]).abs().mean()
                + (t[..., :, 1:] - t[..., :, :-1]).abs().mean()
            )
        return tv(self.sigma_plane_XY) + tv(self.sigma_plane_XZ) + tv(self.sigma_plane_YZ)
