"""Volume rendering — the core of NeRF.

The rendering integral:
    C(r) = ∫ T(t) σ(r(t)) c(r(t), d) dt
where T(t) = exp(-∫₀ᵗ σ(r(s)) ds) is the accumulated transmittance.

Discretised with N quadrature points:
    C(r) ≈ Σᵢ Tᵢ αᵢ cᵢ
    αᵢ  = 1 - exp(-σᵢ δᵢ)          (opacity of sample i)
    Tᵢ  = Πⱼ<ᵢ (1 - αⱼ)             (transmittance up to sample i)
    δᵢ  = tᵢ₊₁ - tᵢ                 (segment length)
"""

import torch
import torch.nn.functional as F


def sample_stratified(
    near: float,
    far: float,
    n_samples: int,
    n_rays: int,
    device: torch.device,
    perturb: bool = True,
) -> torch.Tensor:
    """Stratified sampling along rays — one sample per equal-width bin."""
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals          # (n_samples,)
    z_vals = z_vals.unsqueeze(0).expand(n_rays, -1).clone() # (n_rays, n_samples)
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand
    return z_vals


def sample_importance(
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_importance: int,
    perturb: bool = True,
) -> torch.Tensor:
    """Hierarchical importance sampling — draw samples where coarse model predicts high weight."""
    weights = weights + 1e-5  # prevent nans
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    u = torch.rand(*cdf.shape[:-1], n_importance, device=z_vals.device)
    if not perturb:
        u = u.expand_as(u).contiguous()

    inds = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    below = (inds - 1).clamp(0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # (N_rays, n_importance, 2)

    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(*inds_g.shape[:-1], cdf.shape[-1]),
                         dim=-1, index=inds_g)
    bins_g = torch.gather(z_vals.unsqueeze(-2).expand(*inds_g.shape[:-1], z_vals.shape[-1]),
                          dim=-1, index=inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def volume_render(
    rgb: torch.Tensor,      # (N_rays, N_samples, 3)
    sigma: torch.Tensor,    # (N_rays, N_samples)
    z_vals: torch.Tensor,   # (N_rays, N_samples)
    rays_d: torch.Tensor,   # (N_rays, 3)
    white_bg: bool = True,
):
    """Integrate samples into a pixel colour using the volume rendering equation."""
    dists = z_vals[..., 1:] - z_vals[..., :-1]                          # (N_rays, N_samples-1)
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)  # (N_rays, N_samples)
    # Scale by ray direction magnitude so σ is independent of parameterisation.
    dists = dists * rays_d.norm(dim=-1, keepdim=True)

    alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)                     # (N_rays, N_samples)

    # T_i = prod_{j<i}(1 - alpha_j)  — exclusive cumprod
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]                                                          # (N_rays, N_samples)

    weights = transmittance * alpha                                      # (N_rays, N_samples)
    rgb_map   = (weights.unsqueeze(-1) * rgb).sum(dim=-2)               # (N_rays, 3)
    depth_map = (weights * z_vals).sum(dim=-1)                          # (N_rays,)
    acc_map   = weights.sum(dim=-1)                                     # (N_rays,)

    if white_bg:
        rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1))

    return rgb_map, depth_map, acc_map, weights


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """Generate one ray per pixel given a camera-to-world matrix.

    Camera convention (blender/NeRF): looks down -Z, up is +Y.
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        indexing="xy",
    )
    # Pixel centres → camera-space directions (z = -1 canonical look direction)
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)],
        dim=-1,
    )  # (H, W, 3)

    # Rotate directions into world space (no translation for directions).
    rays_d = (dirs.unsqueeze(-2) @ c2w[:3, :3].T).squeeze(-2)  # (H, W, 3)
    rays_o = c2w[:3, 3].expand_as(rays_d)                       # (H, W, 3)
    return rays_o, rays_d
