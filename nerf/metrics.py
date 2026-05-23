"""PSNR and SSIM for comparing rendered vs ground-truth images."""

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio (dB). Images should be in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0.0:
        return float("inf")
    return -10.0 * (mse + 1e-10) ** 0.5  # avoid log(0)


def psnr_from_mse(mse: float) -> float:
    return 10.0 * (-torch.log10(torch.tensor(mse + 1e-10))).item()


def ssim(
    pred: torch.Tensor,   # (H, W, 3) or (N, H, W, 3)
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Structural similarity index (simplified single-scale)."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    # (N, H, W, 3) → (N, 3, H, W)
    pred   = pred.permute(0, 3, 1, 2).float()
    target = target.permute(0, 3, 1, 2).float()

    # Build Gaussian kernel.
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    C = pred.shape[1]
    kernel = kernel.expand(C, 1, window_size, window_size).to(pred.device)

    pad = window_size // 2
    mu1 = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred   * pred,   kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(pred   * target, kernel, padding=pad, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()
