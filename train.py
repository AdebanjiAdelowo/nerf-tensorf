"""Unified training entry-point for both NeRF and TensoRF.

Usage
-----
    # Train vanilla NeRF
    python train.py --model nerf --data data/sphere --iters 30000

    # Train TensoRF
    python train.py --model tensorf --data data/sphere --iters 15000

    # Full comparison (both models, prints metrics table)
    python compare.py
"""

import argparse
import math
import time
from pathlib import Path

# nerf.device sets PYTORCH_ENABLE_MPS_FALLBACK before torch is used.
from nerf.device import resolve_device

import torch
import torch.nn.functional as F

from nerf.dataset import load_blender_dataset
from nerf.metrics import psnr_from_mse, ssim
from nerf.model_nerf import NeRF
from nerf.model_tensorf import TensoRF


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   choices=["nerf", "tensorf"], default="tensorf")
    p.add_argument("--data",    default="data/sphere")
    p.add_argument("--out",     default="outputs")
    p.add_argument("--iters",   type=int,   default=20_000)
    p.add_argument("--batch",   type=int,   default=2048)
    p.add_argument("--lr",      type=float, default=5e-4)
    p.add_argument("--wh",      type=int,   default=100)
    p.add_argument("--near",    type=float, default=2.0)
    p.add_argument("--far",     type=float, default=6.0)
    p.add_argument("--tv_weight", type=float, default=1e-4,
                   help="TensoRF TV regularisation weight")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--device",  default="")   # empty → auto
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=2000)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model factories
# ──────────────────────────────────────────────────────────────────────────────

def build_nerf(near: float, far: float, device: torch.device) -> NeRF:
    return NeRF(
        n_coarse=64, n_fine=128,
        near=near, far=far,
        white_bg=True,
        pos_freqs=10, dir_freqs=4,
        hidden=256, skip_at=4,
    ).to(device)


def build_tensorf(aabb: torch.Tensor, near: float, far: float, device: torch.device) -> TensoRF:
    return TensoRF(
        aabb=aabb,
        grid_size=128,
        R_sigma=16,
        R_feat=8,      # output feature dim = 3*feat_ch=48; R_feat only affects grid params
        feat_ch=16,
        dir_freqs=4,
        near=near,
        far=far,
        white_bg=True,
        n_samples=128,
        pt_chunk=16384,
    ).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args) -> dict:
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)

    print(f"\n{'='*60}")
    print(f"  Model : {args.model.upper()}")
    print(f"  Device: {device}")
    print(f"  Iters : {args.iters:,}")
    print(f"{'='*60}")

    # ── Data ────────────────────────────────────────────────────────────────
    img_wh = (args.wh, args.wh)
    rays_o, rays_d, rgbs, meta = load_blender_dataset(
        args.data, split="train", img_wh=img_wh, device=device
    )
    rays_o_val, rays_d_val, rgbs_val, _ = load_blender_dataset(
        args.data, split="val", img_wh=img_wh, device=device
    )

    N_train = rays_o.shape[0]
    print(f"  Train rays: {N_train:,}  |  Val rays: {rays_o_val.shape[0]:,}")

    focal = meta["focal"]
    H, W  = meta["H"], meta["W"]

    # ── Model ────────────────────────────────────────────────────────────────
    # AABB: conservatively cover the sphere (radius 1) plus near/far range
    aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], device=device)

    if args.model == "nerf":
        model     = build_nerf(args.near, args.far, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / args.iters))
    else:
        model = build_tensorf(aabb, args.near, args.far, device)
        # TensoRF uses a higher lr for grid params than for the MLP
        grid_params = [p for n, p in model.named_parameters() if "mlp" not in n]
        mlp_params  = [p for n, p in model.named_parameters() if "mlp" in n]
        optimizer = torch.optim.Adam([
            {"params": grid_params, "lr": 0.02},
            {"params": mlp_params,  "lr": args.lr},
        ])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / args.iters))

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # ── Training ─────────────────────────────────────────────────────────────
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {"model": args.model, "psnr_history": [], "time_per_iter": []}
    t0  = time.perf_counter()

    for step in range(1, args.iters + 1):
        model.train()
        idx = torch.randint(0, N_train, (args.batch,), device=device)
        ro, rd, gt = rays_o[idx], rays_d[idx], rgbs[idx]

        t_iter = time.perf_counter()
        out = model.render_rays(ro, rd, perturb=True)

        loss = F.mse_loss(out["rgb_fine"], gt)
        if args.model == "tensorf":
            loss = loss + args.tv_weight * model.tv_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        dt = time.perf_counter() - t_iter
        log["time_per_iter"].append(dt)

        if step % args.log_every == 0:
            psnr_v = psnr_from_mse(F.mse_loss(out["rgb_fine"].detach(),
                                               gt.detach()).item())
            elapsed = time.perf_counter() - t0
            print(f"  step {step:>6d}/{args.iters}  "
                  f"loss {loss.item():.4f}  "
                  f"PSNR {psnr_v:.2f} dB  "
                  f"elapsed {elapsed:.1f}s")
            log["psnr_history"].append((step, psnr_v))

        # ── Validation pass ──────────────────────────────────────────────────
        if step % args.eval_every == 0 or step == args.iters:
            model.eval()
            with torch.no_grad():
                val_psnrs, val_ssims = [], []
                # Evaluate one image at a time to stay in memory.
                n_val_imgs = len(meta["all_c2w"])
                for vi in range(min(n_val_imgs, 5)):
                    c2w = meta["all_c2w"][vi].to(device)
                    pred_img = model.render_image(H, W, focal, c2w)
                    gt_img   = meta["all_images"][vi].to(device)
                    mse_v    = F.mse_loss(pred_img, gt_img).item()
                    val_psnrs.append(psnr_from_mse(mse_v))
                    val_ssims.append(ssim(pred_img.cpu(), gt_img.cpu()))

            mean_psnr = sum(val_psnrs) / len(val_psnrs)
            mean_ssim = sum(val_ssims) / len(val_ssims)
            print(f"  [VAL step {step}]  PSNR {mean_psnr:.2f} dB  SSIM {mean_ssim:.4f}")

            if step == args.iters:
                log["final_psnr"] = mean_psnr
                log["final_ssim"] = mean_ssim

    total_time = time.perf_counter() - t0
    log["total_time"]    = total_time
    log["avg_iter_ms"]   = 1000 * sum(log["time_per_iter"]) / len(log["time_per_iter"])
    log["param_count"]   = param_count

    print(f"\n  Done in {total_time:.1f}s  "
          f"({log['avg_iter_ms']:.1f} ms/iter)  "
          f"final PSNR {log.get('final_psnr', float('nan')):.2f} dB")

    torch.save(model.state_dict(), out_dir / "weights.pt")
    return log


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    train(args)
