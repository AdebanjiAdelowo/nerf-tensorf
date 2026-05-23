"""Render a spiral fly-around GIF from a trained checkpoint.

Usage
-----
    # Render TensoRF (default)
    python render.py --model tensorf --ckpt outputs/tensorf/weights.pt

    # Render vanilla NeRF
    python render.py --model nerf --ckpt outputs/nerf/weights.pt

    # Side-by-side comparison GIF (both checkpoints required)
    python render.py --side_by_side
"""

import argparse
import math
from pathlib import Path

from nerf.device import resolve_device

import numpy as np
import torch
from PIL import Image

from generate_data import spiral_poses
from nerf.dataset import load_blender_dataset
from nerf.model_nerf import NeRF
from nerf.model_tensorf import TensoRF
from nerf.renderer import get_rays
from train import build_nerf, build_tensorf


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, ckpt_path: str, device: torch.device, near, far):
    aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], device=device)
    if model_name == "nerf":
        model = build_nerf(near, far, device)
    else:
        model = build_tensorf(aabb, near, far, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def frames_to_gif(frames: list[np.ndarray], path: str, fps: int = 15):
    imgs = [Image.fromarray((f * 255).clip(0, 255).astype(np.uint8)) for f in frames]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=False,
    )
    print(f"  Saved → {path}  ({len(imgs)} frames, {fps} fps)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       choices=["nerf", "tensorf"], default="tensorf")
    p.add_argument("--ckpt",        default="")
    p.add_argument("--data",        default="data/sphere")
    p.add_argument("--out",         default="outputs")
    p.add_argument("--wh",          type=int,   default=100)
    p.add_argument("--n_frames",    type=int,   default=60)
    p.add_argument("--fps",         type=int,   default=15)
    p.add_argument("--near",        type=float, default=2.0)
    p.add_argument("--far",         type=float, default=6.0)
    p.add_argument("--chunk",       type=int,   default=4096)
    p.add_argument("--side_by_side",action="store_true",
                   help="render both models and stitch horizontally")
    p.add_argument("--device",      default="")
    return p.parse_args()


def render_spiral(model, H, W, focal, n_frames, chunk, device):
    """Return list of (H, W, 3) float32 numpy arrays."""
    poses  = spiral_poses(n_frames)
    frames = []
    for i, c2w in enumerate(poses):
        c2w_t = torch.tensor(c2w[:3, :4], dtype=torch.float32, device=device)
        with torch.no_grad():
            img = model.render_image(H, W, focal, c2w_t, chunk=chunk)
        frames.append(img.cpu().numpy())
        if (i + 1) % 10 == 0:
            print(f"  Rendered {i+1}/{n_frames} frames")
    return frames


def main():
    args = get_args()

    device = resolve_device(args.device)

    # Get focal length from dataset meta
    _, _, _, meta = load_blender_dataset(
        args.data, split="train", img_wh=(args.wh, args.wh)
    )
    H, W, focal = meta["H"], meta["W"], meta["focal"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.side_by_side:
        # Render both models, stitch frames horizontally.
        models, names = [], []
        for mname in ("nerf", "tensorf"):
            ckpt = f"outputs/{mname}/weights.pt"
            if not Path(ckpt).exists():
                print(f"  Warning: {ckpt} not found, skipping {mname}")
                continue
            print(f"\nRendering {mname.upper()} ...")
            m = load_model(mname, ckpt, device, args.near, args.far)
            models.append(m)
            names.append(mname)

        if len(models) < 2:
            print("Need both checkpoints for side-by-side. Run compare.py first.")
            return

        frames_a = render_spiral(models[0], H, W, focal, args.n_frames, args.chunk, device)
        frames_b = render_spiral(models[1], H, W, focal, args.n_frames, args.chunk, device)

        # Add label strips (simple pixel overlay)
        combined = [np.concatenate([a, b], axis=1) for a, b in zip(frames_a, frames_b)]
        frames_to_gif(combined, str(out_dir / "comparison.gif"), args.fps)

    else:
        ckpt = args.ckpt or f"outputs/{args.model}/weights.pt"
        print(f"\nRendering {args.model.upper()} from {ckpt} ...")
        model  = load_model(args.model, ckpt, device, args.near, args.far)
        frames = render_spiral(model, H, W, focal, args.n_frames, args.chunk, device)
        frames_to_gif(frames, str(out_dir / f"{args.model}_spiral.gif"), args.fps)


if __name__ == "__main__":
    main()
