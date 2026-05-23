"""Synthetic dataset generator — no downloads required.

Creates a Phong-shaded sphere scene with a specular highlight so the
radiance field has genuine view-dependent effects to learn.  Images are
saved in the blender NeRF format (transforms_train/test/val.json + PNGs).

Usage
-----
    python generate_data.py               # default: data/sphere, 100×100
    python generate_data.py --out data/sphere --wh 200 --n_train 100
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Ray-sphere intersection
# ──────────────────────────────────────────────────────────────────────────────

def ray_sphere(
    rays_o: np.ndarray,   # (H*W, 3)
    rays_d: np.ndarray,   # (H*W, 3)
    center: np.ndarray,   # (3,)
    radius: float,
):
    """Analytic ray-sphere intersection.  Returns (hit_mask, t, normal)."""
    oc = rays_o - center[None]          # (N, 3)
    a = (rays_d * rays_d).sum(-1)
    b = 2.0 * (oc * rays_d).sum(-1)
    c = (oc * oc).sum(-1) - radius ** 2
    disc = b * b - 4 * a * c

    hit  = disc >= 0
    t    = np.full(len(rays_o), np.inf)
    t[hit] = (-b[hit] - np.sqrt(disc[hit].clip(0))) / (2 * a[hit])
    # Negative t means the sphere is behind the camera — not a hit.
    t = np.where(t > 0, t, np.inf)
    hit = np.isfinite(t)

    t_safe = np.where(hit, t, 0.0)          # avoid inf*ray_d for missed rays
    pts    = rays_o + t_safe[:, None] * rays_d   # (N, 3)
    normal = (pts - center[None]) / (radius + 1e-8)
    return hit, t, normal


# ──────────────────────────────────────────────────────────────────────────────
# Shading
# ──────────────────────────────────────────────────────────────────────────────

LIGHT_DIR  = np.array([1.0, 1.0, 2.0], dtype=np.float64)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)

# Sphere surface colour: mix of ambient, lambertian diffuse, and Phong specular.
KA = np.array([0.05, 0.05, 0.08])   # ambient
KD = np.array([0.6,  0.3,  0.1])    # diffuse (terracotta)
KS = np.array([0.9,  0.9,  0.9])    # specular (white highlight)
SHININESS = 64.0


def shade(
    normal:  np.ndarray,    # (N, 3)
    view_d:  np.ndarray,    # (N, 3)  pointing toward camera
) -> np.ndarray:            # (N, 3) RGB in [0, 1]
    n  = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
    v  = view_d / (np.linalg.norm(view_d, axis=-1, keepdims=True) + 1e-8)

    diff = np.clip((n * LIGHT_DIR[None]).sum(-1, keepdims=True), 0, 1)

    # Phong reflection direction: r = 2(n·l)n - l
    r = 2 * (n * LIGHT_DIR[None]).sum(-1, keepdims=True) * n - LIGHT_DIR[None]
    spec = np.clip((r * v).sum(-1, keepdims=True), 0, 1) ** SHININESS

    rgb = KA[None] + diff * KD[None] + spec * KS[None]
    return np.clip(rgb, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Camera utilities
# ──────────────────────────────────────────────────────────────────────────────

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """4×4 camera-to-world matrix (blender convention: -Z forward, +Y up)."""
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def spherical_poses(
    n: int,
    radius: float = 4.0,
    elevation_range: tuple[float, float] = (-30.0, 60.0),
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    elev_lo, elev_hi = np.radians(elevation_range[0]), np.radians(elevation_range[1])
    poses = []
    for i in range(n):
        azim = 2 * math.pi * i / n
        elev = rng.uniform(elev_lo, elev_hi)
        eye  = radius * np.array([
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ])
        poses.append(look_at(eye, np.zeros(3), np.array([0.0, 0.0, 1.0])))
    return poses


def spiral_poses(n: int, radius: float = 4.0) -> list[np.ndarray]:
    """Smooth spiral path for rendering — used by render.py."""
    poses = []
    for i in range(n):
        t    = i / n
        azim = 2 * math.pi * t
        elev = np.radians(20.0 + 10.0 * math.sin(2 * math.pi * t))
        eye  = radius * np.array([
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ])
        poses.append(look_at(eye, np.zeros(3), np.array([0.0, 0.0, 1.0])))
    return poses


# ──────────────────────────────────────────────────────────────────────────────
# Image rendering
# ──────────────────────────────────────────────────────────────────────────────

def render_view(
    c2w: np.ndarray,
    W: int,
    H: int,
    focal: float,
    bg_color: np.ndarray = np.ones(3),
) -> np.ndarray:
    """Render one image of the sphere analytically. Returns RGBA (H, W, 4) uint8."""
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -np.ones_like(i),
    ], axis=-1).astype(np.float64)  # (H, W, 3)

    # Rotate to world space.
    rays_d = (dirs.reshape(-1, 3) @ c2w[:3, :3].T)
    rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape).copy()

    hit, t, normal = ray_sphere(rays_o, rays_d, np.zeros(3), 1.0)

    # View direction toward camera
    view_d = -rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)

    rgb = np.tile(bg_color, (len(rays_o), 1))
    alpha = np.zeros(len(rays_o))

    if hit.any():
        rgb[hit]   = shade(normal[hit], view_d[hit])
        alpha[hit] = 1.0

    rgba = np.concatenate([rgb, alpha[:, None]], axis=-1)
    rgba = (rgba * 255).clip(0, 255).astype(np.uint8)
    return rgba.reshape(H, W, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset writer
# ──────────────────────────────────────────────────────────────────────────────

def write_split(
    out_root: Path,
    split: str,
    poses: list[np.ndarray],
    W: int,
    H: int,
    focal: float,
):
    img_dir = out_root / split
    img_dir.mkdir(parents=True, exist_ok=True)

    camera_angle_x = 2 * math.atan(W / (2 * focal))
    frames = []

    for idx, c2w in enumerate(poses):
        fname = f"r_{idx:04d}"
        rgba  = render_view(c2w, W, H, focal)
        Image.fromarray(rgba, "RGBA").save(img_dir / f"{fname}.png")

        frames.append({
            "file_path": f"./{split}/{fname}",
            "transform_matrix": c2w.tolist(),
        })

    meta = {"camera_angle_x": camera_angle_x, "frames": frames}
    with open(out_root / f"transforms_{split}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{split}]  {len(poses)} images → {out_root / split}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default="data/sphere")
    parser.add_argument("--wh",     type=int, default=100, help="image width = height")
    parser.add_argument("--n_train",type=int, default=20)
    parser.add_argument("--n_val",  type=int, default=10)
    parser.add_argument("--n_test", type=int, default=40)
    parser.add_argument("--fov",    type=float, default=60.0, help="horizontal FOV degrees")
    args = parser.parse_args()

    W = H = args.wh
    focal = W / (2 * math.tan(math.radians(args.fov) / 2))
    out   = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic sphere dataset → {out}  ({W}×{H})")
    write_split(out, "train", spherical_poses(args.n_train, seed=0), W, H, focal)
    write_split(out, "val",   spherical_poses(args.n_val,   seed=1), W, H, focal)
    write_split(out, "test",  spherical_poses(args.n_test,  seed=2), W, H, focal)
    print("Done.")


if __name__ == "__main__":
    main()
