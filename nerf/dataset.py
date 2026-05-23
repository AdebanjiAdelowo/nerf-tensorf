"""Blender-format NeRF dataset loader.

Reads the transforms_*.json files produced by the Blender NeRF addon
(and our synthetic generator), then pre-computes all rays.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .renderer import get_rays


def load_blender_dataset(
    root: str | Path,
    split: str = "train",
    img_wh: tuple[int, int] = (100, 100),
    white_bg: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """Load a blender-format dataset split.

    Returns
    -------
    rays_o  : (N_rays_total, 3)
    rays_d  : (N_rays_total, 3)
    rgbs    : (N_rays_total, 3)
    meta    : dict with focal, H, W, all_c2w, all_images
    """
    root = Path(root)
    W, H = img_wh

    with open(root / f"transforms_{split}.json") as f:
        meta = json.load(f)

    focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])

    all_rays_o, all_rays_d, all_rgbs = [], [], []
    all_c2w, all_images = [], []

    for frame in meta["frames"]:
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)[:3, :4]

        img_path = root / (frame["file_path"] + ".png")
        img = Image.open(img_path).convert("RGBA").resize((W, H), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0

        if white_bg:
            # Blend onto white background using alpha channel.
            rgb = img[..., :3] * img[..., 3:] + (1.0 - img[..., 3:])
        else:
            rgb = img[..., :3]

        rgb_t = torch.from_numpy(rgb).reshape(-1, 3)

        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_rgbs.append(rgb_t)
        all_c2w.append(c2w)
        all_images.append(torch.from_numpy(rgb))

    rays_o = torch.cat(all_rays_o).to(device)
    rays_d = torch.cat(all_rays_d).to(device)
    rgbs   = torch.cat(all_rgbs).to(device)

    return rays_o, rays_d, rgbs, {
        "focal": focal,
        "H": H,
        "W": W,
        "all_c2w": all_c2w,
        "all_images": all_images,
    }
