"""Device selection shared by all scripts.

Priority: explicit --device arg > CUDA > MPS > CPU.

MPS note
--------
PyTorch's MPS backend does not yet implement `grid_sampler_2d_backward`.
We enable the CPU fallback env var (PYTORCH_ENABLE_MPS_FALLBACK=1) at
import time so that unsupported MPS ops silently fall back to CPU.
This means TensoRF's grid_sample calls use CPU for the backward pass
on Apple Silicon while the rest of the compute (MLP, ray marching) runs
on the GPU.  On CUDA (Kaggle, Colab) everything runs natively.
"""

import os
import warnings

# Must be set before any MPS op is dispatched — module-level import is safe.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# Suppress the per-call warning that fires every backward pass on MPS.
# The fallback is intentional and expected on Apple Silicon.
warnings.filterwarnings(
    "ignore",
    message=".*grid_sampler_2d_backward.*not currently supported on the MPS backend.*",
)


def resolve_device(preference: str = "") -> torch.device:
    """Return the best available device.

    Parameters
    ----------
    preference : str
        If non-empty, used as-is (e.g. "cuda", "mps", "cpu", "cuda:1").
    """
    if preference:
        d = torch.device(preference)
        print(f"[device] forced → {d}")
        return d

    if torch.cuda.is_available():
        d = torch.device("cuda")
    elif torch.backends.mps.is_available():
        d = torch.device("mps")
    else:
        d = torch.device("cpu")

    print(f"[device] auto → {d}")
    return d
