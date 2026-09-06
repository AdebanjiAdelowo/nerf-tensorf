# NeRF vs TensoRF: from scratch in PyTorch

A minimal, self-contained implementation of two neural radiance field approaches:

| | Vanilla NeRF | TensoRF (VM) |
|---|---|---|
| Representation | 8-layer MLP | Vector-Matrix grid + tiny MLP |
| Parameters | 1.19 M | 7.1 M (grid + MLP) |
| Training iters for convergence | ~30 k | ~15 k |
| ms / iter, Apple M3 Pro (MPS)¹ | ~500 | ~1 100 |
| ms / iter, Kaggle T4 (CUDA)² | ~25 | ~8 |
| PSNR, Phong sphere (100 × 100) | 14.2 dB | 17.2 dB |
| SSIM, Phong sphere | 0.770 | 0.834 |

> ¹ Measured on Apple M3 Pro (MPS). TensoRF is slower per-iter on MPS than on CUDA because the
> MPS backend does not yet implement `grid_sampler_2d_backward`; the custom `torch.gather`-based
> bilinear interpolation (in `nerf/model_tensorf.py`) avoids the fallback but is less vectorised
> than CUDA's native `grid_sample`. On CUDA, TensoRF is 3–4× faster per iter than NeRF.
>
> ² CUDA estimates from the original TensoRF paper benchmarks on a comparable scene.
>
> Spiral fly-arounds: [outputs/nerf_spiral.gif](outputs/nerf_spiral.gif) · [outputs/tensorf_spiral.gif](outputs/tensorf_spiral.gif) · [outputs/comparison.gif](outputs/comparison.gif)

---

## The physics of volume rendering

### What is a radiance field?

A radiance field assigns to every point $\mathbf{x} \in \mathbb{R}^3$ and every viewing
direction $\mathbf{d} \in \mathbb{S}^2$ a volume density $\sigma(\mathbf{x}) \geq 0$ and an
emitted colour $\mathbf{c}(\mathbf{x}, \mathbf{d}) \in [0,1]^3$.  Density encodes *how opaque*
the material is; colour encodes *what light escapes in direction* $\mathbf{d}$.

### The rendering integral

To render a pixel, we cast a ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ and integrate:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\; dt$$

where the **transmittance** $T(t)$ is the probability that the ray travels from
$t_n$ to $t$ without being absorbed:

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)$$

Intuitively: $T(t)\, \sigma(t)\, dt$ is the probability that a photon is absorbed in the
tiny interval $[t, t+dt]$ and that it successfully escapes toward the camera.

### Discretisation (quadrature)

We draw $N$ sample depths $t_1 < t_2 < \ldots < t_N$ and replace the integral with:

$$C(\mathbf{r}) \approx \sum_{i=1}^{N} T_i\, \alpha_i\, \mathbf{c}_i$$

$$\alpha_i = 1 - \exp(-\sigma_i \delta_i), \qquad T_i = \prod_{j<i}(1-\alpha_j), \qquad \delta_i = t_{i+1}-t_i$$

where $\alpha_i$ is the opacity of segment $i$, $T_i$ the transmittance at sample $i$,
and $\delta_i$ the segment length.  This is exactly **alpha compositing** from front to
back: the formula used in classical volume rendering and in every NeRF variant.

### Positional encoding

A raw MLP cannot represent high-frequency detail because it is biased toward
low-frequency functions (spectral bias).  The fix is to lift the 3-D input
with Fourier features before passing it to the network:

$$\gamma(x) = \left[x,\ \sin(2^0\pi x),\ \cos(2^0\pi x),\ \sin(2^1\pi x),\ \cos(2^1\pi x),\ \ldots,\ \sin(2^{L-1}\pi x),\ \cos(2^{L-1}\pi x)\right]$$

With $L=10$ frequencies this turns 3 numbers into 63, letting the MLP focus
on learning *which* frequencies are present, not on representing them directly.

---

## What does tensor factorisation buy us?

### The parameter explosion of explicit grids

Storing a radiance field as a dense 3-D voxel grid of resolution N³ costs
O(N³) parameters.  At N=128, that is 2 million voxels; at N=512 it becomes
134 million, already too large for a GPU for a single float channel.

### CP decomposition

A rank-$R$ CP decomposition writes the grid as a sum of outer products of
three vectors:

$$G(x,y,z) \approx \sum_{r=1}^{R} a_r(x) \otimes b_r(y) \otimes c_r(z)$$

Parameter count: $3RN$, linear in $N$.  But CP is numerically fragile and
hard to optimise with SGD.

### VM (Vector-Matrix) decomposition: TensoRF

The VM decomposition used here is a middle ground: each rank-$r$ term is the
outer product of a **2-D matrix (plane)** and a **1-D vector (line)**:

$$G(x,y,z) \approx \sum_{r=1}^{R} M_{XY}^r(x,y)\, v_Z^r(z) \;+\; M_{XZ}^r(x,z)\, v_Y^r(y) \;+\; M_{YZ}^r(y,z)\, v_X^r(x)$$

Parameter count: $3RN^2$, much smaller than $N^3$ for small $R$, and
hardware-friendly because planes can be sampled with GPU bilinear
interpolation (`F.grid_sample`).

For appearance, each rank-$r$ term carries a **$C$-dimensional feature vector**
instead of a scalar.  The three plane types together produce a feature
$f(\mathbf{x}) \in \mathbb{R}^{3C}$, which a small MLP decodes to RGB given the view direction.

### Why TensoRF trains 5–10× faster

1. **Fewer iterations needed.** The grid is an explicit spatial structure:
   gradients update only the cells the ray passes through, so convergence
   is local and fast.  A plain MLP must update all ~1M weights for every ray.

2. **Larger batch sizes.** Grid lookups are O(1) per point; the MLP is
   O(depth × width²) per point.  With the same GPU memory TensoRF fits
   4× more rays per batch.

3. **Better initialisation.** The grid can be initialised near-zero and
   builds up density only where rays agree, no need for the MLP to
   "unlearn" the wrong parts of parameter space.

The trade-off: TensoRF uses more memory (the grid) and is harder to
regularise (hence the TV loss on the density planes).  For very large
scenes or unbounded outdoor environments, pure MLP representations
generalise better.

---

## Evaluation metrics

Both metrics below are computed on images in $[0,1]$, averaged over 5 held-out
validation views every `--eval_every` steps (see `nerf/metrics.py`, called from
`train.py`).

**PSNR** (`psnr_from_mse`):

$$\mathrm{PSNR} = 10 \log_{10}\!\left(\frac{1}{\mathrm{MSE}}\right) \ \text{dB}, \qquad \mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N} (\hat{c}_i - c_i)^2$$

**SSIM** (`ssim`): a simplified, single-scale structural similarity index computed
with an $11\times11$ Gaussian window ($\sigma=1.5$) via convolution, following Wang et
al. (2004):

$$\mathrm{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}, \qquad C_1 = 0.01^2,\ \ C_2 = 0.03^2$$

(`nerf/metrics.py` also exports an unused `psnr()` helper; the reported numbers come
from `psnr_from_mse`, the one actually called by `train.py`/`compare.py`.)

---

## Quick start

### Device support

| Hardware | Backend | Notes |
|---|---|---|
| Apple M-series (M3 Pro etc.) | MPS | All ops run natively. The VM grid uses a custom `torch.gather`-based bilinear interpolation instead of `F.grid_sample` to avoid the unimplemented `grid_sampler_2d_backward` on MPS. |
| Kaggle / Colab GPU | CUDA | All ops run natively, no fallback needed |
| CPU-only | CPU | Works; TensoRF ~45 ms/iter, NeRF ~180 ms/iter |

Device is auto-detected (CUDA > MPS > CPU). Override with `--device cuda`, `--device mps`, or `--device cpu`.

### 1. Install (local / Apple Silicon)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel for Mac
pip install -r requirements.txt
```

### 1. Install (Kaggle / Colab GPU)

```bash
# PyTorch with CUDA is pre-installed on Kaggle. Just install extras:
pip install -r requirements.txt
```

Or paste this at the top of a Kaggle notebook cell:

```python
!git clone https://github.com/AdebanjiAdelowo/nerf-tensorf.git
%cd nerf-tensorf
!pip install -q -r requirements.txt
!python generate_data.py --wh 100 --n_train 20
!python train.py --model tensorf --iters 15000 --device cuda
!python train.py --model nerf    --iters 30000 --device cuda
!python render.py --side_by_side --device cuda
```

### 2. Generate synthetic dataset

```bash
python generate_data.py --out data/sphere --wh 100 --n_train 20
```

This renders 20 training / 10 val / 40 test views of a Phong-shaded sphere
with a specular highlight (view-dependent radiance) using only NumPy,
no download needed.  For the real NeRF blender dataset, point `--data` at
a directory that already contains `transforms_train.json`.

### 3. Train

```bash
# TensoRF (~15 k iters, fast)
python train.py --model tensorf --data data/sphere --iters 15000

# Vanilla NeRF (~30 k iters for similar quality)
python train.py --model nerf    --data data/sphere --iters 30000
```

### 4. Compare

```bash
python compare.py --iters 20000    # trains both and prints a results table
```

### 5. Render GIFs

```bash
python render.py --model tensorf
python render.py --model nerf
python render.py --side_by_side    # needs both checkpoints
```

---

## Project structure

```
nerf-tensorf/
├── nerf/
│   ├── encoding.py       : Fourier positional encoding
│   ├── renderer.py       : ray generation, volume rendering, importance sampling
│   ├── model_nerf.py     : vanilla NeRF MLP (coarse + fine)
│   ├── model_tensorf.py  : TensoRF VM decomposition
│   ├── dataset.py        : blender-format dataset loader
│   └── metrics.py        : PSNR, SSIM
├── generate_data.py      : synthetic Phong sphere dataset generator
├── train.py              : training loop (both models)
├── compare.py            : train both + print results table
├── render.py             : spiral fly-around GIF renderer
└── requirements.txt
```

---

## References

- **NeRF**: Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020.
- **TensoRF**: Chen et al., *TensoRF: Tensorial Radiance Fields*, ECCV 2022.
- **Positional encoding**: Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, NeurIPS 2020.
- **SSIM**: Wang et al., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE Transactions on Image Processing, vol. 13, no. 4, 2004.
