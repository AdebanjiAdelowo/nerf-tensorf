"""Train both models, compare PSNR / SSIM / speed, save a results table.

Usage
-----
    python compare.py                        # defaults
    python compare.py --iters 10000 --wh 80  # faster smoke-test
"""

import argparse
import json
from pathlib import Path

import train as trainer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default="data/sphere")
    p.add_argument("--out",       default="outputs")
    p.add_argument("--iters",     type=int, default=20_000)
    p.add_argument("--batch",     type=int, default=2048)
    p.add_argument("--wh",        type=int, default=100)
    p.add_argument("--near",      type=float, default=2.0)
    p.add_argument("--far",       type=float, default=6.0)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--eval_every",type=int, default=5000)
    p.add_argument("--device",    default="")
    return p.parse_args()


def main():
    args = get_args()
    results = {}

    for model_name in ("nerf", "tensorf"):
        # Patch the model field and hand off to train.train()
        import copy, sys
        train_args = copy.copy(args)
        train_args.model      = model_name
        train_args.seed       = 42
        train_args.tv_weight  = 1e-4
        train_args.lr         = 5e-4

        log = trainer.train(train_args)
        results[model_name] = log

    # ── Print comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Metric':<30} {'NeRF':>12} {'TensoRF':>12}")
    print("=" * 60)

    rows = [
        ("Parameters",      "param_count",  lambda v: f"{v:,}"),
        ("PSNR (dB)",       "final_psnr",   lambda v: f"{v:.2f}"),
        ("SSIM",            "final_ssim",   lambda v: f"{v:.4f}"),
        ("Total time (s)",  "total_time",   lambda v: f"{v:.1f}"),
        ("ms / iteration",  "avg_iter_ms",  lambda v: f"{v:.1f}"),
    ]
    for label, key, fmt in rows:
        nv = results["nerf"].get(key, float("nan"))
        tv = results["tensorf"].get(key, float("nan"))
        print(f"  {label:<30} {fmt(nv):>12} {fmt(tv):>12}")

    print("=" * 60)

    speedup = results["nerf"]["avg_iter_ms"] / max(results["tensorf"]["avg_iter_ms"], 1e-9)
    print(f"\n  TensoRF is {speedup:.1f}× faster per iteration than NeRF.")

    # Save JSON for the README badge
    out_path = Path(args.out) / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip non-serialisable per-iter times to keep the file small.
    save = {
        k: {kk: vv for kk, vv in v.items() if kk != "time_per_iter"}
        for k, v in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
