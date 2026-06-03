#!/usr/bin/env python3
"""Generate paper figures from per-checkpoint .results.json files.

Outputs (papers/paper1/figures/):
- fig_convergence.png  : mIoU / Bnd F1 / Trimap per epoch, 4 variants, 95% CI
- fig_perclass.png     : per-class IoU at epoch 160, 4 variants, sorted by Δ(D-B)
- table_perclass.csv   : per-class IoU mean across seeds at epoch 160
- table_final.csv      : global metrics mean + 95% CI at epoch 160
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT_ROOT = Path("/media/ser/T7/City_Scape/checkpoints")
OUT_DIR = Path("/home/ser/Bureau/City_Scape/papers/paper1/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "A": ("pilot_fullres_A_ce",         "CE",            "#888888"),
    "B": ("pilot_fullres_B_baseline",   "CE+Dice",       "#1f77b4"),
    "C": ("pilot_fullres_C_boundary",   "CE+Dice+Bnd",   "#2ca02c"),
    "D": ("pilot_fullres_D_ce_boundary","CE+Bnd",        "#d62728"),
}
SEEDS = [42, 123, 456]

CITYSCAPES_CLASSES = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle",
]


def load_run(prefix, seed):
    """Return {epoch: dict_metrics} for a run."""
    run_dir = CKPT_ROOT / f"{prefix}_seed{seed}"
    by_epoch = {}
    for jf in sorted(run_dir.glob("epoch_*.results.json")):
        with open(jf) as f:
            d = json.load(f)
        by_epoch[int(d["epoch"]) + 1] = d
    return by_epoch


def mean_ci(values):
    arr = np.array(values, dtype=float)
    mu = arr.mean()
    if len(arr) <= 1:
        return mu, 0.0
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return mu, 1.96 * se


def build_curves():
    """Returns dict variant -> {metric -> (epochs, mean, ci)}"""
    out = {}
    for v, (prefix, _, _) in VARIANTS.items():
        per_seed = [load_run(prefix, s) for s in SEEDS]
        epochs = sorted(set(per_seed[0].keys()) & set(per_seed[1].keys()) & set(per_seed[2].keys()))
        curves = {}
        for metric in ("mIoU", "boundary_f1_mean", "trimap_mIoU_mean"):
            ep_arr, mu_arr, ci_arr = [], [], []
            for e in epochs:
                vals = [per_seed[i][e][metric] for i in range(3)]
                mu, ci = mean_ci(vals)
                ep_arr.append(e); mu_arr.append(mu); ci_arr.append(ci)
            curves[metric] = (np.array(ep_arr), np.array(mu_arr), np.array(ci_arr))
        out[v] = curves
    return out


def plot_convergence(curves):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharex=True)
    metric_titles = [
        ("mIoU",              "mIoU"),
        ("boundary_f1_mean",  "Boundary F1"),
        ("trimap_mIoU_mean",  "Trimap mIoU"),
    ]
    for ax, (metric, title) in zip(axes, metric_titles):
        for v, (_, label, color) in VARIANTS.items():
            ep, mu, ci = curves[v][metric]
            ax.plot(ep, mu * 100, label=f"{v}: {label}", color=color, linewidth=1.6)
            ax.fill_between(ep, (mu - ci) * 100, (mu + ci) * 100, color=color, alpha=0.15)
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = OUT_DIR / "fig_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_convergence.pdf", bbox_inches="tight")
    print(f"-> {out}")


def per_class_at_final():
    """Returns dict variant -> array(19) mean IoU per class at epoch 160."""
    out = {}
    for v, (prefix, _, _) in VARIANTS.items():
        per_seed_arrays = []
        for s in SEEDS:
            jf = CKPT_ROOT / f"{prefix}_seed{s}" / "epoch_160.results.json"
            with open(jf) as f:
                d = json.load(f)
            pc = d["per_class_iou"]
            if isinstance(pc, dict):
                arr = np.array([pc[c] for c in CITYSCAPES_CLASSES], dtype=float)
            else:
                arr = np.array(pc, dtype=float)
            per_seed_arrays.append(arr)
        out[v] = np.stack(per_seed_arrays, axis=0)
    return out


def plot_perclass(per_class):
    means = {v: per_class[v].mean(axis=0) * 100 for v in VARIANTS}
    delta_DA = means["D"] - means["A"]
    order = np.argsort(delta_DA)[::-1]

    classes_ordered = [CITYSCAPES_CLASSES[i] for i in order]

    fig, ax = plt.subplots(figsize=(13, 5.0))
    x = np.arange(len(classes_ordered))
    width = 0.20
    for i, (v, (_, label, color)) in enumerate(VARIANTS.items()):
        ax.bar(x + (i - 1.5) * width, means[v][order], width=width,
               label=f"{v}: {label}", color=color, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(classes_ordered, rotation=40, ha="right")
    ax.set_ylabel("IoU (%) at epoch 160 (mean over 3 seeds)")
    ax.set_title("Per-class IoU — sorted by Δ(D − A)")
    ax.legend(loc="lower left", ncol=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(min(means["A"].min(), means["B"].min(), means["C"].min(), means["D"].min()) - 3, 101)
    fig.tight_layout()
    out = OUT_DIR / "fig_perclass.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_perclass.pdf", bbox_inches="tight")
    print(f"-> {out}")

    csv_lines = ["class," + ",".join(f"{v}_mean,{v}_std" for v in VARIANTS) + ",delta_D_A,delta_D_B,delta_C_B"]
    for c in CITYSCAPES_CLASSES:
        ci = CITYSCAPES_CLASSES.index(c)
        row = [c]
        for v in VARIANTS:
            arr = per_class[v][:, ci] * 100
            row.append(f"{arr.mean():.2f}"); row.append(f"{arr.std(ddof=1):.2f}")
        row.append(f"{means['D'][ci] - means['A'][ci]:+.2f}")
        row.append(f"{means['D'][ci] - means['B'][ci]:+.2f}")
        row.append(f"{means['C'][ci] - means['B'][ci]:+.2f}")
        csv_lines.append(",".join(row))
    (OUT_DIR / "table_perclass.csv").write_text("\n".join(csv_lines) + "\n")
    print(f"-> {OUT_DIR / 'table_perclass.csv'}")


def final_table(curves):
    """Mean + 95% CI per variant at the highest common epoch (typically 160)."""
    csv_lines = ["variant,name,mIoU_mean,mIoU_ci,bnd_f1_mean,bnd_f1_ci,trimap_mean,trimap_ci"]
    for v, (_, label, _) in VARIANTS.items():
        row = [v, label]
        for metric in ("mIoU", "boundary_f1_mean", "trimap_mIoU_mean"):
            ep, mu, ci = curves[v][metric]
            idx = len(ep) - 1
            row.append(f"{mu[idx]*100:.2f}")
            row.append(f"{ci[idx]*100:.2f}")
        csv_lines.append(",".join(row))
    print(f"[final_table] last common epoch: {int(ep[-1])}")
    (OUT_DIR / "table_final.csv").write_text("\n".join(csv_lines) + "\n")
    print(f"-> {OUT_DIR / 'table_final.csv'}")
    print()
    print("Final epoch 160 (mean +/- 95% CI):")
    for line in csv_lines:
        print("  " + line)


def main():
    print("Loading results from", CKPT_ROOT)
    curves = build_curves()
    plot_convergence(curves)
    per_class = per_class_at_final()
    plot_perclass(per_class)
    final_table(curves)


if __name__ == "__main__":
    main()
