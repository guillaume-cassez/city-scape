#!/usr/bin/env python3
"""
Aggregate all <epoch_*.pth>.results.json files under checkpoints/ into
a comparison table + markdown summary B vs C.

Usage:
    python scripts/aggregate_results.py --root checkpoints/ --out results/
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def collect(root: Path):
    """Walk checkpoints/ and load all .results.json. Returns list of dicts."""
    records = []
    for jp in sorted(root.rglob("epoch_*.results.json")):
        try:
            r = json.loads(jp.read_text())
        except Exception as e:
            print(f"  skip {jp}: {e}")
            continue
        run = jp.parent.name  # pilot_fullres_<X>_seed<S>
        m = re.match(r"pilot_fullres_([A-D])_[a-z_]+_seed(\d+)", run)
        if not m:
            continue
        r["variant"] = m.group(1)  # A, B, C, or D
        r["seed"] = int(m.group(2))
        records.append(r)
    return records


def fmt_table_per_epoch(records, metric: str = "mIoU"):
    """Pivot table : rows=epoch, cols=(variant, seed)."""
    by_key = defaultdict(dict)
    seeds_seen = sorted({r["seed"] for r in records})
    variants_seen = sorted({r["variant"] for r in records})
    epochs = sorted({r["epoch"] for r in records})

    for r in records:
        by_key[r["epoch"]][(r["variant"], r["seed"])] = r[metric]

    header = ["epoch"] + [f"mean_{v}" for v in variants_seen]
    if "B" in variants_seen and "C" in variants_seen:
        header.append("Δ(C-B)")
    if "A" in variants_seen and "D" in variants_seen:
        header.append("Δ(D-A)")
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for ep in epochs:
        row = [str(ep)]
        per_var = {v: [] for v in variants_seen}
        for v in variants_seen:
            for s in seeds_seen:
                val = by_key[ep].get((v, s))
                if val is not None:
                    per_var[v].append(val)
        means = {}
        for v in variants_seen:
            if per_var[v]:
                m = np.mean(per_var[v])
                means[v] = m
                row.append(f"{m*100:.2f}")
            else:
                means[v] = None
                row.append("—")
        if "B" in means and "C" in means and means["B"] is not None and means["C"] is not None:
            row.append(f"{(means['C']-means['B'])*100:+.2f}")
        elif "Δ(C-B)" in header:
            row.append("—")
        if "A" in means and "D" in means and means["A"] is not None and means["D"] is not None:
            row.append(f"{(means['D']-means['A'])*100:+.2f}")
        elif "Δ(D-A)" in header:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt_2x2_ablation(records, metric: str = "mIoU", target_epoch: int = 10):
    """2x2 ablation table: rows=Dice on/off, cols=Boundary on/off."""
    final = [r for r in records if r["epoch"] == target_epoch]
    by_var = defaultdict(list)
    for r in final:
        by_var[r["variant"]].append(r[metric])

    def m(v):
        return np.mean(by_var[v]) * 100 if by_var.get(v) else None

    A, B, C, D = m("A"), m("B"), m("C"), m("D")

    def fmt(x):
        return f"{x:.2f}" if x is not None else "—"

    lines = [
        f"### 2×2 ablation — {metric} at epoch {target_epoch} (mean across seeds, %)",
        "",
        "|  | sans Boundary | avec Boundary | Δ(Boundary) |",
        "|---|---|---|---|",
        f"| **sans Dice** | A: {fmt(A)} | D: {fmt(D)} | {fmt(D-A) if (A is not None and D is not None) else '—'} |",
        f"| **avec Dice** | B: {fmt(B)} | C: {fmt(C)} | {fmt(C-B) if (B is not None and C is not None) else '—'} |",
        f"| **Δ(Dice)** | {fmt(B-A) if (A is not None and B is not None) else '—'} | {fmt(C-D) if (C is not None and D is not None) else '—'} | — |",
    ]
    return "\n".join(lines)


def fmt_per_class_final(records, classes, target_epoch=10):
    """Per-class IoU at final epoch, mean across seeds, B vs C."""
    final = [r for r in records if r["epoch"] == target_epoch]
    by_var = defaultdict(list)
    for r in final:
        by_var[r["variant"]].append(r["per_class_iou"])

    if "B" not in by_var or "C" not in by_var:
        return "(missing variant)"

    lines = ["| class | B mean | C mean | Δ(C-B) |", "|---|---|---|---|"]
    cls_names = classes
    rows = []
    for c in cls_names:
        b_vals = [d.get(c, np.nan) for d in by_var["B"]]
        c_vals = [d.get(c, np.nan) for d in by_var["C"]]
        b_m = np.nanmean(b_vals) * 100
        c_m = np.nanmean(c_vals) * 100
        rows.append((c, b_m, c_m, c_m - b_m))
    rows.sort(key=lambda x: x[3], reverse=True)
    for c, b, cv, d in rows:
        lines.append(f"| {c} | {b:.2f} | {cv:.2f} | {d:+.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints", help="Checkpoints root dir")
    ap.add_argument("--out", default="results", help="Output dir for tables")
    ap.add_argument("--target-epoch", type=int, default=10, help="Final epoch for per-class summary")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = collect(root)
    if not records:
        print(f"no results found under {root}")
        return
    print(f"loaded {len(records)} result files")

    classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
               "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
               "truck", "bus", "train", "motorcycle", "bicycle"]

    md = []
    md.append("# Cityscapes pilot — Ablation A/B/C/D × 2 seeds × 10 epochs\n")
    md.append("Variants:")
    md.append("- **A** : CE 1.0 (baseline minimal)")
    md.append("- **B** : CE 0.5 + Dice 0.5 (baseline standard)")
    md.append("- **C** : CE 0.4 + Dice 0.4 + Boundary 0.2 (Kervadec sur Dice)")
    md.append("- **D** : CE 0.8 + Boundary 0.2 (Kervadec sans Dice)")
    md.append("")
    md.append(fmt_2x2_ablation(records, "mIoU", args.target_epoch))
    md.append("")
    md.append(fmt_2x2_ablation(records, "boundary_f1_mean", args.target_epoch))
    md.append("")
    md.append(fmt_2x2_ablation(records, "trimap_mIoU_mean", args.target_epoch))
    md.append("\n## mIoU per epoch (mean across seeds)\n")
    md.append(fmt_table_per_epoch(records, "mIoU"))
    md.append("\n## Boundary F1 per epoch (mean across seeds)\n")
    md.append(fmt_table_per_epoch(records, "boundary_f1_mean"))
    md.append("\n## Trimap mIoU per epoch (mean across seeds)\n")
    md.append(fmt_table_per_epoch(records, "trimap_mIoU_mean"))
    md.append(f"\n## Per-class IoU at epoch {args.target_epoch} (sorted by Δ(C-B) desc)\n")
    md.append(fmt_per_class_final(records, classes, args.target_epoch))

    text = "\n".join(md)
    out_md = out / "pilot_comparison.md"
    out_md.write_text(text)
    print(text)
    print(f"\n-> {out_md}")

    # CSV dump for further analysis
    import csv
    out_csv = out / "pilot_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "epoch", "mIoU", "boundary_f1_mean", "trimap_mIoU_mean"])
        for r in sorted(records, key=lambda x: (x["variant"], x["seed"], x["epoch"])):
            w.writerow([r["variant"], r["seed"], r["epoch"], r["mIoU"],
                        r["boundary_f1_mean"], r["trimap_mIoU_mean"]])
    print(f"-> {out_csv}")


if __name__ == "__main__":
    main()
