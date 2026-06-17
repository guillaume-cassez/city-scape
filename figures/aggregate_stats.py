#!/usr/bin/env python3
"""
Aggregate per-seed metrics into the paper's final table + a significance table.

Reads results/pilot_results.csv (repo-local, robust — no dependency on the
Tower's .results.json paths) and writes, under papers/paper2/figures/:
  - table_final.csv         : mean + 95% CI (Student-t, df=n-1) at epoch 160
  - table_significance.csv  : paired-by-seed t-tests for the key comparisons

Statistical notes
-----------------
* n = 3 seeds (42, 123, 456). The 95% CI uses t_{0.975, df=2} = 4.303, NOT 1.96.
* Significance is a paired t-test across seeds (each seed is a matched block),
  which is more powerful than comparing CI-bar overlap. With n=3 the power is
  still low, so non-significant directional effects are reported as such, not as
  evidence of no effect.
* mIoU is unaffected by the boundary/trimap estimator fix, so its numbers are
  final. boundary_f1_mean / trimap_mIoU_mean central values still come from the
  legacy road-vs-rest estimator and will change after re-evaluation with the
  corrected per-class metric (the CI *factor* correction applies regardless).
"""
import csv
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
# Locate pilot_results.csv across layouts: published bundle (figures/ next to data/)
# or the dev repo (results/ at the project root).
_CANDIDATES = [
    HERE.parent / "data" / "pilot_results.csv",       # bundle: repo/data/
    HERE.parents[2] / "results" / "pilot_results.csv",  # dev: papers/paper2/.. -> repo/results
    HERE.parents[3] / "results" / "pilot_results.csv",
    HERE.parent / "results" / "pilot_results.csv",
]
CSV = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
OUT = HERE

EPOCH = 160
SEEDS = ["42", "123", "456"]
VARIANTS = [("A", "CE"), ("B", "CE+Dice"), ("C", "CE+Dice+Bnd"), ("D", "CE+Bnd")]
METRICS = ["mIoU", "boundary_f1_mean", "trimap_mIoU_mean"]


def load():
    rows = list(csv.DictReader(open(CSV)))
    table = {}  # (variant, metric) -> np.array ordered by SEEDS, in %
    for v, _ in VARIANTS:
        for m in METRICS:
            d = {r["seed"]: float(r[m]) for r in rows
                 if r["variant"] == v and int(r["epoch"]) == EPOCH}
            table[(v, m)] = np.array([d[s] for s in SEEDS]) * 100.0
    return table


def ci_t(arr):
    n = len(arr)
    se = arr.std(ddof=1) / np.sqrt(n)
    return stats.t.ppf(0.975, df=n - 1) * se


def write_final(table):
    lines = ["variant,name,mIoU_mean,mIoU_ci,bnd_f1_mean,bnd_f1_ci,trimap_mean,trimap_ci"]
    for v, name in VARIANTS:
        cells = [v, name]
        for m in METRICS:
            a = table[(v, m)]
            cells += [f"{a.mean():.2f}", f"{ci_t(a):.2f}"]
        lines.append(",".join(cells))
    (OUT / "table_final.csv").write_text("\n".join(lines) + "\n")
    print("-> table_final.csv (95% CI = Student-t, df=2)")
    for ln in lines:
        print("   " + ln)


def write_significance(table):
    comparisons = [
        ("D", "A", "mIoU"), ("D", "B", "mIoU"), ("D", "C", "mIoU"), ("B", "A", "mIoU"),
        ("B", "D", "trimap_mIoU_mean"), ("B", "A", "trimap_mIoU_mean"),
        ("D", "A", "boundary_f1_mean"), ("D", "C", "boundary_f1_mean"),
    ]
    lines = ["comparison,metric,delta_mean,t_stat,p_value,significant_0.05"]
    print("\n-> table_significance.csv (paired t-test by seed, n=3, df=2)")
    for a, b, m in comparisons:
        x, y = table[(a, m)], table[(b, m)]
        t, p = stats.ttest_rel(x, y)
        delta = (x - y).mean()
        sig = "yes" if p < 0.05 else "no"
        lines.append(f"{a}-{b},{m},{delta:+.3f},{t:.3f},{p:.4f},{sig}")
        print(f"   {a}-{b:2} {m:18} Δ={delta:+.3f}  t={t:6.3f}  p={p:.4f}  {sig}")
    (OUT / "table_significance.csv").write_text("\n".join(lines) + "\n")


def main():
    table = load()
    write_final(table)
    write_significance(table)


if __name__ == "__main__":
    main()
