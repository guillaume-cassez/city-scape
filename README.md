# city-scape

**Boundary Loss Ablation for Full-Resolution Cityscapes Segmentation: When Dice Helps and When It Doesn't**

Code, configs, per-epoch metrics, figures, and paper source for a controlled 2×2 loss ablation on Cityscapes at native resolution (1024×2048) with ConvNeXt-V2-Base + UPerNet.

[![paper page](https://img.shields.io/badge/%F0%9F%93%84_paper_landing-guillaume--cassez.fr-blue)](https://guillaume-cassez.fr/voiture-autonome/cityscapes/boundary-loss-kervadec/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20528680.svg)](https://doi.org/10.5281/zenodo.20528680)

> **v0.2.0 erratum.** This release corrects two estimator bugs in v0.1.0: Boundary F1 / Trimap IoU now run **per class** (v0.1.0 measured only the road-vs-rest contour) and **all twelve runs were re-evaluated on the val set**, and 95% CIs use the **Student-t** factor instead of 1.96 (n=3), with Holm-corrected significance. **mIoU is unchanged and is now validated bit-for-bit against the official `cityscapesscripts` evaluation.** The "D is best on mIoU" headline is softened — D significantly beats C (p=0.007) but is a statistical tie with plain CE (A) at n=3. The corrected contour metrics split along the Dice axis (non-Dice A, D lead Boundary F1; Dice B, C lead Trimap IoU), so the v0.1.0 contour numbers should not be cited. Details in [CHANGELOG.md](CHANGELOG.md).

---

## The C-then-D crossover

The setup is narrow on purpose: take the canonical CE + Dice recipe used on Cityscapes, drop Kervadec's MIDL 2019 boundary term on top, and watch four variants (A: CE, B: CE+Dice, C: CE+Dice+Bnd, D: CE+Bnd) for 160 epochs across three seeds each — twelve runs total, evaluated at every checkpoint. At ten epochs, variant C beats B by **+2.26 mIoU**. A short pilot would stop there and call it: ship the Dice + Boundary recipe.

That answer stops being right past epoch 100. Letting all four variants run their full course reshuffles the ranking. Variant D (CE + Boundary, **no Dice**) overtakes everyone from epoch 110 onwards and ends at **81.69 ± 0.25 mIoU** and **77.32 ± 0.13 Boundary F1** — both the highest of the four in the mean. A paired-by-seed test (n=3, Holm-corrected) makes D's win over the joint variant C significant (p=0.007), while D vs plain CE (A) is a statistical tie (p=0.095); C is dethroned at convergence. The two contour metrics split along the Dice axis: the non-Dice variants (A, D) lead **Boundary F1** (D − B = +0.93; Holm p ≤ 0.011) while the Dice variants (B, C) lead **Trimap IoU** (B − D = +1.57, C − D = +1.62; Holm p ≤ 0.012) — Dice trades edge sharpness for intra-region coherence near contours.

The headline is not "use D"; it's the **methodological warning** below it. The C−B gap swings by 2.7 mIoU points between the two training horizons — a magnitude that easily flips a production decision. Per-class breakdown explains the swing: D wins big on large structured classes (truck +5.29, wall +3.73, bus +2.37 mIoU vs B), while B retains thin signal-rich classes (traffic light +2.10, train +1.10, traffic sign +0.86 mIoU vs D). The Dice term works as early regularisation that holds back late-stage convergence on the bulk classes — and bulk classes dominate mIoU, so the long-training picture flips on the global metric.

---

## TL;DR

1. **At 10 epochs on Cityscapes val**, the joint formulation C (CE + Dice + Boundary) leads (mIoU 78.17, +2.26 over B in the mean). A short ablation would recommend it.

2. **At 160 epochs**, D (CE + Boundary, no Dice) has the highest mean mIoU (81.69 ± 0.25) and Boundary F1 (77.32 ± 0.13). Significance (paired by seed, n=3, Holm-corrected): D > C is significant (p=0.007), D > A is not (p=0.095). The contour metrics split along the Dice axis — the non-Dice variants (A, D) lead Boundary F1, the Dice variants (B 53.82, C 53.87) lead Trimap IoU by ≈ +1.6 over A and D (Holm p ≤ 0.012). The Dice term acts as early regularisation that does not translate into long-training gains on the global metric.

3. **Per-class breakdown explains the swing.** D dominates large structured classes (truck +5.29, wall +3.73, bus +2.37 mIoU vs B) where the SDT field provides a consistent gradient. B preserves thin signal-rich classes (traffic light +2.10, train +1.10, traffic sign +0.86 mIoU vs D) where Dice anchors small footprints against CE class imbalance.

4. **Methodological takeaway.** Short-epoch ablations are systematically misleading on this task. A study comparing loss recipes for Cityscapes at ≤ 20 epochs reverses the ranking that holds at 160 epochs. Loss-recipe decisions should be made on at least 80–100 epochs of training.

→ For a deployed Cityscapes model: D is the pragmatic default — simplest of the four (no Dice plumbing, no hyperparameter), highest mean mIoU (significantly above C, tied with plain CE), predictable on large structured classes. If near-boundary region coherence (Trimap IoU) or thin signal-rich classes matter more, prefer a Dice variant (B or C — significant Trimap lead, ≈ +1.6 over the non-Dice variants).

---

## Try the boundary loss on your own pipeline

The Kervadec boundary loss is parameter-free given the signed distance transform (SDT) of the ground truth. Pre-compute per-class EDT once offline, then add as a single term to your existing CE / CE+Dice pipeline:

```python
import torch
from scipy.ndimage import distance_transform_edt
import numpy as np

def signed_distance_transform(mask: np.ndarray) -> np.ndarray:
    """Per-class signed DT. mask shape (C, H, W) one-hot binary.
    Returns float32 array of same shape, negative inside region, positive outside.
    """
    sdt = np.empty_like(mask, dtype=np.float32)
    for c in range(mask.shape[0]):
        m = mask[c].astype(bool)
        if m.any() and not m.all():
            inside  = distance_transform_edt(m)
            outside = distance_transform_edt(~m)
            sdt[c]  = -inside + outside
        else:
            sdt[c] = 0
    # Normalize to [-1, 1] per class
    abs_max = np.abs(sdt).reshape(sdt.shape[0], -1).max(axis=1) + 1e-7
    sdt /= abs_max[:, None, None]
    return sdt

def kervadec_boundary_loss(logits: torch.Tensor, sdt: torch.Tensor) -> torch.Tensor:
    """Kervadec et al. MIDL 2019.
    logits : (B, C, H, W)   raw network output
    sdt    : (B, C, H, W)   per-class signed distance transform, normalised to [-1, 1]
    Returns scalar.
    """
    probs = torch.softmax(logits, dim=1)
    return (probs * sdt).mean()
```

Combine as `loss = ce(logits, target) + 0.2 * kervadec_boundary_loss(logits, sdt)`.

---

## Repository layout

```
├── paper.md                Paper source (English, Markdown)
├── paper.pdf               Compiled paper EN (Pandoc + XeLaTeX)
├── paper_fr.md             Paper source (French, Markdown)
├── paper_fr.pdf            Compiled paper FR (Pandoc + XeLaTeX)
├── README.md               This file
├── LICENSE                 MIT
├── CITATION.cff            Machine-readable citation
├── scripts/                Training, evaluation, aggregation
│   ├── train.py            Main training script (Hydra config, BF16, auto-resume)
│   ├── evaluate.py         Per-checkpoint offline evaluation (corrected per-class metrics)
│   ├── reeval_official.py  Re-eval: official cityscapesscripts mIoU + corrected contour
│   ├── aggregate_official.py   Corrected tables (Student-t CI + Holm tests) from the re-eval
│   ├── evaluate_consensus.py   Multi-seed vote + variant-pair CC veto evaluation
│   └── aggregate_results.py
├── src/                    Corrected metric + consensus code (self-contained)
│   ├── metrics/segmentation_metrics.py   mIoU + per-class Boundary F1 / Trimap IoU
│   └── postprocessing/consensus.py       CC consensus filter (BRATS-adapted)
├── tests/
│   ├── test_metrics_consensus.py   Synthetic unit tests (19/19) for the metric/consensus fixes
│   └── test_official_miou.py       mIoU == official cityscapesscripts (bit-for-bit)
├── data/                   Per-run CSV exports (no raw data)
│   ├── pilot_results.csv   (variant, seed, epoch) -> (mIoU, boundary_f1, trimap), per epoch
│   └── pilot_results_official.csv   epoch-160 official mIoU + corrected contour + per-class
└── figures/                Paper figures + generation/aggregation scripts
    ├── fig_convergence.{png,pdf}    Per-epoch mIoU, 95% Student-t CI
    ├── fig_perclass.{png,pdf}       Per-class IoU at epoch 160, 4 variants
    ├── table_final.csv               Global metrics + Student-t 95% CI at epoch 160 (corrected)
    ├── table_significance.csv        Holm-corrected paired-by-seed t-tests (all pairs)
    ├── table_perclass.csv            Per-class IoU + std at epoch 160 (official)
    ├── aggregate_stats.py            (legacy) rebuild tables from a per-epoch CSV
    └── generate_figures.py           (legacy) 3-panel figures from .results.json
```

Cityscapes raw images are **not** redistributed (Cityscapes Dataset Terms of Use). Obtain them from [cityscapes-dataset.com](https://www.cityscapes-dataset.com/).

---

## Reproduce the numbers

```bash
# 1. Clone and install deps
git clone https://github.com/guillaume-cassez/city-scape
cd city-scape
pip install -r requirements.txt   # see https://github.com/guillaume-cassez/city-scape for the file

# 2. (Optional) Re-train any variant (requires Cityscapes fine annotations + ~28 h on a 96 GB GPU)
python scripts/train.py --config-name=config +experiment=pilot_fullres_D_ce_boundary training.epochs=160 "experiment.seeds=[42,123,456]"

# 3. Evaluate a single checkpoint
python scripts/evaluate.py --checkpoint checkpoints/pilot_fullres_D_ce_boundary_seed42/epoch_160.pth

# 4. Aggregate all .results.json into the comparison tables
python scripts/aggregate_results.py
# -> results/pilot_results.csv (this repo's data/)
# -> results/pilot_comparison.md

# 5. Regenerate all paper figures from the per-checkpoint .results.json files
python figures/generate_figures.py
# -> figures/fig_convergence.{png,pdf}
# -> figures/fig_perclass.{png,pdf}
# -> figures/table_final.csv
# -> figures/table_perclass.csv
```

The CSV in `data/pilot_results.csv` is the ground truth that powers every figure and table in the paper; the figure-generation script reads it (plus the per-checkpoint `.results.json` files for per-class breakdowns) and is self-contained.

---

## Cite

```bibtex
@misc{cassez2026cityscape,
  title  = {Boundary Loss Ablation for Full-Resolution Cityscapes Segmentation:
           When Dice Helps and When It Doesn't},
  author = {Cassez, Guillaume},
  year   = {2026},
  url    = {https://guillaume-cassez.fr/voiture-autonome/cityscapes/boundary-loss-kervadec/},
  doi    = {10.5281/zenodo.20528680},
  version = {0.2.0},
  note   = {Independent research}
}
```

Permanent archive (concept DOI, always latest version): [10.5281/zenodo.20528680](https://doi.org/10.5281/zenodo.20528680). Version history and erratum: [CHANGELOG.md](CHANGELOG.md).

---

## About the author

I'm **Guillaume Cassez** ([ORCID 0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931)), and I built this project in 2026 as **independent research** (outside any institutional framework). This is the second in a series of self-contained research reports on training-loss design for segmentation — the first one ([brats-moe-distmap-fusion-1](https://github.com/guillaume-cassez/brats-moe-distmap-fusion-1)) covered post-hoc fragment filtering for brain tumor segmentation; this one covers loss-design choices for full-resolution Cityscapes.

**I'm currently looking for opportunities**, ideally:

- **ML / research engineering** in computer vision, autonomous-driving perception, or applied research
- **R&D positions** (full-time, contract, industrial post-doc) where loss design, training dynamics, or controlled ablation are first-class concerns
- **MLOps / engineering** in vision-heavy products

If this kind of work is what your team does, I'd love to chat — even just to exchange notes on the C-vs-D crossover or the per-class explanation.

→ [cassez.guillaume@gmail.com](mailto:cassez.guillaume@gmail.com)
→ [guillaume-cassez.fr](https://guillaume-cassez.fr)
→ [Bluesky @guillaume-cassez.bsky.social](https://bsky.app/profile/guillaume-cassez.bsky.social)
→ [ORCID 0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931)

For technical discussion on this work specifically, please open a [GitHub issue](https://github.com/guillaume-cassez/city-scape/issues) — it helps future readers.

---

## License

The **code** in this repository is released under the [MIT License](LICENSE).
The **figures and text of the paper** (paper.md and any derivative figures) are released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/): redistribution and remix permitted with attribution.
Cityscapes raw imaging data is **not** redistributed and remains under the [Cityscapes Dataset Terms of Use](https://www.cityscapes-dataset.com/).
