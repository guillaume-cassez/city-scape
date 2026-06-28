# Changelog

All notable changes to this record are documented here. This project follows
[Zenodo concept versioning](https://help.zenodo.org/docs/deposit/manage-versions/):
the concept DOI [10.5281/zenodo.20528680](https://doi.org/10.5281/zenodo.20528680)
always resolves to the latest version. Earlier version DOIs remain permanently
citable.

## v0.2.2 — 2026-06-19 (primary image-bootstrap + BRATS-faithful canonical consensus C⊘B)

Version DOI: [10.5281/zenodo.21006393](https://doi.org/10.5281/zenodo.21006393) (published 2026-06-28).

The **complete and final** consensus characterisation, plus a pre-specified primary
significance test for mIoU. The mIoU table values are unchanged from v0.2.0; this revision
fixes the *framing* of the significance and the consensus analogy. After this version paper2
is frozen (new DOIs only for a genuine later erratum).

- **mIoU significance — paired image-bootstrap is now the pre-specified primary test.**
  Over the 500 val images (resample with replacement, B = 10 000, recompute the dataset-level
  Δ mIoU per replicate, 95 % CI = pct 2.5/97.5, two-sided p; `analysis/bootstrap_miou.json`).
  Under Holm, **D > B (Δ = +0.59, p = 0.032) and D > A (Δ = +0.40, p = 0.043) are significant**;
  D > C (raw p = 0.023) is **not** significant after Holm (p = 0.092). The 3-seed paired-t test is
  kept as the **robustness layer** (a different variance source — the training seed) and there
  D > C is significant (Holm p = 0.042). The two tests jointly support every "D > {A, B, C}";
  §4.2 now states the primary-endpoint and bootstrap protocol explicitly, §5.1 reports both
  tests side by side.
- **Consensus re-anchored on the BRATS-faithful canonical pairing C⊘B.** This **corrects the
  v0.2.1 "direct analogue" claim**, which used D⊘B: in BRATS the veto is the field-standard
  baseline (`DC_and_CE` = Dice + CE = nnU-Net default) and the pruned generalist (DistMap) *also*
  keeps Dice + CE, so the faithful transcription is a Dice + CE generalist vetoed by the Dice + CE
  baseline. We make explicit that **the baseline is B (CE + Dice = nnU-Net default)**; A (CE) and
  D (CE + Kervadec, no Dice) are Cityscape-specific Dice-axis ablations with no BRATS equivalent.
  D⊘B (used in v0.2.1) had veto B = baseline but generalist D ≠ a Dice variant, so it was *not* the
  direct analogue. The canonical consensus is now **C⊘B**.
- **Full consensus ablation matrix + the veto's boundary-quality effect are now reported** (the
  v0.2.1 "Δ Boundary F1 / Trimap not yet measured" limitation is closed). All four pairings (mean
  over 3 seeds, `analysis/consensus_matrix.json`): C⊘B +0.010 pp mIoU / −18.7 % fragments; D⊘B
  −0.033 pp / −18.6 %; C⊘A +0.085 pp / −8.5 %; D⊘A +0.137 pp / −15.6 % — the baseline veto B prunes
  ≈ 2× more fragments than the CE-only veto A. **C⊘B full characterisation**
  (`analysis/consensus_pair_CB_full.json`, `analysis/bootstrap_consensus_CB.json`): mIoU neutral
  (paired p = 0.747; image-bootstrap Δ = −0.01, p = 0.86), fragments −18.7 % (−111/img, p = 0.007),
  Boundary F1 −0.104 pp (p = 0.009) and Trimap IoU +0.058 pp (p = 0.028) — both significant at n = 3
  but < 0.1 pp, i.e. practically negligible → a **pure spatial-coherence cleanup, boundary-neutral**.
  **D⊘B** (`analysis/consensus_pair_DB_full.json`): same prune and mIoU-neutrality (image-bootstrap
  Δ = +0.03, p = 0.66) **but a Dice-axis shift** (Boundary F1 −0.691 pp, Trimap IoU +0.997 pp, both
  p < 0.001), because reassigning D's fragments to B's labels Dice-ifies the output (D and B are on
  opposite ends of the Dice axis). So C⊘B is the *pure* consensus; D⊘B conflates fragment-cleanup
  with a Dice shift. Abstract contribution (4), §5.5 and §6.4 updated accordingly.
- This is the **complete, final consensus characterisation**; the n = 3 seed-level caveat on the
  contour deltas is retained.

## v0.2.1 — 2026-06-18 (consensus filter quantified)

Adds the val-set numbers for the §5.5 connected-component consensus filter, which
v0.2.0 released as code only ("no numerical claim"). No other change — mIoU, contour
metrics and the methodological finding are identical to v0.2.0.

- **Variant-pair veto (D vetoed by B):** −18.6 % connected-component fragments
  (782.6 → 636.8 per image; paired p = 0.008, consistent across all three seeds) at
  **no mIoU cost** (81.69 → 81.65; Δ = −0.03 pp, p = 0.50) — a spatial-coherence gain,
  not an mIoU gain; the direct analogue of the BRATS Baseline-vetoes-DistMap rule.
- §5.5, abstract contribution (4) and §6.4 updated with these numbers.
  `scripts/evaluate_consensus.py --lean` reproduces them from the trained checkpoints.
- The multi-seed seed-vote an earlier §5.5 draft proposed is **dropped**: it ensembles
  the 3 seeds instead of treating them as statistical replicates, has no BRATS
  equivalent, and its gain is plain ensembling (×3 inference), not the consensus filter.

## v0.2.0 — 2026-06-17 (erratum: re-evaluated contour metrics + official mIoU validation + consensus filter)

This is a **correction release**. Two estimator issues found in v0.1.0 after
publication are fixed, and **all twelve runs were re-evaluated on the val set** with
the corrected metrics. **The mIoU results and the headline methodological finding are
unchanged; the two contour metrics (Boundary F1, Trimap IoU) and all confidence
intervals are revised, and the reported mIoU is now validated bit-for-bit against the
official `cityscapesscripts` evaluation.**

### Fixed — metrics

- **Boundary F1 and Trimap IoU were measuring the road-vs-rest contour only.**
  The previous code passed the multi-class label map to
  `scipy.ndimage.binary_dilation`, which binarises any non-zero label as
  foreground — so only the boundary of class 0 (road) against everything else
  was measured, not the inter-class boundaries the paper claims to average over
  ("averaged across classes"). Both metrics are now computed **per class on
  binary masks**; the trimap band is built from all inter-class transitions
  (`src/metrics/segmentation_metrics.py`). A synthetic unit test
  (`tests/test_metrics_consensus.py`) demonstrates the old estimator returned
  F1 = 1.0 on a wrong building/vegetation seam while the corrected one detects it.
  **All twelve runs were then re-evaluated on the 500-image val set** with the
  corrected per-class estimator (`scripts/reeval_official.py`); the contour values
  in the paper are the corrected ones (Boundary F1 ≈ 58 → ≈ 77, Trimap ≈ 48 → ≈ 53,
  with the qualitative ordering also changing — see "Changed" below).
- The dataset-level confusion matrix is now vectorised (`np.bincount`), removing
  a per-pixel Python loop (no change to values, ~1000x faster — needed for
  re-evaluation).
- **mIoU is validated against the official Cityscapes evaluation.**
  `tests/test_official_miou.py` proves the confusion-matrix mIoU is bit-identical
  (≤ 2·10⁻¹⁶) to `cityscapesscripts.evaluation.getIouScoreForLabel` +
  `getScoreAverage` on random and crafted scenes, so the reported mIoU is the
  official Cityscapes value. (A redundant `+1e-6` denominator term that biased the
  value off the official one was removed.)

### Fixed — statistics

- **95% confidence intervals used 1.96 × SE with n = 3 seeds.** The correct
  factor is the Student-t critical value `t(0.975, df=2) = 4.303`; the previous
  intervals were ~2.2x too narrow (`figures/generate_figures.py`,
  `figures/aggregate_stats.py`).
- **Significance is now established with a paired-by-seed t-test**, not by
  comparing whether confidence-interval bars overlap (which is not a valid
  hypothesis test and relied on the under-estimated intervals above).
  `figures/table_significance.csv` is new.

### Changed — claims (statistics fix + contour re-evaluation)

- The v0.1.0 headline "D reaches the highest mIoU, lead robust" is **softened**:
  D has the highest *mean* mIoU and **significantly beats variant C** (Δ = +0.46,
  raw p = 0.007, Holm p = 0.042), but **D vs A (p = 0.095) and D vs B (p = 0.075)
  are not significant at n = 3** — directionally consistent across all three seeds,
  but underpowered.
- **The contour story changed** with the corrected estimator (the v0.1.0
  road-vs-rest contour numbers should not be cited). Both contour metrics now split
  along the Dice axis, each significant under Holm-corrected paired t-tests: the
  non-Dice variants (A, D) lead **Boundary F1** (D − B = +0.93, A − B = +0.85;
  Holm p ≤ 0.011), while the Dice variants (B, C) lead **Trimap IoU** (B − D = +1.57,
  C − D = +1.62, B − A = +1.65, C − A = +1.70; Holm p ≤ 0.012). Dice trades edge
  sharpness for near-boundary region coherence. v0.1.0 had collapsed Boundary F1 to
  within 0.2 across variants ("no claim") and reported only "B wins Trimap".
- The epoch-10 C−B gap (+2.26 mIoU) is not individually significant (p = 0.10);
  the robust, significant crossover is **D overtaking C** between epoch 10 and
  160 (p = 0.007). The methodological message stands on that axis.

### Done in this revision (was pending in the draft)

- **Boundary F1 / Trimap central values re-evaluated.** All twelve runs were re-run
  on the val set with the corrected per-class estimator (`scripts/reeval_official.py`
  + `scripts/aggregate_official.py`); the † markers are gone and
  `figures/table_final.csv`, `table_significance.csv`, `table_perclass.csv` are
  regenerated (Holm-corrected significance added).
- **Figures regenerated.** `fig_perclass.{png,pdf}` uses the official per-class IoU;
  `fig_convergence.{png,pdf}` now shows per-epoch **mIoU only** with Student-t bands
  (per-epoch contour was not re-evaluated — only epoch-160 checkpoints were retained,
  so intermediate-epoch contour cannot be recomputed).

### Still pending (deferred)

- **Consensus filter quantitative results** — **done in v0.2.1** (variant-pair veto:
  −18.6 % fragments at no mIoU cost).

### Added — consensus filter

- `src/postprocessing/consensus.py`: a connected-component **consensus filter**
  adapted from our BRATS work. Two modes — variant-pair veto (e.g. D vetoed by B)
  and multi-seed majority vote with a per-pixel agreement (uncertainty) map —
  plus a fragment-count metric. Cityscapes-specific differences (2D connectivity,
  19 flat classes, reassignment instead of zeroing, thin-class protection) are
  documented in the module and in §5.5 of the paper. Implementation is unit-tested
  (19/19); quantitative val-set results are pending the same re-run.
- `scripts/evaluate_consensus.py`: runnable consensus evaluation.

## v0.1.0 — 2026-06-03 (initial release)

- Initial public release: paper (EN + FR), companion code, per-epoch metrics for
  the twelve runs, figures, and figure-generation script.
- DOI [10.5281/zenodo.20528681](https://doi.org/10.5281/zenodo.20528681).
- **Note:** contains the metric and confidence-interval issues corrected in
  v0.2.0 above. Cite v0.2.0 or later for the corrected analysis.
