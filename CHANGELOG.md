# Changelog

All notable changes to this record are documented here. This project follows
[Zenodo concept versioning](https://help.zenodo.org/docs/deposit/manage-versions/):
the concept DOI [10.5281/zenodo.20528680](https://doi.org/10.5281/zenodo.20528680)
always resolves to the latest version. Earlier version DOIs remain permanently
citable.

## v0.2.0 — 2026-06-16 (erratum + consensus filter)

This is a **correction release**. Two estimator issues found in v0.1.0 after
publication are fixed in code and reflected in the paper text. **The mIoU results
and the headline methodological finding are unchanged; two contour metrics and
all confidence intervals are revised.**

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
- The dataset-level confusion matrix is now vectorised (`np.bincount`), removing
  a per-pixel Python loop (no change to values, ~1000x faster — needed for
  re-evaluation).

### Fixed — statistics

- **95% confidence intervals used 1.96 × SE with n = 3 seeds.** The correct
  factor is the Student-t critical value `t(0.975, df=2) = 4.303`; the previous
  intervals were ~2.2x too narrow (`figures/generate_figures.py`,
  `figures/aggregate_stats.py`).
- **Significance is now established with a paired-by-seed t-test**, not by
  comparing whether confidence-interval bars overlap (which is not a valid
  hypothesis test and relied on the under-estimated intervals above).
  `figures/table_significance.csv` is new.

### Changed — claims affected by the statistics fix

- The v0.1.0 headline "D reaches the highest mIoU, lead robust" is **softened**:
  D has the highest *mean* mIoU and **significantly beats variant C** (Δ = +0.46,
  p = 0.007), but **D vs A (p = 0.095) and D vs B (p = 0.075) are not significant
  at n = 3** — directionally consistent across all three seeds, but underpowered.
- B's Trimap IoU lead over D **is** significant (Δ = +1.09, p = 0.005).
- The epoch-10 C−B gap (+2.26 mIoU) is not individually significant (p = 0.10);
  the robust, significant crossover is **D overtaking C** between epoch 10 and
  160 (p = 0.007). The methodological message stands on that axis.

### Pending (deferred to a later version)

- **Boundary F1 / Trimap central values** are marked † in the paper. They still
  show the legacy estimator's numbers; re-evaluation on the val set with the
  corrected per-class metric requires re-running inference on the trained
  checkpoints (off-machine) and will land in a follow-up version.
- `figures/fig_convergence.{png,pdf}` still use the 1.96-scaled bands and the
  legacy contour metrics; they will be regenerated at the same time.

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
