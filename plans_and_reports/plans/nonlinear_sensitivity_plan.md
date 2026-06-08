# Probing the Nonlinear Sensitivity of CL Trees vs Matched FC

**Date:** 2026-06-05
**Motivates:** the preliminary finding (`reports/2026-06-05_19roi_motor_network_report.md`) that the
per-subject CL↔FC-MST edge agreement is **0.590** — i.e. ~41% of Chow-Liu tree edges differ from the
maximum spanning tree of |Pearson r|. Under Gaussianity these two trees are identical
(MI = −½·log(1−r²) is monotonic in |r|), so the gap *should* index non-Gaussian / nonlinear
dependence that mutual information captures and correlation cannot. This plan tests whether that gap
is **genuine nonlinearity** (vs estimator noise / edge-ranking ties), localizes it, and asks whether
it carries reproducible signal.

**Core hypothesis (H1):** A non-trivial fraction of CL-tree edges are selected because of *nonlinear*
statistical dependence absent from the second-order (correlation) structure; this fraction exceeds
what would arise from a linear-Gaussian process with the same covariance.

---

## Analysis 1 — Linear-surrogate null for the CL↔FC-MST gap (the decisive test)

The 0.590 agreement is only meaningful against a null: how much would CL and FC-MST disagree *even if
the data were purely linear*, given finite samples and MI estimator noise?

- **Surrogates that kill nonlinearity but preserve linear structure:**
  1. **Multivariate-Gaussian surrogate** — draw `T` samples from `N(0, Σ̂)` with the empirical ROI
     covariance Σ̂. Preserves all pairwise correlations; removes all higher-order structure.
  2. **IAAFT / phase-randomized surrogate** — iterated amplitude-adjusted Fourier transform with a
     *shared* phase randomization across ROIs, preserving each ROI's power spectrum and the pairwise
     cross-spectrum (hence linear FC and autocorrelation) while destroying nonlinear phase coupling.
     More conservative than the Gaussian surrogate — it also preserves the (non-Gaussian) marginal
     amplitude distributions and spectra.
- **Procedure:** per subject, build `M=200` surrogates of each type; recompute MI, the CL tree, and
  FC-MST; record CL↔FC-MST agreement per surrogate. Compare the **observed** agreement to the
  surrogate distribution (one-sided: observed < surrogate ⇒ real nonlinearity).
- **Statistic:** per subject `z = (agreement_obs − mean(agreement_surr)) / sd(agreement_surr)`;
  combine across subjects (Stouffer / sign test). Report the surrogate-mean agreement too — it
  calibrates how much disagreement is just finite-sample noise.
- **Outcome:** if `agreement_obs` is significantly below both surrogate nulls, the CL≠FC-MST gap is
  genuine nonlinearity, not noise. If it sits within the Gaussian-surrogate distribution, the gap is
  an estimation artifact and the nonlinearity claim should be dropped.
- **Implementation:** new `mict.surrogates` (`gaussian_surrogate`, `iaaft_surrogate`); reuse
  `mict.fc_sparse.sparse_network` + `mict.stats.jaccard`.

## Analysis 2 — Per-edge "nonlinear excess" decomposition

Split each pair's MI into its Gaussian-predicted part and a nonlinear remainder:
`MI_excess(i,j) = MI_obs(i,j) − [−½·log(1 − r_ij²)]`.

- Map `MI_excess` across all 171 ROI pairs (group mean + per-subject); FDR-test which pairs exceed
  the Gaussian-surrogate null (Analysis 1).
- **Key test:** are **CL-unique edges** (in the CL tree but not the FC-MST) characterized by higher
  `MI_excess` than CL∩FC-MST shared edges? If yes, MI is demonstrably selecting those edges for
  nonlinear reasons — the mechanism behind H1. (Paired test of `MI_excess`, CL-unique vs shared.)
- **Monotone-nonlinearity split:** compare a **Spearman-MST** to the Pearson FC-MST. Edges appearing
  under Spearman but not Pearson reflect *monotone* nonlinearity; edges unique to MI (not Spearman)
  reflect *non-monotone* dependence — partitioning the nonlinearity into interpretable components.
- **Implementation:** MI and |r| already in `mict.fc_sparse`; add `spearman_mst`; the excess is a
  closed-form transform of the FC matrix.

## Analysis 3 — Estimator robustness (real signal, not a binning artifact?)

Show the nonlinear signal survives a fundamentally different estimator.

- Recompute MI + CL tree with: (a) bins K ∈ {20, 50, 100, 150}; (b) the **KSG k-NN MI estimator**
  (continuous, bias-reduced, no binning); (c) a **Gaussian-copula MI** (rank → Gaussianize → MI from
  the copula correlation), which captures only monotone dependence — so `KSG-MI − copula-MI` is a
  binning-independent nonlinearity estimate.
- **Outcome:** if `MI_excess` and the CL↔FC-MST gap are stable across K and reproduced by KSG, the
  nonlinearity is estimator-robust; if they vanish under KSG / large T / small K, attribute to bias.
- **Implementation:** add a KSG estimator to `mict.mutual_info` (or wrap `npeet`/`sklearn`); add
  `copula_mi`.

## Analysis 4 — Localize and characterize the nonlinear couplings

- **"MI-only" edges:** pairs with low |r| (below the 50th percentile) but high MI (above the 90th) —
  the signature of quadrature / phase-amplitude coupling that correlation misses entirely (cf. the
  simulation `mi_vs_correlation_demo`). Do any CL edges fall here? List + anatomically localize.
- **MI-vs-|r| scatter** for all pairs with the Gaussian curve overlaid; points above the curve = the
  nonlinear-excess pairs, CL-tree edges highlighted — the direct motor-data analog of the old
  `mi_vs_correlation_demo` figure (a strong paper panel).
- **Mechanism probes** on flagged pairs: Hilbert / bandpass analysis to test for phase-amplitude or
  quadrature coupling — does the nonlinear excess correspond to an interpretable coupling mode?

## Analysis 5 — Does the nonlinearity carry reproducible signal?

- **Test-retest:** split sessions odd/even; correlate `MI_excess` matrices and the CL-unique
  (nonlinear) edge sets across halves — reproducible excess ⇒ structured, not noise.
- **Cross-subject consistency:** are the same pairs nonlinear across subjects (consensus of
  `MI_excess > 0`)?
- **Manipulation sensitivity (cast):** does the nonlinear excess on motor edges change
  pre→cast→post? Manipulation-sensitive nonlinearity is strong evidence it is neurally meaningful.

---

## Deliverables & success criteria

- `analyses/nonlinear/`: surrogate-null test (A1), nonlinear-excess maps + CL-unique-vs-shared test
  (A2), estimator sweep (A3), MI-vs-|r| scatter + MI-only edge list (A4), reproducibility (A5).
  Figures: surrogate-null histogram, `MI_excess` heatmap, MI-vs-|r| scatter with the Gaussian curve.
- **Confirms H1 if:** observed CL↔FC-MST agreement is significantly below the linear-surrogate null
  (A1); CL-unique edges carry higher `MI_excess` than shared edges (A2); the excess survives KSG (A3)
  and replicates across session halves (A5).
- **Refutes / tempers H1 if:** the gap sits within the Gaussian-surrogate distribution, or vanishes
  under KSG / large T — then the report's nonlinearity language becomes "edge-rank instability," and
  the project leans on topology (`direct_interpretable_topology.md`).

## Engage with the strongest prior
Hlinka et al. (2011) reported that resting-state fMRI FC is *mostly* linear (Gaussian) — so the null
result is a live possibility and A1 must be decisive. Our motor-task-defined ROIs and longer
concatenated data may differ; either way, an honest surrogate test is the right adjudicator.

## References
- Paninski (2003) *Neural Comput* — MI estimator bias.
- Kraskov, Stögbauer, Grassberger (2004) *Phys Rev E* — KSG k-NN MI.
- Theiler et al. (1992) *Physica D*; Schreiber & Schmitz (2000) *Physica D* — surrogate data / IAAFT.
- Ince et al. (2017) *Hum Brain Mapp* — Gaussian-copula MI.
- Hlinka et al. (2011) *NeuroImage* — (mostly) linear resting-state FC.
- Chow & Liu (1968) *IEEE Trans Inf Theory* — the CL tree.
