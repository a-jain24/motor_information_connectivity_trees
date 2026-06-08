# FC-based sparse networks — apples-to-apples vs the Chow-Liu tree

**Date:** 2026-06-04
**Implements:** an exact, matched-sparsity comparison between the Chow-Liu (CL) tree and
sparse networks derived from *linear* functional connectivity, so we can say precisely what
the information-theoretic tree buys us over standard FC. Code: `mict.fc_sparse`,
`mict.viz.figures`; analysis: `analyses/q1_healthy_conservation/cl_vs_fc_sparse.py`.

---

## 1. Why this comparison, and why it is fair

The CL tree is the **maximum spanning tree (MST) of the pairwise mutual-information graph** —
a sparse (N−1 edges), direct-dependency model that, under the tree assumption, is the
information-maximising approximation to the joint distribution (Chow & Liu 1968). To know
whether its structure reflects something FC misses, we need FC-derived graphs that are
sparse **in the same way** (same edge budget, same graph machinery). Three axes distinguish
the candidates:

- **Marginal vs direct dependence.** Pearson correlation conflates direct and indirect
  (transitive) connections (Zalesky et al. 2012; Smith et al. 2011). Partial correlation /
  the precision matrix removes indirect paths — the linear notion of "direct" connectivity.
- **Linear vs nonlinear.** Correlation captures only linear (second-order) dependence; mutual
  information captures all statistical dependence.
- **Sparsity mechanism.** Thresholding/density-matching vs an MST vs an L1-penalised precision.

### The key fairness result (the clean control)

For jointly Gaussian variables, `MI(i,j) = -½·log(1 − r_ij²)`, a **monotonic function of
|r_ij|**. Because an MST depends only on the *ranking* of edge weights, the **MST of MI (CL
tree) is identical to the MST of |r| (FC-MST) whenever the data are Gaussian.** Therefore:

> **FC-MST is the exact linear twin of the CL tree.** Any CL ≠ FC-MST difference isolates the
> *non-Gaussian / nonlinear* dependence that MI captures and correlation cannot.

We report the per-subject CL↔FC-MST edge agreement as a direct, interpretable scalar for "how
much nonlinearity is in these data" (1.0 ⇒ Gaussian/linear; <1 ⇒ MI adds structure).

---

## 2. The methods (all produce a sparse graph; the MST ones give exactly N−1 edges)

| Method | Edge weight | Removes indirect? | Nonlinear? | Sparsity | Key refs |
|---|---|---|---|---|---|
| **CL tree** | mutual information | yes (tree) | **yes** | N−1 (MST) | Chow & Liu 1968 |
| **FC-MST** | \|Pearson r\| | no | no | N−1 (MST) | Tewarie et al. 2015 |
| **Partial-corr MST** | \|partial r\| | **yes** | no | N−1 (MST) | Smith 2011; Marrelec 2006 |
| **Graphical LASSO** | sparse precision (partial r) | **yes** | no | tunable (L1) | Friedman 2008; Varoquaux & Craddock 2013 |
| **FC top-(N−1)** | \|Pearson r\| thresholded | no | no | N−1 (density-matched) | (baseline) |

- **FC-MST** (Tewarie et al. 2015) — the minimum/maximum spanning tree of FC is an *unbiased,
  threshold-free* sparse network representation that sidesteps the density/threshold confounds
  of correlation-thresholding. It is the structural twin of the CL tree (§1).
- **Partial correlation** — computed from the (Ledoit-Wolf-regularised) precision matrix:
  `pc_ij = −Θ_ij/√(Θ_ii Θ_jj)`. Smith et al. (2011) found partial correlation (especially
  regularised) best recovers *direct* network edges in simulated fMRI, whereas full
  correlation conflates direct and indirect. Its MST is the "direct, linear" analog of CL.
- **Graphical LASSO** (Friedman et al. 2008) — L1-penalised precision estimation yields a
  sparse Gaussian graphical model; the standard sparse-connectome tool (Varoquaux & Craddock
  2013; nilearn `GroupSparseCovariance`). Sparsity is data-driven (CV α), so the edge count is
  *not* fixed to N−1 — useful as the established sparse-FC reference, but interpret its
  conservation/agreement numbers against its (often higher) density.
- **FC top-(N−1)** — keep the strongest N−1 |r| edges: the simplest density-matched control.

**Effective connectivity (noted, deferred).** Directed/causal models (Granger causality;
Dynamic Causal Modelling — Friston 2011) define *effective* connectivity. They are model-heavy
and directional, a poor match for the undirected, graph-scale CL comparison; the
precision/partial-correlation family above is the practical "direct connectivity" analog. DCM
on the small motor-ROI set is a possible future extension if a directed comparison is wanted.

---

## 3. Subject-specific and consensus networks (for every method)

Mirrors the CL pipeline exactly so the figures are interchangeable:

1. **Per subject:** build the method's sparse graph from the ROI timeseries → edge set.
2. **Consensus:** edge-frequency matrix = fraction of subjects containing each edge; the
   consensus backbone is the MST of that frequency graph.
3. **Figures (method-agnostic):** `mict.viz.figures.motor_report(dataset, methods=[...])` emits,
   per method, the **subject grid (F1)** — per-subject networks, nodes coloured by region, hub
   nodes (degree ≥3) outlined, edges styled by consensus tier (canonical ≥70% solid /
   intermediate 40–69% / idiosyncratic <40% dashed) — and the **consensus map (F2)** — edge
   width ∝ frequency, `k/N` labels, node size ∝ degree. These are the same F1/F2 used for CL,
   so CL and each FC method are rendered identically for side-by-side comparison.

---

## 4. Quantitative comparison (`cl_vs_fc_sparse.py`)

Per method: (a) **inter-subject conservation** = mean pairwise edge-set Jaccard; (b)
**canonical bilateral-M1 edge recovery** = consensus frequency of L↔R hand/foot/face; (c)
**CL↔FC-MST per-subject agreement** (§1 nonlinearity probe); (d) **paired tests** of CL vs each
method on conservation (t-test over subject pairs).

### Results so far — MSC, N=10, 12 ROIs (M1 + cerebellar effectors)

| Method | Conservation Jaccard | Edges | Canonical M1 (hand/foot/face) |
|---|---|---|---|
| Chow-Liu (MI MST) | **0.439 ± 0.10** | 11 | **1.0 / 1.0 / 1.0** |
| FC-MST (\|r\|) | 0.414 ± 0.10 | 11 | 0.8 / 1.0 / 1.0 |
| Partial-corr MST | 0.416 ± 0.09 | 11 | 0.8 / 1.0 / 1.0 |
| Graphical LASSO | 0.938 ± 0.03 | 63.9 | 1.0 / 1.0 / 1.0 |
| FC top-(N−1) | 0.446 ± 0.15 | 11 | 0.8 / 1.0 / 1.0 |

- **CL recovers the bilateral hand-M1 edge in 10/10 subjects; every matched-sparsity FC method
  gets it in only 8/10** — a concrete, interpretable CL advantage on a canonical motor edge.
- **CL↔FC-MST edge agreement = 0.608** ⇒ ~40% of CL edges differ from the linear MST: MI is
  capturing substantial non-Gaussian structure in these data.
- At matched sparsity, CL conservation trends above FC-MST/partial-MST (Δ≈+0.025, p≈0.08–0.13;
  not significant at 12 ROIs) — revisit at the full 19-ROI scheme, where the premotor branch
  (less dominated by the universal bilateral pairs) is where CL's advantage may become significant.
- **GLASSO is barely sparse here** (≈64/66 possible edges at 12 ROIs), so its high conservation
  is a density artifact — this is *why* the matched-sparsity MST comparison is the fair control.

---

## 5. Implementation map

| Piece | Location |
|---|---|
| Estimators (partial corr, GLASSO, MSTs) + unified `sparse_network()` | `src/mict/fc_sparse.py` |
| Method-agnostic F1/F2 figures (`subject_tree_grid`, `consensus_tree_figure`) | `src/mict/viz/trees.py` |
| Figure driver (`motor_report`, `network_figures`) | `src/mict/viz/figures.py` |
| Comparison analysis | `analyses/q1_healthy_conservation/cl_vs_fc_sparse.py` |
| Outputs | `results/<dataset>/surface/figures/<method>/{F1_subject_grid,F2_consensus}.{pdf,png}`; `…/q1/cl_vs_fc_sparse.{json,txt}` |

Runs identically for MSC (N=10), cast (per pre/cast/post condition), and PS1 (vs the MSC
consensus) once each is processed.

---

## References

- Chow, C.K., Liu, C.N. (1968). Approximating discrete probability distributions with dependence trees. *IEEE Trans. Inf. Theory* 14:462–467.
- Smith, S.M., Miller, K.L., Salimi-Khorshidi, G., et al. (2011). Network modelling methods for FMRI. *NeuroImage* 54:875–891.
- Tewarie, P., van Dellen, E., Hillebrand, A., Stam, C.J. (2015). The minimum spanning tree: an unbiased method for brain network analysis. *NeuroImage* 104:177–188.
- Friedman, J., Hastie, T., Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics* 9:432–441.
- Varoquaux, G., Craddock, R.C. (2013). Learning and comparing functional connectomes across subjects. *NeuroImage* 80:405–415.
- Marrelec, G., Krainik, A., Duffau, H., et al. (2006). Partial correlation for functional brain interactivity investigation in functional MRI. *NeuroImage* 32:228–237.
- Zalesky, A., Fornito, A., Bullmore, E. (2012). On the use of correlation as a measure of network connectivity. *NeuroImage* 60:2096–2106.
- Friston, K.J. (2011). Functional and effective connectivity: a review. *Brain Connectivity* 1:13–36.
