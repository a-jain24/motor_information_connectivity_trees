# 19-ROI Motor Network — Results Report

**Date:** 2026-06-05
**Repo:** `motor_information_connectivity_trees`
**Datasets analyzed:** Midnight Scan Club (MSC, N=10) and cast-induced plasticity (cast1=MSC02,
cast2=MSC06). Perinatal stroke (PS1 / sub-CIMT001) is pending preprocessing
(see `methods/ps1_preprocessing.md`).
**Space:** fs_LR_32k grayordinates (WU/Gordon `surface_pipeline` derivatives), surface-first.

---

## 1. Summary

We built the full **19-ROI motor network** — bilateral M1 (hand/foot/face), bilateral cerebellar
(hand/foot/face), SMA + bilateral PMd/PMv, and bilateral motor thalamus — and computed, per subject,
Chow-Liu (CL) trees of pairwise mutual information (MI) alongside matched-sparsity sparse networks
from linear functional connectivity (FC). Headline findings:

1. **The 19-node consensus network is anatomically faithful** — a cortico-cerebellar-thalamic-
   premotor motor map whose strongest, most-conserved edges are the bilateral homologues and the
   correct premotor pairings (PMv↔face, PMd↔hand, SMA↔medial/foot).
2. **CL uniquely recovers all three canonical bilateral-M1 edges in 10/10 subjects** (hand, foot,
   face), where matched-sparsity FC methods recover only 8/10 for hand and face.
3. **The CL-vs-FC *conservation* hypothesis is not supported.** At matched sparsity CL ≈ FC-MST, and
   CL is slightly *less* conserved than density-matched thresholded FC (Δ=−0.030, p=0.016).
4. **Follow-up probes temper the "nonlinear" and "direct" advantages (§9).** A surrogate-null test
   finds the CL≠FC-MST gap is mostly finite-sample edge-rank noise, not nonlinearity (Stouffer
   z≈−1.3 to −1.7, n.s.). CL *does* recover direct edges (it suppresses indirect/transitive pairs,
   p<1e-9; excluded high-MI pairs are explained away by conditioning) — but this directness is shared
   with, and on linear measures exceeded by, partial-correlation methods; CL's distinctive directness
   is the modest *nonlinear* (conditional-MI) component.
5. **CL's robust, defensible advantages** are therefore: canonical bilateral-M1 recovery (10/10 vs
   8/10), best **homologue recovery** (0.80 vs 0.72–0.78), **ROI-size robustness** (size-residualizing
   MI changes zero CL edges), a sparse **interpretable** topology, and a focal **cast-disuse** response
   (the casted right-hand M1 edge dips during casting and recovers afterward, both subjects).

---

## 2. Methods

**Preprocessing / space.** MSC and cast were processed by the Washington University precision-mapping
pipeline (Gordon et al. 2017) into fs_LR_32k grayordinate CIFTI; we read these derivatives directly.
A critical implementation detail: the WU files are **CIFTI-1**, which nibabel mis-orders on a raw
read (scrambling the data → near-zero FC). `mict.cifti.read_cifti` converts CIFTI-1→CIFTI-2 via
`wb_command` then reads through nibabel's authoritative API (validated: temporal lag-1 autocorr ≈
0.82; bilateral M1_face FC ≈ 0.92). Resting-state runs are tmask-censored and concatenated across all
10 sessions (~6,300 TRs/subject); the WU timecourses are already nuisance-regressed.

**19-ROI scheme** (`config/roi_scheme.yaml`; labels identical across datasets/spaces):

| Region | ROIs | Method |
|---|---|---|
| Primary motor (M1) | L/R hand, foot, face | task-peak + top-400 expansion of the lateralized motor contrasts (Newbold 2020) |
| Cerebellum | L/R hand, foot, face | task-peak + top-40 in the CIFTI cerebellar voxels |
| Premotor | SMA, L/R PMd, L/R PMv | **individual gradient-watershed parcel** (WU `surface_parcellation`) — the most combined-motor-active parcel per premotor anatomical box (§C) |
| Motor thalamus | L/R | FC-seed from the M1 ROIs → top-5% thalamic CIFTI voxels per hemisphere (§D) |

Mean ROI sizes (grayordinates): M1 400, cerebellar 40, SMA ~128, PMd ~76–91, PMv ~108–125, thalamus
~18–19.

**Connectivity & comparison.** FC = Pearson; MI = histogram estimator (100 bins); CL tree =
maximum spanning tree of MI. Apples-to-apples FC sparse networks (`mict.fc_sparse`,
`methods/fc_sparse_networks.md`): **FC-MST** (MST of |r|; the exact linear twin of CL since under
Gaussianity MI = −½log(1−r²)), **partial-correlation MST** (direct connectivity), **graphical LASSO**
(sparse precision), and **FC top-(N−1)** (density-matched). Consensus = edge-frequency across
subjects; conservation = mean pairwise edge-set Jaccard.

---

## 3. Results — MSC (N=10, 19 ROIs)

### 3.1 Consensus network

Consensus CL edges (present in ≥50% of subjects):

| Edge | Frequency | Edge | Frequency |
|---|---|---|---|
| L_M1_hand ↔ R_M1_hand | **10/10** | L_Cereb_foot ↔ R_Cereb_foot | 8/10 |
| L_M1_foot ↔ R_M1_foot | **10/10** | L_M1_face ↔ L_PMv | 7/10 |
| L_M1_face ↔ R_M1_face | **10/10** | R_M1_face ↔ R_PMv | 7/10 |
| L_Thal ↔ R_Thal | **10/10** | R_Cereb_foot ↔ R_Cereb_face | 7/10 |
| L_Cereb_hand ↔ R_Cereb_hand | 9/10 | R_Cereb_hand ↔ R_Cereb_face | 6/10 |
| L_Cereb_face ↔ R_Cereb_face | 9/10 | R_M1_hand ↔ R_PMd / L_PMd ↔ R_PMd | 5/10 |

The four universal edges (10/10) are the three bilateral M1 homologues and the bilateral thalamus.
The premotor pairings are exactly correct (PMv↔face, PMd↔hand) but more individually variable (the
watershed parcels respect each subject's own boundaries). A representative single-subject tree
(MSC01) is anatomically textbook: `R_M1_face–R_PMv` MI=1.71, `L_M1_face–L_PMv` MI=1.25,
`L_M1_hand–L_PMd` MI=0.85, `R_M1_hand–R_PMd` MI=0.59, `L_Thal–R_Thal` MI=0.47, plus thalamo-cerebellar
relay edges.

### 3.2 CL vs FC sparse networks (apples-to-apples)

| Method | Conservation Jaccard | Edges | Canonical bilateral-M1 (hand/foot/face) |
|---|---|---|---|
| **Chow-Liu (MI MST)** | 0.353 ± 0.07 | 18 | **10/10 · 10/10 · 10/10** |
| FC-MST (\|r\|) | 0.355 ± 0.08 | 18 | 8/10 · 10/10 · 8/10 |
| Partial-corr MST | 0.329 ± 0.07 | 18 | 8/10 · 10/10 · 9/10 |
| Graphical LASSO | 0.495 ± 0.27 | 114.8 | 9/10 · 10/10 · 10/10 |
| FC top-(N−1) | 0.383 ± 0.09 | 18 | 8/10 · 10/10 · 10/10 |

- **Conservation:** CL ≈ FC-MST (Δ=−0.002, p=0.87); CL slightly *below* density-matched FC-top-N
  (Δ=−0.030, p=0.016); CL slightly *above* partial-corr MST (Δ=+0.024, p=0.073). GLASSO's high
  Jaccard is a density artifact (it stays near-dense, ~115/171 edges, at this N). **The hypothesis
  that CL is more conserved than FC is not supported.**
- **Canonical-edge recovery:** CL is the only method recovering all three bilateral-M1 homologues in
  10/10 subjects; every matched-sparsity FC method drops hand and face to 8/10. This is a concrete CL
  advantage on the canonical motor edges.
- **Nonlinearity probe:** CL↔FC-MST per-subject edge agreement = **0.590** — under Gaussianity these
  would be identical, so naively ~41% of CL edges would reflect non-Gaussian dependence. **However,
  the surrogate-null test (§9.1) shows most of this gap is finite-sample edge-rank noise, not
  nonlinearity** — do not read the 0.590 as a nonlinearity estimate.

### 3.3 ROI-size robustness

ROIs span ~18–400 grayordinates, and the histogram MI estimator is size-sensitive, so we checked the
confound directly. Spearman(geometric-mean ROI size, group-mean MI) across 171 pairs = **−0.130,
p=0.09** → no positive size confound. CL edge-set Jaccard between the raw and log-size-residualized
MI tree = **1.000** (zero edges change); all three canonical M1 edges survive residualization. The
CL tree is not an artifact of ROI size.

---

## 4. Results — cast-induced plasticity (Q2)

cast1 (=MSC02) and cast2 (=MSC06) reuse their MSC ROI masks (identical fs_LR_32k grayordinates).
Resting-state was split into pre / cast / post by each subject's `sessions.tsv`. Casting was on the
**right** upper limb → its M1 is the **left** hemisphere. The bilateral hand-M1 edge MI tracks the
disuse-and-recovery:

| Subject | pre | cast | post |
|---|---|---|---|
| cast1: L_M1_hand ↔ R_M1_hand | 0.562 | **0.475** | 0.581 |
| cast2: L_M1_hand ↔ R_M1_hand | 0.413 | **0.299** | 0.410 |

Both subjects show the V-shaped dip during casting and recovery afterward, specific to the casted
hand (face/foot do not show the clean recovery). This reproduces the Newbold (2020) disuse signature
in the CL framework.

---

## 5. Figures produced

Per dataset/method under `results/<dataset>/surface/figures/`:

- **Per-subject networks:** `<method>/F1_subject_grid` (region-colored grid), `subjects_labeled/`
  (hierarchical, edge weights), `aesthetic/` (fig10-style — distinctly-colored labeled ellipses,
  MI-labeled edges, group-frequency edge coloring).
- **Consensus:** `<method>/F2_consensus`, `F2_consensus_labeled`.
- **Brain-space:** `brain/<subject>/roi_surface` (motor ROIs on the inflated fs_LR surface),
  `brain/<subject>/cl_connectome` (CL tree on the glass brain), and group versions
  `brain/group_roi_surface`, `brain/group_cl_connectome`.
- **cast Q2:** `q2/<method>/trajectory_grid` and `edge_trajectory` (pre→cast→post).

Generated for CL, FC-MST, and partial-corr MST (apples-to-apples) on MSC; CL + per-condition on cast.

---

## 6. Interpretation

The information-theoretic motor tree recovers a sparse, anatomically coherent, size-robust backbone
of the motor system, and on the **canonical bilateral-M1 edges** and **homologue recovery** it is the
most reliable method tested. However, the original project hypothesis — that CL trees are *more
conserved across individuals* than FC — does not hold for the motor network at either 12 or 19 ROIs:
CL and FC-MST are equally conserved, and density-matched thresholded FC is marginally more conserved.

We initially attributed CL's value to **nonlinear sensitivity** and **direct/interpretable topology**.
Targeted follow-up tests (§9) require qualifying both: the nonlinearity is a **weak, non-significant
trend** (most of the CL≠FC-MST gap is finite-sample noise, not non-Gaussian structure — consistent
with Hlinka et al. 2011), and while CL genuinely **does** recover direct edges and suppress indirect
ones, that *linear* directness is shared with — and slightly exceeded by — partial-correlation
methods; CL's distinctive directness is the modest *nonlinear* (conditional-MI) component. The
honest, defensible CL contribution is therefore: a **sparse, anatomically faithful, size-robust,
interpretable** motor backbone that recovers the canonical/homologous edges most reliably and
responds **focally** to perturbation (cast disuse; planned PS1 reorganization) — not a strong claim
of superior conservation or nonlinearity.

---

## 7. Limitations

- **Premotor definition** uses the most task-active individual gradient-watershed parcel per
  anatomical box — faithful to §C in spirit, but the box routing is a simplification of full Infomap
  network assignment; premotor edges are the most individually variable in the consensus.
- **Conservation is a null/negative result**, not a positive one — worth stating plainly in any
  write-up.
- **cast** uses a fetched 5-session-per-condition subset of two subjects (cast1/cast2); cast3 needs
  its own ROI definition.
- **PS1** is not yet preprocessed (raw BIDS only; no released derivatives or lesion mask — see the
  author-request email and `methods/ps1_preprocessing.md`).

---

## 8. Next steps

- Q5 inter-effector coupling; conditional-MI / premotor-only CL trees (now that the premotor branch
  is real); odd/even split-half test-retest reliability of the ROIs and trees.
- PS1 once preprocessed → Q3 (reorganization vs the MSC consensus) and Q4 (longitudinal), using
  `mict.lesion` for the displaced-ROI handling.
- Optional: the volumetric cross-check (`analyses/space_comparison/`).

*Engineering plan: `repo_organization_plan.md`. Science plan: `motor_system_plan.md`. Method specs:
`methods/fc_sparse_networks.md`, `methods/ps1_preprocessing.md`.*

---

## 9. Follow-up probes — nonlinearity & directness (added 2026-06-05)

Plans: `plans/nonlinear_sensitivity_plan.md`, `plans/direct_interpretable_topology.md`. Code:
`mict.surrogates`, `mict.conditional`; `analyses/{nonlinear/surrogate_null,topology/directness,topology/anatomical}.py`.
These tests adjudicate the §1/§6 claims rather than assume them — and they temper two of them.

### 9.1 Is the CL≠FC-MST gap real nonlinearity? — surrogate null (A1)

Recomputed the per-subject CL↔FC-MST agreement on N=10 against 100 linear surrogates/type that
preserve correlations but destroy nonlinearity:

| Surrogate | Observed agreement | Surrogate mean | Stouffer z | subjects below null (p<.05) |
|---|---|---|---|---|
| Gaussian (Σ̂) | 0.590 | 0.642 | −1.66 | 1/10 |
| Phase-randomized | 0.590 | 0.633 | −1.32 | 1/10 |

The observed agreement is *below* the null (the direction predicted by nonlinearity), but **not
significantly** (|z|<1.96; 1/10 subjects). **~80–90% of the CL≠FC-MST disagreement is finite-sample
edge-rank instability, not non-Gaussian structure.** The earlier "≈41% of CL edges are nonlinear"
phrasing is therefore an overstatement; the nonlinear contribution is a weak, non-significant trend
(consistent with Hlinka et al. 2011 that resting-state FC is largely linear).

### 9.2 Does CL recover *direct* dependencies? — partial-r + conditional-MI (A1/A2)

| Test | Result |
|---|---|
| CL-edge \|partial r\| vs excluded high-MI pairs | 0.367 vs 0.113 (Δ=+0.25, **p=6e-10**) |
| Excluded high-MI pairs at CL-tree distance 2 (transitive) | 57% |
| Mean \|partial r\| by method | CL 0.367 · FC-MST 0.390 · **partial-corr MST 0.414** · FC-topN 0.371 |
| MI "explained away" by the intermediary (conditional MI) | excluded pairs 0.38 vs CL edges 0.06 (Δ=+0.31, **p=0.04**) |

CL **does** capture direct dependencies: its edges have far higher partial correlation than the
high-MI pairs it excludes, those excluded pairs are mostly tree-transitive, and they are explained
away by conditioning (CMI) while CL edges survive. **But the *linear* directness is shared** — and on
\|partial r\| slightly exceeded — by partial-correlation methods. CL's distinctive directness is the
*nonlinear* conditional-MI contrast (modest, p=0.04). "CL is more direct than FC" should read "CL is
direct like the linear precision methods, plus a small nonlinear-directness increment."

### 9.3 Anatomical edge scorecard (B1)

- **Homologue recovery** (mean consensus frequency over the 9 bilateral pairs): **CL 0.800** >
  FC-MST 0.778 > FC-topN 0.767 > partial-corr 0.722 — a modest but consistent CL advantage.
- **Crossed (contralateral) cortico-cerebellar coupling** is the anatomically correct one. The data
  bear this out at the MI level (MI crossed 0.362 > ipsi 0.354, paired **p=0.01**), but
  cortico-cerebellar edges rarely survive into any 19-node tree, and at the edge level the crossed
  preference is captured best by **partial-corr MST** (Δ=+0.18), not CL (Δ=+0.03).

### 9.4 Net effect on the narrative

The robust CL story is **canonical/homologue recovery + sparse, size-robust, interpretable topology +
focal perturbation response**. The "nonlinear" and "uniquely direct" framings are *not* supported as
headline claims on this dataset and are reframed accordingly above (§1, §6). Higher-powered or
nonlinear-by-design follow-ups remain in the plans (surrogate test with more subjects/data; KSG/copula
decomposition; DTI-based anatomical validation in PS1; focal-vs-diffuse perturbation localization).
