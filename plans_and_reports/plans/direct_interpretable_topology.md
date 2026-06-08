# Probing the Direct / Interpretable Topology of CL Trees vs Matched FC

**Date:** 2026-06-05
**Motivates:** the preliminary findings (`reports/2026-06-05_19roi_motor_network_report.md`) that the
CL tree (i) recovers all three canonical bilateral-M1 edges in 10/10 subjects (vs 8/10 for matched FC
methods), (ii) yields an anatomically faithful 19-node consensus (PMv↔face, PMd↔hand, thalamus as a
relay), and (iii) — in the 12-ROI work — *excludes* high-marginal-MI pairs that are explained by an
intermediary (e.g. a high-MI pair with 0/10 CL frequency). This plan tests, rigorously and against
matched FC, the two claims packed into "direct, interpretable topology."

**H2a (directness):** CL-tree edges preferentially capture *direct* statistical dependencies and
suppress *indirect* (transitive) ones — more so than marginal FC, and comparably to the explicit
direct-connectivity estimators (partial correlation, graphical LASSO).
**H2b (interpretability):** the CL tree's sparse topology maps onto known motor anatomy and onto
focal perturbations more cleanly (more reproducibly, more sparsely) than thresholded FC.

---

## Part A — Directness (H2a)

### A1. Indirect-path suppression vs partial correlation (the linear gold standard)
Partial correlation (precision matrix) is the standard linear definition of a *direct* edge: an
indirect pair has high marginal r but low partial r (Smith 2011). Test whether the CL tree behaves
like a directness filter:
- For every ROI pair, compute marginal MI, marginal |r|, and **partial |r|**.
- **Test 1 (excluded high-MI pairs are indirect):** pairs with high marginal MI but *absent* from the
  CL tree should have low partial correlation. Compare partial-|r| for {CL-tree edges} vs {top-MI
  non-tree pairs} — predict CL edges ≫ excluded high-MI pairs in partial-|r|.
- **Test 2 (excluded pairs are tree-transitive):** for each high-marginal-FC pair not in the CL tree,
  measure its graph distance in the tree. If such pairs are predominantly distance-2 (A–C–B with both
  legs in the tree), CL is correctly identifying them as routed through an intermediary `C`.
- **Cross-method:** repeat the directness scoring for FC-MST and FC-top-N — do they retain more
  indirect (low-partial-r) edges than CL? This quantifies "CL is more direct than marginal FC."

### A2. Conditional mutual information — the MI-native directness test
Partial correlation is linear; the information-theoretic analog is conditional MI. For a CL edge
`A–B`, `CMI(A,B | C)` (conditioning on the tree-path intermediary or on all other nodes) should stay
high; for an excluded high-MI pair, conditioning on its intermediary should *explain it away*
(CMI → 0).
- Implement discrete CMI from the binned joint (extends `mict.mutual_info`).
- **Test:** ΔMI = MI(A,B) − CMI(A,B | intermediary) — large for excluded indirect pairs (collapses on
  conditioning), small for CL edges (survives). This is the direct, nonlinearity-aware version of A1
  and ties to `methods`/the conditional-MI idea. Report per-pair and as a CL-edge-vs-excluded
  contrast.

### A3. Generative / reconstruction quality (Chow-Liu optimality, made empirical)
The CL tree is, by construction, the maximum-likelihood *tree* approximation to the joint
distribution (min KL among trees). Make this concrete and comparative:
- **Tree-implied reconstruction:** from each sparse model, reconstruct the full pairwise dependency
  matrix it implies — for the CL tree, the MI between non-adjacent nodes is bounded by the
  data-processing inequality along the unique tree path; for the Gaussian models (FC-MST, GLASSO),
  the implied full covariance. Compare reconstruction error to the empirical matrix across CL, FC-MST,
  partial-MST, GLASSO at matched edge count.
- **KL / held-out likelihood:** fit each model's distribution (CL tree distribution; sparse Gaussian
  for the FC methods) on half the sessions and evaluate held-out log-likelihood / KL on the other
  half. Predict the CL tree gives the best held-out fit *among tree-sparse models* (its defining
  optimality), and competitive sparse fit overall. This reframes "direct" as "best sparse generative
  model," testable and quantitative.

---

## Part B — Anatomical & perturbational interpretability (H2b)

### B1. Anatomical validity of edges (structure as ground truth)
Direct functional edges should track direct anatomical connections. Score each method on recovery of
*a priori* anatomical edge classes:
- **Bilateral homologues** (callosal/commissural): the three M1 pairs + bilateral cerebellar +
  bilateral thalamus. CL already recovers the M1 set 10/10 — quantify recovery rate for *all*
  homologues, CL vs FC.
- **Crossed cortico-cerebellar loops:** M1_hand(L) ↔ cerebellum(R) is the anatomically correct
  (contralateral, ponto-cerebellar) edge; the ipsilateral edge is not. A *direct* method should
  prefer the **crossed** edge; a marginal method may show both. Test the crossed-vs-ipsilateral edge
  preference per method — a concrete, falsifiable anatomical-directness test.
- **Thalamic relay position:** is the thalamus placed *between* cerebellum and cortex on tree paths
  (relay topology) more often in CL than in FC-thresholded graphs?
- **PS1 / DTI cross-check (when PS1 is processed; it has DWI):** CL edge presence vs corticospinal /
  cortico-cerebellar tract integrity from tractography — the strongest structural validation, and the
  pivot for the reorganization story (displaced R-hand representation).

### B2. Topological interpretability metrics (vs the dense FC matrix)
- **Hub identification + stability:** degree-based hubs in the CL tree vs density-matched FC; are CL
  hubs more anatomically sensible (SMA, thalamus as integrators) and more reproducible across
  subjects/sessions?
- **Module structure:** does the CL tree partition into anatomically meaningful modules (effector- or
  hemisphere-based) more cleanly than FC? Use `mict.clustering` network-coherence (fraction of
  within-region pairs co-clustered) across cut levels, CL-graph-distance vs FC-distance.
- **Path interpretability:** in a tree, every pair has a *unique* path; tabulate the canonical routes
  (e.g. M1_hand → PMd → SMA → …) and check they are anatomically coherent — impossible to state for a
  dense FC matrix.

### B3. Focal-vs-diffuse response to perturbation (the interpretability payoff)
A genuinely *direct/sparse* model should localize a focal manipulation to a few edges, whereas
marginal FC smears it across many pairs.
- **Cast:** quantify the number of edges that change pre→cast→post in the CL tree vs the number of FC
  edges (matched threshold) that change significantly. Predict: CL change is sparse and centered on
  the casted-limb (hand) edges; FC change is diffuse. Report a "localization ratio" (changed-edge
  sparsity) for CL vs FC.
- **PS1 (when processed):** same logic for the displaced right-hand representation — does CL reroute a
  small number of direct edges (R_M1_hand to atypical partners) while FC shows distributed change?
  This is the central interpretability claim for the clinical case.

---

## Deliverables & success criteria

- `analyses/topology/`: directness contrasts A1 (partial-r) + A2 (CMI), reconstruction/held-out fit
  A3, anatomical-edge scorecard B1 (incl. crossed cortico-cerebellar), interpretability metrics B2
  (`mict.clustering`), and the cast localization-ratio B3 (PS1 later). New code: `mict.conditional`
  (CMI), `mict.directness` (partial-r / transitivity scoring), `mict.generative` (tree/Gaussian
  reconstruction + held-out KL).
- **Confirms H2a if:** CL edges have higher partial-|r| and higher surviving-CMI than excluded
  high-MI pairs, and excluded pairs are mostly tree-transitive (A1–A2); CL gives the best held-out
  tree fit (A3).
- **Confirms H2b if:** CL recovers anatomical edge classes (esp. crossed cortico-cerebellar) better
  than marginal FC and yields more reproducible hubs/modules (B1–B2); a focal manipulation produces a
  sparser CL change than FC change (B3).
- **Tempers the claim if:** partial-correlation MST matches or beats CL on directness (A1) — then the
  "direct" advantage is shared with linear precision methods and the distinctive CL contribution is
  the *nonlinear* directness (CMI, A2) plus interpretability (B), which should be framed accordingly.

## References
- Smith et al. (2011) *NeuroImage* — partial correlation recovers direct edges; marginal r conflates.
- Zalesky et al. (2012) *NeuroImage* — correlation conflates direct/indirect.
- Chow & Liu (1968) — tree = ML sparse approximation (optimality behind A3).
- Buckner et al. (2011) *J Neurophysiol*; Stoodley & Schmahmann (2009) *NeuroImage* — cortico-
  cerebellar somatotopy / crossed loops (B1).
- Newbold et al. (2020) *Neuron* — cast disuse (B3). Laumann et al. (2021) *Lancet Neurol* — PS1
  reorganization (B1 DTI / B3).
