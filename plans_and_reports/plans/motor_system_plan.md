Plan for a paper about Chow-Liu trees that uses the Midnight Scan Club, Perinatal Stroke, and Cast Induced Plasticity datasets (similar to https://pmc.ncbi.nlm.nih.gov/articles/PMC10172144/). 

Refs:

Inter-effector study:
https://pmc.ncbi.nlm.nih.gov/articles/PMC10172144/

Cast-induced plasticity, datasets/cast_induced_plasticity:
https://www.sciencedirect.com/science/article/pii/S0896627320303536?via%3Dihub

Perinatal stroke, datasets/perinatal_stroke: 
https://www.clinicalkey.com/#!/content/playContent/1-s2.0-S1474442221000624
Laumann TO, Ortega M, Hoyt CR, Seider NA, Snyder AZ, Dosenbach NUF (2021). Brain network reorganisation in an adolescent after bilateral perinatal strokes. *Lancet Neurology* 20(4):255–256. PMC13182026. OpenNeuro ds004498.

Midnight Scan Club, /mfs/io/groups/dmello/projects/cerebellum_reliability/derivatives/fmriprep/ds000224:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5576360/


---

## Dataset Overview

| Dataset | Subjects | Sessions | Scanner | TR | Resolution | Task available? |
|---------|----------|----------|---------|-----|------------|----------------|
| **MSC** (Gordon et al. 2017) | 10 (MSC01–MSC10) | 10 rest + 10 task per subject | 3T Siemens Trio | 2.2 s | 4 mm isotropic | Yes |
| **Cast-induced plasticity** (Newbold et al. 2020) | 3 (cast1=MSC02, cast2=MSC06, cast3=new) | 42–64 daily scans (pre / cast / post) | Trio (cast1) / Prisma (cast2,3) | 2.2 s (cast1) / 1.1 s (cast2,3) | 4 mm (cast1) / 2.4 mm (cast2,3) | Yes (nightly, before casting) |
| **Perinatal stroke** (Laumann et al. 2021; subject **PS1** = `sub-CIMT001`) | 1 (adolescent male) | 18 sessions, longitudinal (~500-day span); **rest + motor task** | 3T Siemens Trio | 2.2 s | 4 mm isotropic | **Yes — 56 motor-task runs (~15 sessions) + resting-state** |

**Note on overlap:** sub-cast1 = MSC02 and sub-cast2 = MSC06. Their MSC surface parcellations (gradient-watershed) and projection files are already available at `datasets/midnight_scan_club/derivatives/surface_pipeline/sub-MSC0{2,6}/`.

**Clinical context for PS1 (perinatal stroke subject) — critical for ROI definition:**
- **Bilateral** perinatal arterial strokes at ~3 weeks postnatal (dehydration/anaemia), producing **extensive bilateral cystic cortical lesions**; ~259 cm³ (~20%) of supratentorial brain volume lost. This is **not** a unilateral MCA lesion.
- Despite this, near-typical neurodevelopment: only a **slight right upper-limb deficit** (strength, speed, dexterity) and a transient gait asymmetry at 12 months.
- **Motor representations are reorganised/displaced.** Per Laumann et al. 2021: the **right-hand** motor response is **completely displaced** relative to controls — it sits in the **left-hemisphere postcentral gyrus** (posterior, atypical location) rather than canonical left precentral M1. The **left-hand** response is preserved in the spared right-hemisphere central sulcus (typical location). Remapping occurred primarily in frontal/parietal association areas.
- DTI showed intact corticospinal tracts arising from **both** the typical (left-hand) and displaced (right-hand) motor regions.
- Data collected: 285 min RSFC + 137 min task fMRI, beginning age 15 over two summers. Controls = the 10 MSC subjects.
- **Implication:** ROIs for PS1 **must be localised from the subject's own task activation peak wherever it lands** — atlas/group-coordinate seeding will miss displaced representations. This is exactly the precision-mapping rationale and a central scientific point of the comparison.

---

## Preprocessing and Analysis Pipelines

### 1. Inter-effector Study (Reference)

The inter-effector paper (PMC10172144) uses:
- Multi-echo EPI (TR = 1761 ms, MB6, 2.4 mm; TEs = 14.20, 38.93, 63.66, 88.39, 113.12 ms)
- Optimal echo combination before nuisance regression
- Atlas registration → CIFTI space
- Nuisance regression (standard WU pipeline)
- Individual-specific functional network definition via hierarchical data-driven approach (see Gordon et al. 2017)
- Seeds placed "in a continuous line of cortical locations in the left precentral gyrus" — 6 exemplar seeds covering all distinct connectivity patterns observed
- Task labeling: top 1% of vertices activated by each movement, from a 25-movement battery (foot, hand, mouth movements in separate blocks of 15.4 s, 12 movements per block)
- FC: Fisher Z-transformed Pearson correlations; visualized at 80th–97th percentile of values
- Thalamic nuclei identified by FC maps (no explicit atlas): CM (centromedian), VIM (ventral intermediate), VPM (ventral posteromedial), VPI (ventral posterior inferior)
- Cerebellar: lateral lobule V and vermis Crus II, lobule VIIb, VIIIa identified from FC maps

### 2. Cast-Induced Plasticity (Newbold et al. 2020)

**Acquisition:**
- sub-cast1: 3T Siemens Trio, 4 mm isotropic, single-band, TR = 2.2 s (identical to MSC protocol)
- sub-cast2, sub-cast3: 3T Siemens Prisma, 2.4 mm isotropic, multiband-4, TR = 1.1 s
- 42–64 daily 30-minute resting-state fMRI scans
- Block-design motor task (right hand, left hand, right foot, left foot, tongue) performed for 8 minutes every night prior to casting

**Preprocessing** (WU pipeline, as in Gordon et al. 2017):
- Slice-timing correction; odd/even slice intensity correction
- Rigid-body motion correction within run
- Atlas registration: native → T2w → T1w → Talairach 711-2B
- Susceptibility distortion correction (field map-based)
- Temporal bandpass: 0.005–0.1 Hz
- Frame censoring: FD > 0.2 mm (thresholds set individually per participant)
- Nuisance regression: global signal, ventricular signal, white matter signal, 6 motion parameters (low-pass filtered < 0.1 Hz before FD calculation)
- Linearly interpolated frame replacement after censoring
- Surface projection (mid-thickness, ribbon-constrained): 6 mm FWHM smoothing on cortex; 4.7 mm FWHM 3D kernel on cerebellum
- CIFTI format output

**Individual parcellations:**
- gradient-watershed method from Gordon et al. (2017) (see Section 4 below)
- cast1 and cast2: parcellations available from MSC dataset at `derivatives/surface_pipeline/sub-MSC0{2,6}/surface_parcellation/`
- cast3: compute de novo using same method

### 3. Midnight Scan Club (Gordon et al. 2017)

**Acquisition:**
- 10 subjects × (10 rest + 10 task) sessions
- 3T Siemens Trio, 4 mm isotropic, single-band, TR = 2.2 s, 36 slices
- Resting-state: 300 minutes total per subject (30 min/session × 10 sessions)
- Motor task: 2 runs/session, 7.8 min combined; hand / foot / tongue blocks (15.4 s/block, 12 reps/block, 3 blocks/movement type + 3 fixation blocks)

**Preprocessing:**
- Slice-timing correction; odd/even slice intensity correction; intensity normalization (mode = 1000)
- Rigid-body motion correction within run
- Atlas registration: native → T2w → T1w → Talairach 711-2B
- Field map distortion correction (per-session mean field map)
- Frame censoring: FD > 0.2 mm (FD calculated as |Δtrans| + |Δrot × π/180 × 50mm|)
- Nuisance regression: whole-brain signal, ventricular signal, white matter signal + Volterra expansion motion regressors
- Least-squares spectral estimation for censored frames; temporal bandpass: 0.009–0.08 Hz
- Surface projection: mid-thickness, ribbon-constrained; geodesic 2D smoothing σ = 2.55 mm (≈ 6 mm FWHM) on cortex; Euclidean 3D σ = 2.55 mm on subcortex

**Individual parcellations:**
- Already computed for MSC01–MSC10; available at `derivatives/surface_pipeline/sub-MSC{01-10}/surface_parcellation/`

**Network assignment:**
- Infomap community detection at density thresholds 0.3%–5%; consensus across thresholds
- Parcels assigned to 17 canonical networks including: Somatomotor-Hand, Somatomotor-Foot, Somatomotor-Face, Premotor, Cingulo-Opercular, etc.

### 4. Perinatal Stroke (Laumann et al. 2021; subject PS1 = `sub-CIMT001`, OpenNeuro ds004498)

**Subject / clinical:** Adolescent male with **bilateral** perinatal arterial strokes (~3 weeks postnatal); extensive bilateral cystic cortical lesions, ~20% supratentorial volume loss; near-typical neurodevelopment with a mild right upper-limb deficit. Right-hand motor cortex is displaced to the left postcentral gyrus; left-hand cortex preserved. (See clinical context note above.)

**Acquisition (confirmed from on-disk BIDS sidecars):**
- Same scanner and protocol as MSC: 3T Siemens Trio, 4 mm isotropic, TR = 2.2 s, EchoTime 27 ms, 36 slices (`ProtocolName: RSFC_216Frames_36Slices_TR2.2_MoCo`)
- 18 sessions, longitudinal (~500-day span)
- **Both resting-state and motor task fMRI exist on disk:** 64 `task-rest` runs + **56 `task-motor` runs across ~15 sessions** (285 min RSFC + 137 min task per the paper)
- **Motor task design is identical to MSC:** 5 effectors — `RHand`, `LHand`, `RFoot`, `LFoot`, `Tongue` — in 15.4 s blocks (verified from `*_task-motor_*_events.tsv`). This means the **task-defined ROI method (Section A) applies directly to PS1**, not just the RSFC fallback.
- Field maps (phasediff/magnitude) present for distortion correction.
- Source data: `datasets/perinatal_stroke/sub-CIMT001/`

**Preprocessing:**
- Apply the identical pipeline to MSC (same acquisition parameters):
  - Slice-timing correction, motion correction, atlas registration, field map distortion correction
  - Frame censoring: FD > 0.2 mm
  - Nuisance regression: WB + WM + ventricle + Volterra motion
  - Bandpass: 0.009–0.08 Hz
  - Surface projection + 6 mm FWHM smoothing
- Process resting-state and motor-task runs identically; reserve task runs for the GLM localizer, rest runs for connectivity.

**Individual parcellations:**
- Compute de novo using Gordon et al. 2017 gradient-watershed method (see below) on the concatenated resting-state timeseries.
- **Caveat:** With bilateral cystic lesions, gradient-watershed boundaries near lesioned tissue will be unreliable; restrict parcellation to surviving cortex and verify against the subject's own task activation (which Laumann et al. found aligns with individual functional network boundaries even where displaced).

---

## My Preprocessing Plan

**Goal:** Coherent preprocessing across MSC, cast, and perinatal stroke so that CL trees and FC matrices are directly comparable.

### Common pipeline (all datasets)

1. **Motion correction:** rigid-body, within-run
2. **Atlas registration:** Talairach 711-2B via T1w/T2w
3. **Distortion correction:** field map-based (or AP/PA spin-echo EPI for cast2/cast3)
4. **Nuisance regression:** whole-brain signal + WM + ventricle + Volterra-expanded motion parameters (6 params + 6 first derivatives + 6 squared + 6 squared derivatives = 24 motion regressors) — same model as Gordon et al. 2017 / Newbold et al. 2020
5. **Frame censoring:** FD > 0.2 mm; linearly interpolate censored frames before filtering
6. **Temporal filtering:** bandpass 0.009–0.08 Hz (MSC/CIMT001); 0.005–0.1 Hz (cast2/cast3 with TR=1.1s)
7. **Surface projection:** ribbon-constrained sampling to mid-thickness; geodesic 2D σ = 2.55 mm
8. **Cerebellar/subcortical:** 3D Gaussian σ = 2.55 mm (≈ 6 mm FWHM)

### Dataset-specific notes

| Dataset | Special considerations |
|---------|----------------------|
| MSC | Already preprocessed; verify bandpass and confound model match above; re-extract if needed |
| cast1 | Same as MSC; use MSC02 surface files |
| cast2 | Same as MSC; use MSC06 surface files |
| cast3 | Compute surface files de novo; apply distortion correction using AP/PA EPI pairs |
| PS1 (`sub-CIMT001`) | Same parameters as MSC; apply same pipeline; compute individual parcellation. **Has motor task data** → use task-defined ROIs (Section A). Localise from empirical activation peak (representations are displaced); restrict parcellation to surviving cortex. |

### Individual surface parcellations (Gordon et al. 2017 gradient-watershed method)

For cast3 and CIMT001 (parcellations not yet computed):

1. Concatenate preprocessed resting-state timeseries across all sessions
2. Generate seed FC maps from every surface vertex
3. Compute spatial gradient maps of FC similarity between all vertex pairs
4. Apply watershed edge detection to gradient maps; sum resulting edge maps
5. Re-apply watershed to build initial parcels
6. Merge neighboring parcels with edge counts below 50th percentile threshold
7. Calculate mean timecourse per parcel
8. Assign parcels to canonical networks using Infomap at density thresholds 0.3%–5%

For MSC/cast1/cast2: parcellations already exist in the MSC derivatives.

---

## My Analysis Plan: Same Seeds and Regions as Authors

The goal is to define subject-specific ROIs for each of the following regions, **using the same methods as Newbold et al. 2020 (task-based) and the inter-effector study (RSFC-based)**, then compare CL trees vs. FC across datasets.

### ROI Set

| Region | Abbrev | Method | Hemisphere |
|--------|--------|--------|-----------|
| Primary motor cortex, hand | M1_hand | Task (Newbold) | Bilateral |
| Primary motor cortex, foot | M1_foot | Task (Newbold) | Bilateral |
| Primary motor cortex, face/tongue | M1_face | Task (Newbold) | Bilateral |
| Supplementary motor area | SMA | Network parcel (Infomap) | Medial wall |
| Dorsal premotor cortex | PMd | Network parcel (Infomap) | Bilateral |
| Ventral premotor cortex | PMv | Network parcel (Infomap) | Bilateral |
| Motor thalamus | Thal | FC-based (inter-effector method) | Bilateral |
| Superior cerebellar (hand/foot) | Cereb_hand / Cereb_foot | Task (Newbold) / FC-based | Bilateral |
| Inferior cerebellar (tongue) | Cereb_face | Task (Newbold) / FC-based | Bilateral |

**Total: ~18 ROIs** (bilateral for all except SMA). This gives a 17-edge CL tree.

---

### A. Cortical Motor ROIs — Task-Defined (Newbold et al. 2020 Method)

**Applicable datasets:** MSC (all 10 subjects), cast (all 3 subjects), **and PS1 / perinatal stroke (motor task data confirmed on disk — same 5-effector design as MSC).**

**Step 1 — Motor task GLM:**
Run a block-design GLM with separate regressors for each effector condition:
- Conditions (exact `trial_type` labels in the events files, identical across MSC/cast/PS1): `RHand`, `LHand`, `RFoot`, `LFoot`, `Tongue`
- Regressors: boxcar convolved with HRF; duration = 15.4 s per block
- Include standard nuisance regressors: 6 motion params + derivatives, WB/WM/ventricle signals, frame spike regressors
- Concatenate across all task sessions/runs (PS1: ~15 sessions, 56 runs) for a high-SNR per-effector contrast

**Step 2 — Find peak vertex:**
For each effector contrast (e.g., `RHand > rest`), identify the vertex with the highest task synchrony (Z-score or t-value).
- **Healthy subjects (MSC, cast):** restrict the peak search to the somatomotor parcellation boundary (individual gradient-watershed parcel + Infomap assignment) near the expected pre/post-central location.
- **PS1 (perinatal stroke): DO NOT constrain to the expected anatomical location.** Representations are displaced (right hand sits in the *left postcentral gyrus*, not left precentral M1). Search the whole surviving somatomotor + adjacent association cortex for the activation peak, and place the ROI wherever the subject's own peak lands. The displaced location is a finding to report, not an error to correct. (Laumann et al. found PS1's task responses align with his individual functional-network boundaries even where displaced.)

**Step 3 — Expand to 400 vertices (Newbold et al. 2020):**
Starting from the peak vertex, expand to the N = 400 contiguous vertices with the highest task activation. For healthy subjects, constrain to anatomically plausible pre/post-central cortex; for PS1, constrain only to contiguous surviving cortex around the empirical peak (no a priori anatomical constraint).

This is the exact method from Newbold et al. 2020: *"the vertex/voxel showing maximal task synchrony then expanded to a preset size (400 vertices in the somatomotor cortex, 40 voxels in the cerebellum)."*

**Step 4 — Report centroid MNI coordinates:**
Compute the weighted centroid of each 400-vertex ROI and report in MNI space. For PS1, report the **displacement vector** (Δ from the MSC group-mean centroid) per effector — this quantifies the reorganisation directly.

**Expected cortical somatotopy (MNI) — healthy reference:**

| ROI | Expected centroid | Landmark |
|-----|------------------|----------|
| M1_foot (bilateral) | ±5, −25, +68 | Paracentral lobule, area 4 |
| M1_hand (bilateral) | ±38, −22, +58 | Hand knob, area 4 |
| M1_face/tongue (bilateral) | ±56, −6, +28 | Lateral precentral gyrus |

**PS1 deviations to expect (from Laumann et al. 2021):**
- **R_M1_hand:** displaced to the **left postcentral gyrus** (posterior, atypical) — large displacement from the canonical right-hemisphere hand knob frame, reflecting remapping of right-hand control into spared left-hemisphere tissue. (Note the laterality framing carefully: right-hand control normally lives in the *left* hemisphere; in PS1 it is displaced *within/around* the lesioned left hemisphere to the postcentral gyrus.)
- **L_M1_hand (left-hand control):** preserved near the canonical right-hemisphere central sulcus.
- **Foot / tongue:** not characterised in the paper; localise empirically and report whether displaced or preserved.

---

### B. (Reserved) RSFC-Based Seeding — Fallback Only

Originally this section was the perinatal-stroke ROI method under the assumption that PS1 had no task data. **That assumption was wrong: PS1 has full motor-task data, so PS1 uses the task-defined method in Section A.** Keep the RSFC-seed approach only as a *fallback* for any effector whose task activation is too weak/absent in surviving cortex (e.g., if a foot or tongue representation cannot be localised from the GLM):

- Seed from the contralateral preserved homolog (or the inter-effector intermediate region) and follow the inter-effector RSFC procedure: Fisher-Z seed FC map, threshold at the 80th percentile, 400-vertex expansion.
- If an effector ROI still cannot be defined, mark it **absent** (NaN in the connectivity matrix) and compute the CL tree on the remaining nodes — the absence is itself informative.

---

### C. Supplementary Motor Area (SMA) and Premotor Cortex (PMd, PMv)

**Method:** Individual-specific network parcellation (Gordon et al. 2017 + Infomap)

**Step 1 — Network assignment:**
From the individual gradient-watershed parcellation, assign each parcel to one of the 17 canonical networks using Infomap. The "Premotor" network (which in the MSC Infomap solution includes SMA/pre-SMA, PMd, and PMv) provides the initial parcel set.

**Step 2 — Task refinement (all three datasets — PS1 included, since it has task data):**
For MSC, cast, and PS1, verify premotor parcel assignments using the motor task GLM. Parcels in the premotor network that are activated by all effectors (foot, hand, face) are SMA; those activated preferentially by hand/foot are PMd; those activated preferentially by face/hand are PMv. For PS1, allow these parcels to fall in displaced/atypical locations (frontal association cortex remapping was reported).

**Step 3 — FC-based cross-check (especially PS1):**
Additionally identify premotor parcels from the Infomap network assignment and verify by computing FC with the hand/foot/face M1 ROIs. SMA should show high FC with all M1 regions; PMd should show preferential FC with M1_hand and M1_foot. For PS1, use this FC cross-check to resolve premotor parcels in lesion-adjacent cortex where the gradient-watershed boundaries are unreliable.

**ROI size:** Use all vertices within the assigned parcel (not size-matched). This may introduce size heterogeneity — address in ROI size confound analysis (see `roi_size_confound_plan.md`).

---

### D. Motor Thalamus

**Method:** FC-based identification (inter-effector study method) + Morel atlas labeling

The inter-effector study identifies motor thalamic nuclei by seeding from motor cortex ROIs and thresholding the resulting volumetric FC maps. Thalamic nuclei identified: CM, VIM, VPM, VPI. Thresholds are set per-subject: Z(r) > 0.03–0.35 depending on subcortical SNR.

**Step 1 — Seed FC from motor cortex:**
Compute volumetric FC between each cortical motor ROI (M1_hand, M1_foot, M1_face, SMA) and all thalamic voxels.

**Step 2 — Apply per-subject threshold:**
Set the threshold at the 95th percentile of all FC values within the thalamus (per-subject normalization). This matches the inter-effector paper approach of varying thresholds per subject due to variation in subcortical SNR.

**Step 3 — Identify thalamic clusters:**
Find connected clusters of above-threshold thalamic voxels. Cross-reference with the **Morel thalamic atlas** (already in the pipeline at `atlases/MorelAtlasMNI152/`) to assign cluster labels: VLa, VLpd, VLpv (motor relay), VAmc, VApc (basal ganglia relay), CM, VIM, VPM, VPI.

**Step 4 — Define motor thalamus ROI:**
Take the union of VLa + VLpd + VLpv voxels (primary motor relay nuclei) per hemisphere as the motor thalamus ROI. Report centroid MNI coordinates.

**For PS1:** Apply the same procedure, seeding from the (possibly displaced) cortical motor ROIs defined in Section A. Expect altered thalamo-cortical FC given the bilateral cortical injury; thalamus itself may be relatively spared (lesions were cortical/cystic), so motor thalamus may be a useful preserved anchor node. Report whether thalamo-cortical edges track the displaced cortical representations.

---

### E. Cerebellar Motor ROIs

**Method:** Task-defined (Newbold et al. 2020) for MSC/cast **and PS1** (all have motor task data); FC-based only as a fallback.

**Task-defined (MSC, cast, PS1):**
Following Newbold et al. 2020:
1. Run motor task GLM on cerebellar voxels
2. For each effector contrast, find the voxel with maximal task synchrony in the superior cerebellum
3. Expand to 40 contiguous voxels with highest task activation (Newbold exact method: *"expanded to a preset size... 40 voxels in the cerebellum"*)
4. Constrain expansion to the appropriate anatomical region:
   - Hand and foot: superior cerebellum (lobule V, SUIT parcels 5/6), per Buckner et al. 2011
   - Face/tongue: lobule VI (SUIT parcel 7/8), per Stoodley & Schmahmann 2009
5. **PS1:** the cerebellum was not part of the supratentorial injury, so cerebellar somatotopy is likely preserved — making cerebellar ROIs a useful set of intact anchor nodes against which to measure cortical displacement. Still localise from the subject's own peak.

**FC-based fallback (any effector with insufficient cerebellar task activation):**
1. Seed FC from the corresponding cortical motor ROI
2. Apply cerebellar mask (SUIT atlas)
3. Threshold at 95th percentile of cerebellar FC values per subject (inter-effector approach)
4. Take the peak cluster in the expected cerebellar zone as the cerebellar motor ROI

**Expected cerebellar somatotopy (MNI):**

| ROI | Expected MNI | SUIT lobule |
|-----|-------------|-------------|
| Cereb_foot | ±5, −60, −20 | Lobule V (medial) |
| Cereb_hand | ±25, −58, −25 | Lobule V (lateral) |
| Cereb_face | ±20, −55, −28 | Lobule VI |

---

## Coherence Across Datasets

To enable direct comparison of CL trees and FC across MSC, cast, and perinatal stroke:

1. **Same ROI labels:** All subjects receive the same 18-label ROI scheme (M1_hand L/R, M1_foot L/R, M1_face L/R, SMA, PMd L/R, PMv L/R, Thal L/R, Cereb_hand L/R, Cereb_face L/R, Cereb_foot L/R — or subset thereof if some are absent/lesioned)

2. **Same timeseries extraction:** NiftiLabelsMasker with standardization (z-score), bandpass-filtered, frame-censored timeseries; same confound model

3. **Same MI/FC computation:** same histogram MI estimator (K=100 bins), same Pearson FC, same CL tree construction (maximum_spanning_tree with kruskal)

4. **Handling absent/displaced/lesioned ROIs:** If an ROI is absent (cannot be localised in surviving cortex for PS1), treat as NaN in the connectivity matrix and compute the CL tree on the remaining nodes. **Displaced** ROIs (e.g., PS1 right-hand in left postcentral gyrus) keep their effector label but carry an atypical centroid — connectivity is computed from the displaced location, and the displacement is tracked as a separate variable.

---

## Analysis: CL Tree vs. FC Across Conditions

### Question 1: Is the motor CL tree conserved across healthy subjects?
- **Data:** MSC, all 10 subjects, all-session connectivity
- **Method:** Jaccard similarity of CL edge sets across subjects; compare to FC top-N Jaccard
- **Prediction:** CL Jaccard > FC Jaccard (core hypothesis)

### Question 2: How does the CL tree change with disuse (cast)?
- **Data:** cast subjects (pre / cast / post conditions); treat each condition as a separate connectivity estimate
- **Method:** Compare CL tree topology and MI edge weights between pre, cast, and post periods
  - Which edges are disrupted during casting? (expected: ipsilesional/unilateral arm edges)
  - Does the tree hub shift during casting? (SMA or contralesional M1 may gain degree)
  - Does recovery (post) restore the pre-cast topology?
- **Prediction:** Casting disrupts the right-hand bilateral M1_hand edge; right cerebellar-M1_hand connection also disrupted. CL tree detects this more cleanly than FC because it identifies the direct connection loss without propagating through the entire dense FC matrix.
- **Note:** cast1=MSC02 and cast2=MSC06 also have MSC baseline data, providing the cleanest within-subject pre-cast reference

### Question 3: Does the CL tree track motor reorganisation after bilateral perinatal stroke?
- **Data:** PS1 (`sub-CIMT001`, ~18 sessions, rest + task); MSC (10 healthy controls)
- **Method:** Build PS1's motor CL tree from subject-specific (possibly displaced) ROIs and compare to the MSC group consensus CL tree.
  - Does the **displaced** R_M1_hand node (in left postcentral gyrus) still form the bilateral hand edge (R_M1_hand ↔ L_M1_hand), or is that edge broken/rerouted?
  - With displaced right-hand control, does R_M1_hand instead connect to premotor / parietal association nodes (the regions where Laumann et al. reported remapping)?
  - Which canonical edges are preserved (predict: the intact left-hand and cerebellar edges) vs. altered?
  - Does the CL hub shift relative to controls?
- **Prediction (anchored to Laumann et al. 2021):** The right-hand representation's displacement reshapes its CL edges — R_M1_hand attaches to atypical frontal/parietal partners rather than the canonical hand-knob neighbourhood. Cerebellar and motor-thalamus nodes (spared by the cortical injury) remain stable anchors. The CL tree should expose this rerouting more interpretably than the dense FC matrix, because a single displaced node changes a small number of tree edges rather than smearing across all pairwise correlations.
- **Validation tie-in:** Laumann et al. found intact corticospinal tracts from *both* the typical (left-hand) and displaced (right-hand) cortical sites (DTI). If CL edges track direct anatomical connections, the displaced R_M1_hand node should still show a strong CL edge to motor thalamus / cerebellum consistent with its preserved descending tract — a concrete structural-vs-functional cross-check.

### Question 4: Does PS1's CL tree topology change across the longitudinal scan period?
- **Data:** PS1, ~18 sessions spanning ~500 days (rest + task)
- **Method:** Compute the CL tree per session (or per session-pair for stability) and track edge presence and the R_M1_hand centroid over time.
- **Prediction:** If reorganisation is ongoing/stabilising across adolescence, the displaced-node edges and/or its centroid should shift systematically (beyond session-to-session noise); if the remapping is already fixed by age 15, the topology should be stable across sessions (a useful null that also demonstrates CL test–retest reliability in a single subject).

### Question 5: Do CL trees predict inter-effector connectivity?
- **Data:** All datasets
- **Method:** Identify inter-effector hub nodes (per inter-effector paper: CON-connected regions between M1 zones) in the CL tree. Does the CL tree capture the same inter-effector intermediate region as the RSFC analysis?
- **Prediction:** CL tree edges from M1_hand → M1_foot (if present) pass through an inter-effector intermediate; CL tree identifies direct inter-effector coupling

---

## Key References

- Gordon EM et al. (2017). Precision functional mapping of individual human brains. *Neuron* 95:791–807. — MSC dataset and gradient-watershed parcellation
- Newbold DJ et al. (2020). Plasticity and spontaneous activity pulses in disused human brain circuits. *Neuron* 107:580–589. — Cast dataset; 400-vertex task-defined ROI method
- Laumann TO, Ortega M, Hoyt CR, Seider NA, Snyder AZ, Dosenbach NUF (2021). Brain network reorganisation in an adolescent after bilateral perinatal strokes. *Lancet Neurology* 20(4):255–256. PMC13182026. — Perinatal stroke subject PS1 (= `sub-CIMT001`); bilateral cystic cortical lesions (~20% volume loss); right-hand motor cortex displaced to left postcentral gyrus; rest + motor task fMRI; MSC subjects as controls
- PMC10172144 — Inter-effector study: seeds along precentral gyrus; top-1% task activation labeling; thalamic nuclei CM/VIM/VPM/VPI via FC; cerebellar lobule V and VIIb/VIIIa
- Buckner RL et al. (2011). Organization of human cerebellum. *J Neurophysiol* 106:2322–2345. — Cerebellar somatotopy (hand/foot/tongue lobules)
- Morel A (2007). Stereotactic Atlas of the Human Thalamus. — Thalamic atlas for VLa/VLpd/VLpv labeling
