# Transcoder circuit tracing of occlusion features in π0.5 — full research report

**Repository:** `MarthalaSaiKavya/smaj-research` · **Run date:** 2026-09-23 (Colab, 1× A100-SXM4-80GB) · **Status:** Goals 1 passed, Goal 2 did not pass (null), Goal 3 not run (gated)

Every number below comes from a file in this repository. The derived numbers (ratios, per-frame breakdowns, exact tests) are recomputed by `analysis/derive_stats.py`, which writes `analysis/derived_stats.json`. File paths are given next to each result.

---

## 1. Executive summary

We asked whether a vision-language-action (VLA) policy, π0.5, represents *occlusion* of a task-relevant object with sparse, identifiable internal features, and whether those features are what actually changes the robot's action when the object is hidden. We applied **transcoders** (sparse replacements of MLP layers) and **causal patching** to π0.5's 18-layer Gemma-2B language backbone on a LIBERO task, with pixel-controlled image edits of the target bowl.

1. **The instrument works.** All five pre-registered validity checks passed: edits are pixel-clean, the forward pass is bit-exact deterministic, hiding the bowl changes the action by 2.9–4.4× the noise-seed baseline, a full-layer swap recovers up to 91.8% of that change, and the transcoders are faithful (median held-out FVU 0.174; splicing any single transcoder in costs ≤16.1% of the occlusion effect).
2. **Occlusion-selective features exist (Goal 1 passes).** 72 of 9,216 transcoder features (0.78%) across all 18 layers rise when the bowl is painted over and stay comparatively quiet when it is recolored, removed, or when the same paint is placed elsewhere (median specificity 0.64).
3. **They are causal but explain little (Goal 2 null).** Patching only those 72 features into the unoccluded run closes 3.9% of the action gap, 6.9× more than norm-matched random features (0.56%) and 5.9× more than color features (0.66%), and is concentrated on the bowl's image tokens. But the plan required +10 percentage points over controls; the margin is +3.3. With 4 frames the best achievable exact one-sided p-value is 0.0625, so the ~7× ratio is suggestive, not significant.
4. **The effect is distributed.** Swapping all MLP outputs recovers 65.5% of the gap and all transcoder features together recover 40.1%; the occlusion-selective subset carries ~10% of what the transcoder features carry. The occlusion signal flows through many non-selective features.
5. **Important interpretive limit.** The "occlusion" features respond identically to paint with or without the bowl underneath (correlation 0.9986). By construction these two images differ in ≤0.39% of pixels (0 pixels in one frame). The features are best described as "uniform patch at the target location" detectors, not object-permanence representations.

**Paper claim this supports:** *The first transcoder-based circuit analysis of a VLA policy, showing that occlusion-selective sparse features exist in π0.5's LLM backbone and causally move the action above chance, but account for only a small fraction of the occlusion effect, which is carried by a distributed set of non-selective MLP features.*

---

## 2. Problem statement

### 2.1 The scientific question
When a robot's target object becomes hidden, a good policy must either (a) keep acting on a remembered/inferred object, or (b) change its behavior. Internally, a VLA must represent *that* the object is not visible. We ask:

- **Q1 (existence):** Does π0.5 contain sparse, human-identifiable features that selectively encode "the target object is occluded", as opposed to generic visual change, color change, object absence, or occluder appearance?
- **Q2 (causality):** Are those features what drives the change in the action when the object is occluded, measured by patching only those features?
- **Q3 (behavior):** Does ablating or amplifying them change closed-loop task success? (Planned; not run because Q2 did not pass its gate.)

### 2.2 Why it matters
- VLAs are being deployed as generalist robot controllers, but we have very little mechanistic insight into how visual scene structure (visibility, occlusion) reaches the action head.
- Occlusion is a safety-relevant failure mode: a policy that silently mis-handles a hidden object can collide or drop things.
- Existing VLA interpretability uses SAEs, linear probes, steering vectors, activation injection or attention knockout (Section 6). None uses **transcoders**, which decompose the *computation* of MLP layers rather than the state of the residual stream, and therefore allow circuit-style analysis (feature → feature → action).

### 2.3 Pre-registered plan and gates
The study followed a written "fast-track" plan with explicit gates:

| Stage | Question | Gate |
|---|---|---|
| Checks 1–4 | Is the measurement instrument sound (clean edits, determinism, real signal, working ceiling)? | Must pass before Goal 1 |
| Check 5 | Are the transcoders faithful? | Must pass before Goal 1 |
| Goal 1 | Do occlusion-selective features exist? | If none, stop: negative result |
| Goal 2 | Do they move the action ≥10 pts more than random and color features? | If not, stop: Goal 2's null is the finding |
| Check 6 + Goal 3 | Does closed-loop success shift? | Only if Goal 2 passes |

Rule of thumb from the plan: *if checks 1–4 pass, a null at Goal 1/2 is a finding; if any check fails, a null is just a bug.* All checks passed.

---

## 3. Experimental setup (from `code/config.json`, `env/`, `results/status/*.json`)

| Item | Value |
|---|---|
| Policy | `lerobot/pi05_libero_finetuned` (π0.5, PaliGemma-3B backbone + 300M action expert, flow matching, 10 denoising steps, 50-step action chunks) |
| Precision | bfloat16 weights; `torch.compile` disabled so hooks run eagerly |
| Software | lerobot 0.6.1, transformers 5.5.4, torch 2.11.0+cu128, mujoco 3.8.1, robosuite 1.4.0, numpy 2.2.6, Python 3.12.14 |
| Hardware | 1× NVIDIA A100-SXM4-80GB (Colab) |
| Benchmark | LIBERO-Spatial, task 0: *"pick up the black bowl between the plate and the ramekin and place it on the plate"* |
| Target object | `akita_black_bowl_1` (from the task's BDDL `obj_of_interest`; a second, distractor bowl is present) |
| Cameras | agentview + wrist (eye-in-hand), 256×256; the policy also has one empty camera slot |
| Prefix layout | 968 tokens = 3 image slots × 256 SigLIP tokens + 200 language/state tokens (empty camera and padding masked) |
| LLM layers analysed | all 18 Gemma-2B decoder layers (the plan said "32"; the model has 18) |
| Seeds | global seed 0; one fixed flow-matching noise tensor per probe frame |

### 3.1 Probe frames (`results/frames_candidates.json`, `results/status/frames.json`, `results/frames.pkl`)
Frames come from replaying the LIBERO demonstration HDF5 (demo_0: 98 steps, bowl lifted at t=51; demo_1: 84 steps, lifted at t=47). For each demo, 8 evenly spaced pre-grasp candidates were scored by the base-vs-occluded action RMSE, and the two with the largest gap were kept (minimum spacing 5 steps):

| Frame | Episode | t | Bowl px agentview (% of image) | Bowl px wrist (% of image) | Selection gap |
|---|---|---|---|---|---|
| 0 | demo_0 | 35 | 491 (0.75%) | 7,482 (11.4%) | 0.520 |
| 1 | demo_0 | 41 | 354 (0.54%) | 13,257 (20.2%) | 0.516 |
| 2 | demo_1 | 5 | 779 (1.19%) | 2,662 (4.1%) | 0.666 |
| 3 | demo_1 | 32 | 437 (0.67%) | 6,801 (10.4%) | 0.522 |

All 16 candidate gaps ranged 0.152–0.666. Selecting the largest-gap frames maximizes the signal for Checks 3–4, but it is a selection step to disclose: the frames are not a random sample.

23 additional unedited frames (12 + 11 evenly spaced across both demos) were used only for transcoder training.

### 3.2 The six image conditions (`results/fig_probe_conditions_agentview.png`, `..._wrist.png`)
All edits are applied in pixel space to both cameras, using a MuJoCo segmentation mask of the target bowl rendered from the same camera (our own decoder; robosuite 1.4's segmentation path overflows under numpy 2).

| Condition | Definition | What it controls for |
|---|---|---|
| `base` | unedited render | reference |
| `recolor` | bowl pixels shaded red (RGB 200,40,40, luminance-preserving) | object still visible, appearance changed |
| `absent` | bowl moved out of the scene and re-rendered; composited back only inside the bowl mask dilated by 6 px (so its shadow disappears too) | object truly gone |
| `occluded` | bowl mask dilated by 2 px, filled with flat gray (128) | the manipulation of interest |
| `slab_miss` | the identical gray shape translated off the bowl (away from bowl and robot) | "a gray patch appeared somewhere" |
| `occluded_absent` | `absent` image + the same gray paint | paint with nothing underneath |

Pixel facts (`analysis/derived_stats.json → pixels`): edited regions cover 0.5–1.5% of the agentview image but 4–22% of the wrist image. **`occluded` and `occluded_absent` differ in only 0–10 agentview pixels and 0–257 wrist pixels (≤0.39%), and are pixel-identical in frame 2**, because the 2-px-dilated paint already covers the bowl and only a thin shadow ring differs. In frame 1 the wrist `slab_miss` had no free room, so the wrist camera is unedited there. In frames 0 and 3 the wrist `slab_miss` partly covers the plate.

### 3.3 The metric
For each frame we run the policy with the same noise on two inputs and compare the 50×7 normalized action chunks:
- **gap** = RMSE(a_base, a_occluded)
- **gap closed (%)** of a patched run = 100 × (1 − RMSE(a_patched, a_target) / gap), where "inject" patches base → occluded (target = a_occluded) and "remove" patches occluded → base (target = a_base).

---

## 4. Methodology

### 4.1 Hooks and determinism
Forward hooks are placed on every Gemma decoder layer (residual output) and its MLP (input = post-attention RMSNorm output; output = MLP output before the residual add). The hooks fire only during the prefix pass (images + prompt), which fills the KV cache that the action expert reads at every denoising step. The prefix pad mask is captured by wrapping `embed_prefix`. The checkpoint config enables `torch.compile(mode="max-autotune")`, whose CUDA graphs reuse output buffers and bypass Python hooks, so compilation is disabled.

### 4.2 Validity checks (`results/validate.json`, `results/fig_check4_ceiling.png`)
- **Check 1 (clean edits):** for 4 frames × 5 edits × 2 cameras (40 rows), the fraction of changed pixels outside each edit's allowed region is 0.0 everywhere. Robot state and prompt tokens are identical across conditions. The env's pixels equal a direct re-render, and 0 bowl pixels remain in the `absent` render.
- **Check 2 (determinism):** two forward passes on the same input and noise give max |Δ| = 0.0 on all frames.
- **Check 3 (signal):** base-vs-occluded RMSE 0.520, 0.516, 0.666, 0.522 (threshold 0.02), versus 0.127–0.178 when only the noise seed changes (ratio 2.9–4.4×). For reference, removing the bowl changes the action more than occluding it (0.831–1.060; 1.45–2.04× the occlusion gap). Recolor gives 0.074–0.392, slab_miss 0.089–0.337, occluded_absent 0.493–0.666.
- **Check 4 (ceiling):** replacing the entire residual stream of all prefix tokens at layer L with the occluded run's closes 82.3% (L0), 83.8, 85.8, **91.8% (L3)**, 91.3, 80.4, 66.2, 65.9, 56.8, 32.4, 20.4, 19.5, 12.9, 12.1, 1.8, 1.5, 0.4, 0.0% (L17). The information that moves the action is carried in the prefix representations of layers ≤ 8 (closure first drops below 50% at L9 and below 5% at L14). Frame 1 is the outlier at early layers (39.0% at L0 vs 94–100% for the others).
- **Same run, MLP-only swap:** replacing only one layer's MLP output closes 0.0–6.5% (max 6.5% at L8). **This is the ceiling of any single-layer transcoder intervention**, and it becomes central in Section 5.3.

### 4.3 Transcoders (check 5; `results/train.json`, `results/status/train.json`, `results/fig_check5_transcoders.png`, `results/transcoders/`)
- **Architecture:** one TopK transcoder per layer mapping MLP input (2048-d) to MLP output (2048-d): 512 features, k=16, ReLU on the top-k pre-activations, unit-norm decoder rows, inputs and outputs mean-centred and scaled to unit mean norm.
- **Training:** 3,000 Adam steps (lr 2e-3 with warm-up and final decay), batch 1,024, AuxK loss on dead features (k_aux=64, coefficient 1/32). ~9 s per layer.
- **Data:** 26,852 valid prefix tokens from 47 forward passes (4 frames × 6 conditions + 23 extra frames); 24,167 train / 2,685 held-out tokens.
- **Results:** held-out FVU per layer 0.018 (L0), 0.149, 0.158, 0.171, 0.163, 0.156, 0.198, 0.152, 0.202, 0.199, **0.258 (L10, worst)**, 0.199, 0.233, 0.188, 0.182, 0.177, 0.022, 0.076 (L17). Median **0.174** (gate ≤ 0.20). All 512 features are alive in every layer. Train/held-out gaps are small except L17 (0.016 vs 0.076).
- **Functional faithfulness (splice test):** replacing a layer's MLP with its transcoder, without the error term, changes the action by 0.000–0.161 × the occlusion gap (worst L13). L17's splice has zero effect because the last layer's MLP output does not feed any key/value used by the action expert.
- **History:** a first run at 400 steps gave median FVU 0.209 (gate failed by 0.009). The gate was met by training longer (3,000 steps), not by relaxing the threshold.
- **Scale note:** MLP-output RMS norms grow from ~36–60 in early layers to 435 (L15), 1,785 (L16) and 9,375 (L17), consistent with the massive-activation phenomenon in late Gemma layers.

### 4.4 Goal 1: feature selection (`results/goal1_feature_table.csv`, `results/goal1_selected.json`, `fig_goal1_*.png`)
For each feature, activation is summed over all valid prefix tokens per (frame, condition), and Δ_c = A_c − A_base per frame. A feature **passes** if:
1. Δ_occluded > 0 on all 4 frames;
2. specificity = 1 − max_c mean|Δ_c| / mean Δ_occluded ≥ 0.5, over c ∈ {recolor, absent, slab_miss};
3. rise ≥ 1 typical token-firing (mean Δ_occluded / the feature's mean non-zero activation).

Up to 8 features per layer are kept, ranked by specificity × rise. A size-matched **color** control set is ranked the same way with recolor as the target. The top 3 layers by summed score are carried to Goal 2. `occluded_absent` is recorded but not used for selection.

### 4.5 Goal 2: causal patching (`results/goal2_rows.csv` (736 rows), `goal2_circuit_trace_table.{csv,md}`, `goal2_edges.json`, `fig_goal2_contact_sheet.png`)
For each frame, the base and occluded runs are captured. Patches edit the MLP output as y' = y + Σ_{j∈S} (f_j^target − f_j^live) d_j (decoder rows d_j), which keeps the transcoder error term. Methods:
- **occ_features:** the Goal-1 set S.
- **color_features:** the size-matched color set.
- **random_same_norm:** 5 random feature sets of the same size, rescaled per token to the occlusion set's delta norm.
- **random_raw:** the same random sets, unscaled.
- **all_tc_features:** all 512 features.
- **mlp_swap_ceiling:** the whole MLP output.
- **full_token_swap_ceiling:** the whole layer output.

Token scopes: all valid tokens; bowl tokens (image tokens overlapping the painted region, ≥10% of the patch); everything else. Directions: inject (base→occluded) and remove (occluded→base). Also measured: single features, the top-3 layers jointly, cross-layer "edges" (does injecting S at L1 turn on S at L2?), and **all 18 layers jointly** (72 features).

The **all-layers site was added after the first Goal 2 run**, once the data showed the single-layer MLP ceiling (3–6%) makes the +10-pt criterion unreachable at any single layer. The criterion itself was not changed.

---

## 5. Results

### 5.1 Validity (all pass): see 4.2 and 4.3.

### 5.2 Goal 1: occlusion-selective features exist (PASS)
- **Count:** 72 of 9,216 features pass (0.78%): 4, 3, 3, 4, **8 (L4)**, 6, 6, 6, 2, 2, 5, 4, 2, 2, **7 (L14)**, 2, 2, 4 per layer (`fig_goal1_counts.png`). For context, 962 features have Δ_occluded > 0 on all four frames and 171 reach specificity ≥ 0.5; the other criteria bring that down to 72.
- **Strength:** specificity 0.50–0.98 (median 0.64); rise 1.4–127.8 token-firings (median 21.9). The strongest are L15 f96 (rise 127.8, spec 0.71), L0 f274 (69.1, 0.74), L0 f396 (65.7, 0.71), L9 f497 (51.9, 0.86), L1 f69 (52.9, 0.82), L5 f318 (44.3, 0.96) and L3 f76 (40.2, spec 0.98).
- **Relative quietness:** median |Δ|/Δ_occluded = 0.27 for recolor, 0.11 for absent, 0.05 for slab_miss. The features barely respond to the same paint off the bowl, so they are location-bound.
- **Localization:** in the contact sheet (`fig_goal1_contact_sheet.png`, layer 5), the top features fire on the agentview tokens under the painted bowl and are silent there in base/recolor/absent/slab_miss.
- **Caveat:** Δ_occluded_absent / Δ_occluded has median 0.997 (range 0.77–1.15) and correlation 0.9986 across the 72 features (`fig_goal1_features.png`). Given the near-identical inputs (Section 3.2), this says nothing about object permanence; π0.5 sees one frame at a time and cannot know what is under the paint.
- **Top layers:** 5, 0, 4 (score sums 180.8, 128.3, 105.6).

### 5.3 Goal 2: causal effect on the action (NULL under the plan's rule)
Gap closed (inject, all tokens, mean over 4 frames; per-frame values in brackets):

| Site | Occlusion features | Random, same norm | Color | MLP-swap ceiling | Occ. / ceiling | Margin | Pass (+10) |
|---|---|---|---|---|---|---|---|
| Layer 5 (6 feats) | 1.92 [5.88, 0.67, −0.05, 1.16] | 0.27 | 0.46 | 4.15 | 46% | +1.5 | no |
| Layer 0 (4 feats) | 0.18 [0.31, 0.13, −0.12, 0.40] | 0.27 | 0.37 | 3.09 | 6% | −0.2 | no |
| Layer 4 (8 feats) | 1.17 [2.63, 0.44, 0.04, 1.55] | 0.20 | 0.25 | 5.55 | 21% | +0.9 | no |
| Top-3 jointly (18 feats) | 2.47 [5.72, 1.34, −0.16, 2.99] | – | – | – | – | – | – |
| **All 18 layers (72 feats)** | **3.92** [8.63, 1.45, −0.23, 5.82] | 0.56 | 0.66 | **65.53** | 6% | **+3.3** | **no** |

Additional all-layers numbers:
- **All transcoder features: 40.1%** [32.3, 16.0, 62.7, 49.7], i.e. 61% of the MLP ceiling. The occlusion subset is **9.8%** of what all features carry, using 0.78% of the dictionary.
- **Beats every random draw:** the occlusion set beats all 5 random draws in 3/4 frames (frame 0: 8.63 vs max 3.91; frame 1: 1.45 vs 0.87; frame 3: 5.82 vs 1.63). Frame 2 shows no effect (−0.23), although its MLP ceiling is 83%.
- **Exact paired sign-flip test** (occlusion vs random, n=4 frames): p = 0.125 (all layers), 0.0625 (layers 4 and 5), 0.75 (layer 0). **0.0625 is the smallest p-value four frames can produce.**
- **Token structure:** the occlusion features act mostly through the bowl tokens (3.08 of 3.92 pts; non-bowl 0.96). The MLP ceiling splits 20.2% bowl vs 29.4% non-bowl (these do not add up to the 65.5% joint ceiling), so more of the MLP-mediated effect sits at non-bowl tokens than at the bowl's own tokens.
- **Remove direction:** restoring base MLP outputs everywhere recovers **85.6%** toward the base action (MLPs are largely necessary for the effect). Restoring only the occlusion features recovers 7.9% (color: 3.0%). This is noisy: per-frame 13.2, −3.9, −0.7, 23.1.

Single-layer details:
- **Single features:** the largest single-feature effects are L5 f124 (1.12%), L5 f505 (0.87%) and L5 f318 (0.78%). At L5 the single-feature effects sum to 3.0%, more than the joint 1.9% (sub-additive).
- **Cross-layer edges:** injecting the L4 set turns on 2.3% of the L5 set's occlusion activation (random: 0.04%). L0→L4 gives 0.8% (random: −0.3%); L0→L5 gives 0.06% (random: 0.10%). The selected features form at most a weak serial chain.
- **Remove direction (single layers):** noisy and dominated by frame 1 (L0: 36.0% in frame 1, <1% elsewhere).

### 5.4 Goal 3 and check 6: not run
The plan's gate stops after a Goal 2 null (`FORCE_CONTINUE=False`). No closed-loop success numbers exist in this repository.

---

## 6. Novelty and positioning (verified against the literature, Sept 2026)

| Work | Models | Method | Transcoders / circuits? |
|---|---|---|---|
| Häon et al., CoRL 2025 (arXiv 2509.00328) | π0, OpenVLA | project FFN activations onto token embeddings; steering | no |
| Buurmeijer et al., 2026 (2603.05487) | π0.5, OpenVLA | linear probes + optimal-control steering | no |
| Swann et al., 2026 (2603.19183) | VLAs on LIBERO/DROID | SAEs + steering/ablation | no |
| Grant et al., 2026 (2603.19233) | 6 VLAs incl. π0.5 | activation injection, SAEs, probes | no |
| Jin et al., 2026 (2605.17204) | OpenVLA, π0.5 | event-grounded SAEs | no |
| Zhang et al., 2026 (2605.00321) | VLAs | interventional masking attribution | no |
| Shi et al., VLA-Trace 2026 (2605.30117) | π0.5, OpenVLA, LIBERO | CKA, attention knockout, behavior probes | no |
| Damianos et al., 2026 (2605.22902) | Gemma 3 VLM (no actions) | **transcoders + circuit tracing** | yes, but not on a VLA |

**Contributions of this work:**
1. **First transcoder-based analysis of a VLA policy, to our knowledge.** One transcoder per LLM layer of π0.5, validated both by reconstruction (FVU) and by a functional splice test measured on the action.
2. **A controlled occlusion protocol.** Six pixel-exact conditions from simulator segmentation, isolating occlusion from appearance change, absence and occluder appearance, with a pre-registered validity battery (clean edits, bit-exact determinism, signal-over-noise, patching ceiling).
3. **A methodological finding for VLA circuit work: measure each intervention against the ceiling of its own site.** In π0.5, a single layer's MLP carries only 0–6.5% of the occlusion effect, while the full residual stream carries up to 92%. An absolute threshold at a single MLP site is unreachable by construction. We report both the absolute and the ceiling-normalized effect.
4. **A clean negative result.** Selective features exist and are causal above chance, but carry ≈4% of the effect (≈10% of the transcoder-mediated part). The occlusion computation is distributed.

---

## 7. Limitations and threats to validity
1. **Four probe frames, one task, one object.** Error bars are wide; exact tests cannot reach p < 0.05; no across-task generalization.
2. **Frame selection by largest gap.** It maximizes the signal and may bias towards frames where occlusion matters most.
3. **`occluded_absent` is near-degenerate by construction** (≤0.39% pixel difference). The design cannot test object permanence. A multi-frame (temporal) setup with a real occluder that the policy watches move is needed.
4. **Pixel-space occluder.** A flat gray patch is not a physical occluder: it has no shadow, perspective or contact, and the wrist edit is very large (up to 22% of the image).
5. **Selection criterion.** Specificity uses summed activations. The thresholds (0.5 specificity, 8 features per layer) were set in advance but not tuned; other selection rules (e.g. attribution-based) may find more causal sets.
6. **Transcoder capacity.** 512 features and k=16 per layer, trained on ~27k tokens from one scene; ~39% of the MLP-mediated effect lives in the transcoder error terms.
7. **Post-hoc extension.** The all-layers site was added after seeing single-layer results. It is reported as an extension, and the pass threshold was not changed.
8. **Remove-direction instability.** It is dominated by frame 1, which is also the outlier in Check 4.
9. **Goal 3 not run.** No closed-loop evidence either way.

---

## 8. Recommended next steps
1. **More frames and tasks:** e.g. 40 frames across LIBERO-Spatial tasks, so paired tests have power (with 40 frames, sign-flip p can go below 1e-6).
2. **Attribution-ranked features:** rank features by causal effect (activation × gradient of action RMSE) and report how many are needed to reach 25/50/75% of the MLP ceiling. That turns "distributed" into a curve.
3. **Temporal occlusion:** render a physical occluder moving over the bowl across several frames, and compare with a memory-enabled π0.5 variant, to test permanence properly.
4. **Attention-path analysis:** the residual swap shows that early-layer prefix representations carry the signal, yet MLPs carry ~65% of it. Attention-output patching per layer would complete the picture.
5. **Goal 3 as an exploratory run** (`FORCE_CONTINUE=True`), clearly labelled as such.

---

## 9. File-by-file index

| File | Content | Used in |
|---|---|---|
| `RESULTS.md` | auto-generated one-page summary | §1, §5 |
| `code/tc_occlusion.py` | full pipeline (frames, validate, capture, train, goal1–3) | §4 |
| `code/config.json`, `results/config.json` | all hyper-parameters (identical) | §3 |
| `code/notebook.ipynb` | Colab notebook with all cell outputs and logs | §3, runtimes |
| `env/versions.txt`, `packages.txt`, `gpu.txt` | software and hardware | §3 |
| `results/frames.pkl` | 4 probe frames × 6 conditions × 2 cameras, masks, regions, robot state, and 23 extra frames | §3.1–3.2 |
| `results/frames_candidates.json` | all 16 candidate frames and gaps | §3.1 |
| `results/fig_probe_conditions_{agentview,wrist}.png` | the conditions as the policy sees them | §3.2 |
| `results/validate.json`, `status/validate.json` | checks 1–4, per-frame and per-layer | §4.2 |
| `results/fig_check4_ceiling.png` | residual vs MLP swap per layer | §4.2 |
| `results/status/capture.json` | token layout, 26,852 tokens | §3, §4.3 |
| `results/train.json`, `status/train.json`, `fig_check5_transcoders.png` | FVU, splice, training curves | §4.3 |
| `results/transcoders/layer_XX.pt` | 18 trained transcoders | §4.3 |
| `results/goal1_feature_table.csv` | all 9,216 features × every contrast | §5.2 |
| `results/goal1_selected.json`, `status/goal1.json` | selected occlusion/color sets per layer | §5.2 |
| `results/fig_goal1_{counts,features,contact_sheet}.png` | Goal 1 figures | §5.2 |
| `results/goal2_rows.csv` | 736 patching runs (frame × site × method × scope × direction × draw) | §5.3 |
| `results/goal2_circuit_trace_table.{csv,md}` | aggregated circuit-trace table | §5.3 |
| `results/goal2_edges.json` | cross-layer edges | §5.3 |
| `results/fig_goal2_contact_sheet.png`, `status/goal2.json` | Goal 2 figure and verdict | §5.3 |
| `analysis/derive_stats.py`, `analysis/derived_stats.json` | every derived number in this report and the paper | all |
| `paper/` | ICLR 2027 submission draft (LaTeX + PDF) | – |

**Runtimes (A100):** policy load 149 s per step; frames 205 s, validate 285 s, capture 201 s, train 375 s, Goal 1 10 s, Goal 2 688 s. Total ≈ 30 min of compute.
