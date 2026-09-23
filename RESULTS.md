# Transcoder circuit tracing on π0.5 (LIBERO) — results

Bundle created 2026-09-23 05:32 from `/content/groot-run/outputs/permanence/tc_circuit`.

Policy `lerobot/pi05_libero_finetuned` · suite `libero_spatial` task 0 · target `akita_black_bowl_1` · 4 probe frames · transcoders 512 features, k=16, 3000 steps

## Checks and goals

| item | result |
|---|---|
| Check 1: edits are clean | PASS |
| Check 2: determinism | PASS |
| Check 3: the signal exists | PASS |
| Check 4: the ceiling works | PASS |
| Check 5: transcoder is faithful | PASS |
| Goal 1: occlusion features exist | PASS |
| Goal 2: they cause the action | FAIL |
| Check 6: closed loop measurable | not run |
| Goal 3: task success shifts | not run |

## Validation

- Base-vs-occluded action RMSE per frame: 0.5197, 0.5158, 0.6665, 0.5219
- LLM layers: 18 · best full-layer swap (check 4): 91.8% at layer 3

## Transcoders (check 5)

- Median held-out FVU: 0.1739
- Per layer FVU: L0=0.018, L1=0.149, L2=0.158, L3=0.171, L4=0.163, L5=0.156, L6=0.198, L7=0.152, L8=0.202, L9=0.199, L10=0.258, L11=0.199, L12=0.233, L13=0.188, L14=0.182, L15=0.177, L16=0.022, L17=0.076
- Splice error / occlusion gap: L0=0.070, L1=0.099, L2=0.071, L3=0.101, L4=0.092, L5=0.118, L6=0.108, L7=0.110, L8=0.132, L9=0.077, L10=0.118, L11=0.124, L12=0.136, L13=0.161, L14=0.090, L15=0.078, L16=0.010, L17=0.000

## Goal 1

- Top layers: [5, 0, 4]
- Passing features per layer: {'0': 4, '1': 3, '2': 3, '3': 4, '4': 8, '5': 6, '6': 6, '7': 6, '8': 2, '9': 2, '10': 5, '11': 4, '12': 2, '13': 2, '14': 7, '15': 2, '16': 2, '17': 4}
- Selected features (top layers): {'5': [318, 177, 505, 270, 32, 124], '0': [274, 396, 415, 269], '4': [112, 317, 397, 59, 158, 333, 359, 150]}

## Goal 2 (gap closed, base → occluded, mean over frames)

| site | occlusion features | random same-norm | random raw | color | MLP-site ceiling | occlusion as % of ceiling | margin vs best control | pass |
|---|---|---|---|---|---|---|---|---|
| layer 5 | 1.9% | 0.3% | 0.3% | 0.5% | 4.1% | 46.2% | 1.5 pts | FAIL |
| layer 0 | 0.2% | 0.3% | 0.2% | 0.4% | 3.1% | 5.9% | -0.2 pts | FAIL |
| layer 4 | 1.2% | 0.2% | 0.2% | 0.2% | 5.6% | 21.0% | 0.9 pts | FAIL |
| all layers | 3.9% | 0.6% | 0.5% | 0.7% | 65.5% | 6.0% | 3.3 pts | FAIL |

Rule: absolute (+10.0 pts over random and color). Passing sites: []. occlusion features do not beat random/color by the margin at any site (single layers or all layers) -> Goal 2's null is the finding; skip Goal 3

Full table: `results/goal2_circuit_trace_table.md` (all methods, token scopes, reverse direction).

## Caveats to report

- 4 probe frames: the error bars are wide.
- The features respond equally to paint with or without the bowl underneath (`occluded` vs `occluded_absent`). π0.5 sees one frame at a time, so they are 'gray patch at the object location' features, not object permanence.
- The all-layers Goal 2 site was added after the single-layer run (single-MLP ceilings were only 3–6%). Report the single-layer null under the original rule and the all-layers test as an extension.

## What is in this bundle

- `RESULTS.md`: this page.
- `results/`: every figure (`fig_*.png`), table (`*.csv`, `*.md`), status JSON (`status/`), `validate.json`, `train.json`, `goal1_selected.json`, `goal2_edges.json`, trained transcoders (`transcoders/layer_XX.pt`), probe frames with all 6 renders (`frames.pkl`).
- `code/`: `tc_occlusion.py` (the whole pipeline), `config.json`, and `notebook.ipynb` if the export worked.
- `env/`: package versions, GPU info.

Re-run any step on a GPU box with: `python code/tc_occlusion.py <frames|validate|capture|train|goal1|goal2|goal3> --out results` (needs the same lerobot + LIBERO install; `capture` must run before `goal1` if `acts/` is not included).
