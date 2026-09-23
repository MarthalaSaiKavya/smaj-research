# smaj-research — Transcoder circuit analysis of occlusion in π0.5

Does the vision-language-action policy π0.5 represent occlusion of its target object with sparse internal features, and do those features drive the action? We apply per-layer TopK transcoders and causal patching to π0.5's 18-layer Gemma backbone on LIBERO, using a pixel-controlled occlusion protocol and a pre-registered validity battery.

**Result:** 72 occlusion-selective features exist and move the action 6.9× more than matched random features, but they close only 3.9% of the occlusion gap (all transcoder features: 40.1%; full MLP outputs: 65.5%). The effect is distributed. Goal 2 is a null under the pre-registered +10-point rule, so Goal 3 (closed loop) was not run.

| Start here | |
|---|---|
| [`REPORT.md`](REPORT.md) | Full research report: problem, method, novelty, every result, limitations, file index |
| [`paper/main.pdf`](paper/main.pdf) | ICLR 2027 submission draft (anonymous), built from [`paper/main.tex`](paper/main.tex) |
| [`RESULTS.md`](RESULTS.md) | Auto-generated one-page summary from the Colab run |

## Layout
- `code/`: `tc_occlusion.py` (the whole pipeline: `frames`, `validate`, `capture`, `train`, `goal1`, `goal2`, `goal3`), `config.json`, and the Colab `notebook.ipynb` with all outputs.
- `results/`: raw outputs of the run (figures, CSV/JSON tables, status files, 18 trained transcoders, probe frames).
- `analysis/`: `derive_stats.py` recomputes every number quoted in the report and paper into `derived_stats.json`; `make_paper_figures.py` builds the paper figures; `check_claims.py` verifies that the quoted numbers match the data.
- `paper/`: LaTeX source, bibliography, figures and the compiled PDF (ICLR 2027 style).
- `env/`: exact package versions and GPU info.

## Reproduce the analysis (CPU is enough)
```bash
pip install numpy matplotlib torch
python analysis/derive_stats.py        # -> analysis/derived_stats.json
python analysis/make_paper_figures.py  # -> paper/figures/fig_*.pdf
python analysis/check_claims.py        # all quoted numbers vs. the data
cd paper && latexmk -pdf main.tex      # -> paper/main.pdf
```
Re-running the experiments needs a GPU with lerobot 0.6.1 + LIBERO. See `code/notebook.ipynb`.
