#!/usr/bin/env python
"""Cross-check the numbers quoted in paper/main.tex and REPORT.md against the raw result files.

Usage (repo root):  python analysis/check_claims.py      (run derive_stats.py first)
Every claim is recomputed from results/ (via analysis/derived_stats.json) and must appear, formatted as
written, in the documents listed for it. Exit code 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from decimal import ROUND_HALF_UP, Decimal

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "analysis" / "derived_stats.json").read_text())
TEX = (ROOT / "paper" / "main.tex").read_text()
REP = (ROOT / "REPORT.md").read_text()
G = S["goal2"]
A = G["all_layers"]
m = lambda site, k: G[site]["means"][k]

claims = [
    # (description, formatted value, where)
    ("n selective features", f"{S['goal1']['n_pass']}", "TR"),
    ("dictionary size", f"{S['goal1']['n_features_total']:,}", "TR"),
    ("selective share", f"{S['goal1']['pass_frac_pct']:.2f}\\%", "T"),
    ("selective share (md)", f"{S['goal1']['pass_frac_pct']:.2f}%", "R"),
    ("median specificity", f"{S['goal1']['specificity_median']:.2f}", "TR"),
    ("specificity range lo", f"{S['goal1']['specificity_range'][0]:.2f}", "TR"),
    ("specificity range hi", f"{S['goal1']['specificity_range'][1]:.2f}", "TR"),
    ("median rise", f"{S['goal1']['rise_tokens_median']:.1f}", "TR"),
    ("max rise", f"{S['goal1']['rise_tokens_range'][1]:.1f}", "TR"),
    ("corr occ vs occ_absent", f"{S['goal1']['corr_d_occluded_vs_d_occluded_absent']:.4f}", "TR"),
    ("recolor/occ", f"{S['goal1']['median_abs_other_over_occ']['recolor']:.2f}", "TR"),
    ("absent/occ", f"{S['goal1']['median_abs_other_over_occ']['absent']:.2f}", "TR"),
    ("slab/occ", f"{S['goal1']['median_abs_other_over_occ']['slab_miss']:.2f}", "TR"),
    ("all-layer occ", f"{m('all_layers', 'occ_features'):.2f}", "TR"),
    ("all-layer occ (1dp)", f"{m('all_layers', 'occ_features'):.1f}", "TR"),
    ("all-layer random", f"{m('all_layers', 'random_same_norm'):.2f}", "TR"),
    ("all-layer color", f"{m('all_layers', 'color_features'):.2f}", "TR"),
    ("all-layer ceiling", f"{m('all_layers', 'mlp_swap_ceiling'):.2f}", "T"),
    ("all-layer ceiling (1dp)", f"{m('all_layers', 'mlp_swap_ceiling'):.1f}", "TR"),
    ("all tc features", f"{m('all_layers', 'all_tc_features'):.1f}", "TR"),
    ("ratio vs random", f"{m('all_layers', 'occ_features') / m('all_layers', 'random_same_norm'):.1f}", "TR"),
    ("ratio vs color", f"{m('all_layers', 'occ_features') / m('all_layers', 'color_features'):.1f}", "TR"),
    ("occ share of all tc", f"{A['occ_share_of_all_tc_pct']:.1f}", "TR"),
    ("all tc share of ceiling", f"{A['all_tc_share_of_mlp_ceiling_pct']:.0f}", "TR"),
    ("margin all layers", f"+{G['verdict']['all']['margin_vs_best_control']:.1f}", "TR"),
    ("L5 occ", f"{m('layer5', 'occ_features'):.2f}", "T"),
    ("L5 ceiling", f"{m('layer5', 'mlp_swap_ceiling'):.2f}", "T"),
    ("L5 share", f"{G['layer5']['occ_share_of_mlp_ceiling_pct']:.0f}\\%", "T"),
    ("L4 occ", f"{m('layer4', 'occ_features'):.2f}", "T"),
    ("L0 occ", f"{m('layer0', 'occ_features'):.2f}", "T"),
    ("top3 joint", f"{G['top3_joint']['mean']:.2f}", "TR"),
    ("p all layers", f"{G['all_layers']['sign_flip_p_vs_random']:.3f}", "T"),
    ("p L5 (round half up)", str(Decimal(str(G['layer5']['sign_flip_p_vs_random'])).quantize(Decimal("0.001"), ROUND_HALF_UP)), "T"),
    ("p L4 (round half up)", str(Decimal(str(G['layer4']['sign_flip_p_vs_random'])).quantize(Decimal("0.001"), ROUND_HALF_UP)), "T"),
    ("p L0", f"{G['layer0']['sign_flip_p_vs_random']:.3f}", "T"),
    ("frames beating every random draw", f"{A['frames_occ_gt_every_random_draw']} of 4", "T"),
    ("frame2 occ (negative, |value|)", f"{abs(A['occ_features'][2]):.2f}", "TR"),
    ("frame2 ceiling", f"{A['mlp_swap_ceiling'][2]:.0f}\\%", "T"),
    ("bowl-token occ", f"{np.mean(A['bowl']['occ_features']):.2f}", "TR"),
    ("nonbowl occ", f"{np.mean(A['nonbowl']['occ_features']):.2f}", "TR"),
    ("bowl ceiling", f"{np.mean(A['bowl']['mlp_swap_ceiling']):.1f}", "TR"),
    ("nonbowl ceiling", f"{np.mean(A['nonbowl']['mlp_swap_ceiling']):.1f}", "TR"),
    ("remove ceiling", f"{np.mean(A['remove']['mlp_swap_ceiling']):.1f}", "TR"),
    ("remove occ", f"{np.mean(A['remove']['occ_features']):.1f}", "TR"),
    ("remove color", f"{np.mean(A['remove']['color_features']):.1f}", "TR"),
    ("top single feature", f"{G['single_features'][0]['mean']:.2f}", "TR"),
    ("edge L4->L5", f"{100 * next(e['mean_frac_recovered'] for e in G['edges'] if e['from_layer'] == 4 and e['to_layer'] == 5 and e['source'] == 'occ'):.1f}", "TR"),
    ("best resid swap", f"{S['check4']['best_pct']:.1f}", "TR"),
    ("max single MLP swap", f"{S['check4']['max_single_layer_mlp_swap']:.1f}", "TR"),
    ("median FVU", f"{S['transcoders']['median_fvu']:.3f}", "TR"),
    ("max FVU", f"{S['transcoders']['max_fvu']:.3f}", "TR"),
    ("min FVU", f"{S['transcoders']['min_fvu']:.3f}", "TR"),
    ("max splice", f"{S['transcoders']['max_splice']:.2f}", "T"),
    ("gap lo", f"{min(S['check3']['gap_occluded']):.3f}", "TR"),
    ("gap hi", f"{max(S['check3']['gap_occluded']):.3f}", "TR"),
    ("gap/noise lo", f"{min(S['check3']['gap_over_seed_noise']):.1f}", "TR"),
    ("gap/noise hi", f"{max(S['check3']['gap_over_seed_noise']):.1f}", "TR"),
    ("absent gap lo", f"{min(S['check3']['gap_absent']):.2f}", "TR"),
    ("absent gap hi", f"{max(S['check3']['gap_absent']):.2f}", "TR"),
    ("train tokens", f"{S['setup']['n_tokens']:,}", "TR"),
    ("max occluded-vs-occluded_absent pixel diff", f"{max(max(r['agentview_occluded_vs_occluded_absent_pct'], r['wrist_occluded_vs_occluded_absent_pct']) for r in S['pixels']):.2f}", "TR"),
]
fail = 0
for desc, val, where in claims:
    docs = {"T": ("paper", TEX), "R": ("report", REP)}
    for w in where:
        name, txt = docs[w]
        ok = val in txt
        if not ok:
            fail += 1
        print(f"[{'ok ' if ok else 'MISSING'}] {desc:42s} {val:>10s}  in {name}")
# sanity: exact p-value floor for n=4
assert abs(min(G[s]['sign_flip_p_vs_random'] for s in ('layer0', 'layer4', 'layer5', 'all_layers')) - 1 / 16) < 1e-9
print(f"\n{len(claims)} claims, {fail} missing")
sys.exit(1 if fail else 0)
