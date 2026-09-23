#!/usr/bin/env python
"""Derive every statistic quoted in REPORT.md and paper/ from the raw result files.

Usage (from the repo root):  python analysis/derive_stats.py
Writes analysis/derived_stats.json and prints a readable summary. Needs numpy (+ torch only for the
optional transcoder section).
"""

from __future__ import annotations

import collections
import csv
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "analysis" / "derived_stats.json"


def load(p):
    return json.loads((R / p).read_text())


def exact_sign_flip_p(diffs):
    """One-sided exact sign-flip (paired permutation) test that mean(diffs) > 0."""
    d = np.asarray(diffs, float)
    obs = d.mean()
    flips = [np.mean(d * np.array(s)) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean([f >= obs - 1e-12 for f in flips]))


def main():
    S = {}
    cfg = load("config.json")
    val = load("validate.json")
    st_val, st_train, st_g1, st_g2 = (load(f"status/{s}.json") for s in ("validate", "train", "goal1", "goal2"))
    cap = load("status/capture.json")
    frames_st = load("status/frames.json")
    S["setup"] = {
        "policy": cfg["policy_path"], "suite": cfg["suite"], "task_id": cfg["task_id"],
        "target_object": frames_st["target_object"], "n_probe_frames": frames_st["n_probe_frames"],
        "n_extra_train_frames": frames_st["n_extra_train_frames"], "n_layers": cap["n_layers"],
        "n_samples": cap["n_samples"], "n_tokens": cap["n_tokens"], "prefix_len": cap["layout"]["T"],
        "tokens_per_image": cap["layout"]["per_img"], "n_image_slots": cap["layout"]["n_img"],
        "lang_len": cap["layout"]["lang_len"], "tc_features": cfg["tc_features"], "tc_k": cfg["tc_k"],
        "tc_steps": cfg["tc_steps"], "obs_size": cfg["obs_size"],
    }

    # ---------------- probe frames and conditions (pixel-level facts)
    fr = pickle.load(open(R / "frames.pkl", "rb"))
    px = []
    for p in fr["probe"]:
        row = {"frame": p["frame_id"], "episode": p["episode"], "t": p["t"]}
        for key, cam in (("image", "agentview"), ("image2", "wrist")):
            base = p["conds"]["base"][key].astype(int)
            m = p["masks"][key]
            row[f"{cam}_bowl_px"] = int(m.sum())
            row[f"{cam}_bowl_frac_pct"] = 100 * float(m.mean())
            oa = np.abs(p["conds"]["occluded"][key].astype(int) - p["conds"]["occluded_absent"][key].astype(int)).max(-1) > 0
            row[f"{cam}_occluded_vs_occluded_absent_px"] = int(oa.sum())
            row[f"{cam}_occluded_vs_occluded_absent_pct"] = 100 * float(oa.mean())
            for c in ("recolor", "absent", "occluded", "slab_miss", "occluded_absent"):
                ch = np.abs(p["conds"][c][key].astype(int) - base).max(-1) > 0
                row[f"{cam}_{c}_changed_pct"] = 100 * float(ch.mean())
        px.append(row)
    S["pixels"] = px

    # ---------------- validation
    c3 = val["check3"]["rows"]
    S["check3"] = {
        "gap_occluded": [r["gap_occluded"] for r in c3],
        "gap_absent": [r["rmse_absent"] for r in c3],
        "gap_recolor": [r["rmse_recolor"] for r in c3],
        "gap_slab_miss": [r["rmse_slab_miss"] for r in c3],
        "gap_occluded_absent": [r["rmse_occluded_absent"] for r in c3],
        "seed_noise_rmse": [r["seed_noise_rmse"] for r in c3],
    }
    S["check3"]["gap_over_seed_noise"] = [a / b for a, b in zip(S["check3"]["gap_occluded"], S["check3"]["seed_noise_rmse"])]
    S["check3"]["absent_over_occluded"] = [a / b for a, b in zip(S["check3"]["gap_absent"], S["check3"]["gap_occluded"])]
    c4 = val["check4"]
    S["check4"] = {
        "resid_swap_mean": c4["resid_swap_pct_mean"], "mlp_swap_mean": c4["mlp_swap_pct_mean"],
        "best_layer": c4["best_layer"], "best_pct": c4["best_pct"],
        "sum_single_layer_mlp_swap": float(np.sum(c4["mlp_swap_pct_mean"])),
        "max_single_layer_mlp_swap": float(np.max(c4["mlp_swap_pct_mean"])),
        "argmax_single_layer_mlp_swap": int(np.argmax(c4["mlp_swap_pct_mean"])),
        "first_layer_resid_below_50": int(next(i for i, v in enumerate(c4["resid_swap_pct_mean"]) if v < 50)),
        "first_layer_resid_below_5": int(next(i for i, v in enumerate(c4["resid_swap_pct_mean"]) if v < 5)),
        "resid_swap_frame_range_layer0": [float(min(r[0] for r in c4["resid_swap_pct_per_frame"])), float(max(r[0] for r in c4["resid_swap_pct_per_frame"]))],
    }
    S["check1"] = {k: v for k, v in val["check1"].items() if k != "rows"}
    S["check2"] = val["check2"]

    # ---------------- transcoders
    tr = load("train.json")
    fvu = [r["fvu_heldout"] for r in tr["rows"]]
    fvu_tr = [r["fvu_train"] for r in tr["rows"]]
    spl = [r["splice_rmse_over_gap"] for r in tr["rows"]]
    S["transcoders"] = {
        "fvu_heldout": fvu, "fvu_train": fvu_tr, "splice_over_gap": spl,
        "median_fvu": float(np.median(fvu)), "min_fvu": float(np.min(fvu)), "max_fvu": float(np.max(fvu)),
        "argmax_fvu": int(np.argmax(fvu)), "max_splice": float(np.max(spl)), "argmax_splice": int(np.argmax(spl)),
        "n_train_tokens": tr["rows"][0]["n_train_tokens"], "n_heldout_tokens": tr["rows"][0]["n_heldout_tokens"],
        "alive_all": all(r["alive_features"] == 512 for r in tr["rows"]),
        "seconds_per_layer_mean": float(np.mean([r["seconds"] for r in tr["rows"]])),
        "generalization_gap_max": float(np.max(np.array(fvu) - np.array(fvu_tr))),
        "generalization_gap_argmax": int(np.argmax(np.array(fvu) - np.array(fvu_tr))),
    }
    try:
        import torch

        ys = []
        for l in range(S["setup"]["n_layers"]):
            ck = torch.load(R / "transcoders" / f"layer_{l:02d}.pt", map_location="cpu", weights_only=False)
            ys.append(float(ck["state_dict"]["y_scale"]))
        S["transcoders"]["mlp_out_rms_norm_per_layer"] = ys
    except Exception as exc:  # torch optional
        S["transcoders"]["mlp_out_rms_norm_per_layer"] = f"unavailable: {exc}"

    # ---------------- goal 1
    rows = list(csv.DictReader(open(R / "goal1_feature_table.csv")))
    f = lambda r, k: float(r[k]) if r[k] not in ("", "None") else float("nan")
    P = [r for r in rows if r["pass"] == "True"]
    occ = np.array([f(r, "d_occluded") for r in P])
    oa = np.array([f(r, "d_occluded_absent") for r in P])
    S["goal1"] = {
        "n_features_total": len(rows), "n_alive": sum(r["alive"] == "True" for r in rows),
        "n_pass": len(P), "pass_frac_pct": 100 * len(P) / len(rows),
        "n_pass_per_layer": st_g1["n_pass_per_layer"], "top_layers": st_g1["top_layers"],
        "specificity_median": float(np.median([f(r, "specificity") for r in P])),
        "specificity_range": [float(min(f(r, "specificity") for r in P)), float(max(f(r, "specificity") for r in P))],
        "rise_tokens_median": float(np.median([f(r, "rise_tokens") for r in P])),
        "rise_tokens_range": [float(min(f(r, "rise_tokens") for r in P)), float(max(f(r, "rise_tokens") for r in P))],
        "corr_d_occluded_vs_d_occluded_absent": float(np.corrcoef(occ, oa)[0, 1]),
        "ratio_occluded_absent_over_occluded_median": float(np.median(oa / occ)),
        "ratio_occluded_absent_over_occluded_range": [float((oa / occ).min()), float((oa / occ).max())],
        "median_abs_other_over_occ": {c: float(np.median(np.abs([f(r, f"d_{c}") for r in P]) / occ)) for c in ("recolor", "absent", "slab_miss")},
        "n_alive_positive_all_frames": int(sum(1 for r in rows if r["alive"] == "True" and f(r, "frames_positive") == 1.0)),
        "n_spec_ge_05_any": int(sum(1 for r in rows if np.isfinite(f(r, "specificity")) and f(r, "specificity") >= 0.5)),
        "top_features": [{k: r[k] for k in ("layer", "feature", "specificity", "rise_tokens", "d_occluded", "d_occluded_absent", "d_recolor", "d_absent", "d_slab_miss", "d_occluded_min")}
                         for r in sorted(P, key=lambda r: -f(r, "score"))[:10]],
    }

    # ---------------- goal 2
    g2 = list(csv.DictReader(open(R / "goal2_rows.csv")))

    def pf(layer, method, scope="all", direction="inject"):
        d = collections.defaultdict(list)
        for r in g2:
            if r["layer"] == str(layer) and r["method"] == method and r["scope"] == scope and r["direction"] == direction:
                d[int(r["frame"])].append(float(r["closed_pct"]))
        return [float(np.mean(d[k])) for k in sorted(d)]

    sites = {"layer5": 5, "layer0": 0, "layer4": 4, "all_layers": -2}
    G = {}
    for name, l in sites.items():
        s = {m: pf(l, m) for m in ("occ_features", "color_features", "random_same_norm", "random_raw", "all_tc_features", "mlp_swap_ceiling", "full_token_swap_ceiling")}
        s = {k: v for k, v in s.items() if v}
        s["remove"] = {m: pf(l, m, direction="remove") for m in ("occ_features", "color_features", "mlp_swap_ceiling")}
        s["bowl"] = {m: pf(l, m, "bowl") for m in ("occ_features", "mlp_swap_ceiling", "full_token_swap_ceiling") if pf(l, m, "bowl")}
        s["nonbowl"] = {m: pf(l, m, "nonbowl") for m in ("occ_features", "mlp_swap_ceiling", "full_token_swap_ceiling") if pf(l, m, "nonbowl")}
        means = {k: float(np.mean(v)) for k, v in s.items() if isinstance(v, list)}
        diffs_rand = np.array(s["occ_features"]) - np.array(s["random_same_norm"])
        diffs_col = np.array(s["occ_features"]) - np.array(s["color_features"])
        s["means"] = means
        s["paired_diff_vs_random_same_norm"] = diffs_rand.tolist()
        s["paired_diff_vs_color"] = diffs_col.tolist()
        s["sign_flip_p_vs_random"] = exact_sign_flip_p(diffs_rand)
        s["sign_flip_p_vs_color"] = exact_sign_flip_p(diffs_col)
        s["frames_occ_gt_random"] = int((diffs_rand > 0).sum())
        s["occ_over_random_ratio"] = means["occ_features"] / means["random_same_norm"] if means["random_same_norm"] > 0 else None
        s["occ_share_of_mlp_ceiling_pct"] = 100 * means["occ_features"] / means["mlp_swap_ceiling"]
        if "all_tc_features" in means:
            s["occ_share_of_all_tc_pct"] = 100 * means["occ_features"] / means["all_tc_features"]
            s["all_tc_share_of_mlp_ceiling_pct"] = 100 * means["all_tc_features"] / means["mlp_swap_ceiling"]
        G[name] = s
    rd = [float(r["closed_pct"]) for r in g2 if r["layer"] == "-2" and r["method"] == "random_same_norm" and r["scope"] == "all"]
    G["all_layers"]["random_same_norm_all_draws"] = sorted(rd)
    per_frame_max_rand = []
    for fid in range(4):
        per_frame_max_rand.append(max(float(r["closed_pct"]) for r in g2 if r["layer"] == "-2" and r["method"] == "random_same_norm" and r["frame"] == str(fid)))
    G["all_layers"]["per_frame_max_random_draw"] = per_frame_max_rand
    G["all_layers"]["frames_occ_gt_every_random_draw"] = int(sum(o > m for o, m in zip(G["all_layers"]["occ_features"], per_frame_max_rand)))
    G["all_layers"]["n_features_patched"] = int(next(r["n_features"] for r in g2 if r["layer"] == "-2" and r["method"] == "occ_features"))
    G["all_layers"]["n_features_patched_frac_pct"] = 100 * G["all_layers"]["n_features_patched"] / (S["setup"]["n_layers"] * S["setup"]["tc_features"])
    G["top3_joint"] = {"occ_features": pf(-1, "occ_features_all_top_layers"), "mean": float(np.mean(pf(-1, "occ_features_all_top_layers")))}
    singles = {}
    for r in g2:
        if r["method"].startswith("single_feature_"):
            singles.setdefault((int(r["layer"]), int(r["feature"])), []).append(float(r["closed_pct"]))
    G["single_features"] = sorted([{"layer": k[0], "feature": k[1], "mean": float(np.mean(v)), "per_frame": v} for k, v in singles.items()], key=lambda d: -d["mean"])
    G["edges"] = load("goal2_edges.json")
    G["verdict"] = st_g2["verdict"]
    G["margin_required_pts"] = cfg["goal2_margin_pts"]
    G["frame2_note"] = {
        "occ_all_layers": G["all_layers"]["occ_features"][2],
        "mlp_ceiling_all_layers": G["all_layers"]["mlp_swap_ceiling"][2],
        "occluded_eq_occluded_absent_pixels": px[2]["agentview_occluded_vs_occluded_absent_px"] == 0 and px[2]["wrist_occluded_vs_occluded_absent_px"] == 0,
    }
    S["goal2"] = G
    S["runtime_s"] = {"policy_load": 149, "frames": 205, "validate": 285, "capture": 201, "train": 375, "goal1": 10, "goal2": 688}

    OUT.write_text(json.dumps(S, indent=1))
    # ---------------- print summary
    a = G["all_layers"]
    print(f"setup: {S['setup']}")
    print(f"check3 gap/seed-noise: {np.round(S['check3']['gap_over_seed_noise'], 2)} | absent/occluded: {np.round(S['check3']['absent_over_occluded'], 2)}")
    print(f"check4: best {S['check4']['best_pct']:.1f}% @L{S['check4']['best_layer']} | sum single MLP swaps {S['check4']['sum_single_layer_mlp_swap']:.1f}% | first L with resid<50: {S['check4']['first_layer_resid_below_50']}, <5: {S['check4']['first_layer_resid_below_5']}")
    print(f"transcoders: median FVU {S['transcoders']['median_fvu']:.3f} range [{S['transcoders']['min_fvu']:.3f},{S['transcoders']['max_fvu']:.3f}] max splice {S['transcoders']['max_splice']:.3f}@L{S['transcoders']['argmax_splice']}")
    print(f"goal1: {S['goal1']['n_pass']}/{S['goal1']['n_features_total']} pass ({S['goal1']['pass_frac_pct']:.2f}%); corr(occ,occ_absent)={S['goal1']['corr_d_occluded_vs_d_occluded_absent']:.4f}")
    for name in sites:
        s = G[name]
        print(f"{name:10s}: occ {s['means']['occ_features']:.2f} rand {s['means']['random_same_norm']:.2f} color {s['means']['color_features']:.2f} "
              f"ceiling {s['means']['mlp_swap_ceiling']:.2f} | occ/rand {s['occ_over_random_ratio']:.1f}x | frames occ>rand {s['frames_occ_gt_random']}/4 "
              f"| sign-flip p (vs rand) {s['sign_flip_p_vs_random']:.4f} (vs color) {s['sign_flip_p_vs_color']:.4f}")
    print(f"all layers: all_tc {a['means']['all_tc_features']:.1f}% = {a['all_tc_share_of_mlp_ceiling_pct']:.1f}% of MLP ceiling; occ = {a['occ_share_of_all_tc_pct']:.1f}% of all_tc; "
          f"{a['n_features_patched']} features ({a['n_features_patched_frac_pct']:.2f}% of dictionary); occ > every random draw in {a['frames_occ_gt_every_random_draw']}/4 frames")
    print(f"remove (all layers): occ {np.mean(a['remove']['occ_features']):.1f}% color {np.mean(a['remove']['color_features']):.1f}% ceiling {np.mean(a['remove']['mlp_swap_ceiling']):.1f}%")
    print(f"top single features: {[(d['layer'], d['feature'], round(d['mean'], 2)) for d in G['single_features'][:5]]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
