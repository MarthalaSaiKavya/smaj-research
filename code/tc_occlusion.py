#!/usr/bin/env python
"""Transcoder circuit tracing of occlusion features in pi0.5 (LIBERO).

Every subcommand reads <out_dir>/config.json (written by the notebook's controls cell).

  frames    pick the probe frames, render the 6 conditions, save frames.pkl
  validate  checks 1-4 on the unpatched pipeline ("instrument is not broken")
  capture   capture MLP input/output of every LLM layer on every sample
  train     one TopK transcoder per layer + check 5 (faithfulness)
  goal1     do occlusion features exist? (feature table + kill switch)
  goal2     do they cause the action? (patching / circuit-trace table)
  goal3     do they change task success? (closed loop + check 6)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import torch.nn as nn

CONDS = ["base", "recolor", "absent", "occluded", "slab_miss", "occluded_absent"]
QUIET_FOR_OCC = ["recolor", "absent", "slab_miss"]  # the "other 3 contrasts"
QUIET_FOR_COLOR = ["occluded", "absent", "slab_miss"]
CAM_OF_KEY = {"image": "agentview", "image2": "robot0_eye_in_hand"}
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")

CFG: dict = {}
OUT: Path = Path(".")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- utils


def log(*args):
    print(time.strftime("[%H:%M:%S]"), *args, flush=True)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_status(step: str, passed: bool | None, summary: dict):
    save_json(OUT / "status" / f"{step}.json", {"step": step, "passed": passed, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **summary})


def read_status(step: str):
    p = OUT / "status" / f"{step}.json"
    return load_json(p) if p.exists() else None


def require_gate(step: str, what: str):
    st = read_status(step)
    if st is None:
        raise SystemExit(f"[gate] {step} has not been run yet. Run the {step} cell first.")
    if st.get("passed") is False and not CFG.get("force_continue", False):
        raise SystemExit(
            f"[gate] {step} did not pass, so {what} is stopped (plan's gate rule). "
            f"Reason: {st.get('reason', 'see status/' + step + '.json')}. Set FORCE_CONTINUE=True to run anyway."
        )
    return st


def setup_torch(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    warnings.filterwarnings("once")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item())


def closure_pct(a_patch, a_target, a_start) -> float:
    gap = rmse(a_start, a_target)
    if gap <= 0:
        return float("nan")
    return 100.0 * (1.0 - rmse(a_patch, a_target) / gap)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask.copy()
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def token_mask(region: np.ndarray, grid: int, min_frac: float) -> np.ndarray:
    """Region in raw (OpenGL) image coords -> (grid*grid,) bool over the model's image tokens.

    The LIBERO processor flips H and W before the policy sees the image, then the image is
    resized to 224 and cut into grid x grid SigLIP patches in row-major order.
    """
    r = region[::-1, ::-1].astype(np.float32)
    H, W = r.shape
    ys = (np.arange(grid + 1) * H / grid).astype(int)
    xs = (np.arange(grid + 1) * W / grid).astype(int)
    out = np.zeros((grid, grid), dtype=bool)
    for i in range(grid):
        for j in range(grid):
            blk = r[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            out[i, j] = blk.size > 0 and blk.mean() >= min_frac
    return out.reshape(-1)


def model_view(img: np.ndarray) -> np.ndarray:
    """What the policy sees (after the LIBERO 180-degree flip)."""
    return np.ascontiguousarray(img[::-1, ::-1])


def _add_batch_axis(x):
    if isinstance(x, dict):
        return {k: _add_batch_axis(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x[None])
    return x


def bootstrap_paired_diff(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int):
    """95% CI for mean(a) - mean(b) resampling paired episode indices."""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


# ----------------------------------------------------------------------------- image edits


def recolor(img: np.ndarray, mask: np.ndarray, rgb) -> np.ndarray:
    out = img.copy()
    if not mask.any():
        return out
    lum = img[mask].astype(np.float32).mean(axis=1)
    rel = lum / max(float(lum.max()), 1.0)
    shade = 0.35 + 0.65 * rel
    out[mask] = np.clip(shade[:, None] * np.asarray(rgb, np.float32)[None, :], 0, 255).astype(np.uint8)
    return out


def paint(img: np.ndarray, region: np.ndarray, gray: int) -> np.ndarray:
    out = img.copy()
    out[region] = np.uint8(gray)
    return out


def shift_region(region: np.ndarray, forbidden_strict: np.ndarray, forbidden_soft: np.ndarray):
    """Translate `region` to the nearest spot that avoids the object (and, if possible, the robot)."""
    if not region.any():
        return region.copy(), (0, 0), "empty"
    H, W = region.shape
    ys, xs = np.nonzero(region)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    dirs = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for forbidden, tag in ((forbidden_strict | forbidden_soft, "clear"), (forbidden_strict, "overlaps_robot")):
        for mul in (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            for dy, dx in dirs:
                sy, sx = int(round(dy * mul * h)), int(round(dx * mul * w))
                ny, nx = ys + sy, xs + sx
                if ny.min() < 0 or nx.min() < 0 or ny.max() >= H or nx.max() >= W:
                    continue
                new = np.zeros_like(region)
                new[ny, nx] = True
                if (new & forbidden).any():
                    continue
                return new, (sy, sx), tag
    return np.zeros_like(region), (0, 0), "no_room"


def make_conditions(base_px: dict, absent_px: dict, masks: dict, robot_masks: dict):
    """Returns conds[cond][key] -> uint8 image and regions[cond][key] -> allowed-change mask."""
    conds = {c: {} for c in CONDS}
    regions = {c: {} for c in CONDS}
    info = {}
    for key, base in base_px.items():
        m = masks[key]
        occ_r = dilate(m, CFG["occ_pad_px"])
        abs_r = dilate(m, CFG["shadow_pad_px"])
        forbid_obj = dilate(m, CFG["shadow_pad_px"] + 2)
        slab_r, shift, tag = shift_region(occ_r, forbid_obj, robot_masks[key])
        absent = np.where(abs_r[..., None], absent_px[key], base).astype(np.uint8)
        g = CFG["paint_gray"]
        conds["base"][key] = base.copy()
        conds["recolor"][key] = recolor(base, m, CFG["recolor_rgb"])
        conds["absent"][key] = absent
        conds["occluded"][key] = paint(base, occ_r, g)
        conds["slab_miss"][key] = paint(base, slab_r, g)
        conds["occluded_absent"][key] = paint(absent, occ_r, g)
        zeros = np.zeros_like(m)
        regions["base"][key] = zeros
        regions["recolor"][key] = m.copy()
        regions["absent"][key] = abs_r
        regions["occluded"][key] = occ_r
        regions["slab_miss"][key] = slab_r
        regions["occluded_absent"][key] = abs_r | occ_r
        info[key] = {"mask_px": int(m.sum()), "occ_px": int(occ_r.sum()), "slab_shift": shift, "slab_tag": tag}
    return conds, regions, info


# ----------------------------------------------------------------------------- simulator side


def libero_task(env):
    return env.task_description


def make_env(episode_index: int = 0):
    from lerobot.envs.libero import LiberoEnv, _get_suite

    suite = _get_suite(CFG["suite"])
    env = LiberoEnv(
        task_suite=suite,
        task_id=int(CFG["task_id"]),
        task_suite_name=CFG["suite"],
        obs_type="pixels_agent_pos",
        observation_width=int(CFG["obs_size"]),
        observation_height=int(CFG["obs_size"]),
        init_states=True,
        episode_index=episode_index,
    )
    ensure_sim(env)
    return env


def ensure_sim(env):
    """Newer lerobot builds the LIBERO sim lazily (on first reset / _ensure_env); make sure it exists."""
    if getattr(env, "_env", None) is None:
        if callable(getattr(env, "_ensure_env", None)):
            env._ensure_env()
        else:
            env.reset(seed=int(CFG.get("seed", 0)))
    if getattr(env, "_env", None) is None:
        raise RuntimeError("LiberoEnv did not create its simulator (env._env is None).")
    return env


def _bddl_lists(env):
    """obj_of_interest / goal objects / declared objects, from every place LIBERO keeps them."""
    import re

    inner = getattr(env._env, "env", None)
    ooi, goal, objs = [], [], []
    try:
        ooi = [str(o) for o in (env._env.obj_of_interest or [])]
    except Exception:
        pass
    pp = getattr(inner, "parsed_problem", None) or {}
    if not ooi:
        ooi = [str(o) for o in (pp.get("obj_of_interest") or [])]
    for pred in pp.get("goal_state") or []:
        if isinstance(pred, (list, tuple)):
            goal += [str(t) for t in pred[1:] if isinstance(t, str)]
    try:
        objs = [str(k) for k in inner.objects_dict.keys()]
    except Exception:
        pass
    text = ""
    for cand in (getattr(inner, "bddl_file_name", None), getattr(env._env, "bddl_file_name", None)):
        if cand and os.path.exists(str(cand)):
            text = Path(str(cand)).read_text()
            break
    if not text:
        try:
            from libero.libero import get_libero_path

            from lerobot.envs.libero import _get_suite

            task = _get_suite(CFG["suite"]).get_task(int(CFG["task_id"]))
            if task is not None:
                f = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                text = f.read_text() if f.exists() else ""
        except Exception:
            pass
    if text:
        m = re.search(r"\(:obj_of_interest([^()]*)\)", text)
        if m and not ooi:
            ooi = m.group(1).split()
        g = re.search(r"\(:goal(.*)", text, re.S)
        if g and not goal:
            goal = re.findall(r"\(\s*\w+\s+([\w]+)(?:\s+([\w]+))?\s*\)", g.group(1))
            goal = [t for pair in goal for t in pair if t]
    return ooi, goal, objs


def infer_target(env):
    kw = str(CFG.get("target_keyword", "")).strip()
    ooi, goal, objs = _bddl_lists(env)
    explicit = str(CFG.get("target_object", "auto")).strip()
    if explicit and explicit != "auto":
        return explicit, ooi, "TARGET_OBJECT"
    for name, cands in (("obj_of_interest", ooi), ("BDDL goal", goal), ("scene objects", sorted(objs))):
        hit = [o for o in cands if kw in o and (not objs or o in objs)]
        if hit:
            return hit[0], ooi, name
    import mujoco

    m = env._env.sim.model._model
    bodies = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)})
    hit = [b for b in bodies if kw in b and not b.startswith(ROBOT_PREFIXES)]
    if hit:
        root = hit[0]
        for suffix in ("_main", "_body", "_base"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
        return root, ooi, "MuJoCo body names"
    raise RuntimeError(
        f"Could not infer the target object for keyword {kw!r}. obj_of_interest={ooi} goal={goal} objects={objs}. "
        "Set TARGET_OBJECT in the Plan controls cell (e.g. akita_black_bowl_1)."
    )


class Scene:
    """Object bookkeeping + rendering helpers on top of the LIBERO robosuite sim.

    robosuite's hard reset (LIBERO default) builds a new MjSim on every env.reset(), so every
    public method re-binds to the live sim first.
    """

    def __init__(self, env):
        import mujoco

        self.mj = mujoco
        self.env = ensure_sim(env)
        self._sim = None
        self.size = int(CFG["obs_size"])
        target, ooi, how = infer_target(env)
        self.target = target
        self.obj_of_interest = ooi
        log(f"target object {target!r} (found via {how})")
        self._refresh()
        log(f"target object: {target} | bodies={len(self.body_ids)} geoms={len(self.geom_ids)} free_joints={len(self.free_joints)} | obj_of_interest={ooi}")

    def _refresh(self):
        ensure_sim(self.env)
        sim = self.env._env.sim
        if sim is self._sim:
            return
        mujoco = self.mj
        self._sim = sim
        self.sim = sim
        self.m = sim.model._model
        self.d = sim.data._data
        target = self.target
        names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(self.m.nbody)]
        self.body_names = names

        def subtree(pred):
            out = set()
            for b in range(self.m.nbody):
                a = b
                while a > 0:
                    if pred(names[a]):
                        out.add(b)
                        break
                    a = int(self.m.body_parentid[a])
            return out

        self.body_ids = subtree(lambda n: n == target or n.startswith(target + "_"))
        if not self.body_ids:
            hint = sorted({n.rsplit("_", 1)[0] for n in names if n and not n.startswith(ROBOT_PREFIXES)})
            raise RuntimeError(f"No MuJoCo bodies found for target object {target!r}. Body-name stems in this scene: {hint}")
        self.geom_ids = np.array([g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) in self.body_ids], dtype=np.int64)
        robot_bodies = subtree(lambda n: n.startswith(ROBOT_PREFIXES))
        self.robot_geom_ids = np.array([g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) in robot_bodies], dtype=np.int64)
        self.free_joints = [
            (int(self.m.jnt_qposadr[j]), int(self.m.jnt_bodyid[j]))
            for j in range(self.m.njnt)
            if int(self.m.jnt_bodyid[j]) in self.body_ids and int(self.m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        self.root_body = self.free_joints[0][1] if self.free_joints else min(self.body_ids)

    # --- rendering
    def rgb(self, cam: str) -> np.ndarray:
        self._refresh()
        return np.array(self.sim.render(camera_name=cam, width=self.size, height=self.size), copy=True)

    def seg(self, cam: str) -> np.ndarray:
        """(H, W, 2) [objtype, objid]. Own decoder: robosuite 1.4's overflows under numpy 2."""
        self._refresh()
        mj = self.mj
        ctx = self.sim._render_context_offscreen
        cid = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_CAMERA, cam)
        W = H = self.size
        ctx.render(width=W, height=H, camera_id=cid, segmentation=True)
        buf = np.empty((H, W, 3), dtype=np.uint8)
        mj.mjr_readPixels(rgb=buf, depth=None, viewport=mj.MjrRect(0, 0, W, H), con=ctx.con)
        code = buf[..., 0].astype(np.int64) + (buf[..., 1].astype(np.int64) << 8) + (buf[..., 2].astype(np.int64) << 16)
        ng = ctx.scn.ngeom
        code[code >= ng + 1] = 0
        table = np.full((ng + 1, 2), -1, dtype=np.int64)
        for i in range(ng):
            g = ctx.scn.geoms[i]
            if g.segid != -1:
                table[g.segid + 1] = (g.objtype, g.objid)
        return table[code]

    def geom_mask(self, seg: np.ndarray, ids: np.ndarray) -> np.ndarray:
        return (seg[..., 0] == int(self.mj.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 1], ids)

    def masks(self, cams):
        self._refresh()
        out, robot = {}, {}
        for key, cam in cams.items():
            s = self.seg(cam)
            out[key] = self.geom_mask(s, self.geom_ids)
            robot[key] = self.geom_mask(s, self.robot_geom_ids)
        return out, robot

    def render_absent(self, cams: dict):
        """Render with the target object moved out of view; restore the exact state afterwards."""
        self._refresh()
        mj = self.mj
        q = self.d.qpos.copy()
        v = self.d.qvel.copy()
        rgba = None
        if self.free_joints:
            for i, (adr, _) in enumerate(self.free_joints):
                self.d.qpos[adr : adr + 3] = np.array([60.0 + 5 * i, 60.0, -30.0])
        else:  # fixed object: make it transparent instead
            rgba = self.m.geom_rgba[self.geom_ids].copy()
            self.m.geom_rgba[self.geom_ids, 3] = 0.0
        mj.mj_forward(self.m, self.d)
        imgs = {key: self.rgb(cam) for key, cam in cams.items()}
        leftover = {key: int(self.geom_mask(self.seg(cam), self.geom_ids).sum()) for key, cam in cams.items()}
        self.d.qpos[:] = q
        self.d.qvel[:] = v
        if rgba is not None:
            self.m.geom_rgba[self.geom_ids] = rgba
        mj.mj_forward(self.m, self.d)
        assert np.array_equal(self.d.qpos, q), "state restore failed"
        return imgs, leftover

    def set_state_fast(self, state):
        self._refresh()
        self.env._env.sim.set_state_from_flattened(state)
        self.mj.mj_forward(self.m, self.d)

    def object_z(self) -> float:
        self._refresh()
        return float(self.d.xpos[self.root_body][2])


def render_frame(env, scene: Scene, state: np.ndarray) -> dict:
    raw = env._env.set_init_state(state)
    fmt = env._format_raw_obs(raw)
    keys = list(fmt["pixels"].keys())
    cams = {k: CAM_OF_KEY[k] for k in keys}
    base_px, render_diff = {}, {}
    for k in keys:
        base_px[k] = np.ascontiguousarray(fmt["pixels"][k]).copy()
        render_diff[k] = int(np.abs(scene.rgb(cams[k]).astype(int) - base_px[k].astype(int)).max())
    absent_px, leftover = scene.render_absent(cams)
    try:
        masks, robot = scene.masks(cams)
        mask_src = "segmentation"
    except Exception as exc:  # fall back to "what changes when the object is removed"
        log(f"segmentation render failed ({type(exc).__name__}: {exc}); using removal-diff masks")
        masks = {k: (np.abs(absent_px[k].astype(int) - base_px[k].astype(int)).max(-1) > 12) for k in keys}
        robot = {k: np.zeros_like(masks[k]) for k in keys}
        mask_src = "removal_diff"
    conds, regions, info = make_conditions(base_px, absent_px, masks, robot)
    return {
        "state": np.asarray(state).copy(),
        "robot_state": fmt["robot_state"],
        "conds": conds,
        "regions": regions,
        "masks": masks,
        "robot_masks": robot,
        "edit_info": info,
        "render_diff": render_diff,
        "absent_leftover_px": leftover,
        "mask_source": mask_src,
    }


def fmt_obs(frame: dict, cond: str) -> dict:
    return {"pixels": frame["conds"][cond], "robot_state": frame["robot_state"]}


def occlude_live_obs(obs: dict, scene: Scene) -> dict:
    """Closed loop: paint the target object gray in every camera, every step."""
    cams = {k: CAM_OF_KEY[k] for k in obs["pixels"].keys()}
    masks, _ = scene.masks(cams)
    px = {}
    for k, img in obs["pixels"].items():
        px[k] = paint(np.ascontiguousarray(img), dilate(masks[k], CFG["occ_pad_px"]), CFG["paint_gray"])
    out = dict(obs)
    out["pixels"] = px
    return out


# ----------------------------------------------------------------------------- policy side


class Stack:
    def __init__(self):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        path = CFG["policy_path"]
        pcfg = PreTrainedConfig.from_pretrained(path)
        pcfg.pretrained_path = path
        pcfg.device = "cuda"
        if getattr(pcfg, "compile_model", False):
            # torch.compile (max-autotune = CUDA graphs) reuses output buffers and bakes the graph,
            # which breaks forward hooks / patching. Interpretability needs eager mode.
            log("checkpoint config has compile_model=True -> disabling torch.compile (hooks need eager mode)")
            pcfg.compile_model = False
        ecfg = LiberoEnvConfig(task=CFG["suite"], task_ids=[int(CFG["task_id"])])
        t0 = time.time()
        self.policy = make_policy(cfg=pcfg, env_cfg=ecfg)
        self.policy.eval()
        disable_compile(self.policy.model)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=pcfg,
            pretrained_path=path,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=ecfg, policy_cfg=pcfg)
        self.config = self.policy.config
        log(f"policy loaded in {time.time() - t0:.0f}s | dtype={getattr(self.config, 'dtype', '?')} "
            f"chunk={self.config.chunk_size} n_action_steps={self.config.n_action_steps} "
            f"image_features={list(self.config.image_features)}")

    def batch(self, obs: dict, task: str) -> dict:
        from lerobot.envs.utils import preprocess_observation

        o = preprocess_observation(_add_batch_axis(obs))
        o["task"] = [task]
        o = self.env_pre(o)
        o = self.pre(o)
        return o

    def noise(self, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(int(seed))
        shape = (1, int(self.config.chunk_size), int(self.config.max_action_dim))
        return torch.randn(shape, generator=g, dtype=torch.float32).to(DEV)

    def img_keys(self, batch: dict):
        keys = [k for k in self.config.image_features if k in batch]
        keys += [k for k in self.config.image_features if k not in batch]
        return keys


def disable_compile(model):
    """Drop instance-level torch.compile wrappers (sample_actions / forward) so the model runs eagerly."""
    removed = []
    for name in ("sample_actions", "forward", "denoise_step", "embed_prefix"):
        fn = model.__dict__.get(name)
        if fn is not None and (hasattr(fn, "_torchdynamo_orig_callable") or "compile" in type(fn).__name__.lower()
                               or "OptimizedModule" in type(fn).__name__):
            del model.__dict__[name]
            removed.append(name)
    if removed:
        log(f"removed torch.compile wrappers: {removed}")
    try:
        import torch._dynamo

        torch._dynamo.reset()
    except Exception:
        pass
    return removed


def _find_lm_layers(model):
    pwe = model.paligemma_with_expert
    cands = []
    for getter in (
        lambda: pwe.paligemma.language_model.layers,
        lambda: pwe.paligemma.model.language_model.layers,
        lambda: pwe.paligemma.language_model.model.layers,
    ):
        try:
            layers = getter()
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
        except Exception:
            pass
    for name, mod in pwe.paligemma.named_modules():
        if name.endswith("language_model.layers") and isinstance(mod, nn.ModuleList):
            cands.append(mod)
    if cands:
        return cands[0]
    raise RuntimeError("Could not find the PaliGemma language-model decoder layers.")


class Runner:
    """Forward hooks on every LLM (PaliGemma/Gemma-2B) decoder layer and its MLP.

    Hooks only fire during the prefix pass (images + prompt); the action expert has its own layers.
    """

    def __init__(self, stack: Stack):
        self.stack = stack
        self.policy = stack.policy
        self.model = stack.policy.model
        self.layers = _find_lm_layers(self.model)
        self.n_layers = len(self.layers)
        self.capture = None
        self.capture_layers: set = set()
        self.mlp_patch: dict = {}
        self.resid_patch: dict = {}
        self.persistent_mlp_patch: dict = {}
        self.last_prefix: dict = {}
        self.mlp_calls = 0
        orig = self.model.embed_prefix

        def wrapped(*args, **kwargs):
            embs, pad, att = orig(*args, **kwargs)
            images = args[0] if len(args) > 0 else kwargs["images"]
            tokens = args[2] if len(args) > 2 else kwargs["tokens"]
            self.last_prefix = {
                "pad": pad[0].detach().bool().clone(),
                "n_img": len(images),
                "lang_len": int(tokens.shape[1]),
                "T": int(embs.shape[1]),
                "tokens": tokens[0].detach().cpu().clone(),
            }
            return embs, pad, att

        self.model.embed_prefix = wrapped
        for i, layer in enumerate(self.layers):
            layer.mlp.register_forward_hook(self._mlp_hook(i), with_kwargs=True)
            layer.register_forward_hook(self._layer_hook(i))
        log(f"hooked {self.n_layers} LLM decoder layers (MLP + residual output)")

    def _mlp_hook(self, i):
        def hook(mod, args, kwargs, output):
            self.mlp_calls += 1
            x = args[0] if args else next(iter(kwargs.values()))
            if self.capture is not None and i in self.capture_layers:
                self.capture[("mlp_in", i)] = x.detach()
                self.capture[("mlp_out", i)] = output.detach()
            fn = self.mlp_patch.get(i) or self.persistent_mlp_patch.get(i)
            if fn is not None:
                return fn(x, output).to(output.dtype)
            return None

        return hook

    def _layer_hook(self, i):
        def hook(mod, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if self.capture is not None and i in self.capture_layers:
                self.capture[("resid", i)] = h.detach()
            fn = self.resid_patch.get(i)
            if fn is not None:
                h2 = fn(h).to(h.dtype)
                return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
            return None

        return hook

    @torch.no_grad()
    def run(self, batch, noise, capture_layers=None, mlp_patch=None, resid_patch=None):
        self.capture = {} if capture_layers is not None else None
        self.capture_layers = set(capture_layers or [])
        self.mlp_patch = mlp_patch or {}
        self.resid_patch = resid_patch or {}
        try:
            actions = self.policy.predict_action_chunk(batch, noise=noise.clone())
        finally:
            cap = self.capture
            self.capture = None
            self.capture_layers = set()
            self.mlp_patch = {}
            self.resid_patch = {}
        return actions.detach().float()[0], cap

    # token layout ---------------------------------------------------------------
    def layout(self, batch):
        lp = self.last_prefix
        n_img, lang_len, T = lp["n_img"], lp["lang_len"], lp["T"]
        if (T - lang_len) % n_img:
            log(f"WARNING: prefix has {T - lang_len} non-language tokens, not divisible by {n_img} images")
        per_img = (T - lang_len) // n_img
        grid = int(round(math.sqrt(per_img)))
        keys = self.stack.img_keys(batch)
        suffix = [k.split(".")[-1] for k in keys]
        return {"n_img": n_img, "lang_len": lang_len, "T": T, "per_img": per_img, "grid": grid, "img_keys": keys, "img_suffix": suffix}

    def bowl_positions(self, layout, frame) -> torch.Tensor:
        T = layout["T"]
        m = torch.zeros(T, dtype=torch.bool)
        for i, suf in enumerate(layout["img_suffix"]):
            if suf in frame["regions"]["occluded"]:
                tm = token_mask(frame["regions"]["occluded"][suf], layout["grid"], CFG["token_min_frac"])
                idx = np.nonzero(tm)[0] + i * layout["per_img"]
                m[torch.as_tensor(idx, dtype=torch.long)] = True
        return m


# ----------------------------------------------------------------------------- transcoder


class TopKTranscoder(nn.Module):
    """MLP-input -> MLP-output transcoder with TopK sparsity (inputs/outputs mean-centred, unit mean norm)."""

    def __init__(self, d_in: int, d_out: int, n_feat: int, k: int):
        super().__init__()
        self.d_in, self.d_out, self.n_feat, self.k = d_in, d_out, n_feat, k
        self.W_enc = nn.Parameter(torch.empty(d_in, n_feat))
        self.b_enc = nn.Parameter(torch.zeros(n_feat))
        self.W_dec = nn.Parameter(torch.empty(n_feat, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))
        self.register_buffer("x_mean", torch.zeros(d_in))
        self.register_buffer("y_mean", torch.zeros(d_out))
        self.register_buffer("x_scale", torch.ones(()))
        self.register_buffer("y_scale", torch.ones(()))

    def init_weights(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(self.n_feat, self.d_out, generator=g)
        W = W / W.norm(dim=1, keepdim=True)
        self.W_dec.data.copy_(W)
        if self.d_in == self.d_out:
            self.W_enc.data.copy_(W.t())
        else:
            self.W_enc.data.copy_(torch.randn(self.d_in, self.n_feat, generator=g) / math.sqrt(self.d_in))

    def pre(self, x):
        xn = (x.float() - self.x_mean) / self.x_scale
        return xn @ self.W_enc + self.b_enc

    def encode(self, x):
        z = self.pre(x)
        v, i = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter_(-1, i, torch.relu(v))

    def decode(self, f):
        return (f @ self.W_dec + self.b_dec) * self.y_scale + self.y_mean

    def delta(self, df, idx=None):
        W = self.W_dec if idx is None else self.W_dec[idx]
        return (df @ W) * self.y_scale


def train_transcoder(x: torch.Tensor, y: torch.Tensor, seed: int, device=None):
    device = device or DEV
    n = x.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_hold = max(256, int(CFG["tc_holdout_frac"] * n))
    hold, train = perm[:n_hold], perm[n_hold:]
    xt = x[train].to(device).float()
    yt = y[train].to(device).float()
    xh = x[hold].to(device).float()
    yh = y[hold].to(device).float()
    tc = TopKTranscoder(x.shape[1], y.shape[1], int(CFG["tc_features"]), int(CFG["tc_k"])).to(device)
    tc.init_weights(seed)
    with torch.no_grad():
        tc.x_mean.copy_(xt.mean(0))
        tc.y_mean.copy_(yt.mean(0))
        tc.x_scale.copy_(((xt - tc.x_mean) ** 2).sum(-1).mean().sqrt().clamp_min(1e-6))
        tc.y_scale.copy_(((yt - tc.y_mean) ** 2).sum(-1).mean().sqrt().clamp_min(1e-6))
        # scale the encoder so the initial reconstruction has roughly the target norm (1 after normalisation)
        xs0 = xt[: min(4096, len(xt))]
        rec0 = (tc.encode(xs0) @ tc.W_dec).norm(dim=-1).mean()
        tc.W_enc.data /= rec0.clamp_min(1e-6)
    opt = torch.optim.Adam(tc.parameters(), lr=float(CFG["tc_lr"]), betas=(0.9, 0.999))
    steps, bs = int(CFG["tc_steps"]), int(CFG["tc_batch"])
    since_fired = torch.zeros(tc.n_feat, device=device)
    gd = torch.Generator(device=device).manual_seed(seed)
    hist = []
    for step in range(steps):
        lr_mult = min(1.0, (step + 1) / max(1, steps // 20)) * (1.0 if step < 0.8 * steps else max(0.05, (steps - step) / (0.2 * steps)))
        for pg in opt.param_groups:
            pg["lr"] = float(CFG["tc_lr"]) * lr_mult
        idx = torch.randint(0, xt.shape[0], (bs,), generator=gd, device=device)
        xb = xt[idx]
        yb = (yt[idx] - tc.y_mean) / tc.y_scale
        z = tc.pre(xb)
        v, i = z.topk(tc.k, dim=-1)
        f = torch.zeros_like(z).scatter(-1, i, torch.relu(v))
        yhat = f @ tc.W_dec + tc.b_dec
        mse = ((yhat - yb) ** 2).sum(-1).mean()
        loss = mse
        fired = (f > 0).any(0)
        since_fired = torch.where(fired, torch.zeros_like(since_fired), since_fired + 1)
        dead = since_fired > int(CFG["tc_dead_steps"])
        n_dead = int(dead.sum())
        if n_dead > 0 and CFG["tc_aux_coef"] > 0:
            ka = min(int(CFG["tc_aux_k"]), n_dead)
            zd = z.masked_fill(~dead[None, :], float("-inf"))
            va, ia = zd.topk(ka, dim=-1)
            fa = torch.zeros_like(z).scatter(-1, ia, torch.relu(va))
            resid = (yb - yhat).detach()
            ehat = fa @ tc.W_dec
            aux = ((ehat - resid) ** 2).sum(-1).mean() / (resid ** 2).sum(-1).mean().clamp_min(1e-8)
            loss = loss + float(CFG["tc_aux_coef"]) * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            tc.W_dec.data /= tc.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
        if step % 50 == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(mse.item()), "dead": n_dead})
    tc.eval()

    @torch.no_grad()
    def evaluate(xx, yy):
        out = []
        for s in range(0, xx.shape[0], 8192):
            out.append(tc.decode(tc.encode(xx[s : s + 8192])))
        yhat = torch.cat(out)
        fvu = float(((yhat - yy) ** 2).sum() / ((yy - yy.mean(0)) ** 2).sum().clamp_min(1e-12))
        nmse = float((((yhat - yy) / tc.y_scale) ** 2).sum(-1).mean())
        return fvu, nmse

    fvu_h, nmse_h = evaluate(xh, yh)
    fvu_t, nmse_t = evaluate(xt[: min(len(xt), 50000)], yt[: min(len(yt), 50000)])
    with torch.no_grad():
        fires = torch.zeros(tc.n_feat, device=device)
        for s in range(0, xt.shape[0], 8192):
            fires += (tc.encode(xt[s : s + 8192]) > 0).float().sum(0)
    metrics = {
        "fvu_heldout": fvu_h, "loss_heldout": nmse_h, "fvu_train": fvu_t, "loss_train": nmse_t,
        "alive_features": int((fires > 0).sum()), "n_train_tokens": int(len(train)), "n_heldout_tokens": int(n_hold),
        "history": hist,
    }
    return tc.cpu(), metrics


def load_tc(layer: int, device=None) -> TopKTranscoder:
    device = device or DEV
    ck = torch.load(OUT / "transcoders" / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    tc = TopKTranscoder(c["d_in"], c["d_out"], c["n_feat"], c["k"])
    tc.load_state_dict(ck["state_dict"])
    tc.requires_grad_(False)
    return tc.to(device).eval()


# ----------------------------------------------------------------------------- patch functions


def fp_set(tc, feats, target_f, token_mask=None, norm_match=None, runner=None):
    """Set features `feats` to their values in `target_f` (T, n_feat); keeps the transcoder error term."""
    idx = torch.as_tensor(list(feats), dtype=torch.long, device=target_f.device)

    def fn(x, y):
        f_live = tc.encode(x[0])
        df = target_f[:, idx] - f_live[:, idx]
        d = tc.delta(df, idx)
        if norm_match is not None:
            n = d.norm(dim=-1, keepdim=True)
            d = torch.where(n > 1e-8, d / n.clamp_min(1e-8) * norm_match[:, None], torch.zeros_like(d))
        if token_mask is not None:
            d = d * token_mask[:, None].to(d.dtype)
        return (y[0].float() + d)[None]

    return fn


def fp_scale(tc, feats, alpha: float, runner: Runner):
    """Multiply features by alpha on every valid prefix token (alpha=0 ablates)."""
    idx = torch.as_tensor(list(feats), dtype=torch.long, device=DEV)

    def fn(x, y):
        f_live = tc.encode(x[0])
        d = tc.delta((alpha - 1.0) * f_live[:, idx], idx)
        pad = runner.last_prefix.get("pad")
        if pad is not None and pad.shape[0] == d.shape[0]:
            d = d * pad[:, None].to(d.dtype)
        return (y[0].float() + d)[None]

    return fn


def fp_swap(target_y, token_mask):
    def fn(x, y):
        m = token_mask[:, None]
        return torch.where(m, target_y[0].to(y.dtype), y[0]).float()[None]

    return fn


def rp_swap(target_h, token_mask):
    def fn(h):
        m = token_mask[:, None]
        return torch.where(m, target_h[0].to(h.dtype), h[0])[None]

    return fn


def fp_splice(tc, token_mask):
    """Replace the MLP output with the transcoder's reconstruction (no error term)."""

    def fn(x, y):
        rec = tc.decode(tc.encode(x[0]))
        return torch.where(token_mask[:, None], rec, y[0].float())[None]

    return fn


# ----------------------------------------------------------------------------- plotting


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_conditions(frames, path: Path, key="image"):
    plt = _plt()
    n = len(frames)
    fig, axes = plt.subplots(n, len(CONDS), figsize=(2.2 * len(CONDS), 2.3 * n), squeeze=False)
    for r, fr in enumerate(frames):
        for c, cond in enumerate(CONDS):
            ax = axes[r][c]
            ax.imshow(model_view(fr["conds"][cond][key]))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cond, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{fr.get('episode', '?')}\nt={fr.get('t', '?')}", fontsize=8)
    fig.suptitle(f"Probe frames x conditions ({key}, as the policy sees it)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- data I/O


def load_frames():
    with open(OUT / "frames.pkl", "rb") as f:
        return pickle.load(f)


def task_of(fr_data):
    return fr_data["task"]


def find_demo_file(task_name: str):
    roots = [Path(os.environ.get("LIBERO_DATASET_DIR", "")), Path(CFG.get("libero_dataset_dir", ""))]
    for root in roots:
        if root and root.exists():
            hits = sorted(root.rglob(f"{task_name}_demo.hdf5"))
            if hits:
                return hits[0]
    return None


# ============================================================================= steps


def rollout_states(env, scene, stack, init_id: int, seed: int):
    env.init_state_id = init_id
    stack.policy.reset()
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    states, success = [], False
    for t in range(int(env._max_episode_steps)):
        states.append(np.asarray(env._env.get_sim_state()).copy())
        batch = stack.batch(obs, libero_task(env))
        with torch.inference_mode():
            act = stack.policy.select_action(batch)
        act = stack.post(act)
        act = stack.env_post({"action": act})["action"]
        obs, _, term, _, info = env.step(act.detach().to("cpu").numpy()[0])
        if term:
            success = bool(info.get("is_success", False))
            break
    return np.stack(states), success


def cmd_frames():
    setup_torch(CFG["seed"])
    stack = Stack()
    runner = Runner(stack)
    env = make_env(0)
    scene = Scene(env)
    task = libero_task(env)
    log(f"task: {task}")

    episodes = []
    src = CFG["frame_source"]
    if src in ("auto", "demo"):
        path = find_demo_file(env.task)
        if path is None:
            msg = f"no demo file {env.task}_demo.hdf5 under LIBERO_DATASET_DIR"
            if src == "demo":
                raise SystemExit(msg)
            log(msg + " -> falling back to policy rollouts")
        else:
            try:
                import h5py
            except ImportError:
                h5py = None
            log(f"demo file: {path}")
            try:
                if h5py is None:
                    raise ImportError("h5py is not installed in the venv")
                with h5py.File(path, "r") as f:
                    names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
                    for di in CFG["demo_indices"][: CFG["n_episodes"]]:
                        nm = names[int(di)]
                        episodes.append({"name": f"demo:{nm}", "states": np.array(f["data"][nm]["states"])})
            except Exception as exc:
                if src == "demo":
                    raise
                log(f"could not read demos ({type(exc).__name__}: {exc}) -> falling back to policy rollouts")
                episodes = []
            ref = np.asarray(env._env.get_sim_state())
            if episodes and episodes[0]["states"].shape[1] != ref.shape[0]:
                log(f"demo state dim {episodes[0]['states'].shape[1]} != sim state dim {ref.shape[0]} -> using rollouts")
                episodes = []
    if not episodes:
        for ep in range(int(CFG["n_episodes"])):
            states, ok = rollout_states(env, scene, stack, init_id=ep, seed=CFG["seed"] + ep)
            log(f"rollout init_state={ep}: {len(states)} steps, success={ok}")
            episodes.append({"name": f"rollout:init{ep}", "states": states, "success": ok})

    probe, extra, cand_rows = [], [], []
    frame_counter = 0
    for ei, ep in enumerate(episodes):
        S = ep["states"]
        zs = []
        for s in S:
            scene.set_state_fast(s)
            zs.append(scene.object_z())
        zs = np.array(zs)
        z0 = float(np.median(zs[: max(3, min(10, len(zs)))]))
        lifted = np.nonzero(zs > z0 + float(CFG["lift_dz"]))[0]
        t_lift = int(lifted[0]) if len(lifted) else len(S)
        t_lo = int(CFG["min_t"])
        t_hi = max(t_lo + 1, t_lift - 3)
        cand_ts = sorted(set(np.linspace(t_lo, t_hi - 1, int(CFG["n_candidates"])).round().astype(int).tolist()))
        log(f"{ep['name']}: T={len(S)} lift at t={t_lift} -> candidates {cand_ts}")
        scored = []
        for t in cand_ts:
            fr = render_frame(env, scene, S[t])
            fr.update({"episode": ep["name"], "t": int(t)})
            mpx = fr["edit_info"]["image"]["mask_px"]
            row = {"episode": ep["name"], "t": int(t), "mask_px_agentview": mpx,
                   "mask_px_wrist": fr["edit_info"].get("image2", {}).get("mask_px", 0)}
            if mpx < int(CFG["min_mask_px"]):
                row["gap"] = None
                row["note"] = "object too small/invisible"
                cand_rows.append(row)
                continue
            fr["noise_seed"] = int(CFG["seed"] + 7919 * (ei + 1) + t)
            noise = stack.noise(fr["noise_seed"])
            a_b, _ = runner.run(stack.batch(fmt_obs(fr, "base"), task), noise)
            a_o, _ = runner.run(stack.batch(fmt_obs(fr, "occluded"), task), noise)
            row["gap"] = rmse(a_b, a_o)
            cand_rows.append(row)
            scored.append((row["gap"], t, fr))
            log(f"  t={t:4d} mask_px={mpx:5d} base-vs-occluded RMSE={row['gap']:.4f}")
        if CFG["frame_pick"] == "max_gap":
            scored.sort(key=lambda r: -r[0])
        chosen = []
        for gap, t, fr in scored:
            if all(abs(t - c[1]) >= int(CFG["min_frame_spacing"]) for c in chosen):
                chosen.append((gap, t, fr))
            if len(chosen) == int(CFG["frames_per_episode"]):
                break
        if len(chosen) < int(CFG["frames_per_episode"]):
            log(f"WARNING: only {len(chosen)} usable probe frames in {ep['name']}")
        for gap, t, fr in sorted(chosen, key=lambda r: r[1]):
            fr["frame_id"] = frame_counter
            frame_counter += 1
            probe.append(fr)
        chosen_ts = {c[1] for c in chosen}
        n_extra = int(CFG["extra_train_frames_per_episode"])
        if n_extra > 0:
            ts = np.linspace(0, len(S) - 1, n_extra + 2).round().astype(int)[1:-1]
            for t in ts:
                if int(t) in chosen_ts:
                    continue
                raw = env._env.set_init_state(S[int(t)])
                fmt = env._format_raw_obs(raw)
                extra.append({"episode": ep["name"], "t": int(t), "robot_state": fmt["robot_state"],
                              "conds": {"base": {k: np.ascontiguousarray(v).copy() for k, v in fmt["pixels"].items()}}})

    data = {
        "task": task,
        "task_name": env.task,
        "target_object": scene.target,
        "obj_of_interest": scene.obj_of_interest,
        "probe": probe,
        "extra": extra,
        "episodes": [{"name": e["name"], "T": int(len(e["states"])), "success": e.get("success")} for e in episodes],
        "candidates": cand_rows,
    }
    with open(OUT / "frames.pkl", "wb") as f:
        pickle.dump(data, f)
    save_json(OUT / "frames_candidates.json", cand_rows)
    fig_conditions(probe, OUT / "fig_probe_conditions_agentview.png", "image")
    if probe and "image2" in probe[0]["conds"]["base"]:
        fig_conditions(probe, OUT / "fig_probe_conditions_wrist.png", "image2")
    summary = {
        "n_probe_frames": len(probe),
        "probe": [{"frame_id": p["frame_id"], "episode": p["episode"], "t": p["t"], "edit_info": p["edit_info"],
                   "mask_source": p["mask_source"]} for p in probe],
        "n_extra_train_frames": len(extra),
        "target_object": scene.target,
    }
    ok = len(probe) >= 1
    write_status("frames", ok, {**summary, "reason": "" if ok else "no usable probe frames"})
    log(f"saved {len(probe)} probe frames and {len(extra)} extra transcoder-training frames")
    env.close()


def cmd_validate():
    setup_torch(CFG["seed"])
    data = load_frames()
    frames, task = data["probe"], data["task"]
    stack = Stack()
    runner = Runner(stack)
    res = {"frames": []}

    # ---- check 1: edits are clean
    c1_rows, c1_ok = [], True
    for fr in frames:
        base = fr["conds"]["base"]
        for cond in CONDS[1:]:
            for key in base:
                img = fr["conds"][cond][key]
                changed = np.abs(img.astype(int) - base[key].astype(int)).max(-1) > 0
                allowed = fr["regions"][cond][key]
                outside = float((changed & ~allowed).mean())
                inside = int((changed & allowed).sum())
                row = {"frame": fr["frame_id"], "cond": cond, "camera": key, "frac_changed_outside": outside, "changed_inside_px": inside}
                if outside >= float(CFG["clean_edit_max_outside"]):
                    c1_ok = False
                if key == "image" and inside == 0:
                    row["warning"] = "edit changed nothing in agentview"
                    c1_ok = False
                c1_rows.append(row)
    # state + prompt identical across conditions
    state_same, tokens_same = True, True
    batches = {}
    for fr in frames:
        bs = {c: stack.batch(fmt_obs(fr, c), task) for c in CONDS}
        batches[fr["frame_id"]] = bs
        ref = bs["base"]
        for c in CONDS[1:]:
            if not torch.equal(bs[c]["observation.state"], ref["observation.state"]):
                state_same = False
            tk = [k for k in ref if "tokens" in k]
            for k in tk:
                if not torch.equal(bs[c][k], ref[k]):
                    tokens_same = False
    render_ok = all(max(fr["render_diff"].values()) == 0 for fr in frames)
    leftover = max(max(fr["absent_leftover_px"].values()) for fr in frames)
    c1_ok = c1_ok and state_same and tokens_same and leftover == 0
    res["check1"] = {
        "passed": c1_ok, "max_frac_changed_outside": max(r["frac_changed_outside"] for r in c1_rows),
        "robot_state_identical": state_same, "prompt_tokens_identical": tokens_same,
        "env_pixels_equal_direct_render": render_ok, "absent_render_leftover_px": leftover,
        "mask_sources": sorted({fr["mask_source"] for fr in frames}), "rows": c1_rows,
    }
    log(f"check 1 (edits clean): {'PASS' if c1_ok else 'FAIL'} | max outside={res['check1']['max_frac_changed_outside']:.5f} "
        f"| state identical={state_same} | prompt identical={tokens_same} | absent leftover px={leftover}")

    # ---- check 2: determinism
    c2 = []
    acts = {}
    for fr in frames:
        noise = stack.noise(fr["noise_seed"])
        b = batches[fr["frame_id"]]
        a1, _ = runner.run(b["base"], noise)
        a2, _ = runner.run(b["base"], noise)
        a3, _ = runner.run(b["occluded"], noise)
        a4, _ = runner.run(b["occluded"], noise)
        d = max(float((a1 - a2).abs().max()), float((a3 - a4).abs().max()))
        c2.append(d)
        acts[fr["frame_id"]] = {c: runner.run(b[c], noise)[0] for c in CONDS}
    c2_ok = max(c2) <= float(CFG["determinism_tol"])
    res["check2"] = {"passed": c2_ok, "max_abs_diff": max(c2), "per_frame": c2}
    log(f"check 2 (determinism): {'PASS' if c2_ok else 'FAIL'} | max diff = {max(c2):.3e}")

    # ---- check 3: the signal exists
    rows3 = []
    for fr in frames:
        A = acts[fr["frame_id"]]
        other_noise = stack.noise(fr["noise_seed"] + 1)
        a_seed, _ = runner.run(batches[fr["frame_id"]]["base"], other_noise)
        row = {"frame": fr["frame_id"], "gap_occluded": rmse(A["base"], A["occluded"]), "seed_noise_rmse": rmse(A["base"], a_seed)}
        for c in CONDS[1:]:
            row[f"rmse_{c}"] = rmse(A["base"], A[c])
        rows3.append(row)
    gaps = [r["gap_occluded"] for r in rows3]
    c3_ok = min(gaps) >= float(CFG["min_gap"])
    res["check3"] = {"passed": c3_ok, "min_gap": min(gaps), "mean_gap": float(np.mean(gaps)), "rows": rows3}
    log(f"check 3 (signal exists): {'PASS' if c3_ok else 'FAIL'} | base-vs-occluded RMSE per frame = "
        + ", ".join(f"{g:.4f}" for g in gaps) + f" (need >= {CFG['min_gap']})")
    for r in rows3:
        log("   frame {frame}: occluded {gap_occluded:.4f} | recolor {rmse_recolor:.4f} | absent {rmse_absent:.4f} | "
            "slab_miss {rmse_slab_miss:.4f} | occ_absent {rmse_occluded_absent:.4f} | other-noise-seed {seed_noise_rmse:.4f}".format(**r))

    # ---- check 4: the ceiling works (full layer-output swap base -> occluded)
    L = runner.n_layers
    per_layer_resid = np.zeros((len(frames), L))
    per_layer_mlp = np.zeros((len(frames), L))
    for fi, fr in enumerate(frames):
        b = batches[fr["frame_id"]]
        noise = stack.noise(fr["noise_seed"])
        a_occ, cap = runner.run(b["occluded"], noise, capture_layers=list(range(L)))
        a_base = acts[fr["frame_id"]]["base"]
        valid = runner.last_prefix["pad"]
        for li in range(L):
            a_r, _ = runner.run(b["base"], noise, resid_patch={li: rp_swap(cap[("resid", li)], valid)})
            a_m, _ = runner.run(b["base"], noise, mlp_patch={li: fp_swap(cap[("mlp_out", li)], valid)})
            per_layer_resid[fi, li] = closure_pct(a_r, a_occ, a_base)
            per_layer_mlp[fi, li] = closure_pct(a_m, a_occ, a_base)
        del cap
    mean_r = per_layer_resid.mean(0)
    mean_m = per_layer_mlp.mean(0)
    c4_ok = float(mean_r.max()) >= float(CFG["ceiling_min_pct"])
    res["check4"] = {"passed": c4_ok, "n_layers": L, "best_layer": int(mean_r.argmax()), "best_pct": float(mean_r.max()),
                     "resid_swap_pct_mean": mean_r.tolist(), "mlp_swap_pct_mean": mean_m.tolist(),
                     "resid_swap_pct_per_frame": per_layer_resid.tolist(), "mlp_swap_pct_per_frame": per_layer_mlp.tolist()}
    log(f"check 4 (ceiling): {'PASS' if c4_ok else 'FAIL'} | best full-layer swap = {mean_r.max():.1f}% at layer {int(mean_r.argmax())} "
        f"(need >= {CFG['ceiling_min_pct']}%)")
    log("   layer : resid-swap % | mlp-swap %")
    for li in range(L):
        log(f"   {li:5d} : {mean_r[li]:8.1f}     | {mean_m[li]:8.1f}")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(range(L), mean_r, marker="o", label="full layer-output swap (check 4)")
    ax.plot(range(L), mean_m, marker="s", label="MLP-output swap (transcoder site)")
    ax.axhline(float(CFG["ceiling_min_pct"]), color="gray", ls="--", lw=1)
    ax.set_xlabel("LLM layer")
    ax.set_ylabel("gap closed (%)")
    ax.set_title("Ceiling: base -> occluded swaps")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_check4_ceiling.png", dpi=120)
    plt.close(fig)

    passed = c1_ok and c2_ok and c3_ok and c4_ok
    failed = [n for n, ok in (("1", c1_ok), ("2", c2_ok), ("3", c3_ok), ("4", c4_ok)) if not ok]
    save_json(OUT / "validate.json", res)
    write_status("validate", passed, {
        "check1": c1_ok, "check2": c2_ok, "check3": c3_ok, "check4": c4_ok,
        "n_layers": L, "ceiling_best_pct": float(mean_r.max()), "ceiling_best_layer": int(mean_r.argmax()),
        "gaps": gaps, "reason": "" if passed else f"check(s) {','.join(failed)} failed",
    })


def all_samples(data):
    samples = []
    for fr in data["probe"]:
        for c in CONDS:
            samples.append({"kind": "probe", "frame_id": fr["frame_id"], "cond": c, "noise_seed": fr["noise_seed"], "ref": fr})
    for i, ex in enumerate(data["extra"]):
        samples.append({"kind": "extra", "frame_id": -1 - i, "cond": "base", "noise_seed": CFG["seed"] + 31 * (i + 1), "ref": ex})
    return samples


def cmd_capture():
    setup_torch(CFG["seed"])
    data = load_frames()
    stack = Stack()
    runner = Runner(stack)
    L = runner.n_layers
    samples = all_samples(data)
    xs = [[] for _ in range(L)]
    ys = [[] for _ in range(L)]
    sid, pos = [], []
    meta_samples, layout = [], None
    t0 = time.time()
    for si, s in enumerate(samples):
        obs = {"pixels": s["ref"]["conds"][s["cond"]], "robot_state": s["ref"]["robot_state"]}
        b = stack.batch(obs, data["task"])
        _, cap = runner.run(b, stack.noise(s["noise_seed"]), capture_layers=list(range(L)))
        if layout is None:
            layout = runner.layout(b)
        valid = runner.last_prefix["pad"].nonzero().flatten()
        for li in range(L):
            xs[li].append(cap[("mlp_in", li)][0, valid].to(torch.bfloat16).cpu())
            ys[li].append(cap[("mlp_out", li)][0, valid].to(torch.bfloat16).cpu())
        sid.append(torch.full((len(valid),), si, dtype=torch.int32))
        pos.append(valid.cpu().to(torch.int32))
        meta_samples.append({"i": si, "kind": s["kind"], "frame_id": s["frame_id"], "cond": s["cond"], "n_tokens": int(len(valid))})
        del cap
        if si % 8 == 0:
            log(f"captured {si + 1}/{len(samples)} samples ({time.time() - t0:.0f}s)")
    acts = OUT / "acts"
    acts.mkdir(parents=True, exist_ok=True)
    sid = torch.cat(sid)
    pos = torch.cat(pos)
    for li in range(L):
        torch.save({"x": torch.cat(xs[li]), "y": torch.cat(ys[li]), "sample": sid, "pos": pos}, acts / f"layer_{li:02d}.pt")
    layout = {k: v for k, v in layout.items()}
    save_json(acts / "meta.json", {"samples": meta_samples, "layout": layout, "n_layers": L, "n_tokens": int(len(sid))})
    log(f"saved activations for {L} layers, {len(samples)} samples, {len(sid)} tokens -> {acts}")
    write_status("capture", True, {"n_layers": L, "n_samples": len(samples), "n_tokens": int(len(sid)), "layout": layout})


def cmd_train():
    setup_torch(CFG["seed"])
    meta = load_json(OUT / "acts" / "meta.json")
    L = meta["n_layers"]
    layers = list(range(L)) if CFG["tc_layers"] == "all" else [int(v) for v in CFG["tc_layers"]]
    (OUT / "transcoders").mkdir(parents=True, exist_ok=True)
    metrics = {}
    for li in layers:
        t0 = time.time()
        d = torch.load(OUT / "acts" / f"layer_{li:02d}.pt", weights_only=False)
        tc, m = train_transcoder(d["x"], d["y"], seed=CFG["seed"] + li)
        m["seconds"] = time.time() - t0
        metrics[li] = m
        torch.save({"state_dict": tc.state_dict(), "config": {"d_in": tc.d_in, "d_out": tc.d_out, "n_feat": tc.n_feat, "k": tc.k},
                    "layer": li, "metrics": m}, OUT / "transcoders" / f"layer_{li:02d}.pt")
        log(f"layer {li:2d}: held-out FVU={m['fvu_heldout']:.4f} loss={m['loss_heldout']:.4f} alive={m['alive_features']}/{tc.n_feat} "
            f"({m['seconds']:.0f}s)")
        del d
        torch.cuda.empty_cache()

    # ---- check 5: faithful? (a) reconstruction error, (b) splice the transcoder into the model
    data = load_frames()
    stack = Stack()
    runner = Runner(stack)
    splice = {li: [] for li in layers}
    tcs = {li: load_tc(li) for li in layers}
    for fr in data["probe"]:
        noise = stack.noise(fr["noise_seed"])
        bb = stack.batch(fmt_obs(fr, "base"), data["task"])
        bo = stack.batch(fmt_obs(fr, "occluded"), data["task"])
        a_b, _ = runner.run(bb, noise)
        a_o, _ = runner.run(bo, noise)
        gap = max(rmse(a_b, a_o), 1e-8)
        valid = runner.last_prefix["pad"]
        for li in layers:
            a_s, _ = runner.run(bb, noise, mlp_patch={li: fp_splice(tcs[li], valid)})
            splice[li].append(rmse(a_s, a_b) / gap)
    fvus = [metrics[li]["fvu_heldout"] for li in layers]
    med = float(np.median(fvus))
    c5_ok = med <= float(CFG["tc_max_fvu"])
    rows = [{"layer": li, **{k: v for k, v in metrics[li].items() if k != "history"},
             "splice_rmse_over_gap": float(np.mean(splice[li]))} for li in layers]
    save_json(OUT / "train.json", {"rows": rows, "histories": {li: metrics[li]["history"] for li in layers}})
    log(f"check 5 (transcoder faithful): {'PASS' if c5_ok else 'FAIL'} | median held-out FVU={med:.4f} (need <= {CFG['tc_max_fvu']})")
    log("   layer | FVU(held-out) | splice RMSE / occlusion gap")
    for r in rows:
        log(f"   {r['layer']:5d} | {r['fvu_heldout']:12.4f} | {r['splice_rmse_over_gap']:8.3f}")
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].bar(layers, fvus)
    ax[0].axhline(float(CFG["tc_max_fvu"]), color="gray", ls="--", lw=1)
    ax[0].set_title("Held-out FVU per layer")
    ax[0].set_xlabel("layer")
    ax[1].bar(layers, [r["splice_rmse_over_gap"] for r in rows])
    ax[1].axhline(1.0, color="gray", ls="--", lw=1)
    ax[1].set_title("Splice error / occlusion gap (lower is better)")
    ax[1].set_xlabel("layer")
    fig.tight_layout()
    fig.savefig(OUT / "fig_check5_transcoders.png", dpi=120)
    plt.close(fig)
    write_status("train", c5_ok, {"check5": c5_ok, "median_fvu": med, "fvu_per_layer": fvus,
                                  "splice_over_gap": [r["splice_rmse_over_gap"] for r in rows],
                                  "reason": "" if c5_ok else f"median held-out FVU {med:.3f} > {CFG['tc_max_fvu']}"})


def _feature_stats(A, frame_ids, cond_idx, target, quiet):
    """A: dict (frame, cond) -> (n_feat,) summed activation."""
    d = {c: np.stack([A[(f, c)] - A[(f, "base")] for f in frame_ids]) for c in CONDS[1:]}  # (F, n)
    d_t = d[target]
    t_mean = d_t.mean(0)
    pos_frac = (d_t > 0).mean(0)
    other = np.max(np.stack([np.abs(d[c]).mean(0) for c in quiet]), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where(t_mean > 0, 1.0 - other / np.maximum(t_mean, 1e-12), -np.inf)
    return d, t_mean, pos_frac, other, spec


def cmd_goal1():
    require_gate("validate", "Goal 1")
    if CFG.get("check5_blocks", False):
        require_gate("train", "Goal 1")
    if not (OUT / "acts" / "meta.json").exists():
        raise SystemExit("[gate] activation cache (acts/) is missing, probably after a runtime reset. "
                         "Re-run the Transcoders cell (capture + train) first.")
    setup_torch(CFG["seed"])
    meta = load_json(OUT / "acts" / "meta.json")
    L = meta["n_layers"]
    layers = sorted(int(p.stem.split("_")[1]) for p in (OUT / "transcoders").glob("layer_*.pt"))
    samples = meta["samples"]
    probe_ids = sorted({s["frame_id"] for s in samples if s["kind"] == "probe"})
    rows, selected = [], {}
    contact = {}
    for li in layers:
        tc = load_tc(li)
        d = torch.load(OUT / "acts" / f"layer_{li:02d}.pt", weights_only=False)
        x, sid, pos = d["x"], d["sample"].long(), d["pos"].long()
        n = tc.n_feat
        A = torch.zeros(len(samples), n, dtype=torch.float64)
        fire_cnt = torch.zeros(n, dtype=torch.float64)
        fire_sum = torch.zeros(n, dtype=torch.float64)
        F_keep = {}
        with torch.no_grad():
            for s in range(0, x.shape[0], 8192):
                F = tc.encode(x[s : s + 8192].to(DEV)).double().cpu()
                A.index_add_(0, sid[s : s + 8192], F)
                fire_cnt += (F > 0).sum(0)
                fire_sum += F.sum(0)
                F_keep[s] = F.float().to_sparse()
        typical = (fire_sum / fire_cnt.clamp_min(1)).numpy()
        alive = (fire_cnt > 0).numpy()
        Ad = {}
        for s in samples:
            if s["kind"] == "probe":
                Ad[(s["frame_id"], s["cond"])] = A[s["i"]].numpy()
        dd, occ_mean, pos_frac, other, spec = _feature_stats(Ad, probe_ids, None, "occluded", QUIET_FOR_OCC)
        rise = occ_mean / np.maximum(typical, 1e-8)
        passed = (occ_mean > 0) & (pos_frac >= float(CFG["min_pos_frac"])) & (spec >= float(CFG["min_specificity"])) & (rise >= float(CFG["min_rise_tokens"])) & alive
        score = np.zeros(n)
        score[passed] = spec[passed] * rise[passed]
        _, col_mean, col_pos, col_other, col_spec = _feature_stats(Ad, probe_ids, None, "recolor", QUIET_FOR_COLOR)
        col_rise = col_mean / np.maximum(typical, 1e-8)
        col_score = np.where((col_mean > 0) & alive, np.clip(col_spec, -10, 1) * col_rise, -np.inf)
        for j in range(n):
            rows.append({
                "layer": li, "feature": j, "alive": bool(alive[j]), "typical_act": float(typical[j]),
                "d_occluded": float(occ_mean[j]), "d_occluded_min": float(dd["occluded"][:, j].min()),
                "d_recolor": float(dd["recolor"][:, j].mean()), "d_absent": float(dd["absent"][:, j].mean()),
                "d_slab_miss": float(dd["slab_miss"][:, j].mean()), "d_occluded_absent": float(dd["occluded_absent"][:, j].mean()),
                "frames_positive": float(pos_frac[j]), "specificity": float(spec[j]) if np.isfinite(spec[j]) else None,
                "rise_tokens": float(rise[j]), "score": float(score[j]), "pass": bool(passed[j]),
                "color_specificity": float(col_spec[j]) if np.isfinite(col_spec[j]) else None,
            })
        occ_feats = [int(j) for j in np.argsort(-score) if passed[j]][: int(CFG["max_features_per_layer"])]
        n_color = max(1, len(occ_feats))
        col_order = [int(j) for j in np.argsort(-col_score) if np.isfinite(col_score[j]) and j not in occ_feats]
        color_feats = col_order[:n_color]
        changed = [int(j) for j in range(n) if alive[j] and np.abs(np.stack([dd[c][:, j] for c in CONDS[1:]])).max() > 0]
        selected[li] = {
            "occ": occ_feats, "color": color_feats, "n_pass": int(passed.sum()),
            "score_sum": float(np.sort(score)[::-1][: int(CFG["max_features_per_layer"])].sum()),
            "alive": [int(j) for j in np.nonzero(alive)[0]], "changed": changed,
            "occ_stats": [{"feature": j, "specificity": float(spec[j]), "rise_tokens": float(rise[j]), "d_occluded": float(occ_mean[j])} for j in occ_feats],
        }
        # token maps for the contact sheet (first probe frame, agentview)
        if occ_feats:
            lay = meta["layout"]
            f0 = probe_ids[0]
            cam_i = lay["img_suffix"].index("image") if "image" in lay["img_suffix"] else 0
            lo, hi = cam_i * lay["per_img"], (cam_i + 1) * lay["per_img"]
            maps = {}
            F_all = torch.cat([F_keep[k].to_dense() for k in sorted(F_keep)])
            for s in samples:
                if s["kind"] == "probe" and s["frame_id"] == f0:
                    sel = (sid == s["i"]) & (pos >= lo) & (pos < hi)
                    idx = sel.nonzero().flatten()
                    Fm = F_all[idx][:, occ_feats[:4]]
                    grid = np.zeros((len(occ_feats[:4]), lay["per_img"]), dtype=np.float32)
                    grid[:, (pos[idx] - lo).numpy()] = Fm.t().numpy()
                    maps[s["cond"]] = grid.reshape(len(occ_feats[:4]), lay["grid"], lay["grid"])
            contact[li] = {"features": occ_feats[:4], "maps": maps}
            del F_all
        log(f"layer {li:2d}: {int(passed.sum()):3d} passing occlusion features | top {occ_feats[:8]} | color set {color_feats[:8]}")
        del d, F_keep, tc
        torch.cuda.empty_cache()

    ranked = [l for l in sorted(selected, key=lambda l: -selected[l]["score_sum"]) if selected[l]["n_pass"] > 0]
    top = ranked[: int(CFG["top_layers"])]
    with open(OUT / "goal1_feature_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json(OUT / "goal1_selected.json", {"top_layers": top, "layers": selected})
    # figures
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(layers, [selected[l]["n_pass"] for l in layers])
    ax.set_xlabel("LLM layer")
    ax.set_ylabel("# passing features")
    ax.set_title("Goal 1: occlusion-specific features per layer (specificity >= %.2f)" % CFG["min_specificity"])
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal1_counts.png", dpi=120)
    plt.close(fig)
    passing = sorted([r for r in rows if r["pass"]], key=lambda r: -r["score"])[:24]
    if passing:
        M = np.array([[r[f"d_{c}"] / max(r["typical_act"], 1e-8) for c in CONDS[1:]] for r in passing])
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(passing) + 1.5))
        vmax = np.abs(M).max()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(CONDS) - 1))
        ax.set_xticklabels(CONDS[1:], rotation=30, fontsize=8)
        ax.set_yticks(range(len(passing)))
        ax.set_yticklabels([f"L{r['layer']} f{r['feature']}" for r in passing], fontsize=7)
        fig.colorbar(im, ax=ax, label="change vs base (token-firings)")
        ax.set_title("Goal 1: top occlusion features, response per condition")
        fig.tight_layout()
        fig.savefig(OUT / "fig_goal1_features.png", dpi=120)
        plt.close(fig)
    if top and top[0] in contact:
        data = load_frames()
        fr0 = [p for p in data["probe"] if p["frame_id"] == probe_ids[0]][0]
        c = contact[top[0]]
        nf = len(c["features"])
        fig, axes = plt.subplots(nf, len(CONDS), figsize=(2.2 * len(CONDS), 2.3 * nf), squeeze=False)
        for r in range(nf):
            vmax = max(float(c["maps"][cc][r].max()) for cc in CONDS if cc in c["maps"]) or 1.0
            for ci, cond in enumerate(CONDS):
                ax = axes[r][ci]
                ax.imshow(model_view(fr0["conds"][cond]["image"]))
                if cond in c["maps"]:
                    H = fr0["conds"][cond]["image"].shape[0]
                    mp = np.ma.masked_where(c["maps"][cond][r] <= 0, c["maps"][cond][r])
                    ax.imshow(mp, cmap="autumn", alpha=0.6, vmin=0, vmax=vmax, extent=(0, H, H, 0), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(cond, fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"L{top[0]} f{c['features'][r]}", fontsize=9)
        fig.suptitle("Goal 1 contact sheet: occlusion-feature activation over agentview tokens", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "fig_goal1_contact_sheet.png", dpi=110)
        plt.close(fig)
    ok = len(top) > 0
    write_status("goal1", ok, {
        "top_layers": top, "n_pass_per_layer": {l: selected[l]["n_pass"] for l in layers},
        "selected": {l: selected[l]["occ"] for l in top},
        "reason": "" if ok else "KILL SWITCH: no layer has a passing occlusion feature -> paper = 'no sparse occlusion features in pi0.5; the signal is diffuse'",
    })
    if ok:
        log(f"Goal 1: PASS | top layers {top} | features " + "; ".join(f"L{l}:{selected[l]['occ']}" for l in top))
    else:
        log("Goal 1: KILL SWITCH - no layer has any passing feature. Stop here (negative-result paper).")


def cmd_goal2():
    require_gate("goal1", "Goal 2")
    setup_torch(CFG["seed"])
    sel = load_json(OUT / "goal1_selected.json")
    top = [int(l) for l in sel["top_layers"]]
    all_L = sorted(int(k) for k in sel["layers"])
    do_all = bool(CFG.get("goal2_all_layers", True))
    S = {l: sel["layers"][str(l)]["occ"] for l in all_L}
    C = {l: sel["layers"][str(l)]["color"] for l in all_L}
    occ_L = [l for l in all_L if S[l]]
    col_L = [l for l in all_L if C[l]]
    pools = {}
    for l in all_L:
        excl = set(S[l]) | set(C[l])
        pool = [j for j in sel["layers"][str(l)]["changed"] if j not in excl]
        if len(pool) < len(S[l]):
            pool = [j for j in sel["layers"][str(l)]["alive"] if j not in excl]
        pools[l] = pool
    data = load_frames()
    task = data["task"]
    stack = Stack()
    runner = Runner(stack)
    need = all_L if do_all else top
    tcs = {l: load_tc(l) for l in need}
    n_rand = int(CFG["n_random_draws"])
    rand_sets = {}
    for l in need:
        rng = np.random.default_rng(CFG["seed"] + 17 * l)
        rand_sets[l] = [sorted(rng.choice(pools[l], size=min(len(S[l]), len(pools[l])), replace=False).tolist()) for _ in range(n_rand)]
    rows, edges = [], []
    t0 = time.time()
    for fr in data["probe"]:
        fid = fr["frame_id"]
        noise = stack.noise(fr["noise_seed"])
        bb = stack.batch(fmt_obs(fr, "base"), task)
        bo = stack.batch(fmt_obs(fr, "occluded"), task)
        a_base, cap_b = runner.run(bb, noise, capture_layers=need)
        a_occ, cap_o = runner.run(bo, noise, capture_layers=need)
        lay = runner.layout(bb)
        valid = runner.last_prefix["pad"]
        bowl = runner.bowl_positions(lay, fr).to(valid.device) & valid
        scopes = {"all": valid, "bowl": bowl, "nonbowl": valid & ~bowl}
        gap = rmse(a_base, a_occ)
        fo = {l: tcs[l].encode(cap_o[("mlp_in", l)][0]) for l in need}
        fb = {l: tcs[l].encode(cap_b[("mlp_in", l)][0]) for l in need}

        def add(layer, method, scope, direction, a, start, target, draw=None, extra=None):
            r = {"frame": fid, "layer": layer, "method": method, "scope": scope, "direction": direction, "draw": draw,
                 "gap": gap, "closed_pct": closure_pct(a, target, start), "rmse_to_target": rmse(a, target)}
            if extra:
                r.update(extra)
            rows.append(r)

        for l in top:
            tc = tcs[l]
            s_idx = torch.as_tensor(S[l], dtype=torch.long, device=DEV)
            for sc, m in scopes.items():
                if not bool(m.any()):
                    continue
                inj = lambda patch_mlp=None, patch_res=None: runner.run(bb, noise, mlp_patch=patch_mlp, resid_patch=patch_res)[0]
                add(l, "occ_features", sc, "inject", inj({l: fp_set(tc, S[l], fo[l], m)}), a_base, a_occ,
                    extra={"n_features": len(S[l])})
                if C[l]:
                    add(l, "color_features", sc, "inject", inj({l: fp_set(tc, C[l], fo[l], m)}), a_base, a_occ,
                        extra={"n_features": len(C[l])})
                norm_S = (tc.delta(fo[l][:, s_idx] - fb[l][:, s_idx], s_idx).norm(dim=-1)) * m.float()
                for di, R in enumerate(rand_sets[l]):
                    add(l, "random_same_norm", sc, "inject", inj({l: fp_set(tc, R, fo[l], m, norm_match=norm_S)}), a_base, a_occ, draw=di)
                    add(l, "random_raw", sc, "inject", inj({l: fp_set(tc, R, fo[l], m)}), a_base, a_occ, draw=di)
                add(l, "all_tc_features", sc, "inject", inj({l: fp_set(tc, range(tc.n_feat), fo[l], m)}), a_base, a_occ)
                add(l, "mlp_swap_ceiling", sc, "inject", inj({l: fp_swap(cap_o[("mlp_out", l)], m)}), a_base, a_occ)
                add(l, "full_token_swap_ceiling", sc, "inject", inj(None, {l: rp_swap(cap_o[("resid", l)], m)}), a_base, a_occ)
            # single features (scope all)
            for j in S[l]:
                a = runner.run(bb, noise, mlp_patch={l: fp_set(tc, [j], fo[l], valid)})[0]
                add(l, f"single_feature_{j}", "all", "inject", a, a_base, a_occ, extra={"feature": j})
            # reverse direction: occluded run, set features back to base values
            rem = lambda patch: runner.run(bo, noise, mlp_patch=patch)[0]
            add(l, "occ_features", "all", "remove", rem({l: fp_set(tc, S[l], fb[l], valid)}), a_occ, a_base)
            if C[l]:
                add(l, "color_features", "all", "remove", rem({l: fp_set(tc, C[l], fb[l], valid)}), a_occ, a_base)
            add(l, "mlp_swap_ceiling", "all", "remove", rem({l: fp_swap(cap_b[("mlp_out", l)], valid)}), a_occ, a_base)
        # all top layers at once
        if len(top) > 1:
            patch = {l: fp_set(tcs[l], S[l], fo[l], valid) for l in top}
            add(-1, "occ_features_all_top_layers", "all", "inject", runner.run(bb, noise, mlp_patch=patch)[0], a_base, a_occ)
        # ALL layers at once (layer id -2): a single MLP site only carries ~3-6% of the gap, so the
        # plan's +10 pt test is only reachable when the whole MLP circuit is patched together.
        if do_all:
            run_b = lambda patch: runner.run(bb, noise, mlp_patch=patch)[0]
            n_occ = sum(len(S[l]) for l in occ_L)
            for sc, m in scopes.items():
                if not bool(m.any()):
                    continue
                add(-2, "occ_features", sc, "inject", run_b({l: fp_set(tcs[l], S[l], fo[l], m) for l in occ_L}), a_base, a_occ,
                    extra={"n_features": n_occ})
                add(-2, "mlp_swap_ceiling", sc, "inject", run_b({l: fp_swap(cap_o[("mlp_out", l)], m) for l in all_L}), a_base, a_occ)
            if col_L:
                add(-2, "color_features", "all", "inject", run_b({l: fp_set(tcs[l], C[l], fo[l], valid) for l in col_L}), a_base, a_occ,
                    extra={"n_features": sum(len(C[l]) for l in col_L)})
            for di in range(n_rand):
                p_norm, p_raw = {}, {}
                for l in occ_L:
                    if not rand_sets[l] or not rand_sets[l][di]:
                        continue
                    si = torch.as_tensor(S[l], dtype=torch.long, device=DEV)
                    nS = tcs[l].delta(fo[l][:, si] - fb[l][:, si], si).norm(dim=-1) * valid.float()
                    p_norm[l] = fp_set(tcs[l], rand_sets[l][di], fo[l], valid, norm_match=nS)
                    p_raw[l] = fp_set(tcs[l], rand_sets[l][di], fo[l], valid)
                add(-2, "random_same_norm", "all", "inject", run_b(p_norm), a_base, a_occ, draw=di)
                add(-2, "random_raw", "all", "inject", run_b(p_raw), a_base, a_occ, draw=di)
            add(-2, "all_tc_features", "all", "inject", run_b({l: fp_set(tcs[l], range(tcs[l].n_feat), fo[l], valid) for l in all_L}), a_base, a_occ)
            run_o = lambda patch: runner.run(bo, noise, mlp_patch=patch)[0]
            add(-2, "occ_features", "all", "remove", run_o({l: fp_set(tcs[l], S[l], fb[l], valid) for l in occ_L}), a_occ, a_base)
            if col_L:
                add(-2, "color_features", "all", "remove", run_o({l: fp_set(tcs[l], C[l], fb[l], valid) for l in col_L}), a_occ, a_base)
            add(-2, "mlp_swap_ceiling", "all", "remove", run_o({l: fp_swap(cap_b[("mlp_out", l)], valid) for l in all_L}), a_occ, a_base)
        # cross-layer circuit edges: does injecting S at L1 turn on S at L2?
        for i1, l1 in enumerate(sorted(top)):
            for l2 in sorted(top)[i1 + 1 :]:
                s2 = torch.as_tensor(S[l2], dtype=torch.long, device=DEV)
                base_amt = float((fb[l2][:, s2].sum(-1) * valid).sum())
                occ_amt = float((fo[l2][:, s2].sum(-1) * valid).sum())
                denom = occ_amt - base_amt
                sources = [("occ", S[l1])] + ([("random", rand_sets[l1][0])] if rand_sets[l1] else [])
                for kind, feats in sources:
                    s1 = torch.as_tensor(S[l1], dtype=torch.long, device=DEV)
                    norm = (tcs[l1].delta(fo[l1][:, s1] - fb[l1][:, s1], s1).norm(dim=-1)) * valid.float() if kind == "random" else None
                    _, cap = runner.run(bb, noise, capture_layers=[l2], mlp_patch={l1: fp_set(tcs[l1], feats, fo[l1], valid, norm_match=norm)})
                    amt = float((tcs[l2].encode(cap[("mlp_in", l2)][0])[:, s2].sum(-1) * valid).sum())
                    edges.append({"frame": fid, "from_layer": l1, "to_layer": l2, "source": kind,
                                  "frac_of_target_activation_recovered": (amt - base_amt) / denom if abs(denom) > 1e-8 else float("nan")})
        del cap_b, cap_o
        log(f"frame {fid}: done ({time.time() - t0:.0f}s) gap={gap:.4f}")

    # ---- aggregate
    def agg(filter_fn):
        vals = [r["closed_pct"] for r in rows if filter_fn(r)]
        return (float(np.nanmean(vals)), float(np.nanstd(vals)), len(vals)) if vals else (float("nan"), float("nan"), 0)

    table = []
    keys = sorted({(r["layer"], r["method"], r["scope"], r["direction"]) for r in rows}, key=lambda k: (k[0], k[3], k[1], k[2]))
    for (l, meth, sc, dr) in keys:
        mu, sd, n = agg(lambda r: r["layer"] == l and r["method"] == meth and r["scope"] == sc and r["direction"] == dr)
        table.append({"layer": l, "method": meth, "scope": sc, "direction": dr, "gap_closed_mean_pct": mu, "gap_closed_sd_pct": sd, "n": n})
    margin = float(CFG["goal2_margin_pts"])
    rule = str(CFG.get("goal2_rule", "absolute"))
    norm_margin_req = float(CFG.get("goal2_norm_margin_pts", 10.0))
    min_ceiling = float(CFG.get("goal2_min_site_ceiling_pct", 10.0))
    site_label = lambda l: "all" if l == -2 else str(l)
    verdict = {}
    sites = list(top) + ([-2] if do_all else [])
    for l in sites:
        get = lambda meth, dr="inject": next((t["gap_closed_mean_pct"] for t in table if t["layer"] == l and t["method"] == meth and t["scope"] == "all" and t["direction"] == dr), float("nan"))
        occ, rnd, col = get("occ_features"), get("random_same_norm"), get("color_features")
        rnd_raw, ceil = get("random_raw"), get("mlp_swap_ceiling")
        occ_rm, ceil_rm = get("occ_features", "remove"), get("mlp_swap_ceiling", "remove")
        ref = np.nanmax([rnd, col]) if not (np.isnan(rnd) and np.isnan(col)) else float("nan")
        m_abs = occ - ref
        ok_abs = bool(occ - rnd >= margin and (np.isnan(col) or occ - col >= margin))
        m_norm = 100.0 * m_abs / ceil if ceil and ceil > 0 else float("nan")
        ok_norm = bool(ceil >= min_ceiling and m_norm >= norm_margin_req)
        ok = ok_abs if rule == "absolute" else (ok_abs or ok_norm)
        verdict[site_label(l)] = {
            "occ": occ, "random_same_norm": rnd, "random_raw": rnd_raw, "color": col, "margin_vs_best_control": m_abs,
            "mlp_site_ceiling": ceil, "occ_share_of_ceiling_pct": 100.0 * occ / ceil if ceil and ceil > 0 else float("nan"),
            "margin_share_of_ceiling_pts": m_norm, "occ_remove": occ_rm, "mlp_site_ceiling_remove": ceil_rm,
            "passed_absolute": ok_abs, "passed_normalized": ok_norm, "passed": ok,
        }
        name = "ALL layers" if l == -2 else f"layer {l}"
        log(f"{name:10s}: occlusion {occ:6.1f}% | random(same-norm) {rnd:6.1f}% | random(raw) {rnd_raw:6.1f}% | color {col:6.1f}% "
            f"| MLP-site ceiling {ceil:6.1f}% -> occlusion = {verdict[site_label(l)]['occ_share_of_ceiling_pct']:5.1f}% of ceiling "
            f"| {'PASS' if ok else 'fail'} (rule={rule}: +{margin:.0f} pts over random and color)")
    with open(OUT / "goal2_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "goal2_circuit_trace_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    lines = ["| layer | method | scope | direction | gap closed % (mean ± sd) | n |", "|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['layer'] if t['layer'] >= 0 else ('all-layers' if t['layer'] == -2 else 'top-all')} | {t['method']} | {t['scope']} | {t['direction']} | "
                     f"{t['gap_closed_mean_pct']:.1f} ± {t['gap_closed_sd_pct']:.1f} | {t['n']} |")
    (OUT / "goal2_circuit_trace_table.md").write_text("\n".join(lines) + "\n")
    edge_tab = {}
    for e in edges:
        edge_tab.setdefault((e["from_layer"], e["to_layer"], e["source"]), []).append(e["frac_of_target_activation_recovered"])
    edge_rows = [{"from_layer": k[0], "to_layer": k[1], "source": k[2], "mean_frac_recovered": float(np.nanmean(v))} for k, v in edge_tab.items()]
    save_json(OUT / "goal2_edges.json", edge_rows)
    for e in edge_rows:
        log(f"edge L{e['from_layer']} -> L{e['to_layer']} ({e['source']}): {100 * e['mean_frac_recovered']:.1f}% of target-feature activation")

    # ---- contact-sheet figure
    plt = _plt()
    best = max(sites, key=lambda l: verdict[site_label(l)]["occ"] if np.isfinite(verdict[site_label(l)]["occ"]) else -1e9)
    fr0 = data["probe"][0]
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, len(CONDS), height_ratios=[1, 1.3])
    for ci, cond in enumerate(CONDS):
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(model_view(fr0["conds"][cond]["image"]))
        ax.set_title(cond, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    methods = ["occ_features", "color_features", "random_same_norm", "random_raw", "all_tc_features", "mlp_swap_ceiling", "full_token_swap_ceiling"]
    scopes_ = ["all", "bowl", "nonbowl"]
    width = 0.8 / len(scopes_)
    for si, sc in enumerate(scopes_):
        vals, errs = [], []
        for meth in methods:
            t = next((t for t in table if t["layer"] == best and t["method"] == meth and t["scope"] == sc and t["direction"] == "inject"), None)
            vals.append(t["gap_closed_mean_pct"] if t else np.nan)
            errs.append(t["gap_closed_sd_pct"] if t else 0)
        ax.bar(np.arange(len(methods)) + si * width - 0.4 + width / 2, vals, width, yerr=errs, capsize=2, label=f"tokens: {sc}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=8)
    ax.set_ylabel("gap closed (%)  base -> occluded")
    ax.set_title(f"Goal 2, {'all layers' if best == -2 else f'layer {best}'}: patch only the occlusion features vs controls "
                 f"(mean ± sd over {len(data['probe'])} frames)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal2_contact_sheet.png", dpi=110)
    plt.close(fig)

    passing = [site_label(l) for l in sites if verdict[site_label(l)]["passed"]]
    ok = len(passing) > 0
    if "all" in passing:
        goal3_layers = occ_L
    else:
        goal3_layers = [int(x) for x in passing] or ([best] if best >= 0 else occ_L)
    write_status("goal2", ok, {
        "verdict": verdict, "passing_layers": passing, "best_layer": site_label(best), "rule": rule,
        "features": {str(l): S[l] for l in top}, "goal3_layers": goal3_layers,
        "reason": "" if ok else "occlusion features do not beat random/color by the margin at any site (single layers or all layers) "
                                "-> Goal 2's null is the finding; skip Goal 3",
    })
    log(f"Goal 2: {'PASS' if ok else 'FAIL (null result is the finding)'} | passing sites {passing} | rule={rule}")


def run_episode(env, scene, stack, runner, ep: int, hide: bool, patch: dict, video_path: Path | None):
    import cv2

    seed = int(CFG["seed"]) + 1000 + ep
    env.init_state_id = int(CFG["goal3_init_offset"]) + ep
    stack.policy.reset()
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    runner.persistent_mlp_patch = patch or {}
    writer = None
    success, steps = False, 0
    try:
        for t in range(int(CFG["goal3_max_steps"]) or int(env._max_episode_steps)):
            if hide:
                obs = occlude_live_obs(obs, scene)
            if video_path is not None:
                frame = model_view(obs["pixels"]["image"])
                if writer is None:
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (frame.shape[1], frame.shape[0]))
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            batch = stack.batch(obs, libero_task(env))
            with torch.inference_mode():
                act = stack.policy.select_action(batch)
            act = stack.post(act)
            act = stack.env_post({"action": act})["action"]
            obs, _, term, _, info = env.step(act.detach().to("cpu").numpy()[0])
            steps = t + 1
            if term:
                success = bool(info.get("is_success", False))
                break
    finally:
        runner.persistent_mlp_patch = {}
        if writer is not None:
            writer.release()
    return success, steps


def cmd_goal3():
    st2 = require_gate("goal2", "Goal 3")
    setup_torch(CFG["seed"])
    sel = load_json(OUT / "goal1_selected.json")
    layers = CFG["goal3_layers"]
    if layers == "auto":
        layers = st2.get("goal3_layers") or [l for l in (st2.get("passing_layers") or [st2.get("best_layer")]) if str(l).lstrip("-").isdigit()]
    layers = [int(l) for l in layers if l is not None and str(l).lstrip("-").isdigit() and int(l) >= 0]
    layers = [l for l in layers if sel["layers"][str(l)]["occ"]]
    if not layers:
        raise SystemExit("[gate] no layers with occlusion features to intervene on in Goal 3.")
    S = {l: sel["layers"][str(l)]["occ"] for l in layers}
    stack = Stack()
    runner = Runner(stack)
    tcs = {l: load_tc(l) for l in layers}
    env = make_env(0)
    scene = Scene(env)
    alpha = float(CFG["amplify_factor"])
    conds = []
    if CFG["goal3_include_clean"]:
        conds.append(("clean_reference", False, {}))
    conds += [
        ("baseline_hidden", True, {}),
        ("features_ablated", True, {l: fp_scale(tcs[l], S[l], 0.0, runner) for l in layers}),
        ("features_amplified", True, {l: fp_scale(tcs[l], S[l], alpha, runner) for l in layers}),
    ]
    n_ep = int(CFG["goal3_episodes"])
    results = {}
    for name, hide, patch in conds:
        succ = []
        for ep in range(n_ep):
            vp = OUT / "videos" / f"{name}_ep{ep}.mp4" if (CFG["save_videos"] and ep == 0) else None
            ok, steps = run_episode(env, scene, stack, runner, ep, hide, patch, vp)
            succ.append(ok)
            log(f"{name:20s} episode {ep}: success={ok} steps={steps}")
        results[name] = np.array(succ, dtype=float)
    base = results["baseline_hidden"]
    c6_ok = 0.0 < base.mean() < 1.0
    log(f"check 6 (closed loop measurable): {'PASS' if c6_ok else 'FAIL'} | baseline (bowl hidden) success = {base.mean():.2f}")
    sign = 1.0 if CFG["predicted_direction"] == "ablate_up" else -1.0
    stats = {}
    for name, pred_sign in (("features_ablated", sign), ("features_amplified", -sign)):
        diff, (lo, hi) = bootstrap_paired_diff(results[name], base, int(CFG["n_bootstrap"]), CFG["seed"])
        in_dir = (lo > 0) if pred_sign > 0 else (hi < 0)
        stats[name] = {"success": float(results[name].mean()), "diff_vs_baseline": diff, "ci95": [lo, hi],
                       "predicted_sign": "+" if pred_sign > 0 else "-", "ci_excludes_zero_in_predicted_direction": bool(in_dir)}
        log(f"{name}: success {results[name].mean():.2f} | diff vs baseline {diff:+.2f} CI95 [{lo:+.2f}, {hi:+.2f}] "
            f"| predicted {'+' if pred_sign > 0 else '-'} -> {'shift detected' if in_dir else 'no reliable shift'}")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    names = list(results)
    ax.bar(range(len(names)), [results[n].mean() for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title(f"Goal 3: closed loop, {n_ep} episodes/condition, layers {layers}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_goal3_success.png", dpi=120)
    plt.close(fig)
    shift = any(s["ci_excludes_zero_in_predicted_direction"] for s in stats.values())
    save_json(OUT / "goal3.json", {"layers": layers, "features": S, "per_episode": {k: v.tolist() for k, v in results.items()}, "stats": stats, "check6": c6_ok})
    write_status("goal3", shift if c6_ok else False, {
        "check6": c6_ok, "success": {k: float(v.mean()) for k, v in results.items()}, "stats": stats, "layers": layers,
        "reason": ("" if shift else "quantified null: causal effect does not survive closed loop") if c6_ok else
                  "check 6 failed: baseline success is 0% or 100%, so no shift can be detected",
    })
    env.close()


# ============================================================================= main

COMMANDS = {"frames": cmd_frames, "validate": cmd_validate, "capture": cmd_capture, "train": cmd_train,
            "goal1": cmd_goal1, "goal2": cmd_goal2, "goal3": cmd_goal3}


def main():
    global CFG, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=list(COMMANDS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    OUT = Path(args.out)
    CFG = load_json(OUT / "config.json")
    log(f"== {args.command} | out={OUT}")
    t0 = time.time()
    COMMANDS[args.command]()
    log(f"== {args.command} finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
