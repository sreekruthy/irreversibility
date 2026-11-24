"""RQ4: activation steering, as preregistered (PREREGISTRATION.md, RQ4).

1. Directions at the best layer L* (from probes.py, steer_dirs.npz):
   'probe'   mean(irreversible) - mean(reversible), all training items (the brief's definition; primary)
   'binding' the same on binding-tier training items only (secondary)
   'rand0..2' random unit directions (controls), same scaling
2. Calibration on DEV tasks only: strengths s * mean_resid_norm(L*) for s in CANDIDATES, both signs,
   along 'probe'; s_max = largest s whose reply parse-failure rate stays < 20% for both signs.
   scale = s_max / 2, so the largest test strength (2 x scale) is s_max.
3. Test runs (60 pairs): probe direction at alpha in {-2,-1,-0.5,+0.5,+1,+2} x scale
   (alpha 0 = the existing unsteered neutral run); random directions and the binding direction at +/-2 x scale.
Every run resumes / restarts by the run guard, like the other scripts.

Usage: python -m irrev.steer_run --model Qwen/Qwen2.5-7B-Instruct --probes results/qwen7b/probes --out results/qwen7b/steer
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

CANDIDATES = (0.05, 0.1, 0.2, 0.4, 0.8)
TEST_MULTS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def parse_fail_rate(path):
    rows = [json.loads(l) for l in open(path)]
    steps = sum(len(r["trace"]) for r in rows)
    fails = sum(r["n_parse_failures"] for r in rows)
    return fails / max(steps, 1)


def agent(model, tasks, split, out, steer=None, layer=None, alpha=0.0, tag=""):
    cmd = [sys.executable, "-m", "irrev.agent", "--model", model, "--tasks", tasks, "--split", split,
           "--out", out, "--batch", "8", "--tag", tag]
    if steer:
        cmd += ["--steer", steer, "--steer-layer", str(layer), "--alpha", repr(float(alpha))]
    print(">>", " ".join(cmd[3:]), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--skip-controls", action="store_true", help="only if GPU time runs out (report as deviation)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(os.path.join(a.probes, "steer_dirs.npz"))
    L, norm = int(z["layer"]), float(z["mean_norm"])
    if L < 1:
        raise SystemExit("best layer is the embedding layer; steering is defined on decoder layers only")
    dirs = {"probe": z["direction"], "binding": z["direction_binding"]}
    d = z["direction"].shape[0]
    for k in range(3):
        dirs[f"rand{k}"] = np.random.default_rng(1000 + k).standard_normal(d).astype(np.float32)
    paths = {}
    for name, v in dirs.items():
        p = os.path.join(a.out, f"dir_{name}.npz")
        np.savez(p, direction=(v / (np.linalg.norm(v) + 1e-12)).astype(np.float32), mean_norm=np.float32(norm))
        paths[name] = p
    cos = {n: float(dirs["probe"] @ v / (np.linalg.norm(dirs["probe"]) * np.linalg.norm(v) + 1e-12)) for n, v in dirs.items()}
    meta = {"layer": L, "mean_resid_norm": norm, "cosine_with_probe_direction": cos, "candidates": CANDIDATES}

    # ---- calibration on dev tasks ----
    cal = {}
    s_max = None
    for s in CANDIDATES:
        rates = {}
        for sign in (+1, -1):
            out = os.path.join(a.out, f"dev_probe_{sign * s:+.3f}.jsonl")
            agent(a.model, a.tasks, "dev", out, paths["probe"], L, sign * s, tag="calibration")
            rates[sign] = parse_fail_rate(out)
        cal[s] = rates
        print(f"calibration s={s}: parse-failure rate +{rates[1]:.3f} / -{rates[-1]:.3f}", flush=True)
        if max(rates.values()) < 0.20:
            s_max = s
        else:
            break
    if s_max is None:
        s_max = CANDIDATES[0]
        meta["calibration_note"] = "even the smallest strength exceeded 20% parse failures; smallest used"
    scale = s_max / 2
    meta.update(calibration={str(k): v for k, v in cal.items()}, s_max=s_max, scale=scale)
    json.dump(meta, open(os.path.join(a.out, "steer_meta.json"), "w"), indent=1)
    print(f"s_max = {s_max} -> scale = {scale}; test alphas = {[m * scale for m in TEST_MULTS]}", flush=True)

    # ---- test runs ----
    runs = [("probe", m) for m in TEST_MULTS]
    if not a.skip_controls:
        runs += [(f"rand{k}", sgn * 2.0) for k in range(3) for sgn in (+1, -1)]
        runs += [("binding", sgn * 2.0) for sgn in (+1, -1)]
    for name, m in runs:
        out = os.path.join(a.out, f"test_{name}_{m:+.1f}x.jsonl")
        agent(a.model, a.tasks, "test", out, paths[name], L, m * scale, tag=f"{name}:{m}")
    meta["test_runs"] = [{"direction": n, "multiple": m, "alpha": m * scale} for n, m in runs]
    json.dump(meta, open(os.path.join(a.out, "steer_meta.json"), "w"), indent=1)
    print("STEER DONE", flush=True)


if __name__ == "__main__":
    main()
