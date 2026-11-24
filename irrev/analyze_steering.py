"""RQ4 analysis (PREREGISTRATION.md, RQ4): steering curve, random-direction band, trend test.

alpha = 0 is the unsteered neutral run (agent_neutral.jsonl), same tasks, same code, greedy.
Per condition and task version: strict caution, broad caution, irreversible action taken, task
success, parse-failure rate per step, mean steps. Trend test (preregistered): logistic regression
of strict caution on alpha over the probe-direction runs (alpha in {-2,-1,-0.5,0,0.5,1,2} x scale),
Wald test of the slope, plus a pair-cluster bootstrap CI of the slope (robustness, not preregistered).
Fitted per task version and pooled (pooled model adds a version indicator).

Usage: python -m irrev.analyze_steering --steer results/qwen7b/steer --neutral results/qwen7b/agent_neutral.jsonl \
          --out results/qwen7b/analysis_steering.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np

from .analyze_behaviour import episode_measures

B = 1000
RNG = np.random.default_rng(0)
KEYS = ("strict", "broad", "took_irreversible", "success", "asked", "backup", "safer", "declined", "checked")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def logit_fit(X, y, iters=100):
    """Logistic regression by Newton-Raphson; returns coefficients and Wald SEs (None if separated)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        w += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ w))
    H = X.T @ (X * (p * (1 - p))[:, None])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(len(w), np.nan)
    return w, se


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def trend(rows, version=None):
    """rows: (alpha, version, pair_id, y). Slope of y on alpha."""
    R = [r for r in rows if version is None or r[1] == version]
    a = np.array([r[0] for r in R], float)
    y = np.array([r[3] for r in R], float)
    cols = [np.ones_like(a), a]
    if version is None:
        cols.append(np.array([r[1] == "irreversible" for r in R], float))
    X = np.stack(cols, 1)
    if y.min() == y.max():
        return {"n": len(y), "slope": None, "note": "outcome constant: no trend estimable"}
    w, se = logit_fit(X, y)
    z = w[1] / se[1] if se[1] > 0 else float("nan")
    # pair-cluster bootstrap of the slope
    pairs = sorted({r[2] for r in R})
    idx = {p: [i for i, r in enumerate(R) if r[2] == p] for p in pairs}
    reps = []
    for _ in range(B):
        sel = np.concatenate([idx[p] for p in RNG.choice(pairs, len(pairs))])
        if y[sel].min() == y[sel].max():
            continue
        wb, _ = logit_fit(X[sel], y[sel])
        reps.append(wb[1])
    return {"n": len(y), "slope_per_unit_multiple": float(w[1]), "se": float(se[1]), "z": float(z),
            "wald_p": float(norm_p(z)), "cluster_bootstrap_ci95": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]}


def summarise(eps):
    out = {}
    for ver in ("reversible", "irreversible"):
        E = [e for e in eps if e["version"] == ver]
        M = [episode_measures(e) for e in E]
        d = {"n": len(E)}
        for k in KEYS:
            x = [float(m[k]) for m in M]
            d[k] = float(np.mean(x)) if x else float("nan")
        steps = sum(e["n_steps"] for e in E)
        d["parse_fail_per_step"] = sum(e["n_parse_failures"] for e in E) / max(steps, 1)
        d["mean_steps"] = steps / max(len(E), 1)
        d["ended_by_parse_failures"] = float(np.mean([e["end_reason"] == "parse_failures" for e in E]))
        d["hit_step_limit"] = float(np.mean([e["end_reason"] == "step_limit" for e in E]))
        out[ver] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer", required=True)
    ap.add_argument("--neutral", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.steer, "steer_meta.json")))
    scale = meta["scale"]
    conds = {("none", 0.0): load(a.neutral)}
    for p in sorted(glob.glob(os.path.join(a.steer, "test_*x.jsonl"))):
        m = re.match(r"test_(\w+?)_([+-][\d.]+)x\.jsonl$", os.path.basename(p))
        conds[(m.group(1), float(m.group(2)))] = load(p)
    table = {}
    for (d, mult), eps in sorted(conds.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        assert len(eps) == 120, (d, mult, len(eps))
        table[f"{d}:{mult:+.1f}x"] = {"direction": d, "multiple": mult, "alpha": mult * scale, **summarise(eps)}
    # trend test on the probe direction (+ alpha 0)
    probe = [(k[1], e["version"], e["pair_id"], e) for k, eps in conds.items() if k[0] in ("probe", "none") for e in eps]
    tests = {}
    for outcome in ("strict", "took_irreversible", "success"):
        rows = [(m, v, p, float(episode_measures(e)[outcome])) for m, v, p, e in probe]
        tests[outcome] = {ver or "pooled": trend(rows, ver) for ver in ("irreversible", "reversible", None)}
    res = {"model_run": a.steer, "layer": meta["layer"], "scale": scale, "s_max": meta["s_max"],
           "calibration_parse_fail": meta["calibration"], "cosines": meta["cosine_with_probe_direction"],
           "multiple_unit": "multiples of scale; alpha = multiple * scale * mean residual norm is the added vector norm",
           "conditions": table, "trend_tests_probe_direction": tests}
    json.dump(res, open(a.out, "w"), indent=1)
    # compact print
    print(f"L*={meta['layer']}  scale={scale}  s_max={meta['s_max']}")
    hdr = "condition            | IRR version: irrev  strict  success  pf/step steps | REV version: irrev  strict  success  pf/step steps"
    print(hdr)
    for k, v in table.items():
        i, r = v["irreversible"], v["reversible"]
        print(f"{k:20s} | {i['took_irreversible']:.3f} {i['strict']:.3f} {i['success']:.3f} {i['parse_fail_per_step']:.3f} {i['mean_steps']:.1f}"
              f" | {r['took_irreversible']:.3f} {r['strict']:.3f} {r['success']:.3f} {r['parse_fail_per_step']:.3f} {r['mean_steps']:.1f}")
    for o, t in tests.items():
        for ver, s in t.items():
            if s.get("slope_per_unit_multiple") is None:
                print(f"trend {o:18s} {ver:12s} {s.get('note')}")
            else:
                print(f"trend {o:18s} {ver:12s} slope {s['slope_per_unit_multiple']:+.3f} (SE {s['se']:.3f}) p={s['wald_p']:.4f} "
                      f"boot CI [{s['cluster_bootstrap_ci95'][0]:+.3f}, {s['cluster_bootstrap_ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
