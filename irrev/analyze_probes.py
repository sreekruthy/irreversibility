"""RQ1 summary from probes.json (PREREGISTRATION.md, RQ1) + probe-vs-stated agreement (RQ2 kappa).

Outputs analysis_probes.json with, per environment: per-layer mean +/- sd over seeds for binary / 3-class /
mean-token probes, within-pair accuracy per tier (binding = tier3), selectivity, shuffled-label control,
baselines, ablations (minimal pairs only, horizon 1/2), cross-environment transfer; values at L* and at the
best test layer (reported only descriptively; L* is the preregistered layer).
Agreement: Cohen's kappa between probe (3-class, L*, seed-0 test split) and stated answer (A/B/C argmax),
and each against the true label, on the sandbox items that have both.

Usage: python -m irrev.analyze_probes --tag qwen7b
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from .analyze_behaviour import cohen_kappa

LABELS = ["reversible", "partial", "irreversible"]


def per_layer(runs, path):
    by = {}
    for run in runs:
        for r in run["layers"]:
            v = r
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                by.setdefault(r["layer"], []).append(float(v))
    return {int(L): {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n_seeds": len(v)} for L, v in sorted(by.items(), key=lambda kv: int(kv[0]))}


def at(curve, L):
    return curve.get(L) or curve.get(str(L))


def seed0_layer(runs, L):
    return next((r for r in runs[0]["layers"] if r["layer"] == L), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    base = f"results/{a.tag}"
    P = json.load(open(f"{base}/probes/probes.json"))
    L = P["best_layer"]["layer"]
    out = {"model": a.tag, "best_layer_Lstar": L, "solver": P.get("solver"), "envs": {}}
    for env, e in P["envs"].items():
        d = {"n_items": e["n_items"], "n_unique": e["n_unique"], "label_counts": e["label_counts"], "curves": {}}
        for k in ("binary_last", "three_class_last", "binary_mean"):
            d["curves"][k] = per_layer(e[k], ("test", "balanced_accuracy"))
        d["curves"]["binary_last_auroc"] = per_layer(e["binary_last"], ("test", "auroc"))
        for t in ("tier1", "tier2", "tier3"):
            c = per_layer(e["binary_last"], ("pairs", t, "within_pair_accuracy"))
            if c:
                d["curves"][f"pairs_{t}"] = c
        d["curves"]["selectivity"] = per_layer(e["binary_last"], ("control_task", "selectivity"))
        d["curves"]["control_task"] = per_layer(e["binary_last"], ("control_task", "balanced_accuracy"))
        d["curves"]["shuffled"] = per_layer(e["binary_last"], ("shuffled", "balanced_accuracy"))
        for ab in ("binary_last_minimal_pairs_only", "binary_last_h1", "binary_last_h2"):
            if ab in e:
                d["curves"][ab] = per_layer(e[ab], ("test", "balanced_accuracy"))
        # baselines (mean over seeds)
        bl = {}
        for run in e["binary_last"]:
            for k, v in (run.get("baselines") or {}).items():
                bl.setdefault(k, []).append(v["balanced_accuracy"])
        d["baselines_binary_bal_acc"] = {k: float(np.mean(v)) for k, v in bl.items()}
        bp = {}
        for run in e["binary_last"]:
            for k, v in (run.get("baselines") or {}).items():
                for t, x in (v.get("pairs") or {}).items():
                    bp.setdefault(f"{k}:{t}", []).append(x)
        d["baselines_within_pair"] = {k: float(np.mean(v)) for k, v in bp.items()}
        bl3 = {}
        for run in e["three_class_last"]:
            for k, v in (run.get("baselines") or {}).items():
                bl3.setdefault(k, []).append(v["balanced_accuracy"])
        d["baselines_3class_bal_acc"] = {k: float(np.mean(v)) for k, v in bl3.items()}
        # at L*
        s0 = seed0_layer(e["binary_last"], L)
        d["at_Lstar"] = {k: at(c, L) for k, c in d["curves"].items()}
        if s0:
            d["at_Lstar_seed0"] = {"test": s0["test"], "pairs": s0["pairs"]}
        best = max(d["curves"]["binary_last"], key=lambda x: d["curves"]["binary_last"][x]["mean"])
        d["best_test_layer_binary_descriptive"] = {"layer": best, **d["curves"]["binary_last"][best]}
        out["envs"][env] = d
    for k in ("transfer_sandbox_to_sw", "transfer_sw_to_sandbox"):
        if k in P:
            t = {r["layer"]: {"balanced_accuracy": r["balanced_accuracy"], "auroc": r["auroc"], "n": r["n"]} for r in P[k]}
            b = max(t, key=lambda x: t[x]["auroc"])
            out[k] = {"per_layer": t, "at_Lstar": t.get(L), "best_auroc_layer_descriptive": {"layer": b, **t[b]}}
    # agreement with stated answers
    try:
        preds = {r["id"]: r for r in map(json.loads, open(f"{base}/probes/probe_preds.jsonl"))}
        stated = {r["id"]: r for r in map(json.loads, open(f"{base}/stated_sandbox.jsonl"))}
        ids = sorted(set(preds) & set(stated))
        y = [preds[i]["label"] for i in ids]
        p = [preds[i]["probe3"] for i in ids]
        s = [stated[i]["stated_option"] for i in ids]
        out["agreement_sandbox_seed0_test"] = {
            "n": len(ids),
            "probe_accuracy": float(np.mean([a == b for a, b in zip(y, p)])),
            "stated_accuracy": float(np.mean([a == b for a, b in zip(y, s)])),
            "kappa_probe_vs_stated": cohen_kappa(p, s, LABELS),
            "kappa_probe_vs_true": cohen_kappa(p, y, LABELS),
            "kappa_stated_vs_true": cohen_kappa(s, y, LABELS),
            "three_way": {f"probe {'right' if a == b else 'wrong'} / stated {'right' if a == c else 'wrong'}": n
                          for (a, b, c), n in []} or dict(Counter(
                              f"probe {'right' if t == pp else 'wrong'}, stated {'right' if t == ss else 'wrong'}"
                              for t, pp, ss in zip(y, p, s))),
        }
    except FileNotFoundError as ex:
        out["agreement_sandbox_seed0_test"] = {"error": str(ex)}
    json.dump(out, open(f"{base}/analysis_probes.json", "w"), indent=1)
    # print a compact summary
    print(f"{a.tag}: L* = {L}")
    for env, d in out["envs"].items():
        s = d["at_Lstar"]
        f = lambda c: f"{c['mean']:.3f}±{c['sd']:.3f}" if c else "n/a"
        print(f" [{env}] n={d['n_items']} binary {f(s['binary_last'])} | 3-class {f(s['three_class_last'])} | mean-token {f(s['binary_mean'])}"
              f" | AUROC {f(s['binary_last_auroc'])}")
        print(f"   pairs: " + " ".join(f"{t} {f(s.get('pairs_'+t))}" for t in ("tier1", "tier2", "tier3")))
        print(f"   controls: shuffled {f(s['shuffled'])} | control task {f(s['control_task'])} | selectivity {f(s['selectivity'])}")
        print(f"   baselines binary {d['baselines_binary_bal_acc']} | 3-class {d['baselines_3class_bal_acc']}")
        print(f"   baselines within-pair {d['baselines_within_pair']}")
        print(f"   ablations: minimal-pairs-only {f(s.get('binary_last_minimal_pairs_only'))} | h1 {f(s.get('binary_last_h1'))} | h2 {f(s.get('binary_last_h2'))}")
        print(f"   best test layer (descriptive): {d['best_test_layer_binary_descriptive']}")
    for k in ("transfer_sandbox_to_sw", "transfer_sw_to_sandbox"):
        if k in out:
            print(f" {k}: at L* {out[k]['at_Lstar']} | best {out[k]['best_auroc_layer_descriptive']}")
    print(" agreement:", json.dumps(out["agreement_sandbox_seed0_test"]))


if __name__ == "__main__":
    main()
