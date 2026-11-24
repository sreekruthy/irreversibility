"""Offline analysis of the exploratory controls (no GPU).

* ScienceWorld within-verb test: the bag-of-words baseline on the action can exploit the verb (e.g. 'eat' is always
  irreversible). Within each verb that has both classes in a test split, compare the probe's balanced accuracy and
  AUROC with the bag-of-words baseline trained on the same split. Items are pooled over the three splits.
* Sensitivity to suspected ScienceWorld label errors: the same metrics after dropping items whose label changed in
  the 5x-budget re-search (data/scienceworld_recheck.w*.jsonl), if any.
* Collects the harm-direction and held-out-domain results from results/<model>/extra_controls.json.
Usage: python -m irrev.analyze_extra --tag qwen7b
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .probes import labels, split, unique_mask, verb


def bow_preds(rows):
    """Per-item P(irreversible) from bag-of-words on the action text, same splits and training rule as probes.py."""
    y, keep = labels(rows, "binary")
    out = {}
    for seed in (0, 1, 2):
        tr, te = split(rows, seed)
        trm, tem = unique_mask(rows, tr) & keep, te & keep
        vec = CountVectorizer(token_pattern=r"[A-Za-z0-9_@.\-/]+", ngram_range=(1, 2))
        Xtr = vec.fit_transform([rows[i]["action_text"] for i in np.where(trm)[0]])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, y[trm])
        P = m.predict_proba(vec.transform([rows[i]["action_text"] for i in np.where(tem)[0]]))
        col = list(m.classes_).index(1)
        for j, i in enumerate(np.where(tem)[0]):
            out[(rows[i]["id"], seed)] = float(P[j, col])
    return out


def metrics(items):
    y = np.array([t for t, _ in items])
    p = np.array([s for _, s in items])
    r = {"n": len(y), "n_irr": int(y.sum())}
    if len(set(y)) == 2:
        r["balanced_accuracy"] = float(balanced_accuracy_score(y, p > 0.5))
        r["auroc"] = float(roc_auc_score(y, p))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open("data/probe_scienceworld.jsonl")]
    byid = {r["id"]: r for r in rows}
    preds = [json.loads(l) for l in open(f"results/{a.tag}/probes/sw_preds.jsonl")]
    bow = bow_preds(rows)
    flips = set()
    for f in glob.glob("data/scienceworld_recheck.w*.jsonl"):
        for l in open(f):
            x = json.loads(l)
            if x["new"] != x["old"]:
                flips.add(x["id"])
    res = {"model": a.tag, "n_label_flips_in_recheck": len(flips), "flipped_ids": sorted(flips)}
    for name, excl in (("all_items", set()), ("excluding_relabelled", flips)):
        P = [(int(x["label"] == "irreversible"), x["p_irr"], verb(byid[x["id"]]), x["seed"], x["id"]) for x in preds
             if x["id"] not in excl]
        B = [(t, bow[(i, s)], v, s, i) for t, _, v, s, i in P]
        d = {"probe_overall": metrics([(t, p) for t, p, *_ in P]), "bow_overall": metrics([(t, p) for t, p, *_ in B])}
        per_verb = defaultdict(lambda: {"probe": [], "bow": []})
        for (t, p, v, s, i), (_, b, *_r) in zip(P, B):
            per_verb[v]["probe"].append((t, p))
            per_verb[v]["bow"].append((t, b))
        mixed = {v: x for v, x in per_verb.items() if len({t for t, _ in x["probe"]}) == 2}
        d["within_verb"] = {v: {"probe": metrics(x["probe"]), "bow": metrics(x["bow"])} for v, x in sorted(mixed.items())}
        n = sum(x["probe"]["n"] for x in d["within_verb"].values())
        for k in ("probe", "bow"):
            d[f"{k}_within_verb_weighted"] = {
                m: float(sum(x[k][m] * x[k]["n"] for x in d["within_verb"].values()) / n) if n else float("nan")
                for m in ("balanced_accuracy", "auroc")}
        d["n_items_in_mixed_verbs"] = n
        d["verbs_single_class"] = sorted(set(per_verb) - set(mixed))
        res[name] = d
    try:
        res["extra_controls"] = json.load(open(f"results/{a.tag}/extra_controls.json"))
    except FileNotFoundError:
        pass
    json.dump(res, open(f"results/{a.tag}/analysis_extra.json", "w"), indent=1)
    for name in ("all_items", "excluding_relabelled"):
        d = res[name]
        print(f"[{a.tag} | {name}] overall probe {d['probe_overall']} | bow {d['bow_overall']}")
        print(f"   within-verb (weighted, n={d['n_items_in_mixed_verbs']}): probe {d['probe_within_verb_weighted']} | bow {d['bow_within_verb_weighted']}")
        for v, x in d["within_verb"].items():
            print(f"     {v:12s} n={x['probe']['n']:4d} irr={x['probe']['n_irr']:3d} probe AUROC {x['probe'].get('auroc', float('nan')):.3f} | bow AUROC {x['bow'].get('auroc', float('nan')):.3f}")
    if "extra_controls" in res:
        h = res["extra_controls"]["harm_direction"]
        print(f"   harm direction: cos with reversibility {h['cos_harm_vs_reversibility_direction']:.3f}, "
              f"AUROC of harm direction on irr vs rev items {h.get('auroc_harm_direction_on_irreversible_vs_reversible_items', float('nan')):.3f}, "
              f"AUROC of reversibility direction on harmful vs harmless {h['auroc_reversibility_direction_on_harmful_vs_harmless']:.3f}")
        for d, x in res["extra_controls"]["held_out_domain"].items():
            print(f"   held-out {d}: bal acc {x['test']['balanced_accuracy']:.3f} AUROC {x['test']['auroc']:.3f} pairs "
                  f"{ {t: round(v['within_pair_accuracy'], 3) for t, v in x['pairs'].items()} } BoW {x['bow_baselines_same_split']}")


if __name__ == "__main__":
    main()
