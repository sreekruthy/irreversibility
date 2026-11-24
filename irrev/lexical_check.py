"""Lexical baselines on the dataset, split by scenario (context).

Reports balanced accuracy of bag-of-words logistic regression on
  (a) action text only  (the brief's lexical baseline)
  (b) task + state + action (our stricter full-prompt baseline)
per tier, averaged over 3 scenario splits. If (b) is well above chance on tier3,
the binding pairs leak lexical information and must be fixed before probing.
"""
import json
import sys

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ORD = {"reversible": 0, "partial": 1, "irreversible": 2}


def pair_accuracy(items, scores):
    """Within each minimal pair: does the less reversible member get the higher
    irreversibility score? Ties count 0.5. Chance = 0.5."""
    by = {}
    for it, sc in zip(items, scores):
        if it.get("pair_id"):
            by.setdefault(it["pair_id"], []).append((ORD[it["label"]], sc))
    res = []
    for mem in by.values():
        if len(mem) != 2 or mem[0][0] == mem[1][0]:
            continue
        (la, sa), (lb, sb) = mem
        res.append(0.5 if sa == sb else float((sa > sb) == (la > lb)))
    return float(np.mean(res)) if res else float("nan"), len(res)


def split(items, seed, frac=0.2):
    sc = sorted({i["scenario_id"] for i in items})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(sc, size=max(1, round(frac * len(sc))), replace=False))
    return [i for i in items if i["scenario_id"] not in test], [i for i in items if i["scenario_id"] in test]


def run(path, binary=False):
    items = [json.loads(l) for l in open(path)]
    if binary:
        items = [i for i in items if i["label"] != "partial"]
    views = {"action_only": lambda i: i["action_text"],
             "full_prompt": lambda i: i["task_text"] + "\n" + i["state_text"] + "\n" + i["action_text"]}
    tiers = ["all", "tier1", "tier2", "tier3", "single"]
    out = {v: {t: [] for t in tiers} for v in views}
    pairs = {v: {t: [] for t in ("tier1", "tier2", "tier3")} for v in views}
    for seed in range(3):
        tr, te = split(items, seed)
        for v, fn in views.items():
            vec = CountVectorizer(token_pattern=r"[A-Za-z0-9_@.\-/]+", ngram_range=(1, 2), min_df=1)
            Xtr = vec.fit_transform([fn(i) for i in tr])
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xtr, [i["label"] for i in tr])
            classes = list(clf.classes_)
            P = clf.predict_proba(vec.transform([fn(i) for i in te]))
            score = P @ np.array([ORD[c] for c in classes], dtype=float)
            for t in pairs[v]:
                idx = [j for j, i in enumerate(te) if i["tier"] == t]
                pairs[v][t].append(pair_accuracy([te[j] for j in idx], score[idx])[0])
            for t in tiers:
                sub = te if t == "all" else [i for i in te if i["tier"] == t]
                if len({i["label"] for i in sub}) < 2:
                    continue
                pred = clf.predict(vec.transform([fn(i) for i in sub]))
                out[v][t].append(balanced_accuracy_score([i["label"] for i in sub], pred))
    k = len({i["label"] for i in items})
    print(f"{'binary' if binary else '3-class'} | chance = {1/k:.3f} | n = {len(items)}")
    for v in views:
        print("  " + v.ljust(12), "  ".join(f"{t}: {np.mean(s):.3f}±{np.std(s):.3f}" for t, s in out[v].items() if s))
        print("  " + "  pair-acc".ljust(12), "  ".join(f"{t}: {np.nanmean(s):.3f}" for t, s in pairs[v].items()))


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "data/sandbox.jsonl"
    run(p, binary=True)
    run(p, binary=False)
