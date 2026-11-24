"""Inter-annotator agreement for the blind annotation sample (brief, Section 4.2).

Reads the two returned sheets (column B = item_id, column F = label), and reports Cohen's kappa
annotator A vs annotator B, and each annotator vs the automatic (search) label, overall and per
environment, for the 3 labels and for the binary collapse (reversible vs not). 95% CIs by bootstrap
over items (1,000 resamples). Blank or unrecognised labels are listed and excluded pairwise.
Usage: python tools/annotation_agreement.py annotation_sheet_A.xlsx annotation_sheet_B.xlsx [--out results/annotation_agreement.json]
"""
import argparse
import json
from collections import Counter

import numpy as np
from openpyxl import load_workbook

LABELS = ["reversible", "partial", "irreversible"]
ALIASES = {"r": "reversible", "rev": "reversible", "p": "partial", "partially reversible": "partial",
           "i": "irreversible", "irr": "irreversible", "not reversible": "irreversible"}


def read_sheet(path):
    ws = load_workbook(path, data_only=True)["Label"]
    out, bad = {}, []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[1]:
            continue
        iid, lab = str(row[1]).strip(), (str(row[5]).strip().lower() if row[5] is not None else "")
        lab = ALIASES.get(lab, lab)
        if lab in LABELS:
            out[iid] = lab
        else:
            bad.append((iid, lab))
    return out, bad


def kappa(a, b, labels):
    n = len(a)
    if n == 0:
        return float("nan")
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def kappa_ci(a, b, labels, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    reps = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        k = kappa([a[j] for j in i], [b[j] for j in i], labels)
        if not np.isnan(k):
            reps.append(k)
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))] if reps else [float("nan")] * 2


def compare(x, y, ids, binary=False):
    common = [i for i in ids if i in x and i in y]
    a = [x[i] for i in common]
    b = [y[i] for i in common]
    labels = LABELS
    if binary:
        a = ["reversible" if v == "reversible" else "not" for v in a]
        b = ["reversible" if v == "reversible" else "not" for v in b]
        labels = ["reversible", "not"]
    return {"n": len(common), "agreement": float(np.mean([p == q for p, q in zip(a, b)])) if common else float("nan"),
            "kappa": kappa(a, b, labels), "ci95": kappa_ci(a, b, labels),
            "confusion": {f"{p}->{q}": c for (p, q), c in sorted(Counter(zip(a, b)).items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet_a")
    ap.add_argument("sheet_b")
    ap.add_argument("--out", default="results/annotation_agreement.json")
    a = ap.parse_args()
    ids = json.load(open("data/annotation_sample_ids.json"))["ids"]
    auto, env = {}, {}
    for path in ("data/sandbox.jsonl", "data/probe_scienceworld.jsonl"):
        for l in open(path):
            r = json.loads(l)
            auto[r["id"]] = r["label"]
            env[r["id"]] = r["environment"]
    A, badA = read_sheet(a.sheet_a)
    B, badB = read_sheet(a.sheet_b)
    res = {"n_sample": len(ids), "labelled_A": len(A), "labelled_B": len(B),
           "unusable_A": badA, "unusable_B": badB, "missing_auto": [i for i in ids if i not in auto]}
    for scope in ("all", "sandbox", "scienceworld"):
        sids = [i for i in ids if scope == "all" or env.get(i) == scope]
        res[scope] = {}
        for name, x, y in (("A_vs_B", A, B), ("A_vs_auto", A, auto), ("B_vs_auto", B, auto)):
            res[scope][name] = compare(x, y, sids)
            res[scope][name + "_binary"] = compare(x, y, sids, binary=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for scope in ("all", "sandbox", "scienceworld"):
        r = res[scope]
        print(f"[{scope}] " + " | ".join(f"{k}: kappa {v['kappa']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] n={v['n']}"
                                         for k, v in r.items() if not k.endswith("binary")))
    if badA or badB:
        print("unusable labels:", {"A": badA, "B": badB})
    print("saved", a.out)


if __name__ == "__main__":
    main()
