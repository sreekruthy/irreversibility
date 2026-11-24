"""RQ1: linear probes per layer with all preregistered baselines, controls and ablations.
Runs on CPU next to the hidden states (Kaggle) and writes small result files.

Inputs: one or more (hidden_dir, data_file) pairs produced by extract.py.
Outputs (in --out):
  probes.json          every metric, per layer, per seed, with test n and 95% CIs
  probe_preds.jsonl    seed-0 test predictions at the best layer (for kappa with stated answers)
  steer_dirs.npz       steering directions at the best layer (brief's version + binding-only) and the
                       mean residual norm there, for RQ4

Design (PREREGISTRATION.md):
  * L2 logistic regression on standardised features; C from {0.001, 0.01, 0.1, 1} by 5-fold
    grouped CV (groups = scenario) on the training split; binary (rev vs irr) and 3-class.
  * 3 scenario splits (20% of scenarios held out). Per-item metrics on unique (state, action)
    items; within-pair metrics on all pairs.
  * Baselines: majority; bag-of-words on action text; bag-of-words on the full prompt.
    Controls: shuffled-label probe; Hewitt-Liang control task (each verb gets a fixed random
    label with the true class proportions), selectivity = probe - control; layer 0 is in the sweep.
  * Best layer L* = highest grouped-CV balanced accuracy (binary, sandbox, seed 0, training split).
  * Ablations: token position (last vs mean), minimal pairs only, cross-environment transfer,
    horizon 1/2/3, binary vs 3-class.

Usage:
  python -m irrev.probes --out results/qwen7b/probes \
     --set sandbox:results/qwen7b/hidden_sandbox:data/probe_sandbox.jsonl \
     --set sandbox:results/qwen7b/hidden_extra:data/probe_sandbox_extra.jsonl \
     --set scienceworld:results/qwen7b/hidden_sw:data/probe_scienceworld.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ORD = {"reversible": 0, "partial": 1, "irreversible": 2}
CS = (0.001, 0.01, 0.1, 1.0)
SEEDS = (0, 1, 2)
RNG = np.random.default_rng(0)


# ---------------- data ----------------
def load_set(hidden_dir, data_file):
    items = {}
    for l in open(data_file):
        x = json.loads(l)
        items[x["id"]] = x
    last, mean, ids = [], [], []
    for f in sorted(glob.glob(os.path.join(hidden_dir, "shard_*.npz"))):
        z = np.load(f)
        last.append(z["last"]); mean.append(z["mean"]); ids += [str(i) for i in z["ids"]]
    if not ids:
        raise SystemExit(f"no shards in {hidden_dir}")
    rows = [items[i] for i in ids]
    return rows, np.concatenate(last), np.concatenate(mean)


def verb(x):
    a = json.loads(x["action_text"])
    if "tool" in a:
        return a["tool"]
    s = a["action"]
    for v in ("pick up", "put down", "go to"):
        if s.startswith(v):
            return v
    return s.split()[0]


def key(x):
    return (x["state_text"], x["action_text"])


def split(rows, seed, frac=0.2):
    sc = sorted({r["scenario_id"] for r in rows})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(sc, size=max(1, round(frac * len(sc))), replace=False))
    tr = np.array([r["scenario_id"] not in test for r in rows])
    return tr, ~tr


def unique_mask(rows, mask):
    seen, out = set(), np.zeros(len(rows), bool)
    for i in np.where(mask)[0]:
        k = key(rows[i])
        if k not in seen:
            seen.add(k); out[i] = True
    return out


def labels(rows, task, horizon="3"):
    y = np.array([ORD[r["labels_by_horizon"][horizon] if horizon != "3" else r["label"]] for r in rows])
    if task == "binary":
        keep = y != 1
        return (y == 2).astype(int), keep
    return y, np.ones(len(rows), bool)


# ---------------- metrics ----------------
def score_of(P, task):
    return P[:, 1] if task == "binary" else P[:, 1] + 2 * P[:, 2]


def pair_acc(rows, idx, scores, task, tier):
    """Within-pair accuracy over all pairs of a tier inside idx (chance 0.5)."""
    by = {}
    for i, s in zip(idx, scores):
        r = rows[i]
        if r["tier"] == tier and r.get("pair_id"):
            by.setdefault(r["pair_id"], []).append((ORD[r["label"]], s))
    res = []
    for m in by.values():
        if len(m) == 2 and m[0][0] != m[1][0]:
            (la, sa), (lb, sb) = m
            res.append(0.5 if sa == sb else float((sa > sb) == (la > lb)))
    return res


def ci(values, groups=None, b=1000):
    v = np.asarray(values, float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if groups is None:
        reps = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(b)]
    else:
        g = np.asarray(groups); ug = np.unique(g); mem = {u: np.where(g == u)[0] for u in ug}
        reps = [v[np.concatenate([mem[u] for u in RNG.choice(ug, len(ug))])].mean() for _ in range(b)]
    return (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)))


def bal_acc_ci(y, p, groups, b=1000):
    g = np.asarray(groups); ug = np.unique(g); mem = {u: np.where(g == u)[0] for u in ug}
    reps = []
    for _ in range(b):
        i = np.concatenate([mem[u] for u in RNG.choice(ug, len(ug))])
        if len(np.unique(y[i])) > 1:
            reps.append(balanced_accuracy_score(y[i], p[i]))
    return (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))) if reps else (float("nan"),) * 2


# ---------------- one probe ----------------
class RowSpace:
    """Standardise, then express each vector in an orthonormal basis Q of the span of the training rows.
    Exact for L2 logistic regression: the optimal weights lie in that span (representer theorem), so
    w = Q u with ||w|| = ||u|| and w.x = u.(Q^T x) for every x. Same optimum and same predictions,
    but each fit works in n_train instead of d dimensions (n_train < d here)."""

    def __init__(self, Xtr):
        self.sc = StandardScaler().fit(Xtr)
        Z = self.sc.transform(Xtr)
        self.Q = None
        if Z.shape[0] < Z.shape[1]:
            _, sv, Vt = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
            keep = sv > sv[0] * 1e-7
            if not keep.any():          # all training vectors identical at this layer (e.g. layer 0): no information
                keep[0] = True
            self.Q = Vt[keep].T.astype(np.float32)

    def transform(self, X):
        Z = self.sc.transform(X)
        return Z if self.Q is None else Z @ self.Q


def make_lr(C, task="binary"):
    """L2 logistic regression (same objective as the preregistration), fitted in the training-row span.
    Binary: Newton-Cholesky, a few steps whatever the conditioning. 3-class: lbfgs (the multinomial
    Newton Hessian is 3x larger per side and slower here). lbfgs throughout if Newton is unavailable."""
    if task == "binary" and SOLVER == "newton-cholesky":
        return LogisticRegression(C=C, solver="newton-cholesky", max_iter=100, tol=1e-4)
    return LogisticRegression(C=C, max_iter=2000, tol=1e-4)


def _pick_solver():
    try:
        LogisticRegression(solver="newton-cholesky", max_iter=5).fit(np.eye(6), [0, 1, 2, 0, 1, 2])
        return "newton-cholesky"
    except Exception:
        return "lbfgs"


SOLVER = _pick_solver()


def fit_eval(Xtr, ytr, gtr, Xte, yte, task, C=None):
    sc = RowSpace(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    cv = None
    if C is None:
        nf = min(5, len(np.unique(gtr)))
        cv = {}
        for c in CS:
            accs = []
            for a, b in GroupKFold(nf).split(Xtr, ytr, gtr):
                if len(np.unique(ytr[a])) < len(np.unique(ytr)):
                    continue
                m = make_lr(c, task).fit(Xtr[a], ytr[a])
                accs.append(balanced_accuracy_score(ytr[b], m.predict(Xtr[b])))
            cv[c] = float(np.mean(accs)) if accs else -1
        C = max(cv, key=cv.get)
    m = make_lr(C, task).fit(Xtr, ytr)
    P = m.predict_proba(Xte) if len(Xte) else np.zeros((0, len(m.classes_)))
    # ensure columns for all classes
    k = 2 if task == "binary" else 3
    full = np.zeros((len(Xte), k))
    for j, cl in enumerate(m.classes_):
        full[:, cl] = P[:, j]
    return full, C, cv, m, sc


def item_metrics(y, P, task, groups):
    pred = P.argmax(1)
    out = {"n": int(len(y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")), "ci95_bal_acc": bal_acc_ci(y, pred, groups)}
    try:
        out["auroc"] = float(roc_auc_score(y, P[:, 1]) if task == "binary" else roc_auc_score(y, P, multi_class="ovr"))
    except ValueError:
        out["auroc"] = float("nan")
    return out


def run_layer(L, X, rows, task, tr, te, te_u, tr_u, verbs_ctrl, groups, horizon="3", C=None, train_filter=None):
    """C=None: choose C by grouped CV (main sweeps). Ablations pass the C chosen in the main
    sweep for the same layer and split (selected on training data only)."""
    y, keep = labels(rows, task, horizon)
    trm = tr_u & keep if train_filter is None else tr_u & keep & train_filter
    tem = te_u & keep
    if trm.sum() < 10 or len(np.unique(y[trm])) < 2:
        return None
    Xl = X[:, L, :].astype(np.float32)
    P, Cs, cv, m, sc = fit_eval(Xl[trm], y[trm], groups[trm], Xl[tem], y[tem], task, C)
    res = {"layer": L, "C": Cs, "cv": cv, "test": item_metrics(y[tem], P, task, groups[tem])}
    # within-pair on ALL test pairs (duplicates are distinct pairs)
    tall = te & keep
    Pall = m.predict_proba(sc.transform(Xl[tall]))
    full = np.zeros((tall.sum(), P.shape[1]))
    for j, cl in enumerate(m.classes_):
        full[:, cl] = Pall[:, j]
    s_all = score_of(full, task)
    idx = np.where(tall)[0]
    res["pairs"] = {}
    for tier in ("tier1", "tier2", "tier3"):
        pa = pair_acc(rows, idx, s_all, task, tier)
        if pa:
            res["pairs"][tier] = {"n_pairs": len(pa), "within_pair_accuracy": float(np.mean(pa)), "ci95": ci(pa)}
    # controls (same C): shuffled labels; Hewitt-Liang control task
    ysh = y[trm].copy(); np.random.default_rng(L).shuffle(ysh)
    Psh, _, _, _, _ = fit_eval(Xl[trm], ysh, groups[trm], Xl[tem], y[tem], task, Cs)
    res["shuffled"] = {"balanced_accuracy": float(balanced_accuracy_score(y[tem], Psh.argmax(1)))}
    yc = verbs_ctrl[task]
    Pc, _, _, _, _ = fit_eval(Xl[trm], yc[trm], groups[trm], Xl[tem], yc[tem], task, Cs)
    ctrl = float(balanced_accuracy_score(yc[tem], Pc.argmax(1))) if len(np.unique(yc[tem])) > 1 else float("nan")
    res["control_task"] = {"balanced_accuracy": ctrl, "selectivity": res["test"]["balanced_accuracy"] - ctrl}
    res["cv_best"] = max(cv.values()) if cv else None
    return res


def control_labels(rows, task, seed=0):
    """Hewitt-Liang: each verb gets one random fixed label, sampled with the true class proportions."""
    rng = np.random.default_rng(seed)
    y, keep = labels(rows, task)
    vals, counts = np.unique(y[keep], return_counts=True)
    p = counts / counts.sum()
    vmap = {v: int(rng.choice(vals, p=p)) for v in sorted({verb(r) for r in rows})}
    return np.array([vmap[verb(r)] for r in rows])


def bow_baselines(rows, task, tr_u, te_u, te, groups):
    y, keep = labels(rows, task)
    out = {}
    trm, tem = tr_u & keep, te_u & keep
    maj = np.bincount(y[trm]).argmax()
    out["majority"] = {"balanced_accuracy": float(balanced_accuracy_score(y[tem], np.full(tem.sum(), maj))),
                       "accuracy": float(np.mean(y[tem] == maj))}
    views = {"bow_action": lambda r: r["action_text"],
             "bow_full_prompt": lambda r: r["task_text"] + "\n" + r["state_text"] + "\n" + r["action_text"]}
    for name, fn in views.items():
        vec = CountVectorizer(token_pattern=r"[A-Za-z0-9_@.\-/]+", ngram_range=(1, 2))
        Xtr = vec.fit_transform([fn(rows[i]) for i in np.where(trm)[0]])
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, y[trm])
        Xte = vec.transform([fn(rows[i]) for i in np.where(tem)[0]])
        P = np.zeros((tem.sum(), 2 if task == "binary" else 3))
        for j, cl in enumerate(m.classes_):
            P[:, cl] = m.predict_proba(Xte)[:, j]
        r = {"balanced_accuracy": float(balanced_accuracy_score(y[tem], P.argmax(1)))}
        tall = np.where(te & keep)[0]          # all test items incl. duplicates (each pair counts)
        Pa = np.zeros((len(tall), P.shape[1]))
        if len(tall):
            pr = m.predict_proba(vec.transform([fn(rows[i]) for i in tall]))
            for j, cl in enumerate(m.classes_):
                Pa[:, cl] = pr[:, j]
        r["pairs"] = {t: float(np.mean(pa)) for t in ("tier1", "tier2", "tier3")
                      if (pa := pair_acc(rows, tall, score_of(Pa, task), task, t))}
        out[name] = r
    return out


def sweep(X, rows, task, seeds, n_jobs, horizon="3", train_filter=None, layers=None, C_from=None):
    groups = np.array([r["scenario_id"] for r in rows])
    ctrl = {t: control_labels(rows, t) for t in ("binary", "three_class")}
    out = []
    layers = range(X.shape[1]) if layers is None else layers
    for s in seeds:
        tr, te = split(rows, s)
        tr_u, te_u = unique_mask(rows, tr), unique_mask(rows, te)
        cmap = {}
        if C_from is not None:
            ref = next(r for r in C_from if r["seed"] == s)
            cmap = {r["layer"]: r["C"] for r in ref["layers"]}
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_layer)(L, X, rows, task, tr, te, te_u, tr_u, ctrl, groups, horizon, cmap.get(L), train_filter)
            for L in layers)
        base = bow_baselines(rows, task, tr_u, te_u, te, groups) if train_filter is None and horizon == "3" else None
        out.append({"seed": s, "layers": [r for r in res if r], "baselines": base,
                    "test_scenarios": sorted({rows[i]["scenario_id"] for i in np.where(te)[0]})})
    return out


def cached(out, name, fn):
    """Each sweep is saved as soon as it finishes; a restarted run reloads finished sweeps instead of refitting."""
    p = os.path.join(out, "partial", name + ".json")
    if os.path.exists(p):
        print(f"  reusing saved {name}", flush=True)
        return json.load(open(p))
    r = fn()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(r, open(p + ".tmp", "w"), default=float)
    os.replace(p + ".tmp", p)
    return r


def summarise(runs, metric=("test", "balanced_accuracy")):
    by = {}
    for run in runs:
        for r in run["layers"]:
            v = r
            for k in metric:
                v = v.get(k, {}) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                by.setdefault(r["layer"], []).append(v)
    return {L: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n_seeds": len(v)} for L, v in sorted(by.items())}


def transfer(Xa, ra, Xb, rb, n_jobs):
    """Train on all of env A (binary), test on all of env B, per layer."""
    ya, ka = labels(ra, "binary"); yb, kb = labels(rb, "binary")
    ua, ub = unique_mask(ra, np.ones(len(ra), bool)) & ka, unique_mask(rb, np.ones(len(rb), bool)) & kb
    ga = np.array([r["scenario_id"] for r in ra]); gb = np.array([r["scenario_id"] for r in rb])

    def one(L):
        P, C, _, _, _ = fit_eval(Xa[ua, L].astype(np.float32), ya[ua], ga[ua], Xb[ub, L].astype(np.float32), yb[ub], "binary", 0.01)
        return {"layer": L, **item_metrics(yb[ub], P, "binary", gb[ub])}
    return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(one)(L) for L in range(Xa.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True, help="env:hidden_dir:data_file")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 seed, few layers")
    ap.add_argument("--dirs-only", action="store_true",
                    help="only the seed-0 binary sandbox sweep -> L* and steer_dirs.npz (same computation as the full run), so RQ4 can start early")
    ap.add_argument("--horizon-all-seeds", action="store_true",
                    help="ablation 4 on all three splits (the original run used split 0); finished sweeps are reused")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    envs = {}
    for s in a.set:
        env, hd, df = s.split(":")
        rows, last, mean = load_set(hd, df)
        if env in envs:
            r0, l0, m0 = envs[env]
            envs[env] = (r0 + rows, np.concatenate([l0, last]), np.concatenate([m0, mean]))
        else:
            envs[env] = (rows, last, mean)
        print(f"loaded {env}: +{len(rows)} items from {hd}", flush=True)
    seeds = (0,) if a.quick else SEEDS
    res = {"design": "see PREREGISTRATION.md", "solver": SOLVER, "seeds": list(seeds), "Cs": list(CS), "envs": {}}
    nL = next(iter(envs.values()))[1].shape[1]
    layers = sorted({0, 1, nL // 2, nL - 1}) if a.quick else None
    order = sorted(envs, key=lambda k: k != "sandbox")
    for env in order:
        rows, last, mean = envs[env]
        e = {"n_items": len(rows), "n_unique": len({key(r) for r in rows}),
             "label_counts": {k: sum(r["label"] == k for r in rows) for k in ORD}}
        print(f"[{env}] binary/last ...", flush=True)
        e["binary_last"] = cached(a.out, f"{env}_binary_last" + ("_seed0" if a.dirs_only else ""), lambda: sweep(last, rows, "binary", (0,) if a.dirs_only else seeds, a.n_jobs, layers=layers))
        if a.dirs_only:
            res["envs"][env] = e
            break
        print(f"[{env}] 3-class/last ... ({time.time()-t0:.0f}s)", flush=True)
        e["three_class_last"] = cached(a.out, f"{env}_three_class_last", lambda: sweep(last, rows, "three_class", seeds, a.n_jobs, layers=layers))
        print(f"[{env}] binary/mean (ablation 1) ... ({time.time()-t0:.0f}s)", flush=True)
        e["binary_mean"] = cached(a.out, f"{env}_binary_mean", lambda: sweep(mean, rows, "binary", seeds, a.n_jobs, layers=layers))   # own CV: different features
        print(f"[{env}] minimal pairs only (ablation 2) ... ({time.time()-t0:.0f}s)", flush=True)
        mp = np.array([r["tier"] != "single" for r in rows])
        if mp.any() and (~mp).any():
            e["binary_last_minimal_pairs_only"] = cached(a.out, f"{env}_binary_last_minimal_pairs_only", lambda: sweep(
                last, rows, "binary", seeds, a.n_jobs, train_filter=mp, layers=layers, C_from=e["binary_last"]))
        for h in ("1", "2"):
            print(f"[{env}] horizon {h} (ablation 4) ... ({time.time()-t0:.0f}s)", flush=True)
            hs = seeds if a.horizon_all_seeds else (0,)          # the original run used split 0 only
            e[f"binary_last_h{h}"] = cached(a.out, f"{env}_binary_last_h{h}" + ("_3seeds" if a.horizon_all_seeds else ""),
                                            lambda: sweep(last, rows, "binary", hs, a.n_jobs, horizon=h, layers=layers,
                                                          C_from=e["binary_last"]))
        e["summary"] = {k: summarise(e[k]) for k in ("binary_last", "three_class_last", "binary_mean")}
        e["summary_tier3_pairs"] = summarise(e["binary_last"], ("pairs", "tier3", "within_pair_accuracy"))
        e["summary_selectivity"] = summarise(e["binary_last"], ("control_task", "selectivity"))
        res["envs"][env] = e
    # best layer: grouped-CV balanced accuracy, binary, sandbox (or first env), seed 0, training split only
    main_env = "sandbox" if "sandbox" in envs else next(iter(envs))
    cvb = {r["layer"]: r["cv_best"] for r in res["envs"][main_env]["binary_last"][0]["layers"]}
    Lstar = max(cvb, key=cvb.get)
    res["best_layer"] = {"layer": Lstar, "env": main_env, "rule": "max grouped-CV balanced accuracy, binary, seed-0 training split", "cv": cvb}
    print(f"best layer L* = {Lstar}", flush=True)
    # cross-environment transfer (ablation 3)
    if "sandbox" in envs and "scienceworld" in envs and not a.dirs_only:
        print(f"transfer (ablation 3) ... ({time.time()-t0:.0f}s)", flush=True)
        (rs, ls, _), (rw, lw, _) = envs["sandbox"], envs["scienceworld"]
        res["transfer_sandbox_to_sw"] = cached(a.out, "transfer_sandbox_to_sw", lambda: transfer(ls, rs, lw, rw, a.n_jobs))
        res["transfer_sw_to_sandbox"] = cached(a.out, "transfer_sw_to_sandbox", lambda: transfer(lw, rw, ls, rs, a.n_jobs))
    # seed-0 test predictions at L* (3-class and binary) for kappa with stated answers
    rows, last, _ = envs[main_env]
    tr, te = split(rows, 0)
    tr_u = unique_mask(rows, tr)
    preds = {}
    for task in (() if a.dirs_only else ("three_class", "binary")):
        y, keep = labels(rows, task)
        trm = tr_u & keep
        P, C, _, m, sc = fit_eval(last[trm, Lstar].astype(np.float32), y[trm], np.array([r["scenario_id"] for r in rows])[trm],
                                  last[te, Lstar].astype(np.float32), y[te], task)
        preds[task] = (P, C)
    with open(os.path.join(a.out, "probe_preds.jsonl" if not a.dirs_only else os.devnull), "w") as f:
        inv = {v: k for k, v in ORD.items()}
        for j, i in enumerate(np.where(te)[0] if preds else []):
            P3, Pb = preds["three_class"][0][j], preds["binary"][0][j]
            f.write(json.dumps({"id": rows[i]["id"], "label": rows[i]["label"], "layer": Lstar,
                                "probe3": inv[int(P3.argmax())], "probe3_probs": P3.tolist(),
                                "probe_bin_p_irr": float(Pb[1])}) + "\n")
    # steering directions at L* (last token, seed-0 training split)
    ytr = np.array([ORD[r["label"]] for r in rows])
    trm = tr_u
    H = last[:, Lstar].astype(np.float32)
    d_all = H[trm & (ytr == 2)].mean(0) - H[trm & (ytr == 0)].mean(0)
    bt = np.array([r["tier"] == "tier3" for r in rows])
    d_bind = H[trm & bt & (ytr == 2)].mean(0) - H[trm & bt & (ytr == 0)].mean(0) if (trm & bt & (ytr == 2)).any() else d_all * 0
    np.savez(os.path.join(a.out, "steer_dirs.npz"), direction=d_all, direction_binding=d_bind,
             mean_norm=np.float32(np.linalg.norm(H[trm], axis=1).mean()), layer=np.int32(Lstar),
             cos_all_binding=np.float32(d_all @ d_bind / (np.linalg.norm(d_all) * np.linalg.norm(d_bind) + 1e-9)))
    res["steering_direction"] = {"layer": Lstar, "n_irr": int((trm & (ytr == 2)).sum()), "n_rev": int((trm & (ytr == 0)).sum()),
                                 "norm_all": float(np.linalg.norm(d_all)), "norm_binding": float(np.linalg.norm(d_bind)),
                                 "mean_resid_norm": float(np.linalg.norm(H[trm], axis=1).mean())}
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out, "probes_dirs_only.json" if a.dirs_only else "probes.json"), "w"), default=float)
    print(f"PROBES DONE in {time.time()-t0:.0f}s | best layer {Lstar}", flush=True)


if __name__ == "__main__":
    main()
