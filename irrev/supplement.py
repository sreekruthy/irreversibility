"""Supplementary analyses and the LaTeX tables of the report (CPU, seconds).

Computes from the saved outputs in results/ (no model needed):
  * the exploratory probability-based K of the blindness gap (irreversible version's shortcut gets the
    higher option score P(partial) + 2 P(irreversible)), with paired bootstrap CIs, and its gap K - A;
  * caution counted as ask_user or backup_file only (avoids the F1 design asymmetry);
  * preregistered secondary measures not shown in the main tables: broad caution, the 2x2 paired tables,
    parse failures, the blindness gap under the careful prompt, parse failures per steering strength;
  * the power of the exact McNemar test with 60 pairs (simulation, independent outcomes);
  * ScienceWorld search-budget counts and duplicate items in the sandbox data;
  * the annotation agreement (if results/annotation_agreement.json exists).
Writes results/supplement.json and paper/tables/*.tex (included by paper/main.tex).

Usage: python -m irrev.supplement
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

import numpy as np

from .analyze_behaviour import ORD, episode_measures, mcnemar_exact

MODELS = (("qwen7b", "Qwen-7B"), ("llama8b", "Llama-8B"))
B = 1000


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def boot(x, b=B, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    reps = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(b)]
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def boot_diff(a, c, b=B, seed=0):
    rng = np.random.default_rng(seed)
    a, c = np.asarray(a, float), np.asarray(c, float)
    reps = []
    for _ in range(b):
        i = rng.integers(0, len(a), len(a))
        reps.append(a[i].mean() - c[i].mean())
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def pairs_of(episodes):
    by = {}
    for r in episodes:
        by.setdefault(r["pair_id"], {})[r["version"]] = (r, episode_measures(r))
    return {k: v for k, v in sorted(by.items()) if len(v) == 2}


def score(r):
    p = r["option_probs"]
    return p["partial"] + 2 * p["irreversible"]


def gap_block(stated_tasks, pairs, key="strict"):
    st = {r["id"].replace("-shortcut", ""): r for r in stated_tasks}
    K, Kp, A = [], [], []
    for pid, v in pairs.items():
        si, sr = st[f"{pid}-irreversible"], st[f"{pid}-reversible"]
        K.append(float(ORD[si["stated_option"]] > ORD[sr["stated_option"]]))
        Kp.append(float(score(si) > score(sr)))
        A.append(float(v["irreversible"][1][key] and not v["reversible"][1][key]))
    out = {"n_pairs": len(K), "K_argmax": float(np.mean(K)), "K_argmax_ci95": boot(K),
           "K_prob": float(np.mean(Kp)), "K_prob_ci95": boot(Kp), "A": float(np.mean(A)), "A_ci95": boot(A),
           "gap_argmax": float(np.mean(K) - np.mean(A)), "gap_argmax_ci95": boot_diff(K, A),
           "gap_prob": float(np.mean(Kp) - np.mean(A)), "gap_prob_ci95": boot_diff(Kp, A)}
    pi = [st[f"{p}-irreversible"]["option_probs"]["irreversible"] for p in pairs]
    pr = [st[f"{p}-reversible"]["option_probs"]["irreversible"] for p in pairs]
    out["mean_P_irr_irr_version"], out["mean_P_irr_rev_version"] = float(np.mean(pi)), float(np.mean(pr))
    return out


def behaviour_block(pairs):
    out = {"n_pairs": len(pairs)}
    for key in ("strict", "broad", "ask_or_backup"):
        get = (lambda m: m["asked"] or m["backup"]) if key == "ask_or_backup" else (lambda m, k=key: m[k])
        irr = np.array([float(get(v["irreversible"][1])) for v in pairs.values()])
        rev = np.array([float(get(v["reversible"][1])) for v in pairs.values()])
        b10, b01 = int(((irr == 1) & (rev == 0)).sum()), int(((irr == 0) & (rev == 1)).sum())
        out[key] = {"irr": float(irr.mean()), "irr_ci95": boot(irr), "rev": float(rev.mean()), "rev_ci95": boot(rev),
                    "sensitivity": float((irr - rev).mean()), "sensitivity_ci95": boot(irr - rev),
                    "table": {"irr_only": b10, "rev_only": b01, "both": int(((irr == 1) & (rev == 1)).sum()),
                              "neither": int(((irr == 0) & (rev == 0)).sum())},
                    "mcnemar_exact_p": mcnemar_exact(b01, b10)}
    for key in ("took_irreversible", "success"):
        irr = [float(v["irreversible"][1][key]) for v in pairs.values()]
        rev = [float(v["reversible"][1][key]) for v in pairs.values()]
        out[key] = {"irr": float(np.mean(irr)), "irr_ci95": boot(irr), "rev": float(np.mean(rev)), "rev_ci95": boot(rev)}
    eps = [x[0] for v in pairs.values() for x in v.values()]
    out["parse_failures"] = {"total": int(sum(r["n_parse_failures"] for r in eps)),
                             "episodes_with_any": int(sum(r["n_parse_failures"] > 0 for r in eps)),
                             "episodes_ended_by_parse_failures": int(sum(r["end_reason"] == "parse_failures" for r in eps)),
                             "steps": int(sum(r["n_steps"] for r in eps))}
    out["end_reasons"] = dict(Counter(r["end_reason"] for r in eps))
    return out


def power(n=60, bases=(0.05, 0.10), sims=4000, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for base in bases:
        for d in np.round(np.arange(0.05, 0.41, 0.01), 2):
            hits = 0
            for _ in range(sims):
                r = rng.random(n) < base
                i = rng.random(n) < base + d
                hits += mcnemar_exact(int((r & ~i).sum()), int((i & ~r).sum())) < 0.05
            if hits / sims >= 0.8:
                res[str(base)] = {"min_difference_80pct_power": float(d), "power": hits / sims}
                break
    return res


def probes_block(tag):
    p = json.load(open(f"results/{tag}/probes/probes.json"))
    L = p["best_layer"]["layer"]
    e = p["envs"]["sandbox"]

    def stats(name, layer=L, seeds=None):
        runs = e.get(name) or []
        vals, s0 = [], None
        for run in runs:
            lay = next((x for x in run["layers"] if x["layer"] == layer), None)
            if lay is None:
                continue
            vals.append(lay["test"]["balanced_accuracy"])
            if run["seed"] == 0:
                s0 = lay["test"]
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "n_splits": len(vals),
                "seed0_n": s0["n"], "seed0": s0["balanced_accuracy"], "seed0_ci95": s0["ci95_bal_acc"],
                "seed0_auroc": s0.get("auroc")}
    out = {"Lstar": L}
    for name in ("binary_last", "binary_mean", "binary_last_minimal_pairs_only", "binary_last_h1",
                 "binary_last_h2", "three_class_last"):
        out[name] = stats(name)
    out["layer0_last"] = stats("binary_last", 0)
    out["layer0_mean"] = stats("binary_mean", 0)
    for k in ("transfer_sandbox_to_sw", "transfer_sw_to_sandbox"):
        t = next(x for x in p[k] if x["layer"] == L)
        out[k] = {"n": t["n"], "balanced_accuracy": t["balanced_accuracy"], "ci95": t["ci95_bal_acc"], "auroc": t["auroc"]}
    sw = p["envs"]["scienceworld"]
    vals = []
    for run in sw["binary_last"]:
        lay = next(x for x in run["layers"] if x["layer"] == L)
        vals.append(lay["test"]["balanced_accuracy"])
        if run["seed"] == 0:
            s0 = lay["test"]
    out["sw_binary_last"] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "seed0_n": s0["n"],
                             "seed0": s0["balanced_accuracy"], "seed0_ci95": s0["ci95_bal_acc"]}
    return out


def steering_pf(tag):
    a = json.load(open(f"results/{tag}/analysis_steering.json"))
    return {k: {v: round(c[v]["parse_fail_per_step"], 3) for v in ("irreversible", "reversible")}
            for k, c in sorted(a["conditions"].items())}


def data_block():
    sb = load("data/sandbox.jsonl")
    sig = Counter((x["task_text"], x["state_text"], x["action_text"]) for x in sb)
    sw = load("data/probe_scienceworld.jsonl")
    cap = Counter((x["label"], bool(x["search_capped"])) for x in sw)
    return {"sandbox_main_items": len(sb), "sandbox_duplicate_groups": sum(c > 1 for c in sig.values()),
            "sandbox_duplicate_items": sum(c - 1 for c in sig.values() if c > 1),
            "sw_budget_capped": {f"{k[0]}{'_capped' if k[1] else ''}": v for k, v in sorted(cap.items())}}


# ---------------- LaTeX ----------------
def f(x, d=2):
    s = f"{x:.{d}f}"
    return "$-$" + s[1:] if s.startswith("-") and float(s) != 0 else s.lstrip("-")


def ci(c, d=2):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def tex_tables(S):
    os.makedirs("paper/tables", exist_ok=True)
    P = {t: S[t]["probes"] for t, _ in MODELS}
    rows = [("Main: binary, last action token", "binary_last"),
            ("(1) Mean over action tokens", "binary_mean"),
            ("(2) Train on minimal pairs only", "binary_last_minimal_pairs_only"),
            ("(4) Horizon 1 labels$^\\dagger$", "binary_last_h1"),
            ("(4) Horizon 2 labels$^\\dagger$", "binary_last_h2"),
            ("(7) Three classes", "three_class_last"),
            ("Layer 0, last token$^\\ddagger$", "layer0_last"),
            ("Layer 0, mean over action tokens", "layer0_mean")]
    L = ["\\begin{table*}[t]", "\\centering\\small", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{lcccc}", "\\toprule",
         " & \\multicolumn{2}{c}{\\qwen{} ($L^\\ast=%d$)} & \\multicolumn{2}{c}{\\llama{} ($L^\\ast=%d$)} \\\\" % (P["qwen7b"]["Lstar"], P["llama8b"]["Lstar"]),
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         "Sandbox probe at $L^\\ast$ & mean $\\pm$ s.d. & split 0 [95\\% CI] & mean $\\pm$ s.d. & split 0 [95\\% CI] \\\\", "\\midrule"]
    for name, k in rows:
        cells = []
        for t, _ in MODELS:
            s = P[t][k]
            sd = f"{f(s['mean'], 3)} $\\pm$ {f(s['sd'], 3)}" if s["n_splits"] > 1 else f"{f(s['mean'], 3)} (1 split)"
            cells += [sd, f"{f(s['seed0'], 3)} {ci(s['seed0_ci95'])}"]
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
    L.append("\\midrule")
    for name, k in (("(3) Transfer sandbox $\\rightarrow$ ScienceWorld", "transfer_sandbox_to_sw"),
                    ("(3) Transfer ScienceWorld $\\rightarrow$ sandbox", "transfer_sw_to_sandbox")):
        cells = []
        for t, _ in MODELS:
            s = P[t][k]
            cells += [f"AUROC {f(s['auroc'])}", f"{f(s['balanced_accuracy'], 3)} {ci(s['ci95'])}"]
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
    cells = []
    for t, _ in MODELS:
        b = S[t]["behaviour"]
        cells += [f"{f(b['neutral']['took_irreversible']['irr'])} $\\rightarrow$ {f(b['careful']['took_irreversible']['irr'])}",
                  f"success {f(b['neutral']['success']['irr'])} $\\rightarrow$ {f(b['careful']['success']['irr'])}"]
    L.append("(5) Careful prompt: irreversible action, irr.\\ versions & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}",
          "\\caption{Ablations. Probe rows: balanced accuracy (binary unless noted) at $L^\\ast$, mean $\\pm$ s.d.\\ over three "
          "scenario splits, and the split-0 value with its 95\\%% scenario-bootstrap CI (test $n=%d$ items per split, six held-out "
          "contexts). Transfer rows: train on all of one environment, test on all of the other ($n=%d$ and $%d$). "
          "$^\\dagger$One split only (a deviation; see Table~\\ref{tab:deviations}). $^\\ddagger$The last action token is always the "
          "same closing bracket, so this probe is at chance by construction; the mean-pooled layer-0 row is the informative "
          "embedding baseline. Ablation (5): irreversible-action rate and task success on the 60 irreversible versions, neutral "
          "$\\rightarrow$ careful system prompt.}" % (P["qwen7b"]["binary_last"]["seed0_n"], P["qwen7b"]["transfer_sandbox_to_sw"]["n"],
                                                      P["qwen7b"]["transfer_sw_to_sandbox"]["n"]),
          "\\label{tab:ablations}", "\\end{table*}"]
    open("paper/tables/ablations.tex", "w").write("\n".join(L) + "\n")

    # encoded / stated / enacted + blindness gap
    T = ["\\begin{table}[t]", "\\centering\\small", "\\setlength{\\tabcolsep}{4pt}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{lcc}", "\\toprule",
         " & \\qwen & \\llama \\\\", "\\midrule",
         "\\multicolumn{3}{l}{\\emph{Same 312 sandbox test items (split 0, 3 classes)}} \\\\"]
    ag = {t: json.load(open(f"results/{t}/analysis_probes.json"))["agreement_sandbox_seed0_test"] for t, _ in MODELS}
    T.append("Encoded: probe at $L^\\ast$, acc.\\ ($\\kappa$) & " + " & ".join(
        f"{f(ag[t]['probe_accuracy'])} ({f(ag[t]['kappa_probe_vs_true'])})" for t, _ in MODELS) + " \\\\")
    T.append("Stated: A/B/C answer, acc.\\ ($\\kappa$) & " + " & ".join(
        f"{f(ag[t]['stated_accuracy'])} ({f(ag[t]['kappa_stated_vs_true'])})" for t, _ in MODELS) + " \\\\")
    T.append("$\\kappa$ probe vs.\\ stated & " + " & ".join(f(ag[t]["kappa_probe_vs_stated"]) for t, _ in MODELS) + " \\\\")
    T.append("Probe right, stated wrong / reverse & " + " & ".join(
        f"{ag[t]['three_way']['probe right, stated wrong']} / {ag[t]['three_way']['probe wrong, stated right']}" for t, _ in MODELS) + " \\\\")
    T.append("\\midrule\\multicolumn{3}{l}{\\emph{60 test task pairs (95\\% paired bootstrap CI)}} \\\\")
    T.append("Enacted: sensitivity (strict) & " + " & ".join(
        f"{f(S[t]['behaviour']['neutral']['strict']['sensitivity'])} {ci(S[t]['behaviour']['neutral']['strict']['sensitivity_ci95'])}" for t, _ in MODELS) + " \\\\")
    for lab, k, c in (("$K$ (argmax, prereg.)", "K_argmax", "K_argmax_ci95"), ("$K$ (probabilities, expl.)", "K_prob", "K_prob_ci95"),
                      ("$A$", "A", "A_ci95"), ("Gap $K-A$ (prereg.)", "gap_argmax", "gap_argmax_ci95"),
                      ("Gap $K-A$ (probabilities)", "gap_prob", "gap_prob_ci95")):
        T.append(f"{lab} & " + " & ".join(f"{f(S[t]['gap_neutral'][k])} {ci(S[t]['gap_neutral'][c])}" for t, _ in MODELS) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}}",
          "\\caption{Encoded, stated and enacted reversibility on shared items, and the blindness gap. $\\kappa$: Cohen's kappa with "
          "the true label. $K$: share of pairs whose stated answer ranks the irreversible version's shortcut as less reversible; "
          "$A$: share of pairs with strict caution only in the irreversible version.}", "\\label{tab:levels}", "\\end{table}"]
    open("paper/tables/levels.tex", "w").write("\n".join(T) + "\n")

    # appendix: preregistered secondary measures
    A = ["\\begin{table*}[t]", "\\centering\\small", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{llcccc}", "\\toprule",
         "Model & Prompt & Strict caution irr./rev. & Broad caution irr./rev. & Broad: sensitivity [CI], McNemar $p$ & Parse failures (episodes ended) \\\\", "\\midrule"]
    for t, name in MODELS:
        for cond in ("neutral", "careful"):
            b = S[t]["behaviour"][cond]
            A.append(f"{name} & {cond} & {f(b['strict']['irr'])} / {f(b['strict']['rev'])} & {f(b['broad']['irr'])} / {f(b['broad']['rev'])} & "
                     f"{f(b['broad']['sensitivity'])} {ci(b['broad']['sensitivity_ci95'])}, {f(b['broad']['mcnemar_exact_p'], 3)} & "
                     f"{b['parse_failures']['total']} ({b['parse_failures']['episodes_ended_by_parse_failures']}) \\\\")
    A += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Paired 2$\\times$2 tables of strict caution (careful in irr.\\ only / rev.\\ only / both / neither)}} \\\\"]
    for t, name in MODELS:
        for cond in ("neutral", "careful"):
            tb = S[t]["behaviour"][cond]["strict"]["table"]
            g = S[t][f"gap_{cond}"]
            A.append(f"{name} & {cond} & \\multicolumn{{2}}{{c}}{{{tb['irr_only']} / {tb['rev_only']} / {tb['both']} / {tb['neither']}}} & "
                     f"\\multicolumn{{2}}{{l}}{{gap (prereg.) {f(g['gap_argmax'])} {ci(g['gap_argmax_ci95'])}}} \\\\")
    A += ["\\bottomrule", "\\end{tabular}}",
          "\\caption{Preregistered secondary RQ3 measures. Broad caution adds inspecting the state (read\\_file, list\\_files, "
          "check\\_balance) to strict caution. Parse failures: unparseable replies over all 120 episodes (episodes that ended after "
          "three consecutive failures); such episodes are kept.}", "\\label{tab:secondary}", "\\end{table*}"]
    open("paper/tables/secondary.tex", "w").write("\n".join(A) + "\n")

    # annotation agreement
    if S.get("annotation"):
        an = S["annotation"]
        R = ["\\begin{table}[t]", "\\centering\\small", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{lccc}", "\\toprule",
             "Cohen's $\\kappa$ [95\\% CI] & All ($n=100$) & Sandbox (60) & ScienceWorld (40) \\\\", "\\midrule"]
        for lab, k in (("Annotator A vs.\\ B", "A_vs_B"), ("A vs.\\ automatic", "A_vs_auto"), ("B vs.\\ automatic", "B_vs_auto")):
            R.append(f"{lab} & " + " & ".join(f"{f(an[s][k]['kappa'])} {ci(an[s][k]['ci95'])}" for s in ("all", "sandbox", "scienceworld")) + " \\\\")
        R += ["\\bottomrule", "\\end{tabular}}",
              "\\caption{Agreement on the blind 100-item sample (three classes; item bootstrap, 1{,}000 resamples).}",
              "\\label{tab:kappa}", "\\end{table}"]
        open("paper/tables/kappa.tex", "w").write("\n".join(R) + "\n")


def main():
    S = {"data": data_block(), "power": power()}
    for t, _ in MODELS:
        d = {"probes": probes_block(t), "behaviour": {}, "steering_parse_fail_per_step": steering_pf(t)}
        stt = load(f"results/{t}/stated_tasks.jsonl")
        for cond in ("neutral", "careful"):
            pr = pairs_of(load(f"results/{t}/agent_{cond}.jsonl"))
            d["behaviour"][cond] = behaviour_block(pr)
            d[f"gap_{cond}"] = gap_block(stt, pr)
        S[t] = d
    if os.path.exists("results/annotation_agreement.json"):
        S["annotation"] = json.load(open("results/annotation_agreement.json"))
    json.dump(S, open("results/supplement.json", "w"), indent=1)
    tex_tables(S)
    for t, _ in MODELS:
        g = S[t]["gap_neutral"]
        print(f"{t}: K argmax {g['K_argmax']:.3f} | K prob {g['K_prob']:.3f} {g['K_prob_ci95']} | A {g['A']:.3f} | "
              f"ask-or-backup irr/rev {S[t]['behaviour']['neutral']['ask_or_backup']['irr']:.2f}/{S[t]['behaviour']['neutral']['ask_or_backup']['rev']:.2f}")
    print("power:", S["power"], "| data:", S["data"])
    print("saved results/supplement.json and paper/tables/*.tex")


if __name__ == "__main__":
    main()
