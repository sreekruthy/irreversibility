"""Additional controls (exploratory, added after the preregistered analysis; reported as such).

1. Risk/harm direction (specificity of the steering direction). A harmful-vs-harmless direction is computed
   as in Arditi et al. (2024): mean last-token residual at L* over 128 AdvBench requests minus the mean over
   128 Alpaca requests (data/harm_contrast.json). Reported: its cosine with the reversibility direction, how
   well each direction separates the other's contrast (AUROC of the projection), and (with --steer-harm)
   agent runs steered along the harm direction at the same strength as the reversibility direction (+/-2x).
2. Held-out tool domain. Binary probe at L* trained on two of {files, email, wallet} and tested on the third
   (C by grouped CV on the training domains), with balanced accuracy, AUROC and binding-pair accuracy, and the
   full-prompt bag-of-words baseline on the same split.
3. ScienceWorld per-item predictions at L* (3 scenario splits) for the within-verb analysis, which is done
   offline by analyze_extra.py (needs no GPU).

Usage (Kaggle):
  python -m irrev.extra_analyses --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b            # 1 (direction) + 2 + 3
  python -m irrev.extra_analyses --model Qwen/Qwen2.5-7B-Instruct --tag qwen7b --steer-harm   # 1 (agent runs)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from .probes import (ORD, bow_baselines, fit_eval, item_metrics, labels, load_set, pair_acc, score_of, split,
                     unique_mask)


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def harm_direction(model_id, tag, L, out_dir):
    import torch
    from .common import chat_prompt, decoder_layers, first_device, hf_token, load_model
    data = json.load(open("data/harm_contrast.json"))
    tok, model, dtype = load_model(model_id, token=hf_token())
    dev = first_device(model)
    cap = {}

    def hook(m, i, o):
        h = o[0] if isinstance(o, (tuple, list)) else o
        cap["h"] = h[0, -1].float().cpu().numpy()
    hd = decoder_layers(model)[L - 1].register_forward_hook(hook)   # L counts the embedding layer as 0
    H = {}
    for k in ("harmful", "harmless"):
        vs = []
        for x in data[k]:
            ids = tok(chat_prompt(tok, [{"role": "user", "content": x}]), return_tensors="pt", add_special_tokens=False).to(dev)
            with torch.no_grad():
                model(**ids)
            vs.append(cap["h"])
        H[k] = np.stack(vs)
    hd.remove()
    del model
    torch.cuda.empty_cache()
    d_harm = H["harmful"].mean(0) - H["harmless"].mean(0)
    z = np.load(f"results/{tag}/probes/steer_dirs.npz")
    assert int(z["layer"]) == L
    d_rev, d_bind = z["direction"], z["direction_binding"]
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    meta = json.load(open(f"results/{tag}/steer/steer_meta.json"))
    np.savez(os.path.join(out_dir, "dir_harm.npz"), direction=(d_harm / np.linalg.norm(d_harm)).astype(np.float32),
             mean_norm=np.float32(meta["mean_resid_norm"]))
    np.savez(os.path.join(out_dir, "harm_acts.npz"), harmful=H["harmful"].astype(np.float16), harmless=H["harmless"].astype(np.float16))
    return {"layer": L, "n_each": len(data["harmful"]), "norm_harm_direction": float(np.linalg.norm(d_harm)),
            "cos_harm_vs_reversibility_direction": cos(d_harm, d_rev),
            "cos_harm_vs_binding_direction": cos(d_harm, d_bind),
            "auroc_reversibility_direction_on_harmful_vs_harmless": auroc(H["harmful"] @ d_rev, H["harmless"] @ d_rev)}


def held_out_domain(rows, X, L):
    groups = np.array([r["scenario_id"] for r in rows])
    y, keep = labels(rows, "binary")
    dom = np.array([r["domain"] for r in rows])
    Xl = X[:, L, :].astype(np.float32)
    out = {}
    for d in ("files", "email", "wallet"):
        tr_all, te_all = dom != d, dom == d
        tr = unique_mask(rows, tr_all) & keep
        te = unique_mask(rows, te_all) & keep
        P, C, cv, m, sc = fit_eval(Xl[tr], y[tr], groups[tr], Xl[te], y[te], "binary")
        res = {"train_domains": sorted(set(dom[tr_all])), "C": C, "test": item_metrics(y[te], P, "binary", groups[te])}
        tall = te_all & keep
        Pa = m.predict_proba(sc.transform(Xl[tall]))
        full = np.zeros((tall.sum(), 2))
        for j, cl in enumerate(m.classes_):
            full[:, cl] = Pa[:, j]
        idx = np.where(tall)[0]
        res["pairs"] = {t: {"n_pairs": len(pa), "within_pair_accuracy": float(np.mean(pa))}
                        for t in ("tier1", "tier2", "tier3") if (pa := pair_acc(rows, idx, score_of(full, "binary"), "binary", t))}
        res["bow_baselines_same_split"] = {k: v["balanced_accuracy"] for k, v in
                                          bow_baselines(rows, "binary", unique_mask(rows, tr_all), unique_mask(rows, te_all), te_all, groups).items()}
        out[d] = res
        print(f"  held-out {d}: bal acc {res['test']['balanced_accuracy']:.3f} AUROC {res['test']['auroc']:.3f} "
              f"pairs {{{', '.join(f'{t}: {v['within_pair_accuracy']:.3f}' for t, v in res['pairs'].items())}}} "
              f"| BoW {res['bow_baselines_same_split']}", flush=True)
    return out


def sw_predictions(rows, X, L, path):
    groups = np.array([r["scenario_id"] for r in rows])
    y, keep = labels(rows, "binary")
    Xl = X[:, L, :].astype(np.float32)
    with open(path, "w") as f:
        for seed in (0, 1, 2):
            tr, te = split(rows, seed)
            tru = unique_mask(rows, tr) & keep
            tem = te & keep
            P, C, _, _, _ = fit_eval(Xl[tru], y[tru], groups[tru], Xl[tem], y[tem], "binary")
            for j, i in enumerate(np.where(tem)[0]):
                f.write(json.dumps({"id": rows[i]["id"], "seed": seed, "label": rows[i]["label"],
                                    "p_irr": float(P[j, 1]), "C": C}) + "\n")


def steer_harm(model_id, tag):
    meta = json.load(open(f"results/{tag}/steer/steer_meta.json"))
    L, scale = meta["layer"], meta["scale"]
    for mult in (+2.0, -2.0):
        out = f"results/{tag}/steer/test_harm_{mult:+.1f}x.jsonl"
        cmd = [sys.executable, "-m", "irrev.agent", "--model", model_id, "--tasks", "data/tasks.jsonl", "--split", "test",
               "--out", out, "--batch", "8", "--tag", f"harm:{mult}", "--steer", f"results/{tag}/steer/dir_harm.npz",
               "--steer-layer", str(L), "--alpha", repr(mult * scale)]
        print(">>", " ".join(cmd[3:]), flush=True)
        subprocess.run(cmd, check=True)
    print("HARM STEER DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steer-harm", action="store_true")
    a = ap.parse_args()
    if a.steer_harm:
        return steer_harm(a.model, a.tag)
    out_dir = f"results/{a.tag}/steer"
    os.makedirs(out_dir, exist_ok=True)
    L = int(np.load(f"results/{a.tag}/probes/steer_dirs.npz")["layer"])
    res = {"note": "exploratory controls added after the preregistered analysis", "layer": L}
    res["harm_direction"] = harm_direction(a.model, a.tag, L, out_dir)
    print("harm direction:", res["harm_direction"], flush=True)
    T = a.tag
    r1, l1, _ = load_set(f"results/{T}/hidden_sandbox", "data/probe_sandbox.jsonl")
    r2, l2, _ = load_set(f"results/{T}/hidden_extra", "data/probe_sandbox_extra.jsonl")
    rows, X = r1 + r2, np.concatenate([l1, l2])
    # reversibility probe direction projected on the harm contrast, and harm direction on sandbox items
    d_harm = np.load(f"{out_dir}/dir_harm.npz")["direction"]
    yb = np.array([ORD[r["label"]] for r in rows])
    H = X[:, L, :].astype(np.float32)
    res["harm_direction"]["auroc_harm_direction_on_irreversible_vs_reversible_items"] = auroc(H[yb == 2] @ d_harm, H[yb == 0] @ d_harm)
    print("held-out tool domain:", flush=True)
    res["held_out_domain"] = held_out_domain(rows, X, L)
    rs, ls, _ = load_set(f"results/{T}/hidden_sw", "data/probe_scienceworld.jsonl")
    sw_predictions(rs, ls, L, f"results/{T}/probes/sw_preds.jsonl")
    json.dump(res, open(f"results/{T}/extra_controls.json", "w"), indent=1, default=float)
    print("EXTRA DONE", flush=True)


if __name__ == "__main__":
    main()
