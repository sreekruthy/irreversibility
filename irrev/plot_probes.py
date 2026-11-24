"""Figure for RQ1: per-layer probe accuracy with baselines and controls, sandbox and ScienceWorld, per model."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
tags = [t for t in ("qwen7b", "llama8b") if os.path.exists(f"results/{t}/analysis_probes.json")]
names = {"qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B"}
fig, axes = plt.subplots(len(tags), 2, figsize=(6.6, 2.5 * len(tags)), squeeze=False, sharey=True)
for row, tag in enumerate(tags):
    A = json.load(open(f"results/{tag}/analysis_probes.json"))
    Ls = A["best_layer_Lstar"]
    for col, env in enumerate(("sandbox", "scienceworld")):
        ax = axes[row, col]
        d = A["envs"][env]
        c = d["curves"]
        x = sorted(int(k) for k in c["binary_last"])
        g = lambda k: [c[k][str(i)]["mean"] if str(i) in c[k] else c[k][i]["mean"] for i in x]
        ax.plot(x, g("binary_last"), color=BLUE, lw=2, label="probe, balanced accuracy")
        if "pairs_tier3" in c:
            ax.plot(x, g("pairs_tier3"), color=ORANGE, lw=2, label="probe, binding pairs (within-pair)")
        ax.plot(x, g("shuffled"), color=GRAY, lw=1.2, ls=":", label="shuffled-label probe")
        bl = d["baselines_binary_bal_acc"]
        ax.axhline(bl["bow_full_prompt"], color=MUTED, lw=1, ls="--", label="bag-of-words, full prompt")
        ax.axhline(bl["bow_action"], color=MUTED, lw=1, ls="-.", label="bag-of-words, action")
        if "pairs_tier3" in c:
            ax.axhline(d["baselines_within_pair"].get("bow_full_prompt:tier3", 0.5), color=ORANGE, lw=1, ls="--", alpha=0.6)
        ax.axvline(Ls, color="#c3c2b7", lw=0.8)
        ax.set_ylim(0.4, 1.02)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(f"{names[tag]}, {'sandbox' if env == 'sandbox' else 'ScienceWorld'}", fontsize=9, color=INK)
        if col == 0:
            ax.set_ylabel("binary (rev. vs irr.)", fontsize=8, color=INK)
        if row == len(tags) - 1:
            ax.set_xlabel("layer (0 = embeddings)", fontsize=8, color=INK)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.12 / len(tags), 1, 1))
fig.savefig("paper/fig_probes.pdf"); fig.savefig("paper/fig_probes.png", dpi=220)
print("saved", tags)
