"""Figure for RQ4: strict caution and irreversible actions vs. steering strength, both models, with controls."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
MULTS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def val(c, key, which):
    d = c[key]
    if which == "strict":
        return (d["irreversible"]["strict"] + d["reversible"]["strict"]) / 2
    return d["irreversible"]["took_irreversible"]


fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex=True)
for col, (tag, name) in enumerate([("qwen7b", "Qwen2.5-7B"), ("llama8b", "Llama-3.1-8B")]):
    r = json.load(open(f"results/{tag}/analysis_steering.json"))
    c = r["conditions"]
    for row, (which, ylab) in enumerate([("strict", "strict caution"), ("irr", "irreversible action\n(irreversible versions)")]):
        ax = axes[row, col]
        ys = [val(c, "none:+0.0x" if m == 0 else f"probe:{m:+.1f}x", which) for m in MULTS]
        ax.plot(MULTS, ys, color=BLUE, lw=2, marker="o", ms=5, label="probe direction", zorder=3)
        for k in range(3):
            ax.scatter([-2, 2], [val(c, f"rand{k}:{m:+.1f}x", which) for m in (-2.0, 2.0)], color=GRAY, s=22,
                       marker="s", label="random directions" if k == 0 else None, zorder=2)
        ax.scatter([-2, 2], [val(c, f"binding:{m:+.1f}x", which) for m in (-2.0, 2.0)], color=ORANGE, s=30,
                   marker="D", label="binding-only direction", zorder=2)
        ax.set_ylim(-0.02, 0.45 if which == "strict" else 1.03)
        if which == "irr":
            ax.set_ylim(0.6, 1.02)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=8, color=INK)
        if row == 0:
            ax.set_title(f"{name} (layer {r['layer']})", fontsize=9, color=INK)
        if row == 1:
            ax.set_xlabel(r"steering strength ($\times$ scale; $-$ = towards reversible)", fontsize=8, color=INK)
axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("paper/fig_steering.pdf")
fig.savefig("paper/fig_steering.png", dpi=220)
print("saved")
