"""Generate the report figures from the saved result JSONs (reproducible).

Reads results/cvae_norman_delta.json and results/cvae_replogle_delta.json and
writes:

    figures/fig1_per_split.png   per-split delta-Pearson, cVAE vs mean-of-training
    figures/fig2_leakage.png     random vs function-grouped split (the leakage gap)

Every number plotted is read straight from the JSONs produced by train.py; no
values are hard-coded here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

CVAE = "#2c6fbb"   # blue
MOT = "#bdbdbd"    # grey
ACCENT = "#d1495b"  # red, for the gap call-out

SPLIT_LABEL = {
    "A_random": "A: random\nsingles",
    "B_function": "B: function-\ngrouped singles",
    "C_combinatorial": "C: train singles\n→ test doubles",
}


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def per_split_pearson(res: dict, split: str):
    """(cVAE pearson, MoT pearson, n_test) or None if the split is absent."""
    if split not in res:
        return None
    s = res[split]
    return s["cvae"]["mean_pearson"], s["mean_of_training"]["mean_pearson"], int(s["n_test"])


def fig1(norman: dict, replogle: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    panels = [
        (axes[0], "Norman 2019 (combinatorial Perturb-seq)", norman,
         ["A_random", "B_function", "C_combinatorial"]),
        (axes[1], "Replogle 2022 (K562 essential-gene)", replogle,
         ["A_random", "B_function"]),
    ]
    for ax, title, res, order in panels:
        order = [s for s in order if s in res]
        x = np.arange(len(order))
        w = 0.38
        cv = [res[s]["cvae"]["mean_pearson"] for s in order]
        mt = [res[s]["mean_of_training"]["mean_pearson"] for s in order]
        b1 = ax.bar(x - w / 2, cv, w, label="conditional VAE", color=CVAE)
        b2 = ax.bar(x + w / 2, mt, w, label="mean-of-training", color=MOT)
        for bars in (b1, b2):
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012,
                        f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABEL[s] for s in order], fontsize=8.5)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("delta Pearson (held-out perturbations)" if ax is axes[0] else "")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="black", lw=0.6)
        # annotate the one place the learned model wins
        if "C_combinatorial" in order:
            i = order.index("C_combinatorial")
            ax.annotate("cVAE beats\nbaseline on\nunseen doubles",
                        xy=(i - w / 2, cv[i]), xytext=(i - 1.15, 0.78),
                        fontsize=8, color=ACCENT, ha="left",
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    axes[0].legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.16))
    fig.suptitle("Held-out perturbation-response prediction across leakage-controlled splits",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_per_split.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2(norman: dict, replogle: dict) -> None:
    """Slope plot: random (A) -> function-grouped (B) Pearson, per method/dataset."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    series = [
        ("Norman, cVAE", norman, "cvae", CVAE, "-", "o"),
        ("Norman, mean-of-training", norman, "mean_of_training", CVAE, "--", "o"),
        ("Replogle, cVAE", replogle, "cvae", "#e08a2e", "-", "s"),
        ("Replogle, mean-of-training", replogle, "mean_of_training", "#e08a2e", "--", "s"),
    ]
    labels = []  # (y_endpoint, text, color) for dodged right-side annotation
    for label, res, key, color, ls, mk in series:
        a = res["A_random"][key]["mean_pearson"]
        b = res["B_function"][key]["mean_pearson"]
        ax.plot([0, 1], [a, b], ls, color=color, marker=mk, lw=1.8, ms=7, label=label,
                alpha=0.95)
        tag = "cVAE" if key == "cvae" else "MoT"
        labels.append((b, f"{label.split(',')[0]} {tag}  Δ={a - b:+.2f}", color))
    # Dodge the right-side endpoint labels so close endpoints don't overprint.
    labels.sort(key=lambda t: t[0])
    min_sep = 0.035
    ys = [t[0] for t in labels]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_sep:
            ys[i] = ys[i - 1] + min_sep
    for (_, text, color), y in zip(labels, ys):
        ax.text(1.04, y, text, va="center", fontsize=8, color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A: random\nsingles", "B: function-grouped\nsingles (leakage-free)"])
    ax.set_ylabel("delta Pearson (held-out perturbations)")
    ax.set_xlim(-0.15, 1.65)
    ax.set_ylim(0, 0.7)
    ax.set_title("Leakage gap: removing functionally similar perturbations\nfrom training is harder, by a dataset-dependent amount",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_leakage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    norman = load("cvae_norman_delta.json")
    replogle = load("cvae_replogle_delta.json")
    fig1(norman, replogle)
    fig2(norman, replogle)
    print("wrote", FIG / "fig1_per_split.png")
    print("wrote", FIG / "fig2_leakage.png")


if __name__ == "__main__":
    main()
