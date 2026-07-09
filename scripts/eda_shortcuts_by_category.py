#!/usr/bin/env python
"""EgoLongQA — per-category breakdown of the two video-blind shortcuts.

For each of the 13 scene categories:
  (a) share of questions whose correct letter is C            (overall: 63.4%)
  (b) share whose correct option is the shortest of the four  (overall: 40.3%)

Outputs (outputs/eda/):
  figures/shortcuts_by_category.{pdf,png}
  tables/shortcuts_by_category.{csv,tex}
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pubstyle import set_style, savefig, df_to_booktabs, CB, DBL
from src.data import parse_mcq
import matplotlib.pyplot as plt

REPO = "facebook/wearable-ai"; TOK = os.environ.get("HF_TOKEN") or True
OUT = Path("outputs/eda")


def main():
    set_style()
    pq = hf_hub_download(REPO, "egolongqa/val/0000.parquet", repo_type="dataset",
                         revision="refs/convert/parquet", token=TOK)
    df = pd.read_parquet(pq)

    def is_shortest(row):
        # identical to Step 5 (eda_egolongqa.py): stable sort by length, longest
        # first; "shortest" = rank 4 of 4. Ties resolve by A-D order, so this
        # matches the reported 40.3% overall figure.
        opts = parse_mcq(row["mcq_options"])
        gold = str(row["mcq_answer"]).strip().upper()
        if len(opts) < 2 or gold not in opts:
            return np.nan
        order = sorted(opts, key=lambda k: len(opts[k]), reverse=True)
        return float(order.index(gold) + 1 == len(opts))

    df["is_C"] = (df["mcq_answer"] == "C").astype(float)
    df["is_shortest"] = df.apply(is_shortest, axis=1)

    g = (df.groupby("category")
           .agg(n=("is_C", "size"), share_C=("is_C", "mean"),
                share_shortest=("is_shortest", "mean"))
           .sort_values("n", ascending=False).reset_index())
    g["share_C"] *= 100; g["share_shortest"] *= 100
    overall_C = 100 * df["is_C"].mean()
    overall_S = 100 * df["is_shortest"].mean()

    tab = g.rename(columns={"category": "Category", "n": "N",
                            "share_C": "correct = C (%)",
                            "share_shortest": "correct = shortest (%)"})
    tab.to_csv(OUT/"tables/shortcuts_by_category.csv", index=False)
    df_to_booktabs(tab, OUT/"tables/shortcuts_by_category.tex", float_fmt="%.1f",
        caption=f"Per-category strength of the two video-blind shortcuts "
        f"(overall: C {overall_C:.1f}\\%, shortest {overall_S:.1f}\\%; chance 25\\%).",
        label="tab:shortcuts_by_cat")
    print(tab.to_string(index=False))

    # ---- figure: two aligned horizontal-bar panels --------------------------
    y = np.arange(len(g))[::-1]
    labels = [f"{c}  ({n})" for c, n in zip(g.category, g.n)]
    fig, axes = plt.subplots(1, 2, figsize=(DBL, DBL*0.42), sharey=True)
    for ax, col, overall, title, color in [
            (axes[0], "share_C", overall_C,
             "(a) correct letter is C", CB[0]),
            (axes[1], "share_shortest", overall_S,
             "(b) correct option is the shortest", CB[1])]:
        ax.barh(y, g[col], color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(25, color="0.4", lw=1.0, ls=":", label="chance 25%")
        ax.axvline(overall, color=CB[3], lw=1.2, ls="--",
                   label=f"overall {overall:.1f}%")
        for yi, v in zip(y, g[col]):
            ax.text(v + 1.2, yi, f"{v:.0f}", va="center", fontsize=6)
        ax.set_xlim(0, 100); ax.set_xlabel("share of questions (%)")
        ax.set_title(title, fontsize=8)
        ax.legend(frameon=False, fontsize=6.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=6.5)
    fig.suptitle("EgoLongQA video-blind shortcuts by scene category (N per category "
                 "in parentheses)", fontsize=8.5, y=1.03)
    fig.tight_layout()
    print(">> wrote", savefig(fig, OUT/"figures/shortcuts_by_category"))


if __name__ == "__main__":
    main()
