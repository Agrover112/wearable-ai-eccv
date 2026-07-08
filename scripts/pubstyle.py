"""Shared publication style for all figures/tables (ECCV / Springer LNCS).

Usage:
    from pubstyle import set_style, savefig, df_to_booktabs
    set_style()
    ...
    savefig(fig, "outputs/eda/figures/myplot")   # writes .pdf + .png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
      "#56B4E9", "#F0E442", "#000000", "#999999", "#8172B3"]

def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,            # editable/embeddable text in PDF (no type-3)
        "ps.fonttype": 42,
        "font.family": "serif",         # matches LNCS body
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=CB),
    })

# LNCS text widths (inches): single column ~3.3, double ~6.9
COL, DBL = 3.3, 6.9

def savefig(fig, path_noext):
    p = Path(path_noext); p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"))
    fig.savefig(p.with_suffix(".png"))
    plt.close(fig)
    return str(p.with_suffix(".pdf"))

def df_to_booktabs(df, path, caption="", label="", float_fmt="%.1f"):
    """Write a LaTeX booktabs table alongside any CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=False, escape=True, float_format=lambda x: float_fmt % x)
    body = body.replace("\\toprule", "\\toprule").replace("tabular", "tabular")  # booktabs via pandas
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n"
           + (f"\\caption{{{caption}}}\n" if caption else "")
           + (f"\\label{{{label}}}\n" if label else "")
           + body + "\\end{table}\n")
    Path(path).write_text(tex)
    return path
