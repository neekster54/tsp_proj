"""Make graphs from an evaluation results CSV (the output of run_evaluation.py).

Styled to match the LaTeX/pgfplots figures in the progression review: Computer Modern
serif text, marks only, boxed axes with inward ticks, and a small framed legend.

Usage:
    python plot_results.py                          # results/TESTRUN.csv
    python plot_results.py results/my_run.csv       # any results CSV
    python plot_results.py results/TESTRUN.csv --out plots --show

Writes these PNGs into the output directory (default: results/plots/<csv name>/):
    1_error_vs_n.png           error vs. n: constructed tour, after 2-opt, after 3-opt
    2_mean_error_by_stage.png  mean error per algorithm at each improvement stage
    3_runtime_vs_n.png         construction time and 2-opt time vs. n (log-log)
    4_error_heatmap.png        construction error for every instance x algorithm
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

# pgfplots-style marks. NN / DT / CS use exactly the colours and marks of the report.
ALGORITHMS = {
    # name:              (label,  colour,    marker)
    "nearest_neighbour": ("NN",   "#0000ff", "o"),
    "double_tree":       ("DT",   "#ff0000", "s"),
    "christofides":      ("CS",   "#008000", "^"),
    "multi_start_nn":    ("MSNN", "#996633", "D"),
    "greedy_edge":       ("GE",   "#bf00bf", "p"),
    "held_karp":         ("HK",   "#000000", "x"),
}
FALLBACK = [("#00bfbf", "v"), ("#ff8000", "<"), ("#808080", ">")]
EXACT = {"held_karp"}  # always 0% error, so left out of the error plots

STAGES = [
    ("error", "Constructed tour", "init"),
    ("error_two_opt", "After common 2-opt", "2-opt"),
    ("error_three_opt", "After common 3-opt", "3-opt"),
]
STAGE_COLOURS = ["#0000ff", "#ff0000", "#008000"]

HEATMAP_CMAP = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#9999ff", "#0000b3"])
GRID = "#d9d9d9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.labelsize": 10,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "black",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1,
    "legend.borderpad": 0.3,
    "legend.handletextpad": 0.1,
    "legend.handlelength": 1.0,
    "legend.labelspacing": 0.15,
    "legend.borderaxespad": 0.4,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.markersize": 3.5,
})


def load(path):
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["n"] = df["n"].astype(int)
    return df.sort_values(["n", "instance"])


def algorithm_styles(df):
    present = df["algorithm"].unique().tolist()
    ordered = [a for a in ALGORITHMS if a in present]
    extra = sorted(a for a in present if a not in ALGORITHMS)
    if len(extra) > len(FALLBACK):
        raise SystemExit(f"Too many unknown algorithms ({len(extra)}) to style distinctly.")
    styles = {a: ALGORITHMS[a] for a in ordered}
    styles.update({a: (a, *FALLBACK[i]) for i, a in enumerate(extra)})
    return ordered + extra, styles


def label(styles, algo):
    return styles[algo][0]


def marks(ax, x, y, style, **kw):
    _, colour, marker = style
    filled = marker not in ("x", "+")
    ax.plot(x, y, linestyle="none", marker=marker, color=colour,
            markerfacecolor=colour if filled else "none", markeredgewidth=0.6 if filled else 1,
            **kw)


def log_n_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="minor", visible=False)
    ax.set_xlabel("$n$ (cities)")


def fit_n_axis(ax, n):
    """x range fitted to this panel's n, starting at the decade below so 10^k is labelled."""
    ax.set_xlim(10 ** np.floor(np.log10(n.min())), n.max() * 1.25)


def legend(ax, loc, source=None):
    """Framed legend on ax, listing the marks drawn on source (default: ax itself)."""
    handles, labels = (source or ax).get_legend_handles_labels()
    ax.legend(handles, labels, loc=loc, handletextpad=0.1, borderpad=0.25)


def plot_error_vs_n(df, algos, styles, out):
    algos = [a for a in algos if a not in EXACT]
    stages = [s for s in STAGES if s[0] in df.columns and df[s[0]].notna().any()]
    # Each panel gets its own x and y range, fitted to its data, so that the small
    # errors after 2-opt / 3-opt are not squashed against zero by the construction panel.
    fig, axes = plt.subplots(1, len(stages), figsize=(3.0 * len(stages), 2.3))
    axes = np.atleast_1d(axes)
    for ax, (col, title, _) in zip(axes, stages):
        sub = df[df["algorithm"].isin(algos) & df[col].notna()]
        for algo in algos:
            g = sub[sub["algorithm"] == algo]
            marks(ax, g["n"], g[col], styles[algo], label=label(styles, algo))
        ax.set_title(title if sub["n"].max() == df["n"].max()
                     else f"{title} ($n \\leq {sub['n'].max()}$)")
        log_n_axis(ax)
        fit_n_axis(ax, sub["n"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        span = sub[col].max() - sub[col].min()
        ax.set_ylim(sub[col].min() - 0.06 * span, sub[col].max() + 0.08 * span)
    axes[0].set_ylabel("error above optimum (%)")
    legend(axes[-1], "upper left", source=axes[0])
    axes[-1].get_legend().set_bbox_to_anchor((1.02, 1.0))
    fig.tight_layout(w_pad=1.0)
    save(fig, out / "1_error_vs_n")


def plot_mean_error_by_stage(df, algos, styles, out):
    algos = [a for a in algos if a not in EXACT]
    stages = [s for s in STAGES if s[0] in df.columns and df[s[0]].notna().any()]
    # Only average over instances that have every stage, so the bars are comparable.
    complete = df[df["algorithm"].isin(algos)].dropna(subset=[s[0] for s in stages])
    algos = [a for a in algos if a in complete["algorithm"].values]
    means = complete.groupby("algorithm")[[s[0] for s in stages]].mean().loc[algos]

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    width = 0.8 / len(stages)
    x = np.arange(len(algos))
    for i, ((col, _, short), colour) in enumerate(zip(stages, STAGE_COLOURS)):
        pos = x - 0.4 + width * (i + 0.5)
        # pgfplots ybar look: coloured outline, light fill of the same colour.
        bars = ax.bar(pos, means[col], width * 0.85, color=colour + "4d", edgecolor=colour,
                      linewidth=0.6, label=short, zorder=2)
        for bar, v in zip(bars, means[col]):
            ax.annotate(f"{v:.1f}", (bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([label(styles, a) for a in algos])
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(algos) - 0.4)
    ax.set_ylim(0, means.values.max() * 1.18)
    ax.set_ylabel("mean error above optimum (%)")
    ax.set_title(f"Mean error over the {complete['instance'].nunique()} instances with "
                 f"every stage ($n \\leq {complete['n'].max()}$)")
    legend(ax, "upper right")
    ax.get_legend().set_title(None)
    fig.tight_layout()
    save(fig, out / "2_mean_error_by_stage")


def plot_runtime_vs_n(df, algos, styles, out):
    panels = [("algo_time", "Construction time", algos)]
    if "two_opt_time" in df.columns and df["two_opt_time"].notna().any():
        panels.append(("two_opt_time", "2-opt time (from each start)",
                       [a for a in algos if a not in EXACT]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.3))
    axes = np.atleast_1d(axes)
    for ax, (col, title, show) in zip(axes, panels):
        for algo in show:
            g = df[(df["algorithm"] == algo) & df[col].notna()]
            marks(ax, g["n"], g[col], styles[algo], label=label(styles, algo))
        log_n_axis(ax)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="minor", visible=False)
        ax.set_title(title)
        sub = df[df["algorithm"].isin(show) & df[col].notna()]
        fit_n_axis(ax, sub["n"])
        # Own y range per panel, padded by a fraction of a decade on each side.
        lo, hi = np.log10(sub[col].min()), np.log10(sub[col].max())
        pad = 0.06 * (hi - lo)
        ax.set_ylim(10 ** (lo - pad), 10 ** (hi + pad))
        # Label every decade, or every other / third one on tall ranges, like the report.
        decades = np.arange(np.ceil(lo - pad), np.floor(hi + pad) + 1)
        step = 3 if len(decades) > 6 else 2 if len(decades) > 4 else 1
        ax.set_yticks(10.0 ** decades[decades % step == 0])
        ax.set_ylim(10 ** (lo - pad), 10 ** (hi + pad))
    axes[0].set_ylabel("median time (s)")
    legend(axes[-1], "lower right", source=axes[0])
    fig.tight_layout(w_pad=1.0)
    save(fig, out / "3_runtime_vs_n")


def plot_error_heatmap(df, algos, styles, out):
    algos = [a for a in algos
             if a not in EXACT and df.loc[df["algorithm"] == a, "error"].notna().sum() > 3]
    table = (df[df["algorithm"].isin(algos)]
             .pivot_table(index=["n", "instance"], columns="algorithm", values="error")
             [algos])
    fig, ax = plt.subplots(figsize=(0.7 * len(algos) + 2.2, 0.2 * len(table) + 0.9))
    vmax = np.nanmax(table.values)
    im = ax.imshow(table.values, cmap=HEATMAP_CMAP, vmin=0, vmax=vmax, aspect="auto")
    for (r, c), v in np.ndenumerate(table.values):
        if np.isnan(v):
            continue
        ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=7,
                color="white" if v > vmax * 0.6 else "black")
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([label(styles, a) for a in algos])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([f"{inst} ({n})" for n, inst in table.index], fontsize=8)
    ax.tick_params(which="both", length=0)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("error above optimum (%)")
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(direction="in", right=True)
    ax.set_title("Constructed-tour error (%) per instance, sorted by $n$", pad=6)
    fig.tight_layout()
    save(fig, out / "4_error_heatmap")


def save(fig, stem):
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", default="results/TESTRUN.csv")
    parser.add_argument("--out", help="output directory (default: results/plots/<csv name>)")
    parser.add_argument("--show", action="store_true", help="also open the figures in a window")
    args = parser.parse_args()

    df = load(args.csv)
    if df.empty:
        raise SystemExit(f"No rows with status 'ok' in {args.csv}")
    out = Path(args.out) if args.out else Path("results/plots") / Path(args.csv).stem
    out.mkdir(parents=True, exist_ok=True)

    algos, styles = algorithm_styles(df)
    plot_error_vs_n(df, algos, styles, out)
    plot_mean_error_by_stage(df, algos, styles, out)
    plot_runtime_vs_n(df, algos, styles, out)
    plot_error_heatmap(df, algos, styles, out)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
