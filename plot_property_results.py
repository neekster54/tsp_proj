import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot import ALGORITHMS, legend

FAMILIES = {
    "weights": ("Edge weights (increasing spread)", ["unit", "flat", "original", "skewed"], True),
    "bridges": ("Structure", ["bridge2", "bridge4", "cut2", "cut4", "pendant10", "pendant25"], False),
    "degrees": ("Extra edges (increasing degree variance)", ["regular", "random", "preferential", "hub"], True),
}
PANELS = {
    "error": "Constructed tour",
    "error_two_opt": "After 2-opt",
    "error_free": "After 2-opt, forced cost excluded",
    "extra_per_city": "Repeated visits after 2-opt",
}


def load(path, index):
    df = pd.read_csv(path)
    df = df[(df["status"] == "ok") & (df["mode"] != "complete")]
    idx = pd.read_csv(index)[["name", "family", "variant", "forced_cost"]]
    df = df.merge(idx, on="name")
    df["error_free"] = 100 * (df["length_two_opt"] - df["optimum"]) / (df["optimum"] - df["forced_cost"])
    df["extra_per_city"] = df["extra_visits_two_opt"] / df["n"]
    return df


def variant_panel(ax, df, variants, column, styles, join, dots=True):
    """Mean of `column` per variant and algorithm, with the individual graphs behind."""
    algos = [a for a in styles if a in set(df["algorithm"])]
    rng = np.random.default_rng(0)
    for i, algo in enumerate(algos):
        name, colour, marker = styles[algo]
        offset = (i - (len(algos) - 1) / 2) * 0.17
        xs, means = [], []
        for x, variant in enumerate(variants):
            values = df.loc[(df["variant"] == variant) & (df["algorithm"] == algo), column]
            if values.empty:
                continue
            if dots:
                ax.plot(x + offset + rng.uniform(-0.05, 0.05, len(values)), values, linestyle="none",
                        marker="o", markersize=1.8, color=colour, alpha=0.22, markeredgewidth=0, zorder=2)
            xs.append(x + offset)
            means.append(values.mean())
        ax.plot(xs, means, linestyle="-" if join else "none", linewidth=0.7, marker=marker, color=colour,
                markersize=5.5, markeredgecolor="black", markeredgewidth=0.4, label=name, zorder=3)
    ax.set_xticks(range(len(variants)), variants)
    ax.set_xlim(-0.5, len(variants) - 0.5)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", which="both", length=0)


def plot_family(df, family, columns, out, filename):
    xlabel, variants, join = FAMILIES[family]
    sub = df[df["family"] == family]
    fig, axes = plt.subplots(1, len(columns), figsize=(4.3 * len(columns), 3.5), squeeze=False)
    for ax, column in zip(axes[0], columns):
        variant_panel(ax, sub, variants, column, ALGORITHMS, join, dots=column != "extra_per_city")
        ax.set_title(PANELS[column])
        if column == "extra_per_city":
            ax.set_ylabel("Extra visits per city")
        ax.set_xlabel(xlabel)
        if len(variants) > 4:
            ax.tick_params(axis="x", labelsize=8)
    axes[0][0].set_ylabel("Error above optimum (%)")
    legend(axes[0][0], "upper right" if family != "bridges" else "upper left")
    fig.tight_layout()
    fig.savefig(out / filename, dpi=200)


DENSITY_MODES = {"nearest": "Nearest-neighbour edges", "random": "Random edges"}


def load_density(path):
    """Density results with the ratio as a label. The complete-graph baseline of each
    instance is repeated under both modes as the last point, $K_n$."""
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["extra_per_city"] = df["extra_visits_two_opt"] / df["n"]
    sparse = df[df["mode"] != "complete"].copy()
    sparse["variant"] = sparse["ratio"].map("{:g}".format)
    ratios = [f"{r:g}" for r in sorted(sparse["ratio"].unique())]
    complete = df[df["mode"] == "complete"].assign(variant="$K_n$")
    parts = [sparse] + [complete.assign(mode=mode) for mode in DENSITY_MODES]
    return pd.concat(parts, ignore_index=True), ratios + ["$K_n$"]


def plot_density(path, out):
    """One row per edge mode: error of the constructed tour, error after 2-opt, and the
    cities the expanded walk passes through again."""
    df, variants = load_density(path)
    error = "Error above optimum (%)"
    columns = [("error", "constructed tour", error, True), ("error_two_opt", "after 2-opt", error, True),
               ("extra_per_city", "repeated visits after 2-opt", "Extra visits per city", False)]
    fig, axes = plt.subplots(len(DENSITY_MODES), len(columns), figsize=(4.6 * len(columns), 6.2),
                             sharex=True, squeeze=False)
    for row, (mode, mode_name) in zip(axes, DENSITY_MODES.items()):
        for ax, (column, title, ylabel, dots) in zip(row, columns):
            variant_panel(ax, df[df["mode"] == mode], variants, column, ALGORITHMS, True, dots)
            ax.set_title(f"{mode_name}: {title}")
            ax.set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel("Edge-to-vertex ratio $m/n$")
    legend(axes[0][0], "upper right")
    fig.tight_layout()
    fig.savefig(out / "4_density.png", dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default="results/property_results.csv")
    parser.add_argument("--index", default="data/properties/index.csv")
    parser.add_argument("--density", default="results/sparse_results.csv")
    parser.add_argument("--out")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = load(args.csv, args.index)
    out = Path(args.out) if args.out else Path("results/plots") / Path(args.csv).stem
    out.mkdir(parents=True, exist_ok=True)

    plot_family(df, "weights", ["error", "error_two_opt", "error_free", "extra_per_city"], out, "1_weights.png")
    plot_family(df, "bridges", ["error", "error_two_opt", "error_free", "extra_per_city"], out, "2_bridges.png")
    plot_family(df, "degrees", ["error", "error_two_opt", "extra_per_city"], out, "3_degrees.png")
    if Path(args.density).exists():
        plot_density(args.density, out)
    print(f"wrote plots to {out}/")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
