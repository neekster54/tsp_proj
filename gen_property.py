"""
Generate sparse test graphs that vary one structural property at a time. Every graph is built from a TSPLIB instance with its optimal tour planted, and has an
exactly known graphical TSP optimum which can be compared to.
"""

import argparse
import csv
import random
from pathlib import Path

from tsp.instance import load_dataset
from tsp.properties import (
    DEGREE_KINDS, STAT_COLUMNS, WEIGHT_SCHEMES, add_pendants, chain_blocks, degree_edges,
    graph_stats, planted_length, reweight, with_weights, write_graph,
)
from tsp.sparse import MODES, cycle_edges, load_opt_tour, sparsify, target_edges

FAMILIES = ("weights", "bridges", "degrees")
INDEX_COLUMNS = [
    "name", "family", "variant", "base", "mode", "ratio", "seed", "plant", "n", "m",
    "edge_vertex_ratio", *STAT_COLUMNS, "optimum", "forced_cost", "file",
]


def weight_graphs(inst, tour, args):
    """Same edge set under different weighting distributions."""
    for mode in args.modes:
        for ratio in args.ratios:
            for seed in range(args.seeds) if mode == "random" else [0]:
                tag = f"{mode}-{ratio:g}-{seed}"
                edges = sparsify(inst.d, ratio, mode, tour, random.Random(f"{inst.name}-{tag}"))
                for scheme in args.weight_schemes:
                    rng = random.Random(f"{inst.name}-{tag}-{scheme}")
                    graph, optimum = reweight(inst.d, edges, inst.optimum, scheme, rng)
                    if planted_length(graph, tour) != optimum:
                        raise AssertionError(f"{inst.name} {scheme}: planted tour is not the optimum")
                    yield (f"{inst.name}_w-{scheme}_{mode}_r{ratio:g}_s{seed}", scheme, mode,
                           ratio, seed, inst.n, graph, optimum,
                           optimum - inst.optimum if scheme in ("flat", "skewed") else 0)


def bridge_graphs(inst, tour, args):
    """Chains of randomly sparsified copies of the instance, and copies with leaves to force traversal."""
    variants = [(f"{join}{k}", join, k) for join in ("bridge", "cut") for k in args.block_counts]
    variants += [(f"pendant{p}", "pendant", p) for p in args.pendant_percents]
    for ratio in args.ratios:
        for variant, join, size in variants:
            for seed in range(args.seeds):
                rng = random.Random(f"{inst.name}-{variant}-{ratio:g}-{seed}")
                copies = size if join != "pendant" else 1
                blocks = [(inst.n, with_weights(inst.d, sparsify(inst.d, ratio, "random", tour, rng)))
                          for _ in range(copies)]
                if join == "pendant":
                    n, graph, extra = add_pendants(inst.n, blocks[0][1], round(inst.n * size / 100), rng)
                else:
                    n, graph, extra = chain_blocks(blocks, join, rng)
                if n > args.max_n:
                    continue
                yield (f"{inst.name}_{variant}_r{ratio:g}_s{seed}", variant, "random", ratio, seed,
                       n, graph, copies * inst.optimum + 2 * extra, 2 * extra)


def degree_graphs(inst, tour, args):
    """Same n and m, with the extra edges placed to give different degree distributions."""
    for ratio in args.ratios:
        m = target_edges(inst.n, ratio)
        for kind in args.degree_kinds:
            for seed in range(args.seeds):
                rng = random.Random(f"{inst.name}-deg-{kind}-{ratio:g}-{seed}")
                edges = degree_edges(inst.n, m, cycle_edges(tour), kind, rng)
                yield (f"{inst.name}_deg-{kind}_r{ratio:g}_s{seed}", kind, kind, ratio, seed,
                       inst.n, with_weights(inst.d, sorted(edges)), inst.optimum, 0)


GENERATORS = {"weights": weight_graphs, "bridges": bridge_graphs, "degrees": degree_graphs}

# generate graphs fufilling arguments provided
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--instances", nargs="+", help="instance names (default: all with an optimal tour)")
    parser.add_argument("--ratios", type=float, nargs="+", default=[2],
                        help="edge-to-vertex ratios m/n (per block in the bridges family)")
    parser.add_argument("--seeds", type=int, default=2, help="graphs per random configuration")
    parser.add_argument("--max-n", type=int, default=280, help="skip graphs with more cities")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES),
                        help="weights family: how the edge set is chosen")
    parser.add_argument("--weight-schemes", nargs="+", choices=WEIGHT_SCHEMES, default=list(WEIGHT_SCHEMES))
    parser.add_argument("--block-counts", type=int, nargs="+", default=[2, 4],
                        help="bridges family: blocks per chain")
    parser.add_argument("--pendant-percents", type=int, nargs="+", default=[10, 25],
                        help="bridges family: leaves to attach, as a percentage of n")
    parser.add_argument("--degree-kinds", nargs="+", choices=DEGREE_KINDS, default=list(DEGREE_KINDS))
    parser.add_argument("--out", default="data/properties")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.edges"):
        old.unlink()
    rows = []

    for inst in load_dataset(args.instances):
        if inst.n > args.max_n:
            continue
        tour = load_opt_tour(inst.name)
        if tour is None:
            print(f"{inst.name:>10}  skipped: no optimal tour in data/opt_tours")
            continue
        if any(target_edges(inst.n, r) is None for r in args.ratios):
            print(f"{inst.name:>10}  skipped: a ratio would give a complete graph")
            continue
        counts = []
        for family in args.families:
            before = len(rows)
            for name, variant, mode, ratio, seed, n, graph, optimum, forced in GENERATORS[family](inst, tour, args):
                write_graph(out / f"{name}.edges", graph)
                rows.append({
                    "name": name, "family": family, "variant": variant, "base": inst.name,
                    "mode": mode, "ratio": f"{ratio:g}", "seed": seed, "plant": "optimal",
                    "n": n, "m": len(graph), "edge_vertex_ratio": round(len(graph) / n, 4),
                    **graph_stats(n, graph), "optimum": optimum, "forced_cost": forced, "file": f"{name}.edges",
                })
            counts.append(f"{family} {len(rows) - before}")
        print(f"{inst.name:>10}  {', '.join(counts)}")

    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} graphs to {out}/")


if __name__ == "__main__":
    main()