"""Generate sparse test graphs by removing edges from the TSPLIB instances.

Each graph keeps a planted Hamiltonian cycle (by default the instance's optimal tour,
so the published optimum stays valid) plus extra edges chosen by one of two modes:

  nearest  keep each city's shortest edges (road-network-like; deterministic)
  random   keep uniformly random edges (one graph per seed)

The edge-to-vertex ratio m/n is the controlled variable. Graphs go to data/sparse/
as <name>.edges files, listed with their statistics in data/sparse/index.csv.

    python generate_sparse.py
    python generate_sparse.py --ratios 1.2 1.5 2 4 8 --modes nearest --instances berlin52 kroA100
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from tsp.instance import load_dataset
from tsp.sparse import MODES, cycle_edges, load_opt_tour, nearest_edges, sparsify, target_edges, write_edges

INDEX_COLUMNS = [
    "name", "base", "mode", "ratio", "seed", "plant", "n", "m", "edge_vertex_ratio",
    "avg_degree", "min_degree", "max_degree", "planted_share", "forced_edges", "optimum", "file",
]


def degree_stats(n, edges):
    degree = Counter()
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return min(degree[v] for v in range(n)), max(degree.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.25, 1.5, 2, 3, 5, 10],
                        help="edge-to-vertex ratios m/n to generate (each >= 1)")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", type=int, default=3,
                        help="graphs per (instance, ratio) for anything random")
    parser.add_argument("--plant", choices=["optimal", "random"], default="optimal",
                        help="planted cycle: the optimal tour (keeps the optimum known; needs "
                             "data/opt_tours) or a random cycle (any instance; optimum unknown)")
    parser.add_argument("--instances", nargs="+", help="instance names (default: all usable)")
    parser.add_argument("--out", default="data/sparse")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for inst in load_dataset(args.instances):
        n, d = inst.n, inst.d
        if args.plant == "optimal":
            opt_tour = load_opt_tour(inst.name)
            if opt_tour is None:
                print(f"{inst.name:>10}  skipped: no optimal tour in data/opt_tours")
                continue

        for mode in args.modes:
            # Deterministic graphs need only one seed.
            seeds = range(args.seeds) if mode == "random" or args.plant == "random" else [0]
            for ratio in args.ratios:
                m = target_edges(n, ratio)
                if m is None:
                    print(f"{inst.name:>10}  {mode:>7}  ratio {ratio:g}: skipped, would be complete")
                    continue
                for seed in seeds:
                    if args.plant == "optimal":
                        planted, optimum = opt_tour, inst.optimum
                    else:
                        planted = list(range(n))
                        random.Random(f"{inst.name}-plant-{seed}").shuffle(planted)
                        optimum = None
                    rng = random.Random(f"{inst.name}-{mode}-{ratio:g}-{seed}")
                    edges = sparsify(d, ratio, mode, planted, rng)

                    # Planted edges the mode would not have picked by itself: a measure of
                    # how much the planted cycle shapes the graph.
                    forced = len(cycle_edges(planted) - nearest_edges(d, m)) if mode == "nearest" else ""

                    name = f"{inst.name}_{mode}_r{ratio:g}_s{seed}"
                    write_edges(out / f"{name}.edges", d, edges)
                    min_deg, max_deg = degree_stats(n, edges)
                    rows.append({
                        "name": name, "base": inst.name, "mode": mode, "ratio": f"{ratio:g}",
                        "seed": seed, "plant": args.plant, "n": n, "m": len(edges),
                        "edge_vertex_ratio": round(len(edges) / n, 4),
                        "avg_degree": round(2 * len(edges) / n, 3),
                        "min_degree": min_deg, "max_degree": max_deg,
                        "planted_share": round(n / len(edges), 4), "forced_edges": forced,
                        "optimum": optimum if optimum is not None else "", "file": f"{name}.edges",
                    })
                print(f"{inst.name:>10}  {mode:>7}  ratio {ratio:g}: {len(seeds)} graph(s), m = {m}")

    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} graphs to {out}/")


if __name__ == "__main__":
    main()
