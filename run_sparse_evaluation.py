"""Compare the algorithms on sparse graphs, with and without metric closure.

For every graph in data/sparse/index.csv (see generate_sparse.py) and every algorithm:

  raw      run the algorithm on the sparse graph as is (missing edges = inf) and record
           whether its tour, and the tour after 2-opt, uses only edges that exist.
  closure  run it on the metric closure (shortest-path distances), apply 2-opt, expand
           the tour back into a closed walk in the sparse graph, and record its length,
           error against the optimum and the number of repeated city visits.

Each base instance is also run on its original complete graph (mode "complete") as a
baseline. Results go to results/sparse_results.csv and a summary table is printed.

    python run_sparse_evaluation.py
    python run_sparse_evaluation.py --max-n 1002 --algorithms christofides greedy_edge
"""

import argparse
import csv
import math
import statistics
import time
from collections import defaultdict

from algorithms.two_opt import two_opt_best
from evaluate import TSP_ALGORITHMS
from tsp.instance import load_dataset
from tsp.sparse import expand_tour, load_sparse_dataset, metric_closure, walk_length
from tsp.tour import tour_length, validate_tour

DEFAULT_ALGORITHMS = ["nearest_neighbour", "greedy_edge", "double_tree", "christofides"]
OUTPUT_COLUMNS = [
    "name", "base", "mode", "ratio", "seed", "n", "m", "edge_vertex_ratio", "algorithm", "optimum",
    "raw_valid", "raw_valid_two_opt", "closure_time", "length", "length_two_opt", "error",
    "error_two_opt", "extra_visits", "extra_visits_two_opt", "algo_time", "two_opt_time", "status",
]


def uses_only_real_edges(d, tour):
    n = len(tour)
    return all(d[tour[i]][tour[(i + 1) % n]] != math.inf for i in range(n))


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start


def closure_run(algorithm, sparse_d, dist, pred, optimum):
    """Run on the closure, 2-opt, expand to walks in the sparse graph, and check them."""
    tour, algo_time = timed(TSP_ALGORITHMS[algorithm], dist)
    (improved, _), two_opt_time = timed(two_opt_best, dist, tour)
    row = {"algo_time": algo_time, "two_opt_time": two_opt_time}
    for suffix, t in (("", tour), ("_two_opt", improved)):
        length = validate_tour(dist, t)
        walk = expand_tour(t, pred)
        if not math.isclose(walk_length(sparse_d, walk), length):
            raise AssertionError("expanded walk does not match the closure tour length")
        row[f"length{suffix}"] = length
        row[f"error{suffix}"] = 100 * (length - optimum) / optimum if optimum else ""
        row[f"extra_visits{suffix}"] = len(walk) - 1 - len(t)
    return row


def evaluate_graph(meta, inst, algorithms, closure):
    """Rows for one sparse graph. closure is (dist, pred, seconds), or None for a complete graph."""
    rows = []
    for algorithm in algorithms:
        row = {k: meta[k] for k in ("name", "base", "mode", "ratio", "seed", "n", "m", "edge_vertex_ratio")}
        row.update(algorithm=algorithm, optimum=inst.optimum)
        try:
            if closure is None:  # complete graph: the plain TSP, nothing to repair
                dist, pred, seconds = inst.d, None, 0.0
                row.update(raw_valid=True, raw_valid_two_opt=True)
            else:
                dist, pred, seconds = closure
                raw = TSP_ALGORITHMS[algorithm](inst.d)
                raw_improved, _ = two_opt_best(inst.d, raw)
                row.update(raw_valid=uses_only_real_edges(inst.d, raw),
                           raw_valid_two_opt=uses_only_real_edges(inst.d, raw_improved))
            row["closure_time"] = seconds
            if pred is None:
                tour, algo_time = timed(TSP_ALGORITHMS[algorithm], dist)
                (improved, _), two_opt_time = timed(two_opt_best, dist, tour)
                for suffix, t in (("", tour), ("_two_opt", improved)):
                    length = validate_tour(dist, t)
                    row[f"length{suffix}"] = length
                    row[f"error{suffix}"] = 100 * (length - inst.optimum) / inst.optimum
                    row[f"extra_visits{suffix}"] = 0
                row.update(algo_time=algo_time, two_opt_time=two_opt_time)
            else:
                row.update(closure_run(algorithm, inst.d, dist, pred, inst.optimum))
            row["status"] = "ok"
        except Exception as exc:
            row["status"] = f"failed: {exc!r}"
        rows.append(row)
    return rows


def summarise(rows):
    """Per mode and ratio: % raw tours valid, and closure error after 2-opt, per algorithm."""
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            groups[(r["mode"], float(r["edge_vertex_ratio"]) if r["mode"] == "complete" else float(r["ratio"]),
                    r["algorithm"])].append(r)
    algorithms = list(dict.fromkeys(r["algorithm"] for r in rows))
    for mode in ("nearest", "random", "complete"):
        keys = sorted({k[1] for k in groups if k[0] == mode})
        if not keys:
            continue
        print(f"\n{mode}: raw tours valid after 2-opt | closure error after 2-opt | extra visits per city")
        print(f"{'m/n':>7} " + "".join(f"{a[:18]:>30}" for a in algorithms))
        for ratio in (keys if mode != "complete" else [None]):
            cells = []
            for a in algorithms:
                rs = [r for k, g in groups.items() if k[0] == mode and k[2] == a
                      and (ratio is None or k[1] == ratio) for r in g]
                if not rs:
                    cells.append("-")
                    continue
                valid = 100 * sum(r["raw_valid_two_opt"] for r in rs) / len(rs)
                err = statistics.mean(r["error_two_opt"] for r in rs)
                extra = statistics.mean(r["extra_visits_two_opt"] / int(r["n"]) for r in rs)
                cells.append(f"{valid:4.0f}% | {err:5.2f}% | {extra:4.2f}")
            label = "all" if ratio is None else f"{ratio:g}"
            print(f"{label:>7} " + "".join(f"{c:>30}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", choices=list(TSP_ALGORITHMS), default=DEFAULT_ALGORITHMS)
    parser.add_argument("--max-n", type=int, default=280,
                        help="skip graphs with more cities (pr1002 takes over an hour)")
    parser.add_argument("--sparse-dir", default="data/sparse")
    parser.add_argument("--out", default="results/sparse_results.csv")
    args = parser.parse_args()

    graphs = load_sparse_dataset(args.sparse_dir, where=lambda r: int(r["n"]) <= args.max_n)
    bases = sorted({meta["base"] for meta, _ in graphs})
    rows = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for inst in load_dataset(bases):
            n = inst.n
            meta = {"name": inst.name, "base": inst.name, "mode": "complete", "ratio": "", "seed": "",
                    "n": n, "m": n * (n - 1) // 2, "edge_vertex_ratio": (n - 1) / 2}
            new = evaluate_graph(meta, inst, args.algorithms, None)
            writer.writerows(new)
            rows += new
            print(f"{inst.name:>32}  complete graph")

        for meta, inst in graphs:
            (dist, pred), seconds = timed(metric_closure, inst.d)
            new = evaluate_graph(meta, inst, args.algorithms, (dist, pred, seconds))
            writer.writerows(new)
            f.flush()
            rows += new
            failed = [r["algorithm"] for r in new if r["status"] != "ok"]
            print(f"{meta['name']:>32}  {'ok' if not failed else 'FAILED: ' + ', '.join(failed)}")

    summarise(rows)


if __name__ == "__main__":
    main()
