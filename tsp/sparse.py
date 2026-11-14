"""Sparse (non-complete) graphs built from TSPLIB instances by removing edges.

Every sparse graph keeps a planted Hamiltonian cycle, so at least one tour always
exists. When the planted cycle is the instance's optimal tour, the published optimum
is still the optimum: removing edges can only make the best tour longer, and the
optimal tour survives.

Missing edges are stored as math.inf in the distance matrix.
"""

import csv
import heapq
import math
from pathlib import Path

import tsplib95

from tsp.instance import DATA_DIR, Instance

SPARSE_DIR = DATA_DIR / "sparse"
OPT_TOUR_DIR = DATA_DIR / "opt_tours"
MODES = ("nearest", "random")


def edge(u, v):
    return (u, v) if u < v else (v, u)


def cycle_edges(tour):
    n = len(tour)
    return {edge(tour[i], tour[(i + 1) % n]) for i in range(n)}


def load_opt_tour(name):
    """Optimal tour of a TSPLIB instance as 0-indexed cities, or None if there is no tour file."""
    path = OPT_TOUR_DIR / f"{name}.opt.tour"
    if not path.exists():
        return None
    nodes = sorted(tsplib95.load(str(DATA_DIR / "tsplib" / f"{name}.tsp")).get_nodes())
    index = {v: i for i, v in enumerate(nodes)}
    return [index[v] for v in tsplib95.load(str(path)).tours[0]]


def nearest_edges(d, m, keep=frozenset()):
    """Keep `keep`, then add every city's nearest-neighbour edge, then every city's
    second nearest, and so on until there are m edges. Within one round shorter edges
    go first, so the cut-off is not biased towards low city indices."""
    n = len(d)
    order = [sorted((v for v in range(n) if v != u), key=d[u].__getitem__) for u in range(n)]
    edges = set(keep)
    for r in range(n - 1):
        for _, e in sorted((d[u][order[u][r]], edge(u, order[u][r])) for u in range(n)):
            if len(edges) >= m:
                return edges
            edges.add(e)
    return edges


def random_edges(n, m, keep, rng):
    """Keep `keep`, then add uniformly random edges until there are m edges."""
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in keep]
    return set(keep) | set(rng.sample(candidates, m - len(keep)))


def target_edges(n, ratio):
    """Number of edges for an edge-to-vertex ratio, or None if the graph would be complete."""
    m = round(ratio * n)
    if m < n:
        raise ValueError(f"ratio {ratio} < 1: the planted cycle alone has n edges")
    return m if m < n * (n - 1) // 2 else None


def sparsify(d, ratio, mode, planted, rng=None):
    """Edge list (u < v) of a sparse graph with about ratio * n edges that contains `planted`."""
    n = len(d)
    m = target_edges(n, ratio)
    if m is None:
        raise ValueError(f"ratio {ratio} gives a complete graph on {n} cities")
    keep = cycle_edges(planted)
    if mode == "nearest":
        return sorted(nearest_edges(d, m, keep))
    if mode == "random":
        return sorted(random_edges(n, m, keep, rng))
    raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")


def metric_closure(d):
    """Shortest-path distances between every pair of cities, using only existing (finite) edges.

    Runs Dijkstra from every city, O(n m log n) in total. The result is a complete graph
    that satisfies the triangle inequality, so every algorithm and guarantee applies to it.
    Among equally short paths the one with the fewest edges wins, so a direct edge is
    never replaced by a detour of the same length.

    Returns (dist, pred), where pred[s][v] is the city before v on the chosen s-v path.
    """
    n = len(d)
    adj = [[(v, w) for v, w in enumerate(row) if v != u and w != math.inf] for u, row in enumerate(d)]
    dist, pred = [], []
    for s in range(n):
        ds, hops, ps = [math.inf] * n, [n] * n, [-1] * n
        ds[s], hops[s] = 0, 0
        heap = [(0, 0, s)]
        while heap:
            du, hu, u = heapq.heappop(heap)
            if (du, hu) > (ds[u], hops[u]):
                continue
            for v, w in adj[u]:
                key = (du + w, hu + 1)
                if key < (ds[v], hops[v]):
                    ds[v], hops[v], ps[v] = key[0], key[1], u
                    heapq.heappush(heap, (key[0], key[1], v))
        if math.inf in ds:
            raise ValueError("graph is not connected, so no route visits every city")
        dist.append(ds)
        pred.append(ps)
    return dist, pred


def expand_tour(tour, pred):
    """Turn a tour on the metric closure into a closed walk in the sparse graph by
    replacing each step u -> v with the shortest u-v path. The walk starts and ends at
    tour[0]; cities on the paths in between can appear more than once."""
    walk = [tour[0]]
    n = len(tour)
    for i in range(n):
        u, v = tour[i], tour[(i + 1) % n]
        path = []
        while v != u:
            path.append(v)
            v = pred[u][v]
        walk.extend(reversed(path))
    return walk


def walk_length(d, walk):
    """Length of a walk given as a list of cities (closed walks repeat the start at the end)."""
    return sum(d[walk[i]][walk[i + 1]] for i in range(len(walk) - 1))


def write_edges(path, d, edges):
    """One edge per line: u v weight (0-indexed cities)."""
    with open(path, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v} {d[u][v]}\n")


def load_sparse(path, n, name=None, optimum=None, base=None):
    """Load an edge file into an Instance whose missing edges are math.inf."""
    d = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        d[i][i] = 0
    with open(path) as f:
        for line in f:
            u, v, w = line.split()
            u, v, w = int(u), int(v), float(w)
            d[u][v] = d[v][u] = int(w) if w.is_integer() else w
    coords = None
    if base is not None:
        problem = tsplib95.load(str(DATA_DIR / "tsplib" / f"{base}.tsp"))
        coords = [tuple(problem.node_coords[v]) for v in sorted(problem.get_nodes())] or None
    return Instance(name or Path(path).stem, d, optimum, coords)


def load_sparse_dataset(directory=SPARSE_DIR, where=None):
    """Load the graphs listed in <directory>/index.csv as (metadata row, Instance) pairs.

    `where` optionally filters on the metadata row, e.g. lambda r: r["mode"] == "nearest".
    """
    directory = Path(directory)
    graphs = []
    with open(directory / "index.csv") as f:
        for row in csv.DictReader(f):
            if where is not None and not where(row):
                continue
            optimum = float(row["optimum"]) if row["optimum"] else None
            inst = load_sparse(directory / row["file"], int(row["n"]), row["name"], optimum, row["base"])
            graphs.append((row, inst))
    return graphs
