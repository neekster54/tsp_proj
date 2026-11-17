"""Helpers to vary structural properties of a graph"""
import math
import statistics
from collections import Counter

import networkx as nx

from tsp.sparse import cycle_edges, edge, random_edges

# key types of properties
WEIGHT_SCHEMES = ("unit", "flat", "original", "skewed")
DEGREE_KINDS = ("regular", "random", "preferential", "hub")
STAT_COLUMNS = [
    "avg_degree", "min_degree", "max_degree", "degree_variance", "degree_irregularity",
    "weight_min", "weight_max", "weight_mean", "weight_cv", "weight_ratio",
    "bridges", "cut_vertices", "blocks",
]

# helper functions to calculate and save graph information
def graph_stats(n, graph):
    """Degree, weight-spread and connectivity statistics of a graph."""
    degree = Counter()
    for u, v, _ in graph:
        degree[u] += 1
        degree[v] += 1
    degrees = [degree[v] for v in range(n)]
    weights = [w for _, _, w in graph]
    mean = statistics.fmean(weights)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((u, v) for u, v, _ in graph)
    if not nx.is_connected(g):
        raise ValueError("graph is not connected")

    return {
        "avg_degree": round(statistics.fmean(degrees), 3),
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "degree_variance": round(statistics.pvariance(degrees), 3),
        "degree_irregularity": sum(abs(degree[u] - degree[v]) for u, v, _ in graph),
        "weight_min": min(weights),
        "weight_max": max(weights),
        "weight_mean": round(mean, 3),
        "weight_cv": round(statistics.pstdev(weights) / mean, 4),
        "weight_ratio": round(max(weights) / min(weights), 3) if min(weights) > 0 else "",
        "bridges": sum(1 for _ in nx.bridges(g)),
        "cut_vertices": sum(1 for _ in nx.articulation_points(g)),
        "blocks": sum(1 for _ in nx.biconnected_components(g)),
    }


def with_weights(d, edges):
    return [(u, v, d[u][v]) for u, v in edges]


def write_graph(path, graph):
    with open(path, "w") as f:
        for u, v, w in sorted(graph):
            f.write(f"{u} {v} {w}\n")


# --- edge weight spread -----------------------------------------------------------
def reweight(d, edges, optimum, scheme, rng=None):
    """Reweight a sparse graph that contains the optimal tour of d to unit, flat, original or skewed.
    Returns (graph, optimum).
    """
    n = len(d)
    if scheme == "unit":
        return [(u, v, 1) for u, v in edges], n
    if scheme == "original":
        return with_weights(d, edges), optimum
    longest = max(max(row) for row in d)
    if scheme == "flat":
        return [(u, v, d[u][v] + longest) for u, v in edges], optimum + longest * n
    if scheme == "skewed":
        p = [round(4 * longest * (i / (n - 1)) ** 16) for i in range(n)]
        rng.shuffle(p)
        return [(u, v, d[u][v] + p[u] + p[v]) for u, v in edges], optimum + 2 * sum(p)
    raise ValueError(f"unknown weight scheme {scheme!r}, expected one of {WEIGHT_SCHEMES}")


# --- bridges and cut vertices -------------------------------------------------------
def chain_blocks(blocks, join, rng):
    """Join blocks into a chain. Returns (n, graph, extra), where extra is the weight
    added by the joins to one traversal of them.
    """
    n, graph, extra = 0, [], 0
    for size, block in blocks:
        ids = list(range(n, n + size))
        if n and join == "cut":
            shared = rng.randrange(size)
            ids = [n + i - (i > shared) for i in range(size)]
            ids[shared] = rng.randrange(n)
            n -= 1
        elif n and join == "bridge":
            w = rng.choice([w for _, _, w in graph] + [w for _, _, w in block])
            graph.append((rng.randrange(n), n + rng.randrange(size), w))
            extra += w
        elif n:
            raise ValueError(f"unknown join {join!r}, expected 'bridge' or 'cut'")
        graph += [(*edge(ids[u], ids[v]), w) for u, v, w in block]
        n += size
    return n, graph, extra


def add_pendants(n, graph, count, rng):
    """Attach count new degree-1 cities to random existing cities. Returns (n, graph, extra).
    """
    weights = [w for _, _, w in graph]
    graph, extra = list(graph), 0
    for leaf in range(n, n + count):
        w = rng.choice(weights)
        graph.append((rng.randrange(n), leaf, w))
        extra += w
    return n + count, graph, extra


# --- degree distribution ----------------------------------------------------------
def degree_edges(n, m, keep, kind, rng):
    """keep plus extra edges up to m in total, chosen to shape the degree distribution either to regular, random, 
    preferential or hub distributions to see the effects.
    """
    if kind == "random":
        return random_edges(n, m, keep, rng)
    edges = set(keep)
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def add(u, v):
        edges.add(edge(u, v))
        adj[u].add(v)
        adj[v].add(u)

    if kind == "regular":
        while len(edges) < m:
            # Lowest degree first, ties broken at random.
            order = sorted(range(n), key=lambda v: (len(adj[v]), rng.random()))
            u = next(u for u in order if len(adj[u]) < n - 1)
            add(u, next(v for v in order if v != u and v not in adj[u]))
    elif kind == "preferential":
        while len(edges) < m:
            u, v = rng.choices(range(n), weights=[len(a) for a in adj], k=2)
            if v != u and v not in adj[u]:
                add(u, v)
    elif kind == "hub":
        for hub in rng.sample(range(n), n):
            others = [v for v in range(n) if v != hub and v not in adj[hub]]
            rng.shuffle(others)
            for v in others:
                if len(edges) >= m:
                    return edges
                add(hub, v)
    else:
        raise ValueError(f"unknown degree kind {kind!r}, expected one of {DEGREE_KINDS}")
    return edges


def planted_length(graph, tour):
    """Length of the planted cycle in the graph (it must be present)."""
    weight = {(u, v): w for u, v, w in graph}
    return sum(weight[e] for e in cycle_edges(tour)) if len(tour) > 2 else math.inf