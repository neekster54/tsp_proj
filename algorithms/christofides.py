"""Christofides-Serdyukov 3/2-approximation."""

import networkx as nx

from tsp.graph import euler_circuit, prim_mst, shortcut


def odd_degree_vertices(n, edges):
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return [v for v in range(n) if degree[v] % 2 == 1]


def min_weight_perfect_matching(d, vertices):
    """Minimum weight perfect matching on the complete graph over `vertices`.

    Uses networkx's implementation of Edmonds' blossom algorithm, which
    maximises (1 + max w) - w subject to maximum cardinality; on a complete
    graph with an even number of vertices this is a minimum weight perfect
    matching. O(k^3) for k vertices.
    """
    g = nx.Graph()
    for a in range(len(vertices)):
        u = vertices[a]
        for b in range(a + 1, len(vertices)):
            v = vertices[b]
            g.add_edge(u, v, weight=d[u][v])
    matching = nx.min_weight_matching(g)
    if 2 * len(matching) != len(vertices):
        raise RuntimeError("matching is not perfect")
    return sorted(tuple(sorted(e)) for e in matching)


def christofides(d, start=0):
    """MST + min weight perfect matching on odd-degree vertices, Euler tour, shortcut.

    O(n^3) in the worst case because of the matching step. Length <= 3/2 * OPT
    when d is metric.
    """
    n = len(d)
    mst = prim_mst(d)
    matching = min_weight_perfect_matching(d, odd_degree_vertices(n, mst))
    return shortcut(euler_circuit(n, mst + matching, start))