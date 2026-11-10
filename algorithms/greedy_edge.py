"""Greedy edge heuristic, O(n^2 log n) time and O(n^2) space."""

from tsp.graph import euler_circuit, shortcut


def find(parent, v):
    while parent[v] != v:
        parent[v] = parent[parent[v]]
        v = parent[v]
    return v


def greedy_edge(d, start=0):
    n = len(d)
    if n == 0:
        raise ValueError("Graph must contain at least one city.")
    if not 0 <= start < n:
        raise ValueError(f"Start must be between 0 and {n - 1}.")
    if n == 1:
        return [start]

    candidates = sorted((d[u][v], u, v) for u in range(n) for v in range(u + 1, n))
    parent = list(range(n))
    degree = [0] * n
    edges = []

    # Take the shortest edge that maintains a simple path.
    for _, u, v in candidates:
        if degree[u] == 2 or degree[v] == 2:
            continue
        root_u, root_v = find(parent, u), find(parent, v)
        if root_u == root_v:
            continue
        parent[root_u] = root_v
        degree[u] += 1
        degree[v] += 1
        edges.append((u, v))
        if len(edges) == n - 1:
            break

    closing = tuple(v for v in range(n) if degree[v] == 1)
    return shortcut(euler_circuit(n, edges + [closing], start))
