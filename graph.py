"""Graph routines shared by the tree-based algorithms."""

import math

def prim_mst(d):
    n = len(d)
    in_tree = [False] * n
    dist = [math.inf] * n
    parent = [-1] * n
    dist[0] = 0
    edges = []
    for _ in range(n):
        u = -1
        for v in range(n):
            if not in_tree[v] and (u == -1 or dist[v] < dist[u]):
                u = v
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u))
        row = d[u]
        for v in range(n):
            if not in_tree[v] and row[v] < dist[v]:
                dist[v] = row[v]
                parent[v] = u
    return edges

