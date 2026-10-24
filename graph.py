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

def euler_circuit(n, edges, start=0):
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    used = [False] * len(edges)
    pos = [0] * n  # next unexamined entry of adj[v]
    stack = [start]
    circuit = []
    while stack:
        v = stack[-1]
        while pos[v] < len(adj[v]) and used[adj[v][pos[v]][1]]:
            pos[v] += 1
        if pos[v] == len(adj[v]):
            circuit.append(stack.pop())
        else:
            w, idx = adj[v][pos[v]]
            used[idx] = True
            stack.append(w)
    if len(circuit) != len(edges) + 1:
        raise ValueError("multigraph is not connected or has odd-degree vertices")
    return circuit[::-1]

def shortcut(walk):
    seen = set()
    tour = []
    for v in walk:
        if v not in seen:
            seen.add(v)
            tour.append(v)
    return tour
