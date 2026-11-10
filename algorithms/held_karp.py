"""Held-Karp exact dynamic programme, O(2^n n^2) time and O(2^n n) space."""

import math


def held_karp(d, start=0):
    n = len(d)
    if n == 0:
        raise ValueError("Graph must contain at least one city.")
    if not 0 <= start < n:
        raise ValueError(f"Start must be between 0 and {n - 1}.")

    others = [v for v in range(n) if v != start]
    m = len(others)
    if m == 0:
        return [start]
    dist = [[d[u][v] for v in others] for u in others]

    best = [math.inf] * (m << m)
    for j in range(m):
        best[(1 << j) * m + j] = d[start][others[j]]

    for mask in range(1, 1 << m):
        members = [j for j in range(m) if mask >> j & 1]
        if len(members) == 1:
            continue
        for j in members:
            prev = (mask ^ (1 << j)) * m
            best[mask * m + j] = min(
                best[prev + k] + dist[k][j] for k in members if k != j
            )

    # Walk backwards from the best final city
    mask = (1 << m) - 1
    j = min(range(m), key=lambda c: best[mask * m + c] + d[others[c]][start])
    path = [others[j]]
    while mask != 1 << j:
        mask ^= 1 << j
        j = min(
            (k for k in range(m) if mask >> k & 1),
            key=lambda k: best[mask * m + k] + dist[k][j],
        )
        path.append(others[j])

    path.append(start)
    return path[::-1]
