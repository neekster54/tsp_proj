def reconnection_deltas(graph: list[list[float]], path: list[int], i: int, j: int, k: int) -> list[float]:
    """
    Returns the change in tour length for each of the 7 ways to reconnect the tour
    after removing the edges after positions i, j and k.
    """

    n = len(path)
    a, b = path[i], path[i + 1]
    c, d = path[j], path[j + 1]
    e, f = path[k], path[(k + 1) % n]

    removed = graph[a][b] + graph[c][d] + graph[e][f]

    # each line is (the three edges in the new tour) - (the three edges removed)
    return [
        graph[a][c] + graph[b][d] + graph[e][f] - removed, # 0: a [c..b] [d..e] f, reverse S1
        graph[a][b] + graph[c][e] + graph[d][f] - removed, # 1: a [b..c] [e..d] f, reverse S2
        graph[a][e] + graph[d][c] + graph[b][f] - removed, # 2: a [e..d] [c..b] f, reverse S1 and S2 as one block
        graph[a][c] + graph[b][e] + graph[d][f] - removed, # 3: a [c..b] [e..d] f, reverse S1 and S2 separately
        graph[a][d] + graph[e][b] + graph[c][f] - removed, # 4: a [d..e] [b..c] f, swap S1 and S2
        graph[a][d] + graph[e][c] + graph[b][f] - removed, # 5: a [d..e] [c..b] f, swap, then reverse S1
        graph[a][e] + graph[d][b] + graph[c][f] - removed, # 6: a [e..d] [b..c] f, swap, then reverse S2
    ]


def reconnect(path: list[int], i: int, j: int, k: int, case: int) -> None:
    """
    Applies one of the 7 reconnections to the path in place.
    """

    s1 = path[i + 1:j + 1]
    s2 = path[j + 1:k + 1]

    options = [
        s1[::-1] + s2,       # 0: reverse S1
        s1 + s2[::-1],       # 1: reverse S2
        s2[::-1] + s1[::-1], # 2: reverse S1 and S2 as one block
        s1[::-1] + s2[::-1], # 3: reverse S1 and S2 separately
        s2 + s1,             # 4: swap S1 and S2
        s2 + s1[::-1],       # 5: swap, then reverse S1
        s2[::-1] + s1,       # 6: swap, then reverse S2
    ]

    path[i + 1:k + 1] = options[case]


def three_opt_first(graph: list[list[float]], path: list[int], error: float = 1e-9) -> tuple[list[int], int]:
    """
    Implementation of First 3opt, applying a move immediately when a more favourable path is found
    """

    n = len(path)
    if len(graph) == 0:
        raise ValueError("Graph must contain at least one city.")
    if len(graph) != n:
        raise ValueError("Path must contain every city exactly once.")

    path = path.copy()
    moves = 0

    while True:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # out of the 7 reconnections for this triple, take the best one
                    deltas = reconnection_deltas(graph, path, i, j, k)
                    delta = min(deltas)

                    if delta < -error:
                        reconnect(path, i, j, k, deltas.index(delta))
                        moves += 1
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            return path, moves


def three_opt_best(graph: list[list[float]], path: list[int], error: float = 1e-9) -> tuple[list[int], int]:
    """
    Implementation of Best 3opt, applying a move only after you consider all possible triples
    """

    n = len(path)
    if len(graph) == 0:
        raise ValueError("Graph must contain at least one city.")
    if len(graph) != n:
        raise ValueError("Path must contain every city exactly once.")

    path = path.copy()
    moves = 0

    while True:
        # track the best improvement instead of simply picking the first improvement
        best_delta = -error
        best_move = None

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    deltas = reconnection_deltas(graph, path, i, j, k)
                    delta = min(deltas)

                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j, k, deltas.index(delta))

        if best_move is None:
            return path, moves

        # apply the best move after checking ALL candidates.
        reconnect(path, *best_move)
        moves += 1
