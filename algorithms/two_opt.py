def two_opt_first(graph: list[list[float]], path: list[int], error: float = 1e-9) -> list[int]:
    """
    Implementation of First 2opt, applying swap immediately when a more favourable path is found
    """
    
    # NEED TO ADD ERROR CHECKING
    
    n = len(path)
    path = path.copy()
    moves = 0
    
    while True:
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1: # skip starting and ending vertex since not far enough
                    continue

                delta = (
                    -graph[path[i]][path[i + 1]]
                    -graph[path[j]][path[(j + 1) % n]]
                    +graph[path[i]][path[j]]
                    +graph[path[i + 1]][path[(j + 1) % n]]
                )

                if delta < -error:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    moves += 1
                    improved = True
                    break

            if improved:
                break

        if not improved:
            return path, moves
        
        
def two_opt_best(graph: list[list[float]], path: list[int], error: float = 1e-9) -> list[int]: 
    """
    Implementation of Best 2opt, applying swap only after you consider all possible permutations
    """
    
    # NEED TO ADD ERROR CHECKING
    
    n = len(path)
    path = path.copy()
    moves = 0

    while True:
        # track the best improvement instead of simply picking the first improvement
        best_delta = -error
        best_i = None
        best_j = None

        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue

                delta = (
                    -graph[path[i]][path[i + 1]]
                    -graph[path[j]][path[(j + 1) % n]]
                    +graph[path[i]][path[j]]
                    +graph[path[i + 1]][path[(j + 1) % n]]
                )

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i is None:
            return path, moves

        # apply the best move after checking ALL candidates.
        path[best_i + 1:best_j + 1] = reversed(
            path[best_i + 1:best_j + 1]
        )
        moves += 1

