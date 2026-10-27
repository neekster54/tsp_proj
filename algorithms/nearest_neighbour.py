def nearest_neighbour(graph: list[list[float]], start: int = 0) -> list[int]:
    
    """
    Repeatedly finds the nearest unvisited city. Returns the path taken NOT including starting vertex.
    """
    
    n = len(graph)
    if n == 0:
        raise ValueError("Graph must contain at least one city.")
    if not 0 <= start < n:
        raise ValueError(f"Start must be between 0 and {n - 1}.")    
    
    visited = [False] * n
    visited[start] = True
    path = [start]
    current = start
    
    for _ in range(n - 1): 
        best = -1 
        for neighbour in range(n):
            if not visited[neighbour] and (best == -1 or graph[current][neighbour] < graph[current][best]):
                best = neighbour
        
        visited[best] = True
        path.append(best)
        current = best
    
    return path