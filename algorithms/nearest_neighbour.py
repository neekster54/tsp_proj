def nearest_neighbour(graph: list[list[float]], start: int = 0) -> list[int]:
    
    """
    Repeatedly finds the nearest unvisited city. Returns the path taken NOT including starting vertex.
    """
    
    n = len(graph)
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
                
if __name__ == "__main__":
    print(nearest_neighbour([[0, 5, 7, 10], [5, 0, 2, 9], [7, 2, 0, 4], [10, 9, 4, 0]])) # [0, 1, 2, 3]
    print(nearest_neighbour([[0, 5, 7, 10], [5, 0, 2, 1], [7, 2, 0, 4], [10, 1, 4, 0]])) # [0, 1, 3, 2]