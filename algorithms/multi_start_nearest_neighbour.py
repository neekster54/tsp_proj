from algorithms.nearest_neighbour import nearest_neighbour
from tsp.tour import validate_tour


def multi_start_nearest_neighbour(graph: list[list[float]]) -> list[int]:
    """
    Runs nearest neighbour from every city and returns the shortest tour found.
    """

    n = len(graph)
    if n == 0:
        raise ValueError("Graph must contain at least one city.")

    best_path = None
    best_length = float("inf")

    for start in range(n):
        path = nearest_neighbour(graph, start)
        length = validate_tour(graph, path)
        if length < best_length:
            best_length = length
            best_path = path

    return best_path
