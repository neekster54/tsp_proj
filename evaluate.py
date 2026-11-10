import statistics
import time

from algorithms.multi_start_nearest_neighbour import multi_start_nearest_neighbour
from algorithms.nearest_neighbour import nearest_neighbour
from algorithms.double_tree import double_tree
from algorithms.christofides import christofides
from algorithms.greedy_edge import greedy_edge
from algorithms.held_karp import held_karp
from algorithms.two_opt import two_opt_best
from algorithms.three_opt import three_opt_best
from tsp.instance import Instance, load_tsplib
from tsp.tour import validate_tour

# added all algorithms
TSP_ALGORITHMS = {
    "nearest_neighbour": nearest_neighbour,
    "multi_start_nn": multi_start_nearest_neighbour,
    "double_tree": double_tree,
    "greedy_edge": greedy_edge,
    "christofides": christofides,
    "held_karp": held_karp,
}

# Largest instance each algorithm is run on, unlisted algorithms have no limit.
MAX_CITIES = {"held_karp": 22}

def measure_time(fn, *args):
    """Simple timing function"""
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start

def evaluate(instance: Instance, algorithm: str, reps: int = 10, three_opt: bool = True):
    """
    Run TSP algorithm on data, then improve its tour with two opt and (optionally) three opt.
    Both start from the same TSP tour. Time as well as the tour quality and percentage error are evaluated.

    Three opt is cubic per pass, so pass three_opt=False to skip it on large instances.
    """
    tsp_algorithm = TSP_ALGORITHMS[algorithm]
    algo_times, opt_times, three_opt_times = [], [], []
    tour = improved_tour = moves = None
    three_opt_tour = three_opt_moves = None
    for _ in range(reps):
        tour, tsp_time = measure_time(tsp_algorithm, instance.d)
        (improved_tour, moves), two_opt_time = measure_time(two_opt_best, instance.d, tour)
        algo_times.append(tsp_time)
        opt_times.append(two_opt_time)
        if three_opt:
            (three_opt_tour, three_opt_moves), three_opt_time = measure_time(three_opt_best, instance.d, tour)
            three_opt_times.append(three_opt_time)

    length = validate_tour(instance.d, tour)
    improved_length = validate_tour(instance.d, improved_tour)
    if improved_length > length + 1e-9:
        raise AssertionError("2-opt increased the tour length")

    row = {
        "instance": instance.name,
        "n": instance.n,
        "reps": reps,
        "algorithm": algorithm,
        "optimum": instance.optimum,
        "length": length,
        "length_two_opt": improved_length,
        "two_opt_moves": moves,
        "algo_time": statistics.median(algo_times),
        "two_opt_time": statistics.median(opt_times),
        "total_time": statistics.median(a + b for a, b in zip(algo_times, opt_times)),
        "ratio": length / instance.optimum,
        "ratio_two_opt": improved_length / instance.optimum,
        "error": 100 * (length - instance.optimum) / instance.optimum,
        "error_two_opt": 100 * (improved_length - instance.optimum) / instance.optimum,
    }

    if three_opt:
        three_opt_length = validate_tour(instance.d, three_opt_tour)
        if three_opt_length > length + 1e-9:
            raise AssertionError("3-opt increased the tour length")

        row.update({
            "length_three_opt": three_opt_length,
            "three_opt_moves": three_opt_moves,
            "three_opt_time": statistics.median(three_opt_times),
            "total_time_three_opt": statistics.median(a + b for a, b in zip(algo_times, three_opt_times)),
            "ratio_three_opt": three_opt_length / instance.optimum,
            "error_three_opt": 100 * (three_opt_length - instance.optimum) / instance.optimum,
        })

    return row, tour, improved_tour, three_opt_tour

if __name__ == "__main__":
    burma = load_tsplib("data/tsplib/burma14.tsp", optimum=3323, name="burma14")
    print(burma.name)
    print(burma.optimum)
    row, tour, improved_tour, three_opt_tour = evaluate(burma, "nearest_neighbour")
    print(row)