import statistics
import time

from algorithms.nearest_neighbour import nearest_neighbour
from algorithms.two_opt import two_opt_best
from tsp.instance import Instance, load_tsplib
from tsp.tour import validate_tour

TSP_ALGORITHMS = {
    "nearest_neighbour": nearest_neighbour,
}

def measure_time(fn, *args):
    """Simple timing function"""
    start = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - start

def evaluate(instance: Instance, algorithm: str, reps: int = 10):
    """
    Run TSP and two opt algorithm on data. Time as well as the tour quality and percentage error are evaluated.
    """
    tsp_algorithm = TSP_ALGORITHMS[algorithm]
    algo_times, opt_times = [], []
    tour = improved_tour = moves = None
    for _ in range(reps):
        tour, tsp_time = measure_time(tsp_algorithm, instance.d)
        (improved_tour, moves), two_opt_time = measure_time(two_opt_best, instance.d, tour)
        algo_times.append(tsp_time)
        opt_times.append(two_opt_time)

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
    return row, tour, improved_tour

if __name__ == "__main__":
    burma = load_tsplib("data/tsplib/burma14.tsp", optimum=3323, name="burma14")
    print(burma.name)
    print(burma.optimum)
    row, tour, improved_tour = evaluate(burma, "nearest_neighbour")
    print(row)