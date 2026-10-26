import pyperf
from algorithms.nearest_neighbour import nearest_neighbour
from tsp.instance import load_tsplib
from tsp.tour import InvalidTour, tour_length, validate_tour

temp_test = [[0, 5, 7, 10], [5, 0, 2, 9], [7, 2, 0, 4], [10, 9, 4, 0]]

burma = load_tsplib("data/tsplib/burma14.tsp", optimum=3323, name="burma14")
print(burma.name)
print(burma.optimum)

res = nearest_neighbour(burma.d)
print(res)
print(validate_tour(burma.d, res))

# runner = pyperf.Runner() # testing out pyperf
# runner.bench_func("nearest_neighbour", nearest_neighbour, temp_test)