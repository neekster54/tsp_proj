import pyperf
from algorithms.nearest_neighbour import nearest_neighbour

temp_test = [[0, 5, 7, 10], [5, 0, 2, 9], [7, 2, 0, 4], [10, 9, 4, 0]]

runner = pyperf.Runner() # testing out pyperf
runner.bench_func("nearest_neighbour", nearest_neighbour, temp_test)