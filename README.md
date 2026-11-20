# Approximate TSP Algorithms: Performance and Graph Properties

This project compares classic approximation algorithms for the Travelling Salesman Problem (TSP) and investigates their performance on varying sparse graphs. 

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands from the repository root.

## Main Algorithms
| Algorithm | File | Complexity |
|---|---|---|
| Nearest Neighbour | `algorithms/nearest_neighbour.py` | O(n²) |
| Multi-start Nearest Neighbour | `algorithms/multi_start_nearest_neighbour.py` | O(n³) |
| Greedy Edge | `algorithms/greedy_edge.py` | O(n² log n) |
| Double Tree | `algorithms/double_tree.py` | O(n²) |
| Christofides–Serdyukov | `algorithms/christofides.py` | O(n² + k³) |
| Held–Karp | `algorithms/held_karp.py` | O(2ⁿ n²) |
| 2-opt | `algorithms/two_opt.py` | Local search improvement |
| 3-opt | `algorithms/three_opt.py` | Local search improvement |


## Experiment 1: Performance on Complete Instances

**Purpose:** Measure solution quality and runtime of each algorithm on complete, metric TSPLIB instances, with and without 2-opt/3-opt local search.

**Run:**

```bash
python run_evaluation.py --out results/my_results.csv
python plot.py results/my_results.csv
```


## Experiment 2: Graph Properties on Sparse Graphs

**Purpose:** Test how structural properties of a non-complete graph affect approximation quality. 

**Run:**

```bash
python run_sparse_evaluation.py --sparse-dir data/sparse     --out results/sparse_results.csv
python run_sparse_evaluation.py --sparse-dir data/properties --out results/property_results.csv

python plot_property_results.py
```
## Data

The TSPLIB instances and optimal tours are from the [TSPLIB library](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).
