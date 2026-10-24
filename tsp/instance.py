"""Loading TSP instances into a common distance-matrix representation."""

import csv
from dataclasses import dataclass
from pathlib import Path

import tsplib95

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Instance:
    """A symmetric TSP instance on cities 0..n-1."""

    name: str
    d: list
    optimum: float | None = None
    coords: list | None = None

    @property
    def n(self):
        return len(self.d)


def load_tsplib(path, optimum=None, name=None):
    """Load a TSPLIB .tsp file via tsplib95 into an Instance.

    tsplib95 applies TSPLIB's distance functions and rounding (EUC_2D, ATT,
    GEO, EXPLICIT, ...), so lengths match the published optima.
    """
    problem = tsplib95.load(str(path))
    if problem.type != "TSP":
        raise ValueError(f"{path}: expected a symmetric TSP, got {problem.type}")
    
    nodes = sorted(problem.get_nodes())
    n = len(nodes)
    d = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = problem.get_weight(nodes[i], nodes[j])
    
    coords = [tuple(problem.node_coords[v]) for v in nodes] if problem.node_coords else None
    return Instance(name or Path(path).stem, d, optimum, coords)


def load_dataset(names=None):
    """Load the instances listed in data/instances.csv into a list of Instances."""
    instances = []
    with open(DATA_DIR / "instances.csv") as f:
        for row in csv.DictReader(f):
            if names is not None and row["name"] not in names:
                continue
            optimum = float(row["optimum"]) if row["optimum"] else None
            path = DATA_DIR / "tsplib" / f"{row['name']}.tsp"
            instances.append(load_tsplib(path, optimum, row["name"]))
    return instances

