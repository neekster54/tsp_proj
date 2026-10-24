"""Loading TSP instances into a common distance-matrix representation."""

from dataclasses import dataclass
from pathlib import Path

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
    """Load a TSPLIB .tsp file via tsplib95 into an Instance."""
    raise NotImplementedError


def load_dataset(names=None):
    """Load the instances listed in data/instances.csv."""
    raise NotImplementedError

