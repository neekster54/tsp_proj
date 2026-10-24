"""Tour validation and length computation. A tour lists n distinct cities, the start is not repeated and implicitly closed."""


class InvalidTour(ValueError):
    pass


def tour_length(d, tour):
    """Length of the closed tour, including the return edge."""
    raise NotImplementedError


def validate_tour(d, tour, claimed_length=None, tol=1e-6):
    """Check tour is a Hamiltonian cycle, and return its recomputed length."""
    raise NotImplementedError
