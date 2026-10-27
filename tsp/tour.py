"""Tour validation and length computation. A tour lists n distinct cities, the start is not repeated and implicitly closed."""


class InvalidTour(ValueError):
    pass


def tour_length(d, tour):
    """Length of the closed tour, including the return edge."""
    n = len(tour)
    return sum(d[tour[i]][tour[(i + 1) % n]] for i in range(n))


def validate_tour(d, tour):
    """Check tour is a Hamiltonian cycle, and return its length."""
    n = len(d)
    if len(tour) != n:
        raise InvalidTour(f"tour has {len(tour)} cities, expected {n}")
    if sorted(tour) != list(range(n)):
        raise InvalidTour("tour must visit each city exactly once")
    return tour_length(d, tour)

#test
if __name__ == "__main__":
    print(tour_length([[0, 1, 2], [1, 0, 3], [2, 3, 0]], [0, 1, 2]));
    print(validate_tour([[0, 1, 2], [1, 0, 3], [2, 3, 0]], [0, 1, 2]));
    print(validate_tour([[0, 1, 2], [1, 0, 3], [2, 3, 0]], [0, 1, 2, 3]));
    print(validate_tour([[0, 1, 2], [1, 0, 3], [2, 3, 0]], [0, 1, 3, 2]));