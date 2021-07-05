import numpy as np

def get_interatomic_distances(at):
    interatomic_distances = None
    return interatomic_distances


def get_max_interatomic_distances_displacement(at1, at2):
    """maximum difference between interatomic distances"""

    dists_1 = get_interatomic_distances(at1)
    dists_2 = get_interatomic_distances(at2)

    displacement = max(np.abs(dists_1 - dists_2))
    return displacement


