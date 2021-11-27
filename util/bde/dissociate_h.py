import numpy as np

def dissociate_h(atoms, h_idx, c_idx):
    dist_range = np.arange(1, 10, 0.1) * atoms.get_distance(h_idx, c_idx)

    ats_out = []
    for dist in dist_range:
        at = atoms.copy()
        at.set_distance(c_idx, h_idx, dist, fix=0)
        ats_out.append(at)
    return ats_out