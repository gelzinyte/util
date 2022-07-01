from ase.io import read, write
from util import configs
import matplotlib.pyplot as plt
import numpy as np
from util.qm import color_by_pop
from ase import neighborlist
from ase import Atom, Atoms


def find_rad_c(mock_rad, rad_no):
    natural_cutoffs = neighborlist.natural_cutoffs(mock_rad, mult=1.2)
    neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                            self_interaction=False,
                                            skin=0,
                                            bothways=True)
    _ = neighbor_list.update(mock_rad)
    indices, offsets = neighbor_list.get_neighbors(rad_no)
    assert len(indices) == 1, f'got length of {len(indices)}'
    rad_c_num = indices[0]
    return rad_c_num

def find_idc_beyond_cutoff(mock_rad, rad_no, r):

    all_distances = mock_rad.get_all_distances()
    rad_c_num = find_rad_c(mock_rad, rad_no)
    rad_c_dists = all_distances[rad_c_num]

    return np.argwhere(rad_c_dists > r).flatten()

def cleanup_info_arrays(at, keep_info=None, keep_arrays=None):
    if keep_info is None:
        keep_info = []
    if keep_arrays is None:
        keep_arrays = ["numbers", "positions"]

    del_info_keys = [key for key in at.info.keys() if key not in keep_info]
    del_arrays_keys = [key for key in at.arrays.keys() if key not in keep_arrays]

    for del_info in del_info_keys:
        del at.info[del_info]
    for del_array in del_arrays_keys:
        del at.arrays[del_array]



def get_abs_excess_charge_beyond_cutoff(r, pair):
    pair = configs.into_dict_of_labels(pair, "mol_or_rad")

    rad = pair["rad"][0]
    mol = pair['mol'][0]
    rad_no = rad.info["rad_num"]
    mock_rad = prep_mock_rad(mol, rad, rad_no)
    indices_beyond_r = find_idc_beyond_cutoff(mock_rad, rad_no, r)
    if rad_no in indices_beyond_r:
        raise RuntimeError("Cutoff must be larger than the C-H bond")
    # mock_rad_charges = mock_rad.arrays["dft_NPA_charge"]
    # mol_charges = mol.arrays["dft_NPA_charge"]
    mock_rad_charges = mock_rad.arrays["dft_forces"]
    mol_charges = mol.arrays["dft_forces"]
    abs_charge_diff = np.abs(mol_charges - mock_rad_charges)
    return abs_charge_diff[indices_beyond_r]



def prep_mock_rad(mol, rad, rad_no):
    mock_rad = mol.copy()
    cleanup_info_arrays(mock_rad)
    # rad_charges = rad.arrays["dft_NPA_charge"]
    rad_charges = rad.arrays["dft_forces"]
    rad_charges = np.insert(rad_charges, rad_no, [0, 0, 0])
    # mock_rad.arrays["dft_NPA_charge"] = rad_charges
    mock_rad.arrays["dft_forces"] = rad_charges
    # mock_rad = color_by_pop(mock_rad, "dft_NPA_charge", cmap='seismic', vmin=-0.6, vmax=0.6)
    return mock_rad

