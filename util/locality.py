from ase.io import read, write
from util import configs
import matplotlib.pyplot as plt
import numpy as np
from util.qm import color_by_pop
from ase import neighborlist
from ase import Atom, Atoms
from ase.constraints import FixAtoms


def set_constraints(ci, co, cutoff):
    if co.is_done():
        return co.to_ConfigSet()
    
    for at in ci:
        at = set_constraint(at, cutoff)
        co.write(at)
    co.end_write()
    return co.to_ConfigSet()

def set_constraint(at, cutoff):
    at = at.copy()
    c_idx = at.info["rad_c_num"]
    all_c_distances = at.get_all_distances()[c_idx]
    idc_to_freeze = np.argwhere(all_c_distances < cutoff).flatten()
    constraint = FixAtoms(indices=idc_to_freeze)
    at.set_constraint(constraint=constraint)
    at.info["constraint_cutoff"] = cutoff
    return at


def find_rad_c(mol, rad_num):
    natural_cutoffs = neighborlist.natural_cutoffs(mol, mult=1.25)
    neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                            self_interaction=False,
                                            skin=0,
                                            bothways=True)
    _ = neighbor_list.update(mol)
    indices, offsets = neighbor_list.get_neighbors(rad_num)
    assert len(indices) == 1, f'got more than one neighbour ({len(indices)} neighbours) for the "radical hydrogen" '
    rad_c_num = indices[0]
    return rad_c_num


def find_rad_at_indices_byeond_cutoff(rad, r=None, bin=None):
    assert r is None or bin is None
    rad_c_num = rad.info["rad_c_num"]
    all_distances = rad.get_all_distances()
    rad_c_dists = all_distances[rad_c_num]
    
    if r is not None:
        return np.argwhere(rad_c_dists > r).flatten()
    if bin is not None:
        larger = list(np.argwhere(rad_c_dists > bin[0]).flatten())
        smaller = list(np.argwhere(rad_c_dists < bin[1]).flatten())
        return list(set(larger) & set(smaller)) 

def cleanup_info_arrays(at, keep_info=None, keep_arrays=None):
    if keep_info is None:
        keep_info = []
    if keep_arrays is None:
        keep_arrays = ["numbers", "positions"]

    del_info_keys = [key for key in at.info.keys() if key not in keep_info]
    del_prop_arrays_keys = [key for key in at.arrays.keys() if key not in keep_arrays]

    for del_info in del_info_keys:
        del at.info[del_info]
    for del_array in del_prop_arrays_keys:
        del at.arrays[del_array]


def get_prop_diff_beyond_cutoff(pair, prop_arrays_key, r=None, bin=None):
    pair = configs.into_dict_of_labels(pair, "mol_or_rad")
    assert len(pair["rad"]) == 1
    assert len(pair["mol"]) == 1
    rad = pair["rad"][0]
    mol = pair['mol'][0]
    indices_beyond_r = find_rad_at_indices_byeond_cutoff(rad, r=r, bin=bin)
    # print(indices_beyond_r)
    property_difference = get_property_difference(mol=mol, rad=rad, prop_arrays_key=prop_arrays_key)
    assert property_difference.shape[0] == len(rad)
    return property_difference[indices_beyond_r]


def get_property_difference(mol, rad, prop_arrays_key):
    rad_num = rad.info["rad_num"]
    rad_props = rad.arrays[prop_arrays_key]
    mol_props = mol.arrays[prop_arrays_key]
    mol_props = np.delete(mol_props, (rad_num), axis=0)
    return rad_props - mol_props

