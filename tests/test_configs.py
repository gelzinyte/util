import os
from ase.io import read
from ase import Atoms
from util import configs


def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

def test_filter_insane_geometries(tmp_path):

    all_atoms = read(os.path.join(ref_path(), 'files', 'bad_geometries.xyz'), ':')

    bad_geometries_fname = os.path.join(tmp_path, 'filtered_bad_geometries.xyz')

    good_geometries = configs.filter_insane_geometries(atoms_list=all_atoms,
                                                       bad_structures_fname=bad_geometries_fname)

    bad_ats = read(bad_geometries_fname, ':')
    assert len(bad_ats) == 1
    assert len(good_geometries) == 1

