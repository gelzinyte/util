import os
from ase.io import read
from ase import Atoms
from util import configs


def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

def test_filter_insane_geometries(tmp_path):

    all_atoms = read(os.path.join(ref_path(), 'files', 'bad_geometries.xyz'), ':')

    all_geometries = configs.filter_insane_geometries(atoms_list=all_atoms)

    good_geometries = all_geometries["good_geometries"]
    bad_geometries = all_geometries["bad_geometries"]

    assert len(bad_geometries) == 1
    assert len(good_geometries) == 1

