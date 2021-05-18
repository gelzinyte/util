import os
import numpy as np
from ase.io import read, write
from ase import Atoms
import pytest
from util.bde import table



def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

@pytest.fixture()
def fake_atoms():
    at_fname = os.path.join(ref_path(), 'files/bde_artificial_atoms.xyz')
    return read(at_fname, ':')

@pytest.fixture()
def fake_isolated_h():
    isolated_h = Atoms('H', positions=[(0, 0, 0)])
    isolated_h.info['dft_energy'] = 0.75
    isolated_h.info['gap_energy'] = 0.25

    return isolated_h


def test_bde_table(fake_atoms, fake_isolated_h):

    t = table.bde_table(fake_atoms, gap_prefix='gap_',
                            isolated_h=fake_isolated_h,
                            dft_prefix='dft_',
                            printing=True)

    expected_idx = ['H', 'mol', 'rad1', 'rad2']

    assert np.all(t.index == expected_idx)





