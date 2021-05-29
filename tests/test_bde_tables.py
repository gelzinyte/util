import os
import numpy as np
from ase.io import read, write
from ase import Atoms
import pytest
from pytest import approx
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

    # can't really compare np.nan, so expecting pi instead
    t = t.fillna(np.pi)

    expected_idx = ['H', 'mol', 'rad1', 'rad2']

    assert np.all(t.index == expected_idx)

    assert approx(t.loc[:, 'energy_absolute_error']) == \
           [500.0, 2000.0, 1000.0, 3000.0]

    assert approx(t.loc[:, 'force_rmse']) == \
           [np.pi, 100, 100, 100]

    assert approx(t.loc[:, 'max_abs_force_error']) == \
           [np.pi, 100, 100, 100]

    # assert rmsd

    assert approx(t.loc[:, 'dft_energy_difference']) == \
        [np.pi, 14000, 12000, 16000]

    assert approx(t.loc[:, 'absolute_bde_error']) == \
        [np.pi, np.pi, 500, 1500]

    assert approx(t.loc[:, 'dft_bde']) == \
        [np.pi, np.pi, -1.25, 2.75]

    assert approx(t.loc[:, 'gap_bde']) == \
           [np.pi, np.pi, -0.75, 1.25]

    assert approx(t.loc[:, 'dft_opt_dft_energy']) == \
           [0.750, 24, 22, 26]

    assert approx(t.loc[:, 'dft_opt_gap_energy']) == \
           [np.pi, 20, 20, 20]


    assert approx(t.loc[:, 'gap_opt_gap_energy']) == \
           [0.250, 12, 11, 13]

    assert approx(t.loc[:, 'gap_opt_dft_energy']) == \
           [np.pi, 10, 10, 10]




    assert False



