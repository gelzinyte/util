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

@pytest.fixture()
def ats_for_bde(atoms):
    atoms.info["dft_opt_mol_positions_hash"] = "fake_hash"
    atoms.info["compound"] = "fake_compound"
    return atoms



def test_get_atoms_by_hash_dict(ats_for_bde):

    at1 = ats_for_bde.copy()
    at1.info['mol_or_rad'] = "mol"
    at2 = ats_for_bde.copy()
    at2.info["mol_or_rad"] = "rad"

    atoms_by_hash = table.get_atoms_by_hash_dict([at1, at2], 'dft_')
    assert len(atoms_by_hash["fake_hash"]) == 2

    at1.info["mol_or_rad"] = "rad"
    with pytest.warns(UserWarning):
       atoms_by_hash = table.get_atoms_by_hash_dict([at1, at2], 'dft_')
    assert len(atoms_by_hash) == 0


def test_get_bde():
       fake_h_energy = 1
       fake_rad_energy = 5
       fake_mol_energy = 4
       bde = table.get_bde(mol_energy=fake_mol_energy,
                            rad_energy=fake_rad_energy,
                            isolated_h_energy=fake_h_energy)
       assert bde == 2


def test_assign_bde_info(ats_for_bde):
       
       at1 = ats_for_bde.copy()
       at1.info["mol_or_rad"] = "rad"
       at1.info["fake_energy"] = 5

       at2 = ats_for_bde.copy()
       at2.info["mol_or_rad"] = "mol"
       at2.info["fake_energy"] = 4

       fake_h_energy = 1

       ats_in = [at1, at2]

       table.assign_bde_info(all_atoms=ats_in, 
                                   h_energy=fake_h_energy, 
                                   prop_prefix='fake_', 
                                   dft_prop_prefix='dft_')

       # hasn't mixed up the order
       assert ats_in[0].info["mol_or_rad"] == "rad"
       # got the correct bde
       assert ats_in[0].info["fake_bde_energy"] == 2

def test_bde_table(fake_atoms, fake_isolated_h):

    t = table.bde_table(fake_atoms, pred_prop_prefix='gap_',
                            isolated_h=fake_isolated_h,
                            dft_prefix='dft_',
                            printing=True)

    # can't really compare np.nan, so expecting pi instead
    t = t.fillna(np.pi)

    expected_idx = ['H', 'mol', 'rad1', 'rad2']

    assert np.all(t.index == expected_idx)

    assert approx(t.loc[:, 'E_abs_at_err']) == \
           [500.0, 400.0, 250.0, 750.0]

    assert approx(t.loc[:, 'F_rmse']) == \
           [np.pi, 100, 100, 100]

    assert approx(t.loc[:, 'max_abs_F_err']) == \
           [np.pi, 100, 100, 100]

    # assert rmsd

    assert approx(t.loc[:, 'dft_E_diff']) == \
        [np.pi, 14000, 12000, 16000]

    assert approx(t.loc[:, 'abs_bde_err']) == \
        [np.pi, np.pi, 500, 1500]

    assert approx(t.loc[:, 'dft_bde']) == \
        [np.pi, np.pi, -1.25, 2.75]

    assert approx(t.loc[:, 'ip_bde']) == \
           [np.pi, np.pi, -0.75, 1.25]

    assert approx(t.loc[:, 'dft_opt_dft_E']) == \
           [0.750, 24, 22, 26]

    assert approx(t.loc[:, 'dft_opt_ip_E']) == \
           [np.pi, 20, 20, 20]


    assert approx(t.loc[:, 'ip_opt_ip_E']) == \
           [0.250, 12, 11, 13]

    assert approx(t.loc[:, 'ip_opt_dft_E']) == \
           [np.pi, 10, 10, 10]



