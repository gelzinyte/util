import os
import numpy as np
import pytest
from pytest import approx

from ase.io import read, write
from ase.build import molecule
from quippy.potential import Potential

from wfl.calculators import generic
from wfl.configset import ConfigSet_out

import util.bde.generate

@pytest.fixture()
def atoms():
    at = molecule('CH4')
    # add fake data
    at.info['dft_opt_mol_positions_hash'] = 'fake_hash'
    at.arrays['dft_opt_positions'] = at.arrays['positions']
    at.info['dft_opt_dft_energy'] = -1100.314
    at.arrays['dft_opt_dft_forces'] = np.random.rand(len(at), 3)
    return at

@pytest.fixture()
def gap_filename():
    return os.path.join(ref_path(), 'files/', 'tiny_gap.xml')

def ref_path():
    return os.path.abspath(os.path.dirname(__file__))

def test_generate(atoms, gap_filename, tmp_path):

    dft_bde_fname = os.path.join(tmp_path, 'input_atoms.xyz')
    write(dft_bde_fname, [atoms, atoms])

    print(gap_filename)
    calculator = (Potential, [], {'param_filename':gap_filename})
    output_filename_prefix = f'{tmp_path}/out_bde_atoms_'
    dft_prop_prefix='dft_'
    gap_prop_prefix='gappy_'
    wdir=os.path.join(tmp_path, 'bde_wdir')

    util.bde.generate.everything(calculator=calculator,
                                 dft_bde_filename=dft_bde_fname,
                                 output_fname_prefix=output_filename_prefix,
                                 dft_prop_prefix=dft_prop_prefix,
                                 gap_prop_prefix=gap_prop_prefix,
                                 wdir=wdir)

    bde_atoms = read(output_filename_prefix + 'gap_bde.xyz', '-1')

    arrays_keys = list(bde_atoms.arrays.keys())
    info_keys = list(bde_atoms.info.keys())

    assert 'dft_opt_gappy_forces' in arrays_keys
    assert 'dft_opt_gappy_energy' in info_keys
    assert 'gappy_opt_positions' in arrays_keys
    assert 'gappy_opt_gappy_energy' in info_keys
    assert 'gappy_opt_gappy_forces' in arrays_keys
    assert 'gappy_opt_dft_energy' in info_keys
    assert 'gappy_opt_dft_forces' in arrays_keys

    # optimisation log
    if os.path.isfile('log.txt'):
        os.remove('log.txt')
    os.remove(output_filename_prefix + 'gap_bde.xyz')

def test_get_gap_isolated_h(gap_filename, tmp_path):

    calc = (Potential, [], {'param_filename':gap_filename})
    output_fname = os.path.join(tmp_path, 'gap_iso_h.xyz')

    util.bde.generate.gap_isolated_h(calculator=calc,
                                     dft_prop_prefix='dft_',
                                     gap_prop_prefix='gappy_',
                                     output_fname=output_fname)

    at = read(output_fname)

    assert approx(at.info['dft_energy']) == -13.5474497
    assert at.info['config_type'] == 'H'
    assert approx(at.info['gappy_energy']) == -13.547479








