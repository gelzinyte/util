import os
import pytest
from ase.io import read
import util.calculators.gap
from ase import Atoms
from ase.build import molecule
from util.util_config import Config


def ref_path():
    return os.path.abspath(os.path.dirname(__file__))


def test_gap():

    cfg = Config.load()
    scratch_dir = cfg['scratch_path']
    default_kw = Config.from_yaml(os.path.join(cfg['util_root'],
                                               'default_kwargs.yml'))
    default_kw['orca']['scratch_path'] = scratch_dir

    gap_fname = os.path.join(ref_path(), 'files/modified_gap.xml')

    orig_at = read(os.path.join(ref_path(),
                                'files/atoms_for_modified_gap.xyz'))

    mod_gap_calc = util.calculators.gap.PopGAP(gap_filename=gap_fname,
                                              orca_kwargs=default_kw['orca'],
                                              base_rundir='orca_base_rundir',
                              keep_files=['orca.inp', 'orca.out'])

    original_gap_energy = orig_at.info['gap_energy']
    at = Atoms(list(orig_at.symbols), positions=orig_at.positions)
    # at.set_calculator(mod_gap_calc)
    at.calc = mod_gap_calc
    # print(type(mod_gap_calc))
    # dir(mod_gap_calc)
    new_energy = at.get_potential_energy()

    assert pytest.approx(original_gap_energy) == new_energy


