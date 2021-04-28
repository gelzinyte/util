from util import bde
import os
import pytest
from quippy.potential import Potential


def get_gap_filename():

    ref_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'files/')
    gap_filename = os.path.join(ref_path, 'tiny_gap.xml')
    return gap_filename

def test_evaluate_gap_on_h(tmp_path):

    gap_filename = get_gap_filename()
    output_filename = os.path.join(tmp_path, 'evaluated_h.xyz')

    h = bde.evaluate_gap_on_h(gap_filename, gap_prop_prefix='tiny_gap_',
                              output_filename=output_filename)

    h_energy = h.info['tiny_gap_energy']
    expected_h_energy = -13.547479108102433
    assert h_energy == expected_h_energy


def test_gap_prepare_bde_structures():

    smiles = 'C'
    name = 'methane'
    gap_filename = get_gap_filename()
    calculator = (Potential, [], {'param_filename':gap_filename})

    atoms_out = bde.gap_prepare_bde_structures(smiles_str=smiles, name=name,
                                               calculator=calculator,
                                               gap_prop_prefix='tiny_gap_')

    expected_hash = '180788212ca58d1cce92fe2e1369ffb1'
    expected_compound = 'methane'
    expected_configs_types = ['methane_mol', 'methane_rad1', 'methane_rad2',
                              'methane_rad3', 'methane_rad4']
    expected_mol_or_rads = ['mol', 'rad1', 'rad2', 'rad3', 'rad4']
    expected_gap_energies = [-1100.5316088069112, -1082.228159077573,
                             -1082.228159077485, -1082.228159077686,
                             -1082.2281590775763]


    config_types = [at.info['config_type'] for at in atoms_out]
    compounds = [at.info['compound'] for at in atoms_out]
    mol_or_rads = [at.info['mol_or_rad'] for at in atoms_out]
    gap_energies = [at.info['tiny_gap_opt_energy'] for at in atoms_out]
    dft_energies_on_gap_opt = [at.info['tiny_gap_opt_dft_energy'] for at in atoms_out]
    minim_n_steps = [at.info['minim_n_steps'] for at in atoms_out]

    print(f'config_types:\n{config_types}')
    print(f'compounds:\n{compounds}')
    print(f'mol_or_rads:\n{mol_or_rads}')
    print(f'gap_energies:\n{gap_energies}')
    print(f'dft_energies:\n{tiny_gap_opt_energy}')
    print(f'minim_steps:\n{minim_n_steps}')


    print(f'atoms.info')
    print(atoms_out[0].info)
    print(f'atoms.arrays.keys()')
    print(atoms_out[0].arrays.keys())

    assert False






