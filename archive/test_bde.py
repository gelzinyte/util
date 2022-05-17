from util import bde
import numpy as np
from util import configs
import shutil
from ase.io import read, write
import os
import pytest
from quippy.potential import Potential
from util import smiles
from wfl.configset import ConfigSet, ConfigSet_out




def get_gap_filename():
    gap_filename = os.path.join(ref_path(), 'files/', 'tiny_gap.xml')
    return gap_filename

def ref_path():
    return os.path.abspath(os.path.dirname(__file__))



def test_evaluate_gap_on_h(tmp_path):

    gap_filename = get_gap_filename()
    output_filename = os.path.join(tmp_path, 'evaluated_h.xyz')

    h = bde.evaluate_gap_on_h(gap_filename, gap_prop_prefix='tiny_gap_',
                              output_filename=output_filename)

    h_energy = h.info['tiny_gap_energy']
    expected_h_energy = -13.547479108102433
    assert h_energy == expected_h_energy

@pytest.fixture()
def molecule():
    smiles_str = 'C'
    name = 'methane'
    molecule = smiles.run_op(smiles_str)[0]
    molecule.info['config_type'] = name
    return molecule

@pytest.fixture()
def calculator():
    gap_filename = get_gap_filename()
    calc = (Potential, [], {'param_filename': gap_filename})
    return calc


def test_derive_bdes_paralelly(molecule,
                               calculator):

    molecules = [molecule] * 3
    outputs = ConfigSet_out()
    bde.gap_prepare_bde_structures_parallel(molecules,
                                            outputs,
                                            calculator=calculator,
                                            gap_prop_prefix='tiny_gap_',
                                            chunksize=3,
                                            run_dft=False)


    assert len(outputs.output_configs) == 15



@pytest.mark.dft
def test_gap_prepare_bde_structures_dft(molecule, calculator):



    atoms_out = bde.gap_prepare_bde_structures(molecules=molecule,
                                               calculator=calculator,
                                               gap_prop_prefix='tiny_gap_',
                                               run_dft=True)

    atoms_out = atoms_out[0]
    write('atoms_out.xyz', atoms_out)

    expected_dft_energies = [-1100.5313468904906, -1082.2281310870326,
                             -1082.228070840262, -1082.2281073895344,
                             -1082.2280986348424]

    expected_dft_energies_2 = [-1100.5314172674887, -1082.2281010634858,
                               -1082.2280474264426, -1082.2280495074876,
                               -1082.2281222707068]


    expected_dft_forces_0 = np.array([[3.104160e-03, -5.642670e-03, 3.372170e-03],
                                      [3.481110e-03, 3.467000e-05, -1.551680e-03],
                                      [7.838190e-03, 8.232020e-03, 7.995270e-03],
                                      [3.255130e-03, 3.232250e-03, -1.674914e-02],
                                      [-1.767860e-02, -5.856270e-03,
                                       6.933380e-03]])

    expected_dft_forces_0_2 =np.array([[ 0.00946767,  0.00112173, -0.00263947],
                                     [ 0.00563218,  0.00375014, -0.00640182],
                                     [-0.00948052, -0.01912858,  0.00065402],
                                     [ 0.00463663,  0.00160608,  0.00622369],
                                     [-0.01025595,  0.01265063,  0.00216357]])

    expected_dft_forces_1 = np.array([[6.36260e-04, 9.07000e-06, -7.90170e-03],
                                      [6.58440e-03, 8.62625e-03, 9.06377e-03],
                                      [-3.47565e-03, 1.25734e-03, -3.26164e-03],
                                      [-3.74501e-03, -9.89266e-03, 2.09957e-03]])

    expected_dft_forces_1_2 = np.array([[ 0.00638864,  0.00757466, -0.00741292],
                                         [-0.00476431, -0.01317372, -0.00301945],
                                         [ 0.00461214, -0.00136287,  0.01149152],
                                         [-0.00623648,  0.00696193, -0.00105916]])


    expected_orca_inputs = ['! engrad UKS B3LYP def2-SV(P) def2/J D3BJ \n',
                            '%scf Convergence Tight\n',
                            'SmearTemp 5000\n',
                            'maxiter 500\n',
                            'end \n']

    expected_arrays_entries = ['numbers', 'positions', 'tiny_gap_opt_forces',
                               'tiny_gap_opt_positions',
                               'tiny_gap_opt_dft_forces']

    dft_energies_on_gap_opt = [at.info['tiny_gap_opt_dft_energy'] for at in atoms_out]
    dft_forces_0 = atoms_out[0].arrays['tiny_gap_opt_dft_forces']
    dft_forces_1 = atoms_out[1].arrays['tiny_gap_opt_dft_forces']

    print(dft_forces_0)
    print(dft_forces_1)
    assert np.all(pytest.approx(dft_energies_on_gap_opt, abs=1e-6) == expected_dft_energies) or \
           np.all(pytest.approx(dft_energies_on_gap_opt, abs=1e-6) == expected_dft_energies_2)
    assert np.all(pytest.approx(dft_forces_0, abs=1e-6) == expected_dft_forces_0) or \
           np.all(pytest.approx(dft_forces_0, abs=1e-6) == expected_dft_forces_0_2)
    assert np.all(pytest.approx(dft_forces_1, abs=1e-6) == expected_dft_forces_1) or \
           np.all(pytest.approx(dft_forces_1, abs=1e-6) == expected_dft_forces_1_2)


    for at in atoms_out:
        assert np.all(list(at.arrays.keys()) == expected_arrays_entries)

    dirs = os.listdir('orca_outputs')
    input = os.path.join(ref_path(), 'orca_outputs', dirs[0], 'orca.inp')
    with open(input, 'r') as f:
        lines = f.readlines()

    for line, exp_line in zip(lines, expected_orca_inputs):
        assert line == exp_line

    outputs_dir = os.path.join(ref_path(), 'orca_outputs')
    if os.path.isdir(outputs_dir):
        shutil.rmtree(outputs_dir)


    atoms_fname =os.path.join(ref_path(), 'atoms_out.xyz')
    if os.path.isfile(atoms_fname):
        os.remove(atoms_fname)

    log_fname =os.path.join(ref_path(), 'log.txt')
    if os.path.isfile(log_fname):
        os.remove(log_fname)


def test_gap_prepare_bde_structures_no_dft(molecule, calculator):

    gap_filename = get_gap_filename()
    calculator = (Potential, [], {'param_filename':gap_filename})

    atoms_out = bde.gap_prepare_bde_structures(molecules=molecule,
                                               calculator=calculator,
                                               gap_prop_prefix='tiny_gap_',
                                               run_dft=False)

    atoms_out = atoms_out[0]

    hash = atoms_out[0].info['tiny_gap_opt_positions_hash']
    expected_hashes = [hash]*4
    expected_compound = ['methane'] * 5
    expected_config_types = ['methane_mol', 'methane_rad1', 'methane_rad2',
                              'methane_rad3', 'methane_rad4']
    expected_mol_or_rads = ['mol', 'rad1', 'rad2', 'rad3', 'rad4']
    expected_gap_energies = [-1100.5316088069112, -1082.228159077573,
                             -1082.228159077485, -1082.228159077686,
                             -1082.2281590775763]

    expected_minim_n_steps = [0, 7, 7, 7, 7]


    expected_arrays_entries = ['numbers', 'positions', 'tiny_gap_opt_forces',
                               'tiny_gap_opt_positions' ]

    config_types = [at.info['config_type'] for at in atoms_out]
    compounds = [at.info['compound'] for at in atoms_out]
    mol_or_rads = [at.info['mol_or_rad'] for at in atoms_out]
    gap_energies = [at.info['tiny_gap_opt_energy'] for at in atoms_out]
    minim_n_steps = [at.info['minim_n_steps'] for at in atoms_out]
    hashes_rad = [at.info['mol_tiny_gap_opt_positions_hash'] for at in atoms_out[1:]]

    assert np.all(config_types == expected_config_types)
    assert np.all(compounds == expected_compound)
    assert np.all(mol_or_rads == expected_mol_or_rads)
    assert np.all(pytest.approx(gap_energies) == expected_gap_energies)
    assert np.all(minim_n_steps == expected_minim_n_steps)
    assert atoms_out[0].info['tiny_gap_opt_positions_hash'] == hash
    assert np.all(hashes_rad == expected_hashes)

    for at in atoms_out:
        assert np.all(list(at.arrays.keys()) == expected_arrays_entries)


    atoms_fname =os.path.join(ref_path(), 'atoms_out.xyz')
    if os.path.isfile(atoms_fname):
        os.remove(atoms_fname)

    log_fname =os.path.join(ref_path(), 'log.txt')
    if os.path.isfile(log_fname):
        os.remove(log_fname)


def test_orca_kwargs():

    orca_kwargs = bde.setup_orca_kwargs()
    assert orca_kwargs['orcasimpleinput'] == 'UKS B3LYP def2-SV(P) def2/J D3BJ'
    assert orca_kwargs['orcablocks'] == '%scf Convergence Tight\nSmearTemp 5000\nmaxiter 500\nend'


@pytest.mark.dft
def test_orca_reopt():

    start = read(os.path.join(ref_path(), 'files/rattled_methane.xyz'), ':')
    inputs = ConfigSet(input_configs=start)
    outputs = ConfigSet_out()
    bde.dft_reoptimise(inputs, outputs, 'test_dft_')

    expected_positions = np.array([[ 0.08772669, -0.03961696, -0.04497313],
                                    [ 0.72546274,  0.65307911,  0.53049333],
                                    [-0.5315631 , -0.63244701,  0.6498379 ],
                                    [ 0.72372765, -0.71713214, -0.64012435],
                                    [-0.56700605,  0.53794892, -0.7198908 ]])

    expected_orca_entries = ['! opt UKS B3LYP def2-SV(P) def2/J D3BJ \n',
                             '%scf Convergence Tight\n',
                             'SmearTemp 5000\n']

    at = outputs.output_configs[0]
    assert at.info['test_dft_opt_energy'] == -1100.5313712727905
    assert np.all(pytest.approx(expected_positions) == at.arrays['test_dft_opt_positions'])


    dirs = os.listdir('orca_opt_outputs')
    input = os.path.join(ref_path(), 'orca_opt_outputs', dirs[0], 'orca.inp')
    with open(input, 'r') as f:
        lines = f.readlines()

    for line, exp_line in zip(lines, expected_orca_entries):
        assert line == exp_line

    opt_dir = os.path.join(ref_path(), 'orca_opt_outputs')
    if  os.path.exists(opt_dir):
        shutil.rmtree(opt_dir)





