from ase import Atoms
from ase.io import read, write

from quippy.potential import Potential

from wfl.generate_configs import minim
from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import generic
from wfl.calculators import orca
from wfl.pipeline import iterable_loop

from util import smiles  # TODO change to wfl smiles once merged
from util import radicals  # TODO change to wfl once merged
from util import configs
from util.util_config import Config

cfg = Config.load()

def evaluate_gap_on_h(gap_filename, gap_prop_prefix, output_filename):
    """Evaluates isolted atom with gap needed to derive BDEs"""
    H = Atoms('H', position=[(0, 0, 0)])
    gap = Potential(param_filename=gap_filename)
    H.calc = gap
    H.info[f'{gap_prop_prefix}energy'] = H.get_potential_energy()
    write(output_filename, H)

    return H


def gap_prepare_bde_structures_parallel(smiles, outputs, calculator,
                                        gap_prop_prefix, chunksize=1):
    """using iterable_loop"""
    return iterable_loop(iterable=smiles, configset_out=outputs,
                         op=gap_prepare_bde_structures, chunksize=chunksize)
    

def gap_prepare_bde_structures(smiles, name, calculator, gap_prop_prefix):
    """takes in single smiles, makes 3D structure, optimises with GAP,
    removes sp3 hydrogens, optimises with GAP"""

    # make molecule
    mol = smiles.run_op(smiles)[0]
    mol.info['config_type'] = name

    # gap-optimise
    mol = gap_optimise(mol, calculator)

    # make radicals
    mol_and_rads = ConfigSet_out()
    radicals.abstract_sp3_hydrogen_atoms(inputs=mol,
                            outputs=mols_and_rads)

    # gap-optimise
    mol_and_rads = gap_optimise(mol_and_rads.to_ConfigSet_in(),
                                          calculator)

    # evaluate everyone with gap
    output_prefix = f'{gap_prop_prefix}_opt_'
    mol_and_rads = generic.run_op(mol_and_rads,
                  calculator,
                  properties=['energy', 'forces'],
                  output_prefix=output_prefix)

    # process config_type, compound and mol_or_rad labels
    mol_and_rads = configs.process_config_info_on_atoms(mol_and_rads,
                                                        verbose=False)

    # label gap_optimised positions
    for at in mol_and_rads:
        at.arrays[f'{gap_prop_prefix}_opt_positions'] = at.positions.copy()

    # assign molecule positions' hash
    assert mol_and_rads[0].info['mol_or_rad'] == 'mol'
    mol_hash = configs.hash_atoms(mol_and_rads[0])
    mol_and_rads[0].info[f'{gap_proP_prefix}_opt_positions_hash'] = mol_hash
    for at in mol_and_rads[1:]:
        at.info[f'mol_{gap_prop_prefix}_opt_positions_hash'] = mol_hash

    # reevaluate with dft
    default_kw = Config.from_yaml(os.path.join(cfg['util_root'], 'default_kwargs.yml'))
    smearing = default_kw['orca']['smearing']
    orcasimpleinput = default_kw['orca']['orcasimpleinput']
    orcablocks=f'%scf Convergence Tight\nSmearTemp {smearing}\nmaxiter 500\nend'
    orca_kwargs = {'orcasimpleinput':orcasimpleinput,
                   'orcablocks':orcablocks}

    output_prefix = f'{gap_prop_prefix}_opt_dft_'
    mol_and_rads = orca.evalute_op(atoms, base_rundir='orca_outputs',
                                   orca_kwargs=orca_kwargs,
                                   output_prefix=output_prefix)

    return mol_and_rads


def dft_reoptimise(inputs, outputs, dft_prefix):
    """reoptimises all the structures given and puts them in appropriate
    atoms.info/arrays entries"""

    default_kw = Config.from_yaml(
        os.path.join(cfg['util_root'], 'default_kwargs.yml'))
    smearing = default_kw['orca']['smearing']
    orcasimpleinput = default_kw['orca']['orcasimpleinput']
    orcablocks = f'%scf Convergence Tight\nSmearTemp {smearing}\nmaxiter 500\nend'


    orca_kwargs = {'orcasimpleinput': orcasimpleinput,
                   'orcablocks': orcablocks,
                   'task':'opt'}

    output_prefix = f'{dft_prefix}_opt_'

    orca.evaluate(inputs, outputs, base_rundir='orca_opt_outputs',
                  orca_kwargs=orca_kwargs, output_prefix=output_prefix)

    atoms_out = []
    for at in outputs.to_ConfigSet_in():
        at.arrays[f'{dft_prefix}_opt_positions'] = at.positions.copy()
        atoms_out.append(at)

    return atoms_out


def gap_optimise(atoms, calculator):

    opt_kwargs = {'logfile':'log.txt', 'master':True, 'precon':None,
                    'use_armijo':False}

    atoms_opt_traj = minim.run_op(atoms, calculator, keep_symmetry=False,
                             update_config_type=False, **opt_kwargs)

    atoms_opt = [traj[-1] for traj in atoms_opt_traj]

    return atoms_opt
