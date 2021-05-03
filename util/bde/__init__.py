import os
import logging

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

logger = logging.getLogger(__name__)

cfg = Config.load()

def evaluate_gap_on_h(gap_filename, gap_prop_prefix, output_filename):
    """Evaluates isolted atom with gap needed to derive BDEs"""
    H = Atoms('H', positions=[(0, 0, 0)])
    gap = Potential(param_filename=gap_filename)
    H.calc = gap
    H.info[f'{gap_prop_prefix}energy'] = H.get_potential_energy()
    H.info['config_type'] = 'isolated_atom'
    write(output_filename, H)

    return H


def gap_prepare_bde_structures_parallel(molecules, outputs, calculator,
                                        gap_prop_prefix, chunksize=1,
                                        run_dft=True):
    """using iterable_loop"""
    return iterable_loop(iterable=molecules, configset_out=outputs,
                         op=gap_prepare_bde_structures, chunksize=chunksize,
                         gap_prop_prefix=gap_prop_prefix,
                         calculator=calculator,
                         run_dft=run_dft)


def gap_prepare_bde_structures(molecules, calculator, gap_prop_prefix,
                               run_dft=True):
    """takes in single molecule,  optimises with GAP,
    removes sp3 hydrogens, optimises with GAP"""

    if isinstance(molecules, Atoms):
        molecules = [molecules]

    configs_out = []

    for mol in molecules:
        # gap-optimise
        mol = gap_optimise(mol, calculator)


        # make radicals
        mol_and_rads = ConfigSet_out()
        radicals.abstract_sp3_hydrogen_atoms(inputs=mol,
                                outputs=mol_and_rads)


        # gap-optimise
        mol_and_rads = gap_optimise(mol_and_rads.to_ConfigSet_in(),
                                              calculator)



        # evaluate everyone with gap
        output_prefix = f'{gap_prop_prefix}opt_'
        mol_and_rads = generic.run_op(mol_and_rads,
                      calculator,
                      properties=['energy', 'forces'],
                      output_prefix=output_prefix)

        # process config_type, compound and mol_or_rad labels
        mol_and_rads = configs.process_config_info_on_atoms(mol_and_rads,
                                                            verbose=False)

        # label gap_optimised positions
        for at in mol_and_rads:
            at.arrays[f'{gap_prop_prefix}opt_positions'] = at.positions.copy()

        # assign molecule positions' hash
        mol_and_rads = assign_hash(mol_and_rads, gap_prop_prefix)

        if run_dft:
            # reevaluate with dft
            mol_and_rads = setup_evaluate_orca(mol_and_rads, gap_prop_prefix)

        configs_out.append(mol_and_rads)

    return configs_out


def dft_reoptimise(inputs, outputs, dft_prefix):
    """reoptimises all the structures given and puts them in appropriate
    atoms.info/arrays entries"""


    orca_kwargs = setup_orca_kwargs()
    orca_kwargs['task'] = 'opt'

    output_prefix = f'{dft_prefix}opt_'

    orca.evaluate(inputs, outputs, base_rundir='orca_opt_outputs',
                  orca_kwargs=orca_kwargs, output_prefix=output_prefix)

    atoms_out = []
    for at in outputs.to_ConfigSet_in():
        at.arrays[f'{dft_prefix}positions'] = at.positions.copy()
        atoms_out.append(at)

    return atoms_out


def gap_optimise(atoms, calculator):

    opt_kwargs = {'logfile':'log.txt', 'master':True, 'precon':None,
                    'use_armijo':False}

    atoms_opt_traj = minim.run_op(atoms, calculator, keep_symmetry=False,
                             update_config_type=False, **opt_kwargs)

    atoms_opt = [traj[-1] for traj in atoms_opt_traj]

    return atoms_opt


def assign_hash(mol_and_rads, gap_prop_prefix):

    assert mol_and_rads[0].info['mol_or_rad'] == 'mol'

    mol_hash = configs.hash_atoms(mol_and_rads[0])
    mol_and_rads[0].info[f'{gap_prop_prefix}opt_positions_hash'] = mol_hash

    for at in mol_and_rads[1:]:
        at.info[f'mol_{gap_prop_prefix}opt_positions_hash'] = mol_hash

    return mol_and_rads

def setup_orca_kwargs():

    default_kw = Config.from_yaml(
        os.path.join(cfg['util_root'], 'default_kwargs.yml'))

    smearing = default_kw['orca']['smearing']
    orcasimpleinput = default_kw['orca']['orcasimpleinput']
    orcablocks = f'%scf Convergence Tight\nSmearTemp {smearing}\nmaxiter ' \
                 f'500\nend'

    orca_kwargs = {'orcasimpleinput': orcasimpleinput,
                   'orcablocks': orcablocks}
    print(orca_kwargs)

    return orca_kwargs

def setup_evaluate_orca(mol_and_rads, gap_prop_prefix):

    orca_kwargs = setup_orca_kwargs()

    output_prefix = f'{gap_prop_prefix}opt_dft_'

    logger.info(f'Running ORCA with orca_kwargs {orca_kwargs}')

    mol_and_rads = orca.evaluate_op(mol_and_rads, base_rundir='orca_outputs',
                                    orca_kwargs=orca_kwargs,
                                    output_prefix=output_prefix)
    return mol_and_rads

