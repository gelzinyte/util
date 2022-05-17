
import os
from ase.io import read, write
import click
from wfl.calculators import orca

# @click.command()
# @click.option('--opt_xyzs_dir', default='/data/eg475/carbyne/optimised_structures')
# @click.option('--outputs_dir',
#               default='/data/eg475/carbyne/calculate/aligned_structures')
# @click.option('--output_fname', default='opt_aligned_with_energies.xyz')
def assign_energies(opt_xyzs_dir, outputs_dir, output_fname):

    opt_methods = ['triplet_opt', 'singlet_opt']
    structures = ['HC30H', 'HC31H', 'HC32H', 'HC33H', 'H2C30H2_flat',
                'H2C30H2_angle', 'H2C31H2_flat', 'H2C31H2_angle']

    eval_methods = ['dlpno-ccsd_cc-pvdz', 'uks_cc-pvdz']
    eval_multiplicities = ['1-let', '3-let']
    eval_mult_names = ['singlet', 'triplet']

    atoms_out = []

    for structure in structures:
        for opt_method in opt_methods:

            opt_fname = f'{structure}.{opt_method}.uks_def2-svp_opt.xyz'
            at = read(os.path.join(opt_xyzs_dir, opt_fname))

            at.info['opt_method'] = opt_method
            at.info['structure'] = structure
            at.info['ful_opt_fname'] = opt_fname

            for eval_method in eval_methods:
                for eval_mult, eval_mult_name in zip(eval_multiplicities,
                                                     eval_mult_names):

                    eval_dir_name = f'{structure}.' \
                                    f'{opt_method}.uks_def2-svp_opt'

                    orca_output_label = eval_dir_name \
                                       + f'.{eval_method}.{eval_mult_name}'

                    orca_output_label = os.path.join(outputs_dir,
                                                    eval_dir_name,
                                                    orca_output_label,
                                                     orca_output_label)

                    energy = read_energy(orca_output_label)

                    at.info[f'{eval_method}.{eval_mult_name}_energy'] = \
                        energy

            atoms_out.append(at)
    write(output_fname, atoms_out)


def read_energy(out_label):
    calc = orca.ORCA()
    calc.label = out_label
    calc.read_energy()
    return calc.results['energy']

