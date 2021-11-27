import click

from quippy.potential import Potential

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import generic

@click.command('evaluate')
@click.argument('config-file')
@click.option('--gap-fname', '-g', help='gap xml filename')
@click.option('--output-fname', '-o')
@click.option('--gap-prop-prefix', '-p',
              help='prefix for gap properties in xyz')
@click.option('--force', is_flag=True, help='for configset_out')
@click.option('--all_or_none', is_flag=True, help='for configset_out')
def evaluate_gap(config_file, gap_fname, output_fname,
                                  gap_prop_prefix, force, all_or_none):

    # logger.info('Evaluating GAP on DFT structures')


    calculator = (Potential, [], {'param_filename':gap_fname})

    inputs = ConfigSet_in(input_files=config_file)
    outputs_gap_energies = ConfigSet_out(output_files=output_fname,
                                         force=force,
                                         all_or_none=all_or_none)
    generic.run(inputs=inputs, outputs=outputs_gap_energies, calculator=calculator,
                properties=['energy', 'forces'], output_prefix=gap_prop_prefix)


@click.command('eval-h')
@click.argument('gap-fname')
@click.option('--output', '-o')
@click.option('--prop-prefix', '-p')
def eval_h(gap_fname, output, prop_prefix):
    from quippy.potential import Potential

    gap = Potential(param_filename=gap_fname)
    at = Atoms('H', positions=[(0, 0, 0)])
    at.calc = gap
    at.info[f'{prop_prefix}energy'] = at.get_potential_energy()
    write(output, at)


@click.command('fit')
@click.option('--num-cycles', type=click.INT,
              help='number of gap_fit - optimise cycles')
@click.option('--train-fname', default='train.xyz', show_default=True,
              help='fname of first training set file')
@click.option('--gap-param-filename', help='yml with gap descriptor params')
@click.option('--smiles-csv', help='smiles to optimise')
@click.option('--num-smiles-opt', type=click.INT,
              help='number of optimisations per smiles' )
@click.option('--num-nm-displacements-per-temp', type=click.INT,
              help='number of normal modes displacements per structure per temperature')
@click.option('--num-nm-temps', type=click.INT, help='how many nm temps to sample from')
@click.option('--energy-filter-threshold', type=click.FLOAT,
              help='Error threshold (eV per *structure*) above which to '
                   'take structures for next iteration')
@click.option('--max-force-filter-threshold', type=click.FLOAT,
              help='Error threshold (eV/Å) for maximum force component '
                   'error for taking structures for next iteration')
@click.option('--ref-type', type=click.Choice(['dft', 'dft-xtb2']),
              default='dft',
              help='reference type: either fit directly to DFT '
                   'energies/forces or to DFT-XTB2. ')
@click.option('--traj-step-interval', type=click.INT, help='sampling for '
                   'optimisation trajectory, None takes the last config '
                                                           'only')
def fit(num_cycles, train_fname, gap_param_filename, smiles_csv,
        num_smiles_opt, num_nm_displacements_per_temp, num_nm_temps,
        energy_filter_threshold, max_force_filter_threshold,
        ref_type, traj_step_interval):

    import util.iterations.fit

    util.iterations.fit.fit(no_cycles=num_cycles,
                 given_train_fname=train_fname,
                 gap_param_filename=gap_param_filename,
                 smiles_csv=smiles_csv,
                 num_smiles_opt=num_smiles_opt,
                 num_nm_displacements_per_temp=num_nm_displacements_per_temp,
                 num_nm_temps=num_nm_temps,
                 energy_filter_threshold=energy_filter_threshold,
                 max_force_filter_threshold=max_force_filter_threshold,
                 ref_type=ref_type,
                 traj_step_interval=traj_step_interval)