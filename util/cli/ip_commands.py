import click

from ase import Atoms
from ase.io import read, write

from quippy.potential import Potential

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import generic


@click.command('evaluate-gap')
@click.argument('config-file')
@click.option('--gap-fname', '-g', help='gap xml filename')
@click.option('--output-fname', '-o')
@click.option('--gap-prop-prefix', '-p',
              help='prefix for gap properties in xyz')
@click.option('--force', is_flag=True, help='for configset_out')
@click.option('--all_or_none', is_flag=True, help='for configset_out')
def evaluate_ip(config_file, gap_fname, output_fname,
                gap_prop_prefix, force, all_or_none):

    # logger.info('Evaluating GAP on DFT structures')


    calculator = (Potential, [], {'param_filename':gap_fname})

    inputs = ConfigSet_in(input_files=config_file)
    outputs_gap_energies = ConfigSet_out(output_files=output_fname,
                                         force=force,
                                         all_or_none=all_or_none)
    generic.run(inputs=inputs, outputs=outputs_gap_energies, calculator=calculator,
                properties=['energy', 'forces'], output_prefix=gap_prop_prefix)


# @click.command('eval-h')
# @click.argument('gap-fname')
# @click.option('--output', '-o')
# @click.option('--prop-prefix', '-p')
# def eval_h(gap_fname, output, prop_prefix):
#     from quippy.potential import Potential

#     gap = Potential(param_filename=gap_fname)
#     at = Atoms('H', positions=[(0, 0, 0)])
#     at.calc = gap
#     at.info[f'{prop_prefix}energy'] = at.get_potential_energy()
#     write(output, at)


@click.command('fit')
@click.option('--num-cycles', type=click.INT,
              help='number of gap_fit - optimise cycles')
@click.option('--train-fname', default='train.xyz', show_default=True,
              help='fname of the first training set file')
@click.option('--test-fname', default='test.xyz', show_default=True)
@click.option('--fit-param-fname', help='(base) parameters for fitting the potential')
@click.option('--all-smiles-csv', help='csv with ALL smiles to be added')
@click.option('--md-temp', type=click.FLOAT, default=500, show_default=True,
              help='temperature to run md at')
# @click.option('--energy-error-per-atom-threshold', type=click.FLOAT,
#               help='Threshold to accept predictions as accurate')
# @click.option("--energy-error-total-threshold", type=click.FLOAT,
#               help='Threshold to accept predictions as accurate')
# @click.option('--max-f-comp-error-threshold', type=click.FLOAT,
#               help='Threshold to accept predictions as accurate')
@click.option('--wdir', default='runs', show_default=True)
@click.option('--ref-type', default='dft', help='fit to dft data only for now')
@click.option('--ip-type', type=click.Choice(["ace"]), help='fit ACE only for now.')
@click.option('--bde-test-fname', help='xyz with configs to test bdes on')
@click.option('--soap-params-for-cur-fname', help='soap param filename')
@click.option('--num-train-env-per-cycle', default=10, type=click.INT, show_default=True)
@click.option('--num-test-env-per-cycle', default=10, type=click.INT, show_default=True)
@click.option('--num-extra-smiles-per-cycle', default=10, type=click.INT, show_default=True, 
              help="How many structures to generate each cycle")
def fit(num_cycles, train_fname, test_fname, fit_param_fname, all_smiles_csv, md_temp, 
        # energy_error_per_atom_threshold, energy_error_total_threshold, max_f_comp_error_threshold,
        wdir, ref_type, ip_type, bde_test_fname, soap_params_for_cur_fname,
        num_train_env_per_cycle, num_test_env_per_cycle, num_extra_smiles_per_cycle):

    import util.iterations.fit

    util.iterations.fit.fit(num_cycles=num_cycles,
                            base_train_fname=train_fname,
                            base_test_fname=test_fname,
                            fit_param_fname=fit_param_fname,
                            all_extra_smiles_csv=all_smiles_csv,
                            md_temp=md_temp,
                            # energy_error_per_atom_threshold=energy_error_per_atom_threshold,
                            # energy_error_total_threshold=energy_error_total_threshold,
                            # max_f_comp_error_threshold=max_f_comp_error_threshold,
                            wdir=wdir,
                            ref_type=ref_type,
                            ip_type=ip_type,
                            bde_test_fname=bde_test_fname,
                            soap_params_for_cur_fname=soap_params_for_cur_fname,
                            num_train_environments_per_cycle=num_train_env_per_cycle,
                            num_test_environments_per_cycle=num_test_env_per_cycle,
                            num_extra_smiles_per_cycle=num_extra_smiles_per_cycle)
