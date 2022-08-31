import click

from pathlib import Path

from ase import Atoms
from ase.io import read, write


from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic


@click.command('evaluate-gap')
@click.argument('config-file')
@click.option('--gap-fname', '-g', help='gap xml filename')
@click.option('--output-fname', '-o')
@click.option('--gap-prop-prefix', '-p',
              help='prefix for gap properties in xyz')
@click.option('--force', is_flag=True, help='for OutputSpec')
@click.option('--all_or_none', is_flag=True, help='for OutputSpec')
def evaluate_ip(config_file, gap_fname, output_fname,
                gap_prop_prefix, force, all_or_none):

    # logger.info('Evaluating GAP on DFT structures')


    from quippy.potential import Potential

    calculator = (Potential, [], {'param_filename':gap_fname})

    inputs = ConfigSet(input_files=config_file)
    outputs_gap_energies = OutputSpec(output_files=output_fname,
                                         force=force,
                                         all_or_none=all_or_none)
    generic.run(inputs=inputs, outputs=outputs_gap_energies, calculator=calculator,
                properties=['energy', 'forces'], output_prefix=gap_prop_prefix)


@click.command('fit')
@click.option('--num-cycles', type=click.INT,
              help='number of gap_fit - optimise cycles')
@click.option('--train-fname', default='train.xyz', show_default=True,
              help='fname of the first training set file')
@click.option('--validation-fname')
@click.option('--fit-param-fname', help='(base) parameters for fitting the potential')
@click.option('--all-smiles-csv', help='csv with ALL smiles to be added')
@click.option('--md-temp', type=click.FLOAT, default=500, show_default=True,
              help='temperature to run md at')
@click.option('--wdir', default='runs', show_default=True)
@click.option('--ref-type', default='dft', help='fit to dft data only for now')
@click.option('--ip-type', type=click.Choice(["ace"]), help='fit ACE only for now.')
@click.option('--num-extra-smiles-per-cycle', default=10, type=click.INT, show_default=True, 
              help="How many structures to generate each cycle")
@click.option('--num-rads-per-mol', default=0, type=click.INT, show_default=True)
@click.option('--md-steps', type=click.INT, default=2000)
@click.option('--cur-soap-params')
def fit(num_cycles, train_fname, fit_param_fname, all_smiles_csv, md_temp, 
        wdir, ref_type, ip_type, # bde_test_fname, 
        num_extra_smiles_per_cycle, num_rads_per_mol, validation_fname,
        md_steps, cur_soap_params):

    import util.iterations.fit

    util.iterations.fit.fit(num_cycles=num_cycles,
                            base_train_fname=train_fname,
                            validation_fname=Path(validation_fname),
                            fit_param_fname=fit_param_fname,
                            all_extra_smiles_csv=all_smiles_csv,
                            md_temp=md_temp,
                            wdir=wdir,
                            ref_type=ref_type,
                            ip_type=ip_type,
                            num_extra_smiles_per_cycle=num_extra_smiles_per_cycle,
                            num_rads_per_mol=num_rads_per_mol, 
                            cur_soap_params=cur_soap_params,
                            md_steps=md_steps)
