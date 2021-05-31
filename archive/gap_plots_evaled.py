#!/usr/bin/env python3

import click
from util.plot import rmse_scatter_evaled
from util import plot
import os

@click.command()
@click.option('--ref_energy_name', type=str)
@click.option('--pred_energy_name', type=str)
@click.option('--ref_force_name', type=str)
@click.option('--pred_force_name', type=str)
@click.option('--evaluated_train_fname', type=click.Path(exists=True))
@click.option('--evaluated_test_fname', type=click.Path(exists=True))
@click.option('--output_dir', default='pictures', show_default=True, type=click.Path(), help='directory for figures. Create if not-existent')
@click.option('--prefix', help='prefix to label plots')
@click.option('--by_config_type', type=bool, help='if structures should be coloured by config_type in plots')
@click.option('--force_by_element', default=True, type=bool, help='whether to evaluate force on each element separately')
def make_plots(ref_energy_name, pred_energy_name, ref_force_name, pred_force_name, evaluated_train_fname, evaluated_test_fname,
               output_dir, prefix, by_config_type, force_by_element):
    """Makes energy and force scatter plots and dimer curves"""

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if evaluated_test_fname is None and by_config_type!=False:
        by_config_type=True

    print('Scatter plotting')
    rmse_scatter_evaled.make_scatter_plots_from_evaluated_atoms(ref_energy_name=ref_energy_name, pred_energy_name=pred_energy_name,
    ref_force_name=ref_force_name, pred_force_name=pred_force_name, evaluated_train_fname=evaluated_train_fname,
    evaluated_test_fname=evaluated_test_fname,  output_dir=output_dir, prefix=prefix, by_config_type=by_config_type, force_by_element=force_by_element)


if __name__=='__main__':
    make_plots()

