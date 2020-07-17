#!/usr/bin/env python3

import click
from util import plot
import os

@click.command()
@click.option('--param_fname',  type=click.Path(exists=True), required=True, help='GAP xml to test')
# @click.option('--train_fname', type=click.Path(exists=True), help='.xyz file used for training. If not given, will be read from gap.xml')
@click.option('--test_fname', type=click.Path(exists=True), help='.xyz file to test GAP on')
@click.option('--output_dir', default='pictures', show_default=True, type=click.Path(), help='directory for figures. Create if not-existent')
@click.option('--prefix', help='prefix to label plots')
@click.option('--by_config_type', type=bool, help='if structures should be coloured by config_type in plots')
@click.option('--glue_fname', type=click.Path(exists=True), help='glue potential\'s xml to be evaluated for dimers')
@click.option('--plot_2b_contribution', type=bool, default='True', show_default=True, help='whether to plot the 2b only bit of gap')
@click.option('--plot_ref_curve', type=bool, default='True', show_default=True, help='whether to plot the reference DFT dimer curve')
@click.option('--isolated_atoms_fname',  default='xyzs/isolated_atoms.xyz', show_default=True, help='isolated atoms to shift glue')
@click.option('--ref_name', default='dft', show_default=True, help='prefix to \'_forces\' and \'_energy\' to take as a reference')
# TODO take this out maybe
@click.option('--dimer_scatter', help='dimer data in training set to be scattered on top of dimer curves')
def make_plots(param_fname, test_fname=None, output_dir=None, prefix=None, by_config_type=None, glue_fname=False, \
               plot_2b_contribution=True, plot_ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None):
    """Makes energy and force scatter plots and dimer curves"""
    # TODO get .xyz files from GAP xml file!!

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if test_fname is None and by_config_type!=False:
        by_config_type=True

    print('Scatter plotting')
    plot.make_scatter_plots_from_file(param_fname=param_fname, test_fname=test_fname, \
                       output_dir=output_dir, prefix=prefix, by_config_type=by_config_type, ref_name=ref_name)
    print('Ploting dimers')
    plot.make_dimer_curves(param_fnames=[param_fname],  output_dir=output_dir, prefix=prefix,\
                      glue_fname=glue_fname, plot_2b_contribution=plot_2b_contribution, plot_ref_curve=plot_ref_curve,\
                      isolated_atoms_fname=isolated_atoms_fname, ref_name=ref_name, dimer_scatter=dimer_scatter)



if __name__=='__main__':
    make_plots()
    print('\n\n-----------------CONGRATS YOU MADE IT')

