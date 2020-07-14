#!/usr/bin/env python3

import click
import os
from util import plot


@click.command()
@click.option('--train_fname',  type=click.Path(exists=True), required=True, \
              help='.xyz file with multiple config_types to evaluate against')
@click.option('--gaps_dir',  type=click.Path(exists=True), help='Directory that stores all GAP .xml files to evaluate')
@click.option('--output_dir', default='pictures', show_default=True, type=click.Path(), help='directory for figures. Create if not-existent')
@click.option('--prefix', help='prefix to label plots')
@click.option('--rmse', type=bool, default=True, show_default=True, help='Whether to plot rmse plots')
@click.option('--dimers', type=bool, default=True, show_default=True, help='Whether to plot dimer plots')
@click.option('--glue_fname', type=click.Path(exists=True), help='glue potential\'s xml to be evaluated for dimers')
@click.option('--plot_2b_contribution', type=bool, default='True', show_default=True, help='whether to plot the 2b only bit of gap')
@click.option('--plot_ref_curve', type=bool, default='True', show_default=True, help='whether to plot the reference DFT dimer curve')
@click.option('--isolated_atoms_fname',  default='xyzs/isolated_atoms.xyz', show_default=True, help='isolated atoms to shift glue')
@click.option('--ref_name', default='dft', show_default=True, help='prefix to \'_forces\' and \'_energy\' to take as a reference')
# TODO take this out maybe
@click.option('--dimer_scatter', help='dimer data in training set to be scattered on top of dimer curves')
def make_plots(train_fname, gaps_dir=None, output_dir=None, prefix=None, rmse=True, dimers=True, \
               glue_fname=False, plot_2b_contribution=True, plot_ref_curve=True, isolated_atoms_fname=None,\
               ref_name='dft', dimer_scatter=None):
    """Makes evaluates """

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if gaps_dir == None:
        gaps_dir = os.path.join(os.getcwd(), 'gaps')

    if rmse:
        print('Plotting RMSE plots')
        plot.rmse_plots(train_fname=train_fname, gaps_dir=gaps_dir, output_dir=output_dir, prefix=prefix)
    if dimers:
        print('Plotting dimer plots')
        print('WARNING the distributions will correspond to the given training file, which is not the same for all GAPs')
        dimer_summary_plots(gaps_dir=gaps_dir, train_fname=train_fname, output_dir=output_dir, prefix=prefix, \
                    glue_fname=glue_fname, plot_2b_contribution=plot_2b_contribution, \
                    plot_ref_curve=plot_ref_curve, isolated_atoms_fname=isolated_atoms_fname, \
                    ref_name=ref_name, dimer_scatter=dimer_scatter)

if __name__=='__main__':
    make_plots()
    # print('\n\n-----------------CONGRATS YOU MADE IT')

