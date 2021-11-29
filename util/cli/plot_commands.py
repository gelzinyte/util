import click
import os

from pathlib import Path

from ase.io import read, write

@click.command('data-summary')
@click.argument('in-fname')
@click.option('--fig-prefix', '-p', default='dataset_summary')
@click.option('--isolated_at_fname', '-i')
@click.option('--cutoff', '-c', default=6.0, type=click.FLOAT,
              help='cutoff for counting distances')
@click.option('--info', default='config_type', help='label to color by')
@click.option('--prop-prefix', default='dft_', type=click.STRING)
def plot_dataset(in_fname, fig_prefix, isolated_at_fname, cutoff,
                 info, prop_prefix):

    from util.plot import dataset

    atoms = read(in_fname, ':')

    if prop_prefix is None:
        prop_prefix=""

    fname_stem = os.path.splitext(os.path.basename(in_fname))[0]

    if cutoff == 0:
        cutoff = None

    isolated_ats = None
    if isolated_at_fname:
        isolated_ats = read(isolated_at_fname, ':')

    if fig_prefix:
        title_energy = f'{fig_prefix}_{fname_stem}_by_{info}_energies'
        title_forces = f'{fig_prefix}_{fname_stem}_by_{info}_forces'

    dataset.energy_by_idx(atoms, title=title_energy,
                          isolated_atoms=isolated_ats,
                          info_label=info, prop_prefix=prop_prefix)
    dataset.forces_by_idx(atoms, title=title_forces, info_label=info,
                          prop_prefix=prop_prefix)



@click.command('error-table')
@click.pass_context
@click.argument("inputs", nargs=-1)
@click.option("--ref_prefix", "-r", help='Prefix to "energy" and "forces" to take as reference')
@click.option("--pred_prefix", "-p",
              help='Prefix to label calculator-predicted energies and forces')
@click.option("--calc_kwargs", "--kw", help='quippy.potential.Potential keyword arguments')
@click.option("--output_fname", "-o",
              help="where to save calculator-evaluated configs for quick re-plotting")
@click.option("--chunksize", type=click.INT, default=500, help='For calculators.generic.run')
def plot_error_table(ctx, inputs, ref_prefix, pred_prefix, calc_kwargs, output_fname, chunksize):
    """Prints force and energy error by config_type"""

    from util import error_table

    inputs = ConfigSet_in(input_files=inputs)

    if calc_kwargs is not None:

        from quippy.potential import Potential
        calc = (Potential, [], key_val_str_to_dict(calc_kwargs))
    else:
        calc = None

    fnames = [fname for fname, idx in inputs.input_files]
    if len(fnames) == 1:
        fnames = fnames[0]
    print('-' * 60)
    print(fnames)
    print('-' * 60)

    error_table.plot(data=inputs, ref_prefix=ref_prefix, pred_prefix=pred_prefix, calculator=calc,
                     output_fname=output_fname, chunksize=chunksize)

    @click.command('error-scatter')
    @click.argument('atoms-filename')
    @click.option('--ref-energy-name', '-re', type=str)
    @click.option('--pred-energy-name', '-pe', type=str)
    @click.option('--ref-force-name', '-rf', type=str)
    @click.option('--pred-force-name', '-pf', type=str)
    @click.option('--output-dir', default='.', show_default=True,
                  type=click.Path(),
                  help='directory for figures. Create if not-existent')
    @click.option('--prefix', '-p', help='prefix to label plots')
    @click.option('--info-label',
                  help='info entry to label by')
    @click.option('--isolated-at-fname')
    @click.option('--total-energy', '-te', 'energy_type',
                  flag_value='total_energy',
                  help='plot total energy, not binding energy per atom')
    @click.option('--binding-energy', '-be', 'energy_type', default=True,
                  flag_value='binding_energy', help='Binding energy per atom')
    @click.option('--mean-shifted-energy', '-sft', 'energy_shift',
                  is_flag=True,
                  help='shift energies by the mean. ')
    @click.option('--no-legend', is_flag=True,
                  help='doesn\'t plot the legend')
    @click.option('--error-type', default='rmse')
    def make_plots(ref_energy_name, pred_energy_name, ref_force_name,
                   pred_force_name,
                   atoms_filename,
                   output_dir, prefix, info_label, isolated_at_fname,
                   energy_type, energy_shift, no_legend, error_type):
        """Makes energy and force scatter plots and dimer curves"""

        from util.plot import rmse_scatter_evaled

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        all_atoms = read(atoms_filename, ':')
        if energy_type == 'binding_energy':

            if isolated_at_fname is not None:
                isolated_atoms = read(isolated_at_fname, ':')
            else:
                isolated_atoms = [at for at in all_atoms if len(at) == 1]

        else:
            isolated_atoms = None

        rmse_scatter_evaled.scatter_plot(ref_energy_name=ref_energy_name,
                                         pred_energy_name=pred_energy_name,
                                         ref_force_name=ref_force_name,
                                         pred_force_name=pred_force_name,
                                         all_atoms=all_atoms,
                                         output_dir=output_dir,
                                         prefix=prefix,
                                         color_info_name=info_label,
                                         isolated_atoms=isolated_atoms,
                                         energy_type=energy_type,
                                         energy_shift=energy_shift,
                                         no_legend=no_legend,
                                         error_type=error_type)



@click.command('error-scatter')
@click.argument('atoms-filename')
@click.option('--ref-energy-name', '-re', type=str)
@click.option('--pred-energy-name', '-pe', type=str)
@click.option('--ref-force-name', '-rf', type=str)
@click.option('--pred-force-name', '-pf', type=str)
@click.option('--output-dir', default='.', show_default=True,
              type=click.Path(),
              help='directory for figures. Create if not-existent')
@click.option('--prefix', '-p', help='prefix to label plots')
@click.option('--info-label',
              help='info entry to label by')
@click.option('--isolated-at-fname')
@click.option('--total-energy', '-te', 'energy_type',
              flag_value='total_energy',
              help='plot total energy, not binding energy per atom')
@click.option('--binding-energy', '-be', 'energy_type', default=True,
              flag_value='binding_energy', help='Binding energy per atom')
@click.option('--mean-shifted-energy', '-sft', 'energy_shift',is_flag=True,
              help='shift energies by the mean. ')
@click.option('--no-legend', is_flag=True, help='doesn\'t plot the legend')
@click.option('--error-type', default='rmse')
def scatter(ref_energy_name, pred_energy_name, ref_force_name,
            pred_force_name,
               atoms_filename,
               output_dir, prefix, info_label, isolated_at_fname,
               energy_type, energy_shift, no_legend, error_type):
    """Makes energy and force scatter plots and dimer curves"""

    from util.plot import rmse_scatter_evaled

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    all_atoms = read(atoms_filename, ':')
    if energy_type=='binding_energy':

        if isolated_at_fname is not None:
            isolated_atoms = read(isolated_at_fname, ':')
        else:
            isolated_atoms = [at for at in all_atoms if len(at) == 1]

    else:
        isolated_atoms = None

    rmse_scatter_evaled.scatter_plot(ref_energy_name=ref_energy_name,
                                     pred_energy_name=pred_energy_name,
                                     ref_force_name=ref_force_name,
                                     pred_force_name=pred_force_name,
                                     all_atoms=all_atoms,
                                     output_dir=output_dir,
                                     prefix=prefix,
                                     color_info_name=info_label,
                                     isolated_atoms=isolated_atoms,
                                     energy_type=energy_type,
                                     energy_shift=energy_shift,
                                     no_legend=no_legend,
                                     error_type=error_type)


@click.command("dist-corr")
@click.argument('trajectory-fname')
@click.option('--idx1', '-i1', type=click.INT, help='index for atom 1 in '
                                                    'molecule')
@click.option("--idx2", "-i2", type=click.INT, help="index for atom 2 in "
                                                    "molecule")
@click.option("--vector-distance", is_flag=True,
              help="get distance as vector")
def distance_autocorrelation(trajectory_fname, idx1, idx2, vector_distance):
    """plots distance autocorrelation between two atoms"""

    from util.plot.autocorrelation import plot_autocorrelation

    stem = Path(trajectory_fname).stem

    title = f'{stem} distance autocorrelation between atoms {idx1} and ' \
            f'{idx2} distances as vectors- {vector_distance}'

    ats = read(trajectory_fname, ":")

    plot_autocorrelation(ats=ats, idx1=idx1, idx2=idx2, title=title,
                             vector_distance=vector_distance)

