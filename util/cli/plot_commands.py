import click
import subprocess 
import logging
import util
from pathlib import Path
import os
import glob
import logging
from util.plot import mace_loss 

from pathlib import Path

from ase.io import read, write

logger = logging.getLogger(__name__)

@click.command("mace-loss")
@click.option('--fig-name', '-n', default="train_summary.png", show_default=True, help='filename/prefix to save figure to')
@click.option('--skip-first-n', '-s', type=click.INT, help='how many epochs to skip and not plot')
@click.option('--x-log-scale', is_flag=True, help="plot x in log scale")
@click.argument("in-fnames", nargs=-1)
def plot_mace_loss(fig_name, in_fnames, skip_first_n, x_log_scale):
    mace_loss.plot_mace_train_summary(in_fnames=in_fnames, fig_name=fig_name, skip_first_n=skip_first_n, x_log_scale=x_log_scale)

@click.command('quick-dimer')
@click.argument("dimer-fns", nargs=-1)
@click.option('--isolated-at-fn', help='isolated atoms filename')
@click.option('--pred-prop-prefix')
@click.option('--output-fn', default='dimer_curves.png', show_default=True)
@click.option('--isolated-at-prop-prefix')
def plot_quick_dimer(dimer_fns, isolated_at_fn, pred_prop_prefix, output_fn, isolated_at_prop_prefix):
    from util.plot import quick_dimer
    isolated_ats = read(isolated_at_fn, ":")
    quick_dimer.main(dimer_fns=dimer_fns, isolated_ats=isolated_ats, pred_prop_prefix=pred_prop_prefix,
                        output_fn=output_fn, isolated_at_prop_prefix=isolated_at_prop_prefix)

@click.command('quick-md')
@click.option("--root-dir", '-r', help='path to "md_trajs" or similar')
@click.option('--ace-fname', '-a', default='ace.json')
@click.option('--temps', '-t', type=click.FLOAT, multiple=True, default=[300, 500, 800])
def md_test_summary(root_dir, ace_fname, temps):

    from util.single_use import md_test

    root_dir = Path(root_dir)
    graphs = [d.name for d in root_dir.iterdir() if "mol" in str(d) or "rad" in str(d)] 

    extra_info = {"plot_kwargs": {"color": "tab:orange"} }    

    md_test.plot_mol_graph(graphs, extra_info, temps=temps, ace_fname=ace_fname, traj_root=root_dir)


@click.command('ard-scores')
@click.argument("fname")
def plot_ard_scores(fname):
    from util.plot import ard_scores
    ard_scores.plot_ard_scores(fname)

@click.command("ace-2b")
@click.argument("ace_fname")
@click.option('--plot-type', '-t', help="2b or full")
@click.option('--cc-in')
@click.option('--ch-in')
@click.option('--hh-in')
def ace_2b(ace_fname, plot_type, cc_in, ch_in, hh_in):
    from util.plot import julia_plots
    julia_plots.plot_ace_2b(ace_fname, plot_type, cc_in, ch_in, hh_in)


@click.command("dissociate-h")
@click.argument("fname")
@click.option("--pred-prefix", default='ace_')
@click.option("--rin", type=click.FLOAT)
@click.option("--rout", type=click.FLOAT)
@click.option('--isolated-at-fn')
@click.option('--out-prefix', default='')
def dissociate(fname, pred_prefix, rin, rout, isolated_at_fn, out_prefix):
    from util.plot import dissociate_h_test
    ats = read(fname, ':')
    if isolated_at_fn is not None:
        isolated_ats = read(isolated_at_fn, ':')
        isolated_ats = [at for at in isolated_ats if len(at) == 1]
        isolated_at = [at for at in isolated_ats if list(at.symbols)[0] == "H"][0]
    else:
        isolated_at = None

    dissociate_h_test.curves_from_all_atoms(ats, pred_prefix, isolated_at, rin, rout, out_prefix)


@click.command("dimer-curve")
@click.argument("ip-fname", nargs=-1)
@click.option("--ip-type",
              help="which potential to call")
@click.option("--train-fname", help="to get distances hist")
@click.option("--ref-isolated-fname", help="to plot dft reference")
@click.option("--ref-prefix", default="dft_")
@click.option("--pred-label",help='for y label')
def plot_dimer(ip_fname, ip_type, train_fname, ref_isolated_fname, ref_prefix,
               pred_label):

    from util.plot.dimer import calc_on_dimer

    info_fnames = '\n'.join(ip_fname)
    logger.info(f"filenames: {info_fnames}")

    train_ats = read(train_fname, ':')
    if ref_isolated_fname is not None:
        ref_isolated_ats = read(ref_isolated_fname, ':')
    else:
        ref_isolated_ats = [at for at in train_ats if len(at) == 1]

    if pred_label is None:
        pred_label = ip_type + '_'

    if ip_type.lower() == 'gap':
        from quippy.potential import Potential
    elif ip_type.lower() == 'ace':
        import pyjulip


    for fname in ip_fname:
        print(fname)
        if ip_type.lower() == 'gap':
            calc = Potential(param_filename=fname)
        elif ip_type.lower() == 'ace':
            calc = pyjulip.ACE(fname)

        title = Path(fname).stem

        calc_on_dimer(calc=calc,
                      train_ats=train_ats,
                      ref_isolated_ats=ref_isolated_ats,
                      ref_prefix=ref_prefix,
                      pred_label=pred_label,
                      title=title)


@click.command('data-summary')
@click.argument('in-fname')
@click.option('--fig-prefix', '-p', default='dataset_summary')
@click.option('--isolated_at_fname', '-i')
@click.option('--cutoff', '-c', default=6.0, type=click.FLOAT,
              help='cutoff for counting distances')
@click.option('--info', default='config_type', help='label to color by')
@click.option('--prop-prefix', default='dft_', type=click.STRING)
@click.option('--no-ef', is_flag=True, help='dont plot energy and forces')
@click.option('--cmap', default='tab10')
def plot_dataset(in_fname, fig_prefix, isolated_at_fname, cutoff,
                 info, prop_prefix, no_ef, cmap):

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

    dists_fig_name = f'{fig_prefix}_{fname_stem}_distances_hist.pdf'


    if not no_ef:
        dataset.energy_by_idx(atoms, title=title_energy,
                              isolated_atoms=isolated_ats,
                              info_label=info, prop_prefix=prop_prefix, cmap=cmap)
        dataset.forces_by_idx(atoms, title=title_forces, info_label=info,
                              prop_prefix=prop_prefix, cmap=cmap)

    dataset.pairwise_distances_hist(atoms, dists_fig_name)



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

    inputs = ConfigSet(input_files=inputs)

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
@click.argument('atoms-filenames', nargs=-1)
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
@click.option('--per-atom-energy', '-pae', 'energy_type',
              flag_value='per_atom_energy', help='plot energy per atom (not binding energy per atom)')
@click.option('--mean-shifted-energy', '-sft', 'energy_shift',is_flag=True,
              help='shift energies by the mean. ')
@click.option('--scatter-absolute-error', 'error_scatter_type', 
              flag_value='absolute', help="scatter absolute error in the error plot")
@click.option('--scatter-signed-error', 'error_scatter_type', default=True,
              flag_value='signed', help="scatter signed error in the error plot")
@click.option('--no-legend', is_flag=True, help='doesn\'t plot the legend')
@click.option('--error-type', default='rmse')
@click.option('--xvals', help="values for x axis for multi-file plot")
@click.option('--xlabel', help='x axis label ')
@click.option('--skip', is_flag=True, help="skip if property not present")
def scatter(ref_energy_name, pred_energy_name, ref_force_name,
            pred_force_name,
               atoms_filenames,
               output_dir, prefix, info_label, isolated_at_fname,
               energy_type, energy_shift, no_legend, error_type, xvals, xlabel,
               skip, error_scatter_type):
    """Makes energy and force scatter plots and dimer curves"""

    from util.plot import rmse_scatter_evaled, multiple_error_files

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    if prefix is None:
        prefix = Path(atoms_filenames[0]).stem

    if len(atoms_filenames) == 1:
        atoms_filename = atoms_filenames[0]
        all_atoms = read(atoms_filename, ":")

        isolated_atoms = None
        if isolated_at_fname is not None:
            isolated_atoms = read(isolated_at_fname, ':')

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
                                         error_type=error_type, 
                                         skip_if_prop_not_present=skip,
                                         error_scatter_type=error_scatter_type)
    else:
        if xvals is not None:
            xvals = [float(x) for x in xvals.split()]
        multiple_error_files.main(ref_energy_name=ref_energy_name,
                             pred_energy_name=pred_energy_name,
                             ref_force_name=ref_force_name,
                             pred_force_name=pred_force_name,
                             atoms_filenames=atoms_filenames,
                             output_dir=output_dir,
                             prefix=prefix,
                             color_info_name=info_label,
                                  xvals=xvals, xlabel=xlabel)



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


@click.command('mols')
@click.argument('input-csv')
@click.option('--name-col', default='name',
              help='csv column for mol names')
@click.option('--smiles-col', default='smiles',
              help='csv column for smiles')
def plot_mols(input_csv, name_col, smiles_col):

    from util.plot import mols

    mols.main(input_csv=input_csv,
              name_col=name_col,
              smiles_col=smiles_col)

