import click
from util import compare_minima
from util import error_table
from util import plot
from util.plot import rmse_scatter_evaled
from util import data
from util import select_configs
from util import old_nms_to_new
from util import configs_ops
import os
from quippy.potential import Potential
from util import bde
from util import mem_tracker
from util import iter_fit
from util import iter_tools
import util
from ase.io import read, write
from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import orca
from ase.io.extxyz import key_val_str_to_dict


@click.group('util')
@click.pass_context
def cli(ctx):
    pass


@cli.group("bde")
@click.pass_context
def subcli_bde(ctx):
    pass


@cli.group('plot')
@click.pass_context
def subcli_plot(ctx):
    pass

@cli.group('track')
@click.pass_context
def subcli_track(ctx):
    pass

@cli.group('data')
@click.pass_context
def subcli_data(ctx):
    pass

@cli.group('gap')
@click.pass_context
def subcli_gap(ctx):
    pass

@cli.group('configs')
@click.pass_context
def subcli_configs(ctx):
    pass

@subcli_configs.command('distribute')
@click.argument('in-fname')
@click.option('--num-tasks', '-n', default=8, type=click.INT, help='number of files to distribute across')
@click.option('--prefix', '-p', default='in_', help='prefix for individual files')
def distribute_configs(in_fname, num_tasks, prefix):
    configs_ops.batch_configs(in_fname=in_fname, num_tasks=num_tasks,
                              batch_in_fname_prefix=prefix)


@subcli_configs.command('gather')
@click.option('--out-fname', '-o', help='output for gathered configs')
@click.option('--num-tasks', '-n', default=8, type=click.INT,
              help='number of files to gather configs from')
@click.option('--prefix', '-p', default='out_',
              help='prefix for individual files')
def gather_configs(out_fname, num_tasks, prefix):
    configs_ops.collect_configs(out_fname=out_fname, num_tasks=num_tasks,
                              batch_in_fname_prefix=prefix)


@subcli_data.command('reevaluate-dir')
@click.argument('dirs')
@click.option('--smearing', type=click.INT, default=2000)
def reevaluate_dir(dirs, smearing=2000):
    iter_tools.reeval_dft(dirs, smearing)

@subcli_gap.command('opt-and-nm')
@click.option('--dft-dir', '-d')
@click.option('--gap-fnames', '-g')
@click.option('--output-dir', '-o')
def gap_reopt_dft_and_derive_normal_modes(dft_dir, gap_fnames, output_dir):
    compare_minima.opt_and_normal_modes(dft_dir, gap_fnames, output_dir)


@subcli_plot.command('error-table')
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

    inputs = ConfigSet_in(input_files=inputs)

    if calc_kwargs is not None:
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


@subcli_gap.command('fit')
@click.option('--no_cycles', type=click.INT, help='number of gap_fit - optimise cycles')
@click.option('--test_fname', default='test.xyz', help='xyz to test gaps on')
@click.option('--train_fname', default='train.xyz', help='fname of first training set file')
@click.option('--e_sigma', default=0.0005, type=click.FLOAT, help='energy default sigma')
@click.option('--f_sigma', default=0.02, type=click.FLOAT, help='force default sigma')
@click.option('--n_sparse', default=600, type=click.INT)
@click.option('--smiles_csv', help='smiles to optimise')
@click.option('--num_smiles_opt', type=click.INT, help='number of optimisations per smiles' )
@click.option('--opt_starts_fname', help='filename where to optimise structures from')
@click.option('--num_nm_displacements', type=click.INT, help='number of normal modes displacements per structure')
def fit(no_cycles, test_fname, train_fname, e_sigma, n_sparse,
        f_sigma,  smiles_csv, num_smiles_opt, opt_starts_fname,
        num_nm_displacements):

    iter_fit.fit(no_cycles=no_cycles,
                      test_fname=test_fname, first_train_fname=train_fname,
                      e_sigma=e_sigma, f_sigma=f_sigma, smiles_csv=smiles_csv,
                 num_smiles_opt=num_smiles_opt, opt_starts_fname=opt_starts_fname,
                 num_nm_displacements=num_nm_displacements, n_sparse=n_sparse)

@subcli_gap.command('optimize')
@click.option('--start-dir' )
@click.option('--start-fname')
@click.option('--output', '-o')
@click.option('--traj-fname')
@click.option('--gap-fname', '-g')
@click.option('--chunksize', type=click.INT, default=10)
def gap_optimise(gap_fname, output, traj_fname=None, start_dir=None, start_fname=None, chunksize=10):

    assert start_dir is None or start_fname is None

    dft_fnames, _, _ = bde.dirs_to_fnames(dft_dir)

    # optimise
    opt_trajectory = ConfigSet_out(output_files=traj_fname)
    inputs = ConfigSet_in(input_files=dft_fnames)

    calculator = (Potential, [], {'param_fliename':gap_fname})

    run_opt(inputs=inputs,
            outputs=opt_trajectory, chunksize=chunksize,
            calculator=calculator, return_traj=True, logfile=logfile)

    opt_trajectory = read(traj_fname, ':')

    opt_ats = [at for at in opt_trajectory if 'minim_config_type' in
               at.info.keys() and 'converged' in at.info['minim_config_type']]

    write(output, opt_ats)


@subcli_data.command('convert-nm')
@click.argument('fname_in')
@click.option('--fname_out', '-o')
@click.option('--prefix', '-p', help='properties\' prefix')
@click.option('--arrays-to-keep', '-a', help='string of arrays keys to keep')
@click.option('--info-to-keep', '-i', help='string of info keys to keep')
def convert_nms(fname_in, fname_out, prefix, info_to_keep, arrays_to_keep):
    """converts atoms with old-style nm data (evals/evecs) to new
    (frequencies/modes)"""

    if prefix is None:
        prefix = ''

    old_nms_to_new.convert_all(fname_in, fname_out, prefix, info_to_keep, arrays_to_keep)


@subcli_data.command('gen')
@click.argument('df-fname')
@click.option('--how-many', '-n', type=click.INT, help='how many structures to submit')
@click.option('--skip_first', '-s', type=click.INT, help='how many structures to skip')
@click.option('--submit', type=click.BOOL, help='whether to submit the jobs')
@click.option('--overwrite', type=click.BOOL, help='whether to overwrite stuff in non-empty dirs')
@click.option('--script', help='everything or more_data')
@click.option('--hours', type=click.INT, default=48)
@click.option('--no-cores', type=click.INT, default=16)
@click.option('--script-name', default='sub.sh')
def submit_data(df_fname, how_many, skip_first, submit, overwrite, script,
                hours, no_cores, script_name):
    data.sub_data(df_name=df_fname, how_many=how_many, skip_first=skip_first,
                  submit=submit, overwrite_sub=overwrite, script=script,
                  hours=hours, no_cores=no_cores, script_name=script_name)


@subcli_data.command('select')
@click.argument('input_fn')
@click.option('--output-fname', '-o', help='where_to_put_sampled configs')
@click.option('--n-configs', '-n', type=click.INT, help='number of configs')
@click.option('--sample-type', '-t', help='either "per_config",  "total" or "all" - how to count configs')
@click.option('--include-config-type',  help='whih config_type to return (if sample_type==all)')
@click.option('--exclude-config-type', help='string of config_types to exclude')
def sample_configs(input_fn, output_fname, n_configs, sample_type, include_config_type, exclude_config_type):
    """takes a number of structures evenly from each config"""
    select_configs.sample_configs(input_fn=input_fn, output_fname=output_fname,
                                  n_configs=n_configs, sample_type=sample_type,
                                  include_config_type=include_config_type,
                                  exclude_config_type=exclude_config_type)

@subcli_data.command('dftb')
@click.argument('input_fn')
@click.option('--output_fn', '-o')
@click.option('--prefix', '-p', default='dftb_')
def recalc_dftb_ef(input_fn, output_fn, prefix='dftb_'):

    dftb = Potential('TB DFTB', param_filename='/home/eg475/scripts/source_files/tightbind.parms.DFTB.mio-0-1.xml')

    ats_in = read(input_fn, ':')
    ats_out = []
    for atoms in ats_in:
        at = atoms.copy()
        at.calc = dftb
        at.info[f'{prefix}energy'] = at.get_potential_energy()
        at.arrays[f'{prefix}forces'] = at.get_forces()
        ats_out.append(at)

    write(output_fn, ats_out)


@subcli_track.command('mem')
@click.argument('my_job_id')
@click.option('--period', '-p', type=click.INT, default=60, help='how often to check')
@click.option('--max_time', type=click.INT, default=1e5, help='How long to keep checking for, s')
@click.option('--out_fname_prefix', default='mem_usage', help='output\'s prefix')
def track_mem(my_job_id, period=10, max_time=100000, out_fname_prefix='mem_usage'):
    out_fname = f'{out_fname_prefix}_{my_job_id}.txt'
    mem_tracker.track_mem(my_job_id=my_job_id, period=period, max_time=max_time,
                     out_fname=out_fname   )


@subcli_bde.command("multi")
@click.option('--dft_dir', '-d', type=click.Path())
@click.option('--gap_dir', '-g', type=click.Path())
@click.option('--gap_xml_fname', type=click.Path())
@click.option('--start_dir', '-s', type=click.Path())
@click.option('--precision', '-p', type=int, default=3)
@click.option('--exclude', '-e', help='list of thing to exclude')
def multi_bde_summaries(dft_dir, gap_dir=None, gap_xml_fname=None, start_dir=None,
                        precision=3, exclude=None):
    if gap_xml_fname is not None:
        calculator = (Potential, [], {'param_filename': gap_xml_fname})
    else:
        calculator = None

    bde.multi_bde_summaries(dft_dir=dft_dir, gap_dir=gap_dir, calculator=calculator,
                            start_dir=start_dir,  precision=precision,
                            exclude=exclude)


@subcli_bde.command('single')
@click.option('--dft_fname', '-d', type=click.Path(exists=True), help='dft reference fname')
@click.option('--gap_fname', '-g', type=click.Path(), help='gap bde filename')
@click.option('--gap_xml_fname', '-x', type=click.Path(exists=True),
              help='Gap.xml for optimising structures')
@click.option('--start_fname', '-s', type=click.Path(exists=True),
              help='structures to optimise with gap')
@click.option('--precision', '-p', type=click.INT, default=3,
              help='how many digits to print in the table')
@click.option('--printing', type=click.BOOL, default=True, help='whether to print the table')
@click.option('--no-dft-prefix', is_flag=True )
def single_bde_summary(dft_fname, gap_fname=None, gap_xml_fname=None, start_fname=None, precision=3,
                       printing=True, no_dft_prefix=False):
    """ BDE for single file"""

    if gap_xml_fname is not None:
        calculator = (Potential, [], {'param_filename': gap_xml_fname})
    else:
        calculator = None

    if no_dft_prefix:
        dft_prefix=''
    else:
        dft_prefix='dft_'

    bde.bde_summary(dft_fname=dft_fname, gap_fname=gap_fname, calculator=calculator,
                    start_fname=start_fname, precision=precision, printing=printing,
                    dft_prefix=dft_prefix)


@subcli_bde.command("bar-plot")
@click.option('--dft_dir', '-d', type=click.Path())
@click.option('--gap_dir', '-g', type=click.Path())
@click.option('--start_dir', '-s', type=click.Path())
@click.option('--gap_xml_fname', type=click.Path())
@click.option('--plot_title', '-t', type=click.STRING, default='bde_bar_plot')
@click.option('--output_dir', type=click.Path(), default='pictures')
@click.option('--exclude', '-e', help='string of dft fnames to exclude')
def bde_bar_plot(dft_dir, gap_dir, start_dir=None, gap_xml_fname=None,
                 plot_title='bde_bar_plot', output_dir='pictures',
                 exclude=None):
    if gap_dir is not None:
        if not os.path.isdir(gap_dir):
            os.makedirs(gap_dir)

    if gap_xml_fname is not None:
        calculator = (Potential, [], {'param_filename': gap_xml_fname})
    else:
        calculator = None

    dft_basenames = util.natural_sort(os.listdir(dft_dir))

    if exclude is not None:
        exclude = exclude.split(' ')
        dft_basenames = [bn for bn in dft_basenames if bn not in exclude]


    dft_fnames = [os.path.join(dft_dir, fname) for fname in dft_basenames]
    gap_fnames = [os.path.join(gap_dir, basename.replace('optimised', 'gap_optimised')) for
                  basename in dft_basenames]

    if start_dir is not None:
        start_fnames = [os.path.join(start_dir, basename.replace('optimised', 'non_optimised')) for
                        basename in dft_basenames]
    else:
        start_fnames = [None for _ in dft_fnames]

    bde.bde_bar_plot(gap_fnames=gap_fnames, dft_fnames=dft_fnames, plot_title=plot_title,
                     start_fnames=start_fnames, calculator=calculator, output_dir=output_dir)


@subcli_bde.command("scatter")
@click.option('--dft_dir', '-d', type=click.Path())
@click.option('--gap_dir', '-g', type=click.Path())
@click.option('--start_dir', '-s', type=click.Path())
@click.option('--gap_xml_fname', type=click.Path())
@click.option('--plot_title', '-t', type=click.STRING, default='bde_bar_plot')
@click.option('--output_dir', type=click.Path(), default='pictures')
@click.option('--exclude', '-e', help='string of dft fnames to exclude')
@click.option('--rmsd', is_flag=True, help='scatters rmsds instead of bdes')
def bde_scatter_plot(dft_dir, gap_dir, start_dir=None, gap_xml_fname=None,
                 plot_title='bde_bar_plot', output_dir='pictures',
                exclude=None, rmsd=False):
    if gap_dir is not None:
        if not os.path.isdir(gap_dir):
            os.makedirs(gap_dir)

    if gap_xml_fname is not None:
        calculator = (Potential, [], {'param_filename': gap_xml_fname})
    else:
        calculator = None

    dft_basenames = util.natural_sort(os.listdir(dft_dir))

    if exclude is not None:
        exclude = exclude.split(' ')
        dft_basenames = [bn for bn in dft_basenames if bn not in exclude]

    dft_fnames = [os.path.join(dft_dir, fname) for fname in dft_basenames]
    gap_fnames = [os.path.join(gap_dir, basename.replace('optimised', 'gap_optimised')) for
                  basename in dft_basenames]

    if start_dir is not None:
        start_fnames = [os.path.join(start_dir, basename.replace('optimised', 'non_optimised')) for
                        basename in dft_basenames]
    else:
        start_fnames = [None for _ in dft_fnames]

    data = bde.get_data(dft_fnames, gap_fnames, start_fnames=start_fnames,
                        calculator=calculator, which_data=which_data)

    bde.scatter_plot(all_data=data, plot_title=plot_title,
                         output_dir=output_dir,
                     rmsd=rmsd)



@subcli_bde.command("from-mols")
@click.argument('mols_fname')
@click.option('--starts-dir', '-s', help='where to put start files')
@click.option('--dft-dir', '-d', help='optimised files destination')
@click.option('--calc-kwargs', help='orca kw arguments for optimisation')
@click.option('--h_to_remove', '-h', type=click.INT, help='h_idx to remove')
def multi_bdes(mols_fname, starts_dir, dft_dir, calc_kwargs, h_to_remove):
    """makes radicals and optimises them """
    # TODO - exclude something if dft file exists

    start_fname_end = '_non_optimised.xyz'

    for dir_name in [starts_dir, dft_dir]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    atoms = read(mols_fname, ':')
    start_fnames = []
    dft_fnames = []
    for atom in atoms:

        start_basename = os.path.join(starts_dir, atom.info['config_type'])
        for idx in range(1, 10):
            if os.path.exists(start_basename + start_fname_end):
                continue
                # start_basename += f'_{idx}'
            else:
                start_fname = start_basename + start_fname_end
                break
        else:
            raise RuntimeError(f'All attempted fnames were taken, last one: {start_basename}')

        start_fnames.append(start_fname)
        dft_fname = os.path.basename(start_fname).replace('_non', '')
        dft_fnames.append(os.path.join(dft_dir, dft_fname))

        mol = atom.copy()
        mol.info['config_type'] += '_mol'
        rad = atom.copy()
        rad.info['config_type'] += f'_rad{h_to_remove}'
        del rad[h_to_remove]

        write(start_fname, [mol, rad])

    output_dict = {}
    for st, dft in zip(start_fnames, dft_fnames):
        output_dict[st] = dft

    # do optimisation
    inputs = ConfigSet_in(input_files=start_fnames)
    outputs = ConfigSet_out(output_files=output_dict)

    calc_kwargs = key_val_str_to_dict(calc_kwargs)
    calc_kwargs['orca_kwargs'] = key_val_str_to_dict(calc_kwargs['orca_kwargs'])
    if 'task' not in calc_kwargs['orca_kwargs'].keys():
        calc_kwargs['orca_kwargs']['task'] = 'optimise'

    print(f'orca_parameters: {calc_kwargs}')

    orca.evaluate(inputs=inputs, outputs=outputs, **calc_kwargs)


@subcli_plot.command('dimer')
@click.option('--gap-fname', '-g', type=click.Path(exists=True),  help='GAP xml to test')
@click.option('--gap-dir', help='directory with all the gaps to test')
@click.option('--output-dir', default='pictures', show_default=True, type=click.Path(), help='directory for figures. Create if not-existent')
@click.option('--prefix', '-p', help='prefix to label plots')
@click.option('--glue-fname', type=click.Path(exists=True), help='glue potential\'s xml to be evaluated for dimers')
@click.option('--plot_2b_contribution', is_flag=True, help='whether to plot the 2b only bit of gap')
@click.option('--ref-curve', is_flag=True, help='whether to plot the reference DFT dimer curve')
@click.option('--isolated_atoms_fname',  default='/home/eg475/scripts/source_files/isolated_atoms.xyz', show_default=True, help='isolated atoms to shift glue')
@click.option('--ref-name', default='dft', show_default=True, help='prefix to \'_forces\' and \'_energy\' to take as a reference')
@click.option('--dimer_scatter', help='dimer data in training set to be scattered on top of dimer curves')
def make_plots(gap_fname=None, gap_dir=None, output_dir=None, prefix=None, glue_fname=False,
               plot_2b_contribution=True, ref_curve=True, isolated_atoms_fname=None, ref_name='dft', dimer_scatter=None):
    """Makes energy and force scatter plots and dimer curves"""

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    assert gap_fname is None or gap_dir is None

    if gap_dir is not None:
        gap_fnames = os.path.listdir(gap_dir)
        gap_fnames = [os.path.join(gap_dir, gap)  for gap in gap_fnames if 'xml' in gap]
        gap_fnames = util.natural_sort(gap_fnames)
    elif gap_fname is not None:
        gap_fnames = [gap_fname]


    plot.make_dimer_curves(param_fnames=gap_fnames,  output_dir=output_dir, prefix=prefix,
                      glue_fname=glue_fname, plot_2b_contribution=plot_2b_contribution, plot_ref_curve=ref_curve,
                      isolated_atoms_fname=isolated_atoms_fname, ref_name=ref_name, dimer_scatter=dimer_scatter)



@subcli_plot.command('error-scatter')
@click.option('--ref_energy_name', '-re', type=str)
@click.option('--pred_energy_name', '-pe', type=str)
@click.option('--ref_force_name', '-rf', type=str)
@click.option('--pred_force_name', '-pf', type=str)
@click.option('--evaluated_train_fname', '-te', type=click.Path(exists=True))
@click.option('--evaluated_test_fname', '-tr', type=click.Path(exists=True))
@click.option('--output_dir', default='pictures', show_default=True, type=click.Path(),
              help='directory for figures. Create if not-existent')
@click.option('--prefix', '-p', help='prefix to label plots')
@click.option('--by_config_type', is_flag=True,
              help='if structures should be coloured by config_type in plots')
@click.option('--force_by_element', default=True, type=bool,
              help='whether to evaluate force on each element separately')
def make_plots(ref_energy_name, pred_energy_name, ref_force_name, pred_force_name,
               evaluated_train_fname, evaluated_test_fname,
               output_dir, prefix, by_config_type, force_by_element):
    """Makes energy and force scatter plots and dimer curves"""

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if evaluated_test_fname is None and by_config_type != False:
        by_config_type = True

    print('Scatter plotting')
    rmse_scatter_evaled.make_scatter_plots_from_evaluated_atoms(ref_energy_name=ref_energy_name,
                                                                pred_energy_name=pred_energy_name,
                                                                ref_force_name=ref_force_name,
                                                                pred_force_name=pred_force_name,
                                                                evaluated_train_fname=evaluated_train_fname,
                                                                evaluated_test_fname=evaluated_test_fname,
                                                                output_dir=output_dir,
                                                                prefix=prefix,
                                                                by_config_type=by_config_type,
                                                                force_by_element=force_by_element)
