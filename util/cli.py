import click
import subprocess
import logging
import warnings
import os
import numpy as np

from ase import Atoms
from ase.io import read, write
from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import orca
from wfl.calculators import generic
from ase.io.extxyz import key_val_str_to_dict

try:
    from quippy.potential import Potential
except ModuleNotFoundError:
    pass

from util import compare_minima
import util.bde.generate
import util.bde.table
import util.bde.plot
from util import radicals
from util import error_table
from util import plot
from util.plot import dimer
from util import iter_tools as it
from util.plot import rmse_scatter_evaled
from util.plot import iterations
from util import data
from util import select_configs
from util.plot import rmsd_table
from util import old_nms_to_new
from util import configs
from util import atom_types
from util import mem_tracker
from util import iter_fit
from util import iter_tools
from util import md
from util import qm
import util
from util.plot import dataset


@click.group('util')
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def cli(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    # ignore calculator writing warnings
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.extxyz")

    warnings.filterwarnings("ignore", category=FutureWarning,
                            module="ase.calculators.calculator")

    if verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s')


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

@cli.group('tmp')
def subcli_tmp():
    pass

@cli.group('jobs')
def subcli_jobs():
    pass

@cli.group('qm')
def subcli_qm():
    pass

@subcli_qm.command('read-orca')
@click.argument('input-xyz')
@click.option('--output-xyz', '-o', help='output filename')
@click.option('--orca-label', '-l', help='prefix to orca files to read from')
def read_orca_stuff(input_xyz, output_xyz, orca_label):

    at = qm.read_orca_output(input_xyz, orca_label)
    write(output_xyz, at)

@subcli_configs.command('remove-info')
@click.argument('in-filename')
@click.option('--output-file', '-o')
@click.option('--info-key', '-i')
def remove_info_entries(in_filename, output_file, info_key):

    ats = read(in_filename, ':')
    for at in ats:
        del at.info[info_key]

    write(output_file, ats)


@subcli_bde.command('print-tables')
@click.pass_context
@click.argument('gap-bde-file')
@click.option('--isolated-h-fname', '-h', help='GAP evaluated on isolated H')
@click.option('--gap-prefix', '-g', help='prefix for gap properties')
@click.option('--dft-prefix', '-d', help='prefix for dft properties')
@click.option('--precision', default=3, type=click.INT, help='number of digits in the table')
def print_tables(ctx, gap_bde_file, isolated_h_fname, gap_prefix, dft_prefix,
                 precision):

    isolated_h = read(isolated_h_fname)
    all_atoms = read(gap_bde_file, ':')

    _ = util.bde.table.multiple_tables_from_atoms(all_atoms=all_atoms,
                                             isolated_h=isolated_h,
                                             gap_prefix=gap_prefix,
                                             dft_prefix=dft_prefix,
                                             printing=True,
                                             precision=precision)

@subcli_bde.command('scatter')
@click.argument('gap-bde-file')
@click.option('--isolated-h-fname', '-h', required=True,
              help='Gap evaluated on isolated H')
@click.option('--gap-prefix', '-g', required=True, help='prefix for gap properties')
@click.option('--dft-prefix', '-d', default='dft_', show_default=True,
              help='prefix for dft properties')
@click.option('--measure', '-m', required=True,
              type=click.Choice(['bde_error', 'bde_correlation', 'rmsd', 'energy_error']),
              help='what to plot')
@click.option('--plot-title', help='optional plot title')
@click.option('--output_dir', default='pictures', show_default=True)
@click.option('--color-by', default='compound_type', show_default=True,
              help='Atoms.info key to color by')
def scatter_plot(gap_bde_file, isolated_h_fname, gap_prefix, dft_prefix,
                 measure, plot_title, output_dir, color_by):

    isolated_h = read(isolated_h_fname)
    all_atoms = read(gap_bde_file, ':')

    labeled_atoms = util.sort_atoms_by_label(all_atoms, color_by)

    data_to_scatter = {}
    for label, atoms in labeled_atoms.items():
        data_to_scatter[label] = \
            util.bde.table.multiple_tables_from_atoms(all_atoms=atoms,
                                                      isolated_h=isolated_h,
                                                      gap_prefix=gap_prefix,
                                                      dft_prefix=dft_prefix,
                                                      printing=False)

    util.bde.plot.scatter(measure=measure,
                          all_data=data_to_scatter,
                          plot_title=plot_title,
                          output_dir=output_dir)



@subcli_bde.command('generate')
@click.pass_context
@click.argument('dft-bde-file')
@click.option('--gap-fname', '-g', help='gap xml filename')
@click.option('--iso-h-fname',
              help='if not None, evaluate isolated H energy with GAP'
                   'and save to given file.')
@click.option('--output-fname-prefix', '-o',
              help='prefix for main and all working files with all gap and dft properties')
@click.option('--dft-prop-prefix', default='dft_', show_default=True,
              help='label for all dft properties')
@click.option('--gap-prop-prefix', help='label for all gap properties')
@click.option('--wdir', default='gap_bde_wdir', show_default=True,
              help='working directory for all interrim files')
def generate_gap_bdes(ctx, dft_bde_file, gap_fname, iso_h_fname, output_fname_prefix,
                      dft_prop_prefix, gap_prop_prefix, wdir):

    logging.info('Generating gap bde things')

    calculator = (Potential, [], {'param_filename':gap_fname})

    if iso_h_fname is not None and not os.path.isfile(iso_h_fname):
        # generate isolated atom stuff
        util.bde.generate.gap_isolated_h(calculator=calculator,
                                         dft_prop_prefix=dft_prop_prefix,
                                         gap_prop_prefix=gap_prop_prefix,
                                         output_fname=iso_h_fname,
                                         wdir=wdir)

    # generate all molecule stuff
    util.bde.generate.everything(calculator=calculator,
                                 dft_bde_filename=dft_bde_file,
                                 output_fname_prefix=output_fname_prefix,
                                 dft_prop_prefix=dft_prop_prefix,
                                 gap_prop_prefix=gap_prop_prefix,
                                 wdir=wdir)


@subcli_gap.command('evaluate')
@click.argument('config-file')
@click.option('--gap-fname', '-g', help='gap xml filename')
@click.option('--output-fname', '-o')
@click.option('--gap-prop-prefix', help='prefix for gap properties in xyz')
def evaluate_gap_on_dft_bde_files(config_file, gap_fname, output_fname, gap_prop_prefix):

    # logger.info('Evaluating GAP on DFT structures')

    calculator = (Potential, [], {'param_filename':gap_fname})

    inputs = ConfigSet_in(input_files=config_file)
    outputs_gap_energies = ConfigSet_out(output_files=output_fname)
    generic.run(inputs=inputs, outputs=outputs_gap_energies, calculator=calculator,
                properties=['energy', 'forces'], output_prefix=gap_prop_prefix)




@subcli_jobs.command('from-pattern')
@click.argument('pattern-fname')
@click.option('--start', '-s',  type=click.INT, default=1, show_default=True,
                help='no to start with')
@click.option('--num', '-n', type=click.INT, help='how many scripts to make')
@click.option('--submit', is_flag=True, help='whether to submit the scripts')
def sub_from_pattern(pattern_fname, start, num, submit):
    """makes submission scripts from pattern"""

    with open(pattern_fname, 'r') as f:
        pattern = f.read()

    for idx in range(start, num+1):

        text = pattern.replace('{idx}', str(idx))
        sub_name = f'sub_{idx}.sh'

        with open(sub_name, 'w') as f:
            f.write(text)

        if submit:
            subprocess.run(f'qsub {sub_name}', shell=True)




@subcli_configs.command('csv-to-mols')
@click.argument('smiles-csv')
@click.option('--num-repeats', '-n', type=click.INT)
@click.option('--output-fname', '-o')
def smiles_to_molecules(smiles_csv, num_repeats, output_fname):

    molecules = configs.smiles_csv_to_molecules(smiles_csv, repeat=num_repeats)
    write(output_fname, molecules)

@subcli_bde.command('rads-from-opt-mols')
@click.argument('molecules-fname')
@click.option('--output-fname', '-o')
@click.option('--info-to-keep', '-i', help='info entries in molecules to keep in radicals')
@click.option('--arrays-to-keep', '-a')
def rads_from_opt_mols(molecules_fname, output_fname, info_to_keep, arrays_to_keep):

    if info_to_keep is not None:
        info_to_keep = info_to_keep.split()
    if arrays_to_keep is not None:
        arrays_to_keep = arrays_to_keep.split()

    molecules = read(molecules_fname, ':')
    outputs = ConfigSet_out()

    radicals.abstract_sp3_hydrogen_atoms(inputs = molecules,
                                         outputs = outputs)

    mols_and_rads = []
    for at in outputs.output_configs:

        if 'mol' in at.info['config_type']:
            mols_and_rads.append(at)
        else:
            rad = configs.strip_info_arrays(at, info_to_keep, arrays_to_keep)
            mols_and_rads.append(rad)

    write(output_fname, mols_and_rads)



@subcli_configs.command('hash')
@click.argument('input-fname')
@click.option('--output-fname', '-o')
@click.option('--prefix', '-p', help='prefix for info entries')
def hash_structures(input_fname, output_fname, prefix):
    """assigns positions/numbers hash to all structures in the file"""

    atoms = read(input_fname, ':')
    for at in atoms:
        at.info[f'{prefix}hash'] = configs.hash_atoms(at)

    write(output_fname, atoms)






@subcli_bde.command('gap-generate')
@click.option('--smiles-csv', '-s', help='csv file with smiles and structures names')
@click.option('--molecules-fname', '-m', help='filename with non-optimised molecule structures')
@click.option('--num-repeats', '-n', type=click.INT, help='number of conformers to generate for each smiles')
@click.option('--gap-prefix', '-p', help='how to name all gap entries')
@click.option('--gap-filename', '-g')
@click.option('--non-opt-filename', help='where to save non-optimised molecules')
@click.option('--output-filename', '-o')
def derive_gap_bdes(smiles_csv, molecules_fname, num_repeats, gap_prefix, gap_filename, non_opt_filename,
                    output_filename):

    assert smiles_csv is None or molecules_fname is None

    if smiles_csv:
        molecules = configs.smiles_csv_to_molecules(smiles_csv, repeat=num_repeats)
    elif molecules_fname:
        molecules = read(molecules_fname, ':')

    if non_opt_filename is not None:
        write(non_opt_filename, molecules)

    outputs = ConfigSet_out(output_files=output_filename)
    calculator = (Potential, [], {'param_filename':gap_filename})

    bde.gap_prepare_bde_structures_parallel(molecules, outputs=outputs,
                                             calculator=calculator,
                                             gap_prop_prefix=gap_prefix)


@subcli_bde.command('dft-reoptimise')
@click.argument('input-filename')
@click.option('--output-filename', '-o')
@click.option('--dft-prop-prefix', '-p', default='dft_reopt_')
def reoptimise_with_dft(input_filename, output_filename, dft_prop_prefix):

    inputs = ConfigSet_in(input_files=input_filename)
    outputs = ConfigSet_out(output_files=output_filename)

    bde.dft_optimise(inputs=inputs, outputs=outputs, dft_prefix=dft_prop_prefix)





@subcli_gap.command('eval-h')
@click.argument('gap-fname')
@click.option('--output', '-o')
@click.option('--prop-prefix', '-p')
def eval_h(gap_fname, output, prop_prefix):
    gap = Potential(param_filename=gap_fname)
    at = Atoms('H', positions=[(0, 0, 0)])
    at.calc = gap
    at.info[f'{prop_prefix}energy'] = at.get_potential_energy()
    write(output, at)

@subcli_plot.command('rmsd-table')
@click.option('--fname1', '-f1')
@click.option('--fname2', '-f2')
@click.option('--key', '-k', default='compound')
def print_rmsd_table(fname1, fname2, key):
    rmsd_table.rmsd_table(fname1, fname2, group_key=key)


@subcli_plot.command('dft-dimer')
@click.argument('dimer-fnames', nargs=-1)
@click.option('--prefix', '-p', default=dimer)
def plot_dft_dimers(dimer_fnames, prefix):
    """plots all results from all given evaluated xyzs"""
    dimer.dimer(dimer_fnames, prefix)

@subcli_configs.command('cleanup-info-entries')
@click.argument('input-fname')
@click.option('--output-fname', '-o')
def cleanup_info_entries(input_fname, output_fname):
    configs.process_config_info(input_fname, output_fname)

@subcli_configs.command('sample-normal-modes')
@click.argument('input-fname')
@click.option('--output-fname', '-o', help='where to write sampled structures')
@click.option('--temperature', '-t', type=click.FLOAT,  help='target temperature to generate normal modes at')
@click.option('--sample-size', '-n', type=click.INT, help='number of sampled structures per input structure')
@click.option('--prop-prefix', '-p', help='prefix for properties in xyz')
@click.option('--info-to-keep', help='space separated string of info keys to keep')
@click.option('--arrays-to-keep', help='space separated string of arrays keys to keep')
def sample_normal_modes(input_fname, output_fname, temperature, sample_size,
                        prop_prefix, info_to_keep, arrays_to_keep):

    inputs = ConfigSet_in(input_files=input_fname)
    outputs = ConfigSet_out(output_files=output_fname)
    if info_to_keep is not None:
        info_to_keep = info_to_keep.split()
    if arrays_to_keep is not None:
        arrays_to_keep = arrays_to_keep.split()

    configs.sample_downweighted_normal_modes(inputs=inputs, outputs=outputs,
                                             temp=temperature, sample_size=sample_size,
                                             prop_prefix=prop_prefix,
                                             info_to_keep=info_to_keep,
                                             arrays_to_keep=arrays_to_keep)




@subcli_tmp.command('sample-nms-test-set')
@click.option('--output-prefix', default='normal_modes_test_sample')
@click.option('--num-temps', type=click.INT)
@click.option('--num-displacements-per-temp', type=click.INT)
@click.option('--num-cycles', type=click.INT)
@click.option('--output-dir', default='xyzs', show_default=True)
def make_test_sets(output_prefix, num_temps, num_displacements_per_temp, num_cycles,
                   output_dir):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    temperatures = np.random.randint(1, 500, num_temps)
    for iter_no in range(1, num_cycles+1):

        current_test_set = []
        for temp in temperatures:
            sample = it.testing_set_from_gap_normal_modes(iter_no, temp,
                                                         num_displacements_per_temp)
            current_test_set += sample

        write(os.path.join(output_dir, f'{output_prefix}_{iter_no}.xyz'), current_test_set)



@subcli_plot.command('iterations')
@click.option('--test-fname-pattern', default='xyzs/gap_{idx}_on_test_{idx}.xyz', show_default=True)
@click.option('--train-fname-pattern', default='xyzs/gap_{idx}_on_train_{idx}.xyz',
              show_default=True, help='pattern with "{idx}" in the title to be replaced')
@click.option('--ref-prefix', default='dft_', show_default=True)
@click.option('--pred-prefix-pattern', default='gap_{idx}_', show_default=True,
              help='pattern with {idx} to be replaced')
@click.option('--plot-prefix')
@click.option('--num-cycles', type=click.INT, help='number of iterations', required=True)
@click.option('--gap-bde-dir-pattern', default='bdes_from_dft_min/gap_{idx}_bdes_from_dft',
              show_default=True)
@click.option('--dft-bde-dir', default='/home/eg475/dsets/small_cho/bde_files/train/dft/')
@click.option('--measure', default='bde_correlation',
              type=click.Choice(['bde', 'rmsd', 'soap_dist', 'gap_e_error', \
                                'gap_f_rmse', 'gap_f_max', 'bde_correlation']),
             help='which property to look at')
@click.option('--no-lc', is_flag=True, help='turns off learning curves')
@click.option('--no-means', is_flag=True, help='plot all lines, not just means')
@click.option('--no-bde', is_flag=True, help='turns off bde related plots')
def plot_cycles(test_fname_pattern, train_fname_pattern, ref_prefix, pred_prefix_pattern, plot_prefix,
                num_cycles, gap_bde_dir_pattern, dft_bde_dir, measure, no_lc, no_means, no_bde):

    num_cycles += 1

    train_fnames = [train_fname_pattern.replace('{idx}', str(idx)) for idx in range(num_cycles)]
    test_fnames = [test_fname_pattern.replace('{idx}', str(idx)) for idx in
                    range(num_cycles)]
    pred_prefixes = [pred_prefix_pattern.replace('{idx}', str(idx)) for idx in range(num_cycles)]

    if not no_lc:
        iterations.learning_curves(train_fnames=train_fnames, test_fnames=test_fnames,
                                   ref_prefix=ref_prefix, pred_prefix_list=pred_prefixes,
                               plot_prefix=plot_prefix)

    means = True
    if no_means:
        means=False

    if not no_bde:
        iterations.bde_related_plots(num_cycles=num_cycles, gap_dir_pattern=gap_bde_dir_pattern,
                                    dft_dir=dft_bde_dir, metric=measure, plot_prefix=plot_prefix,
                                 means=means)


@subcli_plot.command('data-summary')
@click.argument('in-fname')
@click.option('--fig-prefix', '-p', default='dataset_summary')
@click.option('--isolated_at_fname', '-i')
@click.option('--cutoff', '-c', default=6.0, type=click.FLOAT,
              help='cutoff for counting distances')
def plot_dataset(in_fname, fig_prefix, isolated_at_fname, cutoff):

    atoms = read(in_fname, ':')

    if cutoff == 0:
        cutoff = None

    isolated_ats = None
    if isolated_at_fname:
        isolated_ats = read(isolated_at_fname, ':')

    if fig_prefix:
        title_energy = f'{fig_prefix}_energy_by_idx'
        title_forces = f'{fig_prefix}_forces_by_idx'
        title_geometry = f'{fig_prefix}_distances_distribution'

    dataset.energy_by_idx(atoms, title=title_energy, isolated_atoms=isolated_ats)
    dataset.forces_by_idx(atoms, title=title_forces)
    dataset.distances_distributions(atoms, title=title_geometry, cutoff=cutoff)


@subcli_configs.command('filter-geometry')
@click.argument('in-fname')
@click.option('--mult', type=click.FLOAT, default=1.2, help='factor for bond cutoffs')
@click.option('--out_fname', '-o', help='output fname')
def filter_geometry(in_fname, mult, out_fname):
    ats_in = read(in_fname, ':')
    ats_out = configs.filter_expanded_geometries(ats_in, mult)
    write(out_fname, ats_out)



@subcli_configs.command('atom-type')
@click.argument('in-fname')
@click.option('--output', '-o')
@click.option('--cutoff-multiplier', '-m', type=click.FLOAT, default=1,
              help='multiplier for cutoffs')
@click.option('--elements', '-e', help='List of elements to atom type in aromatic and non')
@click.option('--force', '-f', is_flag=True, help='whether to force overwriting configs_out')
def atom_type(in_fname, output, cutoff_multiplier, elements, force):
    elements = elements.split(" ")
    inputs = ConfigSet_in(input_files=in_fname)
    outputs = ConfigSet_out(output_files=output, force=force)
    atom_types.assign_aromatic(inputs=inputs, outputs=outputs,
                               elements_to_type=elements,
                               mult=cutoff_multiplier)

@subcli_configs.command('distribute')
@click.argument('in-fname')
@click.option('--num-tasks', '-n', default=8, type=click.INT, help='number of files to distribute across')
@click.option('--prefix', '-p', default='in_', help='prefix for individual files')
def distribute_configs(in_fname, num_tasks, prefix):
    configs.batch_configs(in_fname=in_fname, num_tasks=num_tasks,
                              batch_in_fname_prefix=prefix)


@subcli_configs.command('gather')
@click.option('--out-fname', '-o', help='output for gathered configs')
@click.option('--num-tasks', '-n', default=8, type=click.INT,
              help='number of files to gather configs from')
@click.option('--prefix', '-p', default='out_',
              help='prefix for individual files')
def gather_configs(out_fname, num_tasks, prefix):
    configs.collect_configs(out_fname=out_fname, num_tasks=num_tasks,
                              batch_out_fname_prefix=prefix)

@subcli_configs.command('remove')
@click.option('--num-tasks', '-n', default=8, type=click.INT,
              help='number of files')
@click.option('--out-prefix',  default='out_',
              help='prefix for individual output files')
@click.option('--in-prefix', default='in_',
              help='prefix for individual files')
def cleanup_configs(num_tasks, out_prefix, in_prefix):
    configs.cleanup_configs(num_tasks=num_tasks,
                                batch_in_fname_prefix=in_prefix,
                                batch_out_fname_prefix=out_prefix)


@subcli_data.command('reevaluate-dir')
@click.argument('dirs')
@click.option('--smearing', type=click.INT, default=5000)
def reevaluate_dir(dirs, smearing=5000):
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
@click.option('--no-cycles', type=click.INT, help='number of gap_fit - optimise cycles')
@click.option('--train-fname', default='train.xyz', show_default=True, help='fname of first training set file')
@click.option('--e-sigma', default=0.0005, type=click.FLOAT, show_default=True, help='energy default sigma')
@click.option('--f-sigma', default=0.02, type=click.FLOAT, show_default=True, help='force default sigma')
@click.option('--descriptor-fname', default='descriptors.yml', show_default=True)
@click.option('--smiles-csv', help='smiles to optimise')
@click.option('--num-smiles-opt', type=click.INT, help='number of optimisations per smiles' )
@click.option('--opt-starts-fname', help='filename where to optimise structures from')
@click.option('--num-nm-displacements-per-temp', type=click.INT,
              help='number of normal modes displacements per structure per temperature')
@click.option('--num-nm-temps', type=click.INT, help='how many nm temps to sample from')
def fit(no_cycles, train_fname, e_sigma, descriptor_fname,
        f_sigma,  smiles_csv, num_smiles_opt, opt_starts_fname,
        num_nm_displacements_per_temp, num_nm_temps):

    iter_fit.fit(no_cycles=no_cycles,
                      first_train_fname=train_fname,
                      e_sigma=e_sigma, f_sigma=f_sigma, smiles_csv=smiles_csv,
                 num_smiles_opt=num_smiles_opt, opt_starts_fname=opt_starts_fname,
                 num_nm_displacements_per_temp=num_nm_displacements_per_temp,
                 gap_descriptor_filename=descriptor_fname,
                  num_nm_temps=num_nm_temps)

@subcli_gap.command('md-stability')
@click.option('--gap-filename', '-g')
@click.option('--mol-filename', '-m')
def run_md(gap_filename, mol_filename):
    md.run_md(gap_filename, mol_filename)


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


@subcli_tmp.command('opt-and-nm')
@click.argument('df-fname')
@click.option('--how-many', '-n', type=click.INT, help='how many structures to submit')
@click.option('--skip_first', '-s', type=click.INT, help='how many structures to skip')
@click.option('--submit', type=click.BOOL, help='whether to submit the jobs')
@click.option('--overwrite', type=click.BOOL, help='whether to overwrite stuff in non-empty dirs')
@click.option('--hours', type=click.INT, default=168)
@click.option('--no-cores', type=click.INT, default=16)
@click.option('--script-name', default='sub.sh')
@click.option('--generate', '-g', is_flag=True, help='whether to generate radicals from smiles')
@click.option('--optimise', '-o', is_flag=True, help='whether to optimise')
@click.option('--modes', '-m', is_flag=True, help='whether to generate modes')
def submit_data(df_fname, how_many, skip_first, submit, overwrite,
                hours, no_cores, script_name, generate, optimise, modes):
    data.sub_data(df_name=df_fname, how_many=how_many, skip_first=skip_first,
                  submit=submit, overwrite_sub=overwrite,
                  generate=generate, optimise=optimise, modes=modes,
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
