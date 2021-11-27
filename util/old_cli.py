import click
import subprocess
import logging
import warnings
import os
import yaml
import numpy as np
from copy import deepcopy

from tqdm import tqdm
import pandas as pd

from ase import Atoms
from ase.io import read, write


from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import orca
from wfl.calculators import generic
from wfl.generate_configs import vib
from wfl.utils import gap_xml_tools
from ase.io.extxyz import key_val_str_to_dict


# from util import compare_minima
# import util.bde.generate
# import util.bde.table
# import util.bde.plot
# from util import radicals
# from util import error_table
# from util import plot
# from util.plot import dimer
# from util.plot import mols
# from util import iter_tools as it
# from util.plot import rmse_scatter_evaled
# from util.plot import iterations
# from util import data
# from util import calculators
# from util import select_configs
# from util.plot import rmsd_table
# from util import old_nms_to_new
# from util import configs
# from util import atom_types
# from util import mem_tracker
# import util.iterations.fit
# from util import md
# from util import qm
# from util.util_config import Config
# from util import normal_modes
# import util
# from util.plot import dataset
# from util import cc_results
# from util.configs import max_similarity

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

@cli.group("ace")
def subcli_ace():
    pass

logger = logging.getLogger(__name__)


@subcli_data.command("zinc")
@click.argument("wget_fname")
@click.option('--output-label', '-l',
              help='label for all the output files ')
@click.option("--wdir", default='wdir')
def get_smiles_from_zinc(wget_fname, output_label, wdir):
    from util.single_use import zinc
    zinc.main(wget_fname, output_label, wdir=wdir)






@subcli_qm.command('orca-gradient-test')
@click.argument("input_fname")
def orca_gradient_test(input_fname):

    from util.single_use import orca_gradient_test

    atoms = read(input_fname, ':')
    for at in atoms:
        logger.info(f'atoms.info: {at.info}')
        orca_gradient_test.do_orca_grad_test(at)


@subcli_qm.command('prepare-q')
@click.argument('input_fname')
@click.option('--output-fname', '-o')
def prepare_q(input_fname, output_fname):

    from util.calculators.gap import PopGAP

    ats = read(input_fname, ':')
    for at in tqdm(ats):
        PopGAP.prepare_local_q(at, 'dft_mulliken_gross_atomic_charge')
    write(output_fname, ats)




@subcli_data.command('assign-data-type')
@click.argument('all-data')
@click.option('--train-csv', help='csv with all of the training set '
                                  'structures and names')
@click.option('--output', '-o', help='output to write everything to')
@click.option('--info-key', '-i', default='dataset_type', show_default=True,
              help='info key to assign train/test labels to')
def split_train_test(all_data, train_csv, output, info_key):
    """assigns train/test to at.info based on training set csv"""

    df = pd.read_csv(train_csv, index_col="Unnamed: 0")

    train_names = list(df['Name'])

    outputs = ConfigSet_out(output_files=output, force=True)

    all_data = read(all_data, ':')
    for at in tqdm(all_data):
        if info_key in at.info.keys():
            raise RuntimeError(f'{info_key} already present')

        if at.info['compound'] in train_names:
            at.info[info_key] = 'train_compound'
        else:
            at.info[info_key] = 'test_compound'

        outputs.write(at)
    outputs.end_write()




@subcli_data.command('split')
@click.argument('all-data')
@click.option('--train-csv', help='csv with all of the training set '
                                  'structures and names')
@click.option('--train-xyz', default='train.xyz', show_default=True,
                    help='xyz for training set')
@click.option('--test-xyz', default='test.xyz', show_default=True)
def split_train_test(all_data, train_csv, train_xyz, test_xyz):

    df = pd.read_csv(train_csv, index_col="Unnamed: 0")

    train_names = list(df['Name'])

    ats_test = ConfigSet_out(output_files=test_xyz)
    ats_train = ConfigSet_out(output_files=train_xyz)

    all_data = read(all_data, ':')
    for at in tqdm(all_data):
        if at.info['compound'] in train_names:
            if 'dataset' not in at.info.keys():
                at.info['dataset'] = 'train'
            ats_train.write(at)
        else:
            if 'dataset' not in at.info.keys():
                at.info['dataset'] = 'test'
            ats_test.write(at)

    ats_test.end_write()
    ats_train.end_write()




@subcli_plot.command('mols')
@click.argument('input-csv')
@click.option('--name-col', default='name', help='csv column for mol names' )
@click.option('--smiles-col', default='smiles', help='csv column for smiles')
def plot_mols(input_csv, name_col, smiles_col):
    mols.main(input_csv=input_csv,
              name_col=name_col,
              smiles_col=smiles_col)



@subcli_configs.command('assign-max-similarity')
@click.argument('set-to-eval')
@click.option('--train-set', '-t', help='xyz to which to compare similarity')
@click.option('--output-name', '-o')
@click.option('--zeta', default=1, show_default=True, help='kernel exponent')
@click.option('--remove-descriptor', is_flag=True)
def assign_similarity(set_to_eval, train_set, output_name, zeta,
                      remove_descriptor):
    max_similarity.main(train_set=train_set,
                        set_to_eval=set_to_eval,
                        output_fname=output_name,
                        zeta=zeta,
                        remove_descriptor=remove_descriptor)







@subcli_qm.command('normal-modes-orca')
@click.argument("inputs", nargs=-1)
@click.option('--prop-prefix', '-p', default='dft_', show_default=True,
              help='Prefix for "energy", "forces" and normal mode '
                   'properties',)
@click.option("--outputs", "-o", help="Output filename, see Configset for details", required=True)
@click.option("--dir-prefix", default='orca_')
def generate_nm_reference(inputs, prop_prefix, outputs, dir_prefix):
    """Generates normal mode frequencies and displacements from a
     finite difference approximation of the mass-weighted Hessian matrix """

    configset_in = ConfigSet_in(input_files=inputs)
    configset_out = ConfigSet_out(output_files=outputs)

    calc_kwargs = {"orcasimpleinput" : 'UKS B3LYP def2-SV(P) def2/J D3BJ',
                   "orcablocks" : '%scf SmearTemp 5000 maxiter 500 end'}

    cfg = Config.load()
    scratch_path = cfg['other_paths']['scratch']


    calc = (orca.ExtendedORCA, [], calc_kwargs)

    generic_calc_kwargs = {'use_wdir':True,
                           'scratch_path':scratch_path,
                           'keep_files':'default',
                           'base_rundir':'orca_normal_mode_calc_outputs',
                           'dir_prefix':dir_prefix}

    vib.generate_normal_modes_parallel_hessian(inputs=configset_in,
                                          outputs=configset_out,
                                          calculator=calc,
                                          prop_prefix=prop_prefix,
                                generic_calc_kwargs=generic_calc_kwargs)




@subcli_qm.command('pop-get')
@click.option('--orca-out', help='orca output to read populations from')
@click.option('--xyz-in', help='xyz to read geometry from ')
@click.option('--orca-in', help='orca input to read geometry from, '
                                'alternative to xyz-in')
@click.option('--xyz-out', help='where to write populations to')
@click.option('--pop', help='which population to color by')
@click.option('--cmap', help='matplotlib cmap to color by')
def get_populations(orca_out, xyz_in, orca_in, xyz_out, pop, cmap):
    '''  NA   - Mulliken gross atomic population
   ZA   - Total nuclear charge
   QA   - Mulliken gross atomic charge
   VA   - Mayer's total valence
   BVA  - Mayer's bonded valence
   FA   - Mayer's free valence
 '''
    from util import qm
    qm.print_populations(orca_out=orca_out,
                         xyz_in=xyz_in,
                         orca_in=orca_in,
                         xyz_out=xyz_out,
                         pop=pop,
                         cmap=cmap)

@subcli_qm.command('pop-colour')
@click.option('--in-fname', '-i', help='xyz with populations.')
@click.option('--out-fname', '-o', help='xyz to write to')
@click.option('--pop', '-p', help='population to color by')
@click.option('--cmap', '-c', help='matplotlib colormap')
@click.option('--min', type=click.FLOAT, default=-0.5)
@click.option('--max', type=click.FLOAT, default=0.4)
def color_by_populations(in_fname, out_fname, pop, cmap, min, max):
    from util import qm
    all_ats_in = read(in_fname, ':')
    all_ats_out = [qm.color_by_pop(at, pop, cmap, min, max) for at in
                   all_ats_in]
    write(out_fname, all_ats_out)


@subcli_configs.command('info-to-no')
@click.argument('fname_in')
@click.option('--info-key', '-i', help='atosm.info key to nubmer')
@click.option('--output', '-o', help='output filename')
def info_to_numbers(fname_in, info_key, output):

    ats = read(fname_in, ':')

    entries = list(set([at.info[info_key] for at in ats]))
    entries_dict = {}
    for idx, entry in enumerate(entries):
        entries_dict[entry] = idx

    for at in ats:
        at.info[info_key + '_no'] = entries_dict[at.info[info_key]]

    print(entries_dict)

    write(output, ats)

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

    isolated_h = read(isolated_h_fname, ':')
    for at in isolated_h:
        if list(at.symbols)[0] == 'H':
            isolated_h = at
            break
    else:
        raise RuntimeError(f"Could not find isolated H in {isolated_h_fname}")

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





@subcli_configs.command('select-with-info')
@click.option('--input-filename', '-i',
                       help='input filename with all info entries')
@click.option('--output-filename', '-o')
@click.option('--info-key', '-k', help='info key to select configs by')
def select_with_info(input_filename, output_filename, info_key):

    ats_out = []
    ats = read(input_filename, ':')
    for at in ats:
        if info_key in at.info.keys():
            ats_out.append(at)
    write(output_filename, ats_out)


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







@subcli_bde.command('dft-reoptimise')
@click.argument('input-filename')
@click.option('--output-filename', '-o')
@click.option('--dft-prop-prefix', '-p', default='dft_reopt_')
def reoptimise_with_dft(input_filename, output_filename, dft_prop_prefix):

    inputs = ConfigSet_in(input_files=input_filename)
    outputs = ConfigSet_out(output_files=output_filename)

    bde.dft_optimise(inputs=inputs, outputs=outputs, dft_prefix=dft_prop_prefix)





@subcli_plot.command('rmsd-table')
@click.option('--fname1', '-f1')
@click.option('--fname2', '-f2')
@click.option('--key', '-k', default='compound')
def print_rmsd_table(fname1, fname2, key):
    from util.plot import rmsd_table
    rmsd_table.rmsd_table(fname1, fname2, group_key=key)


@subcli_configs.command('cleanup-info-entries')
@click.argument('input-fname')
@click.option('--output-fname', '-o')
def cleanup_info_entries(input_fname, output_fname):
    from util import configs
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

    from util import normal_modes
    inputs = ConfigSet_in(input_files=input_fname)
    outputs = ConfigSet_out(output_files=output_fname)
    if info_to_keep is not None:
        info_to_keep = info_to_keep.split()
    if arrays_to_keep is not None:
        arrays_to_keep = arrays_to_keep.split()

    normal_modes.sample_downweighted_normal_modes(inputs=inputs,
                                                outputs=outputs,
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


#
# @subcli_plot.command('iterations')
# @click.option('--test-fname-pattern', default='xyzs/gap_{idx}_on_test_{idx}.xyz', show_default=True)
# @click.option('--train-fname-pattern', default='xyzs/gap_{idx}_on_train_{idx}.xyz',
#               show_default=True, help='pattern with "{idx}" in the title to be replaced')
# @click.option('--ref-prefix', default='dft_', show_default=True)
# @click.option('--pred-prefix-pattern', default='gap_{idx}_', show_default=True,
#               help='pattern with {idx} to be replaced')
# @click.option('--plot-prefix')
# @click.option('--num-cycles', type=click.INT, help='number of iterations', required=True)
# @click.option('--gap-bde-dir-pattern', default='bdes_from_dft_min/gap_{idx}_bdes_from_dft',
#               show_default=True)
# @click.option('--dft-bde-dir', default='/home/eg475/dsets/small_cho/bde_files/train/dft/')
# @click.option('--measure', default='bde_correlation',
#               type=click.Choice(['bde', 'rmsd', 'soap_dist', 'gap_e_error', \
#                                 'gap_f_rmse', 'gap_f_max', 'bde_correlation']),
#              help='which property to look at')
# @click.option('--no-lc', is_flag=True, help='turns off learning curves')
# @click.option('--no-means', is_flag=True, help='plot all lines, not just means')
# @click.option('--no-bde', is_flag=True, help='turns off bde related plots')
# def plot_cycles(test_fname_pattern, train_fname_pattern, ref_prefix, pred_prefix_pattern, plot_prefix,
#                 num_cycles, gap_bde_dir_pattern, dft_bde_dir, measure, no_lc, no_means, no_bde):
#
#     num_cycles += 1
#
#     train_fnames = [train_fname_pattern.replace('{idx}', str(idx)) for idx in range(num_cycles)]
#     test_fnames = [test_fname_pattern.replace('{idx}', str(idx)) for idx in
#                     range(num_cycles)]
#     pred_prefixes = [pred_prefix_pattern.replace('{idx}', str(idx)) for idx in range(num_cycles)]
#
#     if not no_lc:
#         iterations.learning_curves(train_fnames=train_fnames, test_fnames=test_fnames,
#                                    ref_prefix=ref_prefix, pred_prefix_list=pred_prefixes,
#                                plot_prefix=plot_prefix)
#
#     means = True
#     if no_means:
#         means=False
#
#     if not no_bde:
#         iterations.bde_related_plots(num_cycles=num_cycles, gap_dir_pattern=gap_bde_dir_pattern,
#                                     dft_dir=dft_bde_dir, metric=measure, plot_prefix=plot_prefix,
#                                  means=means)



@subcli_configs.command('filter-geometry')
@click.argument('in-fname')
@click.option('--mult', type=click.FLOAT, default=1.2, help='factor for bond cutoffs')
@click.option('--out_fname', '-o', help='output fname')
def filter_geometry(in_fname, mult, out_fname):
    ats_in = read(in_fname, ':')
    ats_out = configs.filter_expanded_geometries(ats_in, mult)
    write(out_fname, ats_out)



@subcli_configs.command('atom-type-aromatic')
@click.argument('in-fname')
@click.option('--output', '-o')
@click.option('--cutoff-multiplier', '-m', type=click.FLOAT, default=1,
              help='multiplier for cutoffs')
@click.option('--elements', '-e', help='List of elements to atom type in aromatic and non')
@click.option('--force', '-f', is_flag=True, help='whether to force overwriting configs_out')
def atom_type_aromatic(in_fname, output, cutoff_multiplier, elements,
                         force):
    """atom types aromatic vs not elements, based on neighbour count"""

    elements = elements.split(" ")
    inputs = ConfigSet_in(input_files=in_fname)
    outputs = ConfigSet_out(output_files=output, force=force)
    atom_types.assign_aromatic(inputs=inputs, outputs=outputs,
                               elements_to_type=elements,
                               mult=cutoff_multiplier)

@subcli_configs.command('atom-type')
@click.argument('input-fname')
@click.option('--output-fname', '-o')
@click.option('--isolated_at', is_flag=True, help='whether should append '
                                                  'isolated atoms')
def atom_type(input_fname, output_fname, isolated_at):
    """assigns atom types based on given reference .yml files"""

    ats_in = read(input_fname, ':')
    ats_out = [atom_types.atom_type(at) for at in ats_in if len(at) != 1]
    if isolated_at:
        ats_out += atom_types.atom_type_isolated_at()

    write(output_fname, ats_out)



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


@subcli_gap.command('opt-and-nm')
@click.option('--dft-dir', '-d')
@click.option('--gap-fnames', '-g')
@click.option('--output-dir', '-o')
def gap_reopt_dft_and_derive_normal_modes(dft_dir, gap_fnames, output_dir):
    compare_minima.opt_and_normal_modes(dft_dir, gap_fnames, output_dir)




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

    from quippy.potential import Potential

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





@subcli_data.command('target-from-diff')
@click.argument('input-fn')
@click.option('--output-fn', '-o')
@click.option('--prop-prefix-1', '-p1', help='property prefix for first set of values')
@click.option('--prop-prefix-2', '-p2')
@click.option('--out-prop-prefix', '-po', help='prop prefix for output')
def target_values_from_predicted_diff(input_fn, output_fn, prop_prefix_1,
                                      prop_prefix_2, out_prop_prefix):

    if out_prop_prefix is None:
        out_prop_prefix = prop_prefix_1 + 'plus_' + prop_prefix_2

    ats = read(input_fn, ':')
    for at in ats:
        at.info[f'{out_prop_prefix}energy'] = \
            at.info[f'{prop_prefix_1}energy'] + \
            at.info[f'{prop_prefix_2}energy']

        if f'{prop_prefix_1}forces' in at.arrays.keys() and \
                f'{prop_prefix_2}forces' in at.arrays.keys():
            at.arrays[f'{out_prop_prefix}forces'] = \
                at.arrays[f'{prop_prefix_1}forces'] + \
                at.arrays[f'{prop_prefix_2}forces']

    write(output_fn, ats)




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

    from util import plot

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





