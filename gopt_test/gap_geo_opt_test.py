#!/usr/bin/env python3

import sys
from util import tests, ugap
from ase.io import read, write
import time
import click
import util
import re
import os
from quippy.potential import Potential
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import shutil
from os.path import join as pj

@click.command()
@click.option('--gap_fname', type=click.Path(exists=True), help='GAP xml to test.')
@click.option('--dft_eq_xyz', type=click.Path(), help='.xyz of DFT equilibrium structures')
@click.option('--fmax', type=float, default=0.01, show_default=True, help='Fmax for optimisation.')
@click.option('--max_steps', type=int, default=500, show_default=True,help='Maximum number of steps allowed in optimisation.')
@click.option('--stds', type=str, default='[0.1]', help='list of standard deviations to test')
@click.option('--dft', type=bool, default=False, help='whether should be looking for and plotting dft-optimised structures')
@click.option('--no_cores', type=int, help='number of cores to parallelise over')
@click.option('--cleanup', type=bool, default=True, show_default=True, help='whether to delete all individual optimisation paths')
@click.option('--temps', type=str, help='list of temps for normal mode displaceents')
@click.argument('smiles', nargs=-1)
def do_gap_geometry_optimisation_test(gap_fname, dft_eq_xyz,  stds,
                fmax, max_steps,  dft, no_cores, cleanup, temps, smiles):

    # set up
    stds = util.str_to_list(stds)
    temps = [int(t) for t in util.str_to_list(temps)]

    print(f'Standard deviations for this run: {stds}')
    print(f'SMILES for this run: {smiles}')

    db_path = '/home/eg475/programs/my_scripts/gopt_test/'


    print(f'Parameters: gap {gap_fname}, dft equilibrium fname {dft_eq_xyz}, '
          f' fmax {fmax}, max_steps {max_steps}, stds for rattle: {stds}')


    if not os.path.exists('xyzs'):
        os.makedirs('xyzs')

    dft_confs = read(pj(db_path, 'dft_minima', dft_eq_xyz), ':')

    opt_wdir = 'xyzs/opt_xyzs'
    if not os.path.isdir(opt_wdir):
        os.makedirs(opt_wdir)

    std_optimisations(stds, dft_confs, gap_fname, dft, db_path, opt_wdir=opt_wdir, fmax=fmax, max_steps=max_steps, no_cores=no_cores)
    smi_optimisations(smiles, gap_fname, db_path, opt_wdir=opt_wdir, fmax=fmax, max_steps=max_steps, no_cores=no_cores)
    nm_optimisations(temps=temps, gap_fname=gap_fname, dft_confs=dft_confs, opt_wdir=opt_wdir,
                     db_path=db_path, fmax=fmax, max_steps=max_steps, no_cores=no_cores)

    if cleanup:
        print(f'Removing working dir {opt_wdir}.')
        shutil.rmtree(opt_wdir)

    #
    # for dft_at in dft_confs:
    #     print(f'Summary plot for {dft_at.info["name"]}')
    #     make_rmsd_vs_std_summary_plot(gap_fname, dft_at, stds, dft)

def nm_optimisations(temps, gap_fname, dft_confs, opt_wdir, db_path, fmax, max_steps, no_cores):

    for temp in temps:

        all_opt_ends_fnames = []

        for conf in dft_confs:

            conf_name = conf.info['name']
            ends_fname =  f'xyzs/opt_ends_{conf_name}_{temp}K.xyz'
            all_opt_ends_fnames.append(ends_fname)

            if not os.path.isfile(ends_fname):

                all_start_fnames = []
                all_finish_fnames = []
                all_traj_fnames = []

                start_at_fname = pj(db_path, f'starts/NM_starts_{conf_name}_{temp}K.xyz')
                start_ats = read(start_at_fname, ':')

                finishes_at_fname = f'xyzs/finishes_{conf_name}_{temp}K.xyz'
                local_starts_fname = f'xyzs/starts_{conf_name}_{temp}K.xyz'

                for idx, at in enumerate(start_ats):
                    start_at_fname = pj(opt_wdir, f'start_{conf_name}_{temp}K_{idx}.xyz')
                    finish_at_fname = pj(opt_wdir, f'finish_{conf_name}_{temp}K_{idx}.xyz')
                    traj_name = pj(opt_wdir, f'opt_{conf_name}_{temp}K_{idx}')

                    write(start_at_fname, at, 'extxyz', write_results=False)

                    all_start_fnames.append(start_at_fname)
                    all_finish_fnames.append(finish_at_fname)
                    all_traj_fnames.append(traj_name)


                geo_opt_in_batches(gap_fname=gap_fname,
                                   start_fnames=all_start_fnames,
                                   finish_fnames=all_finish_fnames,
                                   traj_fnames=all_traj_fnames,
                                   steps=max_steps, fmax=fmax, no_cores=no_cores)


                shuffle_opt_trajs_around(all_start_fnames, all_finish_fnames,
                                         ends_fname, finishes_at_fname, local_starts_fname)

            else:
                print(f'Found {ends_fname}, not re-optimising')

        # make_kpca_picture(gap_fname, dft_confs, all_opt_ends_fnames, smiles=smiles)


def geo_opt_in_batches(gap_fname, start_fnames, finish_fnames, traj_fnames, steps, fmax, no_cores):


    for batch in util.grouper(zip(start_fnames, finish_fnames, traj_fnames), no_cores):

        batch = [b for b in batch if b is not None]
        bash_call = ''

        for start_fname, finish_fname, traj_fname in batch:

           bash_call += f"python /home/eg475/programs/my_scripts/gopt_test/single_optimisation.py " \
                       f"--gap_fname {gap_fname} " \
                       f"--start_fname '{start_fname}' " \
                        f"--finish_fname '{finish_fname}' " \
                        f"--traj_name '{traj_fname}' " \
                        f"--steps {steps} " \
                        f"--fmax {fmax} " \
                        f"&\n"

        bash_call += 'wait\n'

        stdout, stderr = util.shell_stdouterr(bash_call)

        print(f'---opt stdout:\n {stdout}')
        print(f'---opt stderr:\n {stderr}')

def shuffle_opt_trajs_around(start_fnames, finish_fnames, ends_fname, finishes_at_fname, local_starts_fname):

    ends = []
    finishes = []
    starts = []
    for idx, (start_fn, finish_fn) in enumerate(zip(start_fnames, finish_fnames)):
        if os.path.isfile(finish_fn):
            at_s = read(start_fn)
            # at_s.info['config_type'] = f'start_{idx}'
            #at_s should have all info already (e.g. start_4)

            at_f = read(finish_fn)
            at_f.info['config_type'] = f'finish_{idx}'

            ends += [at_s, at_f]
            finishes.append(at_f)
            starts.append(at_s)
        else:
            print(f'Do not have optimised file {finish_fn}, skipping')

    write(ends_fname, ends, 'extxyz', write_results=False)
    write(finishes_at_fname, finishes, 'extxyz', write_results=False)
    write(local_starts_fname, starts, 'extxyz', write_results=False)


def smi_optimisations(smiles, gap_fname, db_path, opt_wdir, fmax, max_steps, no_cores):

    all_opt_ends_fnames = []

    for smi in smiles:

        ends_fname = f'xyzs/opt_ends_{smi}.xyz'
        all_opt_ends_fnames.append(ends_fname)

        if not os.path.isfile(ends_fname):

            all_start_fnames = []
            all_finish_fnames = []
            all_traj_fnames = []

            start_at_fname = pj(db_path, f'starts/rdkit_starts_{smi}.xyz')
            start_ats = read(start_at_fname, ':')

            finishes_at_fname = f'xyzs/finishes_{smi}.xyz'
            local_starts_fname = f'xyzs/starts_{smi}.xyz'

            for idx, at in enumerate(start_ats):
                start_at_fname = pj(opt_wdir, f'start_{smi}_{idx}.xyz')
                finish_at_fname = pj(opt_wdir, f'finish_{smi}_{idx}.xyz')
                traj_name = pj(opt_wdir, f'opt_{smi}_{idx}')


                write(start_at_fname, at, 'extxyz', write_results=False)

                all_start_fnames.append(start_at_fname)
                all_finish_fnames.append(finish_at_fname)
                all_traj_fnames.append(traj_name)


            print('running geometry optimisation in batches')
            geo_opt_in_batches(gap_fname=gap_fname, start_fnames=all_start_fnames,
                               finish_fnames=all_finish_fnames, traj_fnames=all_traj_fnames,
                           steps=max_steps, fmax=fmax, no_cores=no_cores)
            print('finished geometry optimisation in batches')

            shuffle_opt_trajs_around(all_start_fnames, all_finish_fnames, ends_fname, finishes_at_fname, local_starts_fname)

        else:
            print(f'Found {ends_fname}, not re-optimising')

    # make_kpca_picture(gap_fname, dft_confs, all_opt_ends_fnames, smiles=smiles)


def std_optimisations(stds, dft_confs, gap_fname, dft, db_path, opt_wdir, fmax, max_steps, no_cores):


    for std in stds:

        all_opt_ends_fnames = []

        for conf in dft_confs:

            conf_name = conf.info['name']
            ends_fname =  f'xyzs/opt_ends_{conf_name}_{std}A_std.xyz'
            all_opt_ends_fnames.append(ends_fname)

            if not os.path.isfile(ends_fname):

                all_start_fnames = []
                all_finish_fnames = []
                all_traj_fnames = []

                start_at_fname = pj(db_path, f'starts/starts_{conf_name}_{std}A_std.xyz')
                start_ats = read(start_at_fname, ':')

                finishes_at_fname = f'xyzs/finishes_{conf_name}_{std}A_std.xyz'
                local_starts_fname = f'xyzs/starts_{conf_name}_{std}A_std.xyz'

                for idx, at in enumerate(start_ats):
                    start_at_fname = pj(opt_wdir, f'start_{conf_name}_{std}A_std_{idx}.xyz')
                    finish_at_fname = pj(opt_wdir, f'finish_{conf_name}_{std}A_std_{idx}.xyz')
                    traj_name = pj(opt_wdir, f'opt_{conf_name}_{std}A_std_{idx}')

                    write(start_at_fname, at, 'extxyz', write_results=False)

                    all_start_fnames.append(start_at_fname)
                    all_finish_fnames.append(finish_at_fname)
                    all_traj_fnames.append(traj_name)


                geo_opt_in_batches(gap_fname=gap_fname,
                                   start_fnames=all_start_fnames,
                                   finish_fnames=all_finish_fnames,
                                   traj_fnames=all_traj_fnames,
                                   steps=max_steps, fmax=fmax, no_cores=no_cores)


                shuffle_opt_trajs_around(all_start_fnames, all_finish_fnames,
                                         ends_fname, finishes_at_fname, local_starts_fname)

            else:
                print(f'Found {ends_fname}, not re-optimising')


        # make_kpca_picture(gap_fname, dft_confs, all_opt_ends_fname, std=std, dft=dft)



def make_kpca_picture(gap_fname, dft_eq_ats, all_opt_ends_fname, std=None, smiles=None, dft=None):
    '''main function to do the kpca'''

    kpca_name = os.path.splitext(ends_name)[0] + '_kpca.xyz'

    prepare_xyz_for_kpca(ends_fname, dft_eq_ats, kpca_name, std, dft)

    util.do_kpca(kpca_name)

    end_pairs, dft_min, dft_finishes = sort_kpca_atoms(kpca_name)
    plot_actual_kpca_plot(end_pairs, dft_min, dft_finishes, gap_fname, std=std, smiles=smiles)


def prepare_xyz_for_kpca(ends_fname, dft_eq_ats, kpca_name, std, dft):
    # make kpca dataset
    for at in dft_eq_ats:
        at.info['config_type'] = at.info['name']
    opt_ends = read(f'{ends_fname}.xyz', ':')

    dft_optimised_ats = []
    if dft:
        for dft_at in dft_eq_ats:
            dft_optimised_ats += read(f'dft_finishes_{dft_at.info["name"]}_{std}A_std.xyz', ':')

    write(kpca_name,  opt_ends + dft_eq_ats + dft_optimised_ats, 'extxyz',
          write_results=False)
    '''labels different datapoints and puts into a single xyz to do kpca on'''

def sort_kpca_atoms(kpca_name):
    '''sorts atoms with kpca coords into different groups for plotting'''
    atoms = read(kpca_name, ':')
    dft_min = []
    starts = []
    finishes = []
    end_pairs = []
    dft_finishes = []


    for at in atoms:
        # if at.info['name'] == at.info['config_type']:
        #     dft_min.append(at)
        # TODO just make a config_type=dft
        if 'name' in at.info.keys():
            if at.info['name'] == at.info['config_type']:
                dft_min.append(at)

        elif 'start' in at.info['config_type']:
            starts.append(at)
        elif 'finish' in at.info['config_type']:
            finishes.append(at)
        elif 'dft_optimised' in at.info['config_type'] or 'failed_dft_opt' in at.info['config_type']:
            dft_finishes.append(at)
        else:
            raise RuntimeError(
                'Couldn\'t assign an atom to either training, '
                'dft equilibrium '
                'or gap optimisation start or finish')

    # group pairs of start and finish
    for start_at, finish_at in zip(starts, finishes):
        no_start = int(re.findall('\d+', start_at.info['config_type'])[-1])
        no_finish = int(re.findall('\d+', finish_at.info['config_type'])[-1])
        if no_start != no_finish:
            print(f'no_start: {no_start}')
            print(f'no_finish: {no_finish}')
            raise RuntimeError(
                'start and finish atoms\' indices do not match')

        pair = (start_at, finish_at)
        end_pairs.append(pair)

    # TODO group dft in ends and beginnings

    return  end_pairs, dft_min, dft_finishes

def plot_actual_kpca_plot(end_pairs, dft_min, dft_finishes, gap_fname, std=None, smiles=None):
    '''plots the pca of three groups of points from what's in ther at.ino['pca_coord']'''

    try:
        gap_no = int(re.findall('\d+', gap_fname)[0])
    except IndexError:
        gap_no = os.path.splitext(gap_fname)[0]
    pca_dict_key = 'pca_d_10'

    pcax = 0
    pcay = 1

    marker_shapes = ['X', '^', 'o', 'D', 's']
    dft_min_marker_shape_dict = {}


    fig = plt.figure(figsize=(10, 7))


    # dft minima
    for idx, at in enumerate(dft_min):


        dft_min_name = at.info['name']
        dft_min_smiles = at.info['smiles']
        marker = marker_shapes[idx]
        dft_min_marker_shape_dict[dft_min_name] = marker
        dft_min_marker_shape_dict[dft_min_smiles] = marker
        label = f'{dft_min_name} DFT minimum'

        x = at.info[pca_dict_key][pcax]
        y = at.info[pca_dict_key][pcay]

        plt.scatter(x, y, marker=marker, color='crimson', s=80, linewidth=0.5,
                    linewidths=10, edgecolors='k', label=label, zorder=4)


    # optimisation ends
    for idx, (at_s, at_f) in enumerate(end_pairs):

        label_s = None
        label_f = None
        if idx == 0:
            label_s = 'Opt. start'
            label_f = 'Opt. end'

        xs = [at_s.info[pca_dict_key][pcax], at_f.info[pca_dict_key][pcax]]
        ys = [at_s.info[pca_dict_key][pcay], at_f.info[pca_dict_key][pcay]]

        if 'name' in at_s.info.keys():
            parent_dft_min_name = at_s.info['name']
        elif 'smiles' in at_s.info.keys():
            parent_dft_min_name = at_s.info['smiles']
        else:
            raise RuntimeError('Could not find neither name nor smiles in dft minimum')

        marker = dft_min_marker_shape_dict[parent_dft_min_name]


        plt.plot(xs, ys, c='grey', linewidth=0.4 , zorder=1)
        plt.scatter(xs[0], ys[0], color='tab:orange', marker=marker, label=label_s, edgecolors='k', linewidth=0.4,  zorder=2)
        plt.scatter(xs[1], ys[1], color='tab:green', marker=marker, edgecolors='k',label=label_f, linewidth=0.4, zorder=3)

    # dft optimisation ends

    successful_dft_x = [at.info[pca_dict_key][pcax] for at in dft_finishes if 'failed' not in at.info['config_type']]
    successful_dft_y = [at.info[pca_dict_key][pcay] for at in dft_finishes if 'failed' not in at.info['config_type']]

    failed_dft_x = [at.info[pca_dict_key][pcax] for at in dft_finishes if 'failed' in at.info['config_type']]
    failed_dft_y = [at.info[pca_dict_key][pcay] for at in dft_finishes if 'failed' in at.info['config_type']]

    if len(successful_dft_x)!=0:
        plt.scatter(successful_dft_x, successful_dft_y, color='deepskyblue', edgecolors='k', linewidth=0.4, marker='*', s=50,  label='DFT-optimised', zorder=5)
    if len(failed_dft_x)!=0:
        plt.scatter(failed_dft_x, failed_dft_y, color='fuchsia', marker='*',edgecolors='k', linewidth=0.4,  s=50, label='DFT non-converged', zorder=5)

    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.legend()
    if std:
        plt.title(f'kPCA of GAP {gap_no} optimisation test, displacement std {std} Å')
        plt.tight_layout()
        plt.savefig(f'GAP_{gap_no}_kpca_{std}A_std.png', dpi=300)
    elif smiles:
        plt.title(
            f'kPCA of GAP {gap_no} optimisation test, RDKit conformers')
        plt.tight_layout()
        plt.savefig(f'GAP_{gap_no}_kpca_rdkit_confs.png', dpi=300)




def make_rmsd_vs_std_summary_plot(gap_fname, dft_at, stds, dft):

    try:
        gap_no = int(re.findall('\d+', gap_fname)[0])
    except IndexError:
        gap_no = os.path.splitext(gap_fname)[0]

    struct_name = dft_at.info['name']
    plot_start_means = []
    plot_start_stds = []
    plot_finish_means = []
    plot_finish_stds = []
    if dft:
        plot_dft_means = []
        plot_dft_stds = []
        dft_stds = []

    for std in stds:
        ends = read(f'all_opt_ends_{std}A_std.xyz', ':')
        ends = [at for at in ends if struct_name == at.info['name']]
        starts = [at for at in ends if 'start' in at.info['config_type']]
        finishes = [at for at in ends if 'finish' in at.info['config_type']]

        if dft:
            dft_finishes_fname = f'dft_finishes_{dft_at.info["name"]}_{std}A_std.xyz'
            dft_finishes = read(dft_finishes_fname, ':')
            dft_finishes = [at for at in dft_finishes if not 'failed_dft_opt' in at.info['config_type']]


        # write(f'starts_{struct_name}_{std}A_std.xyz', starts, 'extxyz', write_results=False)
        write(f'finishes_{struct_name}_{std}A_std.xyz', finishes, 'extxyz', write_results=False)
        # write(f'both_ends_{struct_name}_{std}A_std.xyz', ends, 'extxyz', write_results=False)

        if len(starts) != len(finishes):
            print(f'len(ends) should be four times that of starts and '
                f'finishes for conf*2_confs*2_ends')
            print( f'len(ends): {len(ends)}; len(starts): {len(starts)}; len(finishes): {len(finishes)}')
            raise RuntimeError("Have differing numbers of start and finish configurations, reconsider")

        rmsds_start = [util.get_rmse(dft_at.positions, at.positions) for at in starts]
        rmsds_finish = [util.get_rmse(dft_at.positions, at.positions) for at in finishes]

        plot_start_means.append(np.mean(rmsds_start))
        plot_start_stds.append(np.std(rmsds_start))
        plot_finish_means.append(np.mean(rmsds_finish))
        plot_finish_stds.append(np.std(rmsds_finish))

        if dft:
            rmsds_dft = [util.get_rmse(dft_at.positions, at.positions) for at in dft_finishes]
            if len(rmsds_dft)!=0:
                plot_dft_means.append(np.mean(rmsds_dft))
                plot_dft_stds.append(np.std(rmsds_dft))
                dft_stds.append(std)



    #Actual plot
    fig = plt.figure(figsize=(10, 7))

    kw_args = {'fmt': 'X', 'ms': 6, 'capsize': 6, 'elinewidth': 1, 'capthick': 2, 'markeredgewidth': 2}
    plt.errorbar(stds, plot_start_means, yerr=plot_start_stds, c='tab:orange', **kw_args)
    plt.plot(stds, plot_start_means, c='tab:orange', linewidth=0.8, linestyle='--', label='Opt. traj. beginning')

    plt.errorbar(stds, plot_finish_means, yerr=plot_finish_stds, c='tab:green', **kw_args)
    plt.plot(stds, plot_finish_means, c='tab:green', linewidth=0.8, linestyle='--', label='Opt. traj. end')

    if dft:
        plt.errorbar(dft_stds, plot_dft_means, yerr=plot_dft_means,
                     c='tab:blue', **kw_args)
        plt.plot(dft_stds, plot_dft_means, c='tab:blue', linewidth=0.8,
                 linestyle='--', label='DFT-optimised')

    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim(bottom=0)
    plt.grid(color='lightgrey', which='both')

    plt.title(f'{struct_name} final structures\' RMSD mean $\pm$ std.')
    plt.xlabel('Random displacement\'s standard deviation, Å')
    plt.ylabel('RMSD, Å')

    plt.legend()
    plt.savefig(f'GAP_{gap_no}_{struct_name}_test_summary.png', dpi=300)

if __name__ == '__main__':
    do_gap_geometry_optimisation_test()
