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
    if stds is not None:
        stds = util.str_to_list(stds)
    else:
        stds = []

    if temps is not None:
        temps = [int(t) for t in util.str_to_list(temps)]
    else:
        temps=[]

    print(f'Standard deviations for this run: {stds}')
    print(f'SMILES for this run: {smiles}')

    db_path = '/home/eg475/scripts/'


    print(f'Parameters: gap {gap_fname}, dft equilibrium fname {dft_eq_xyz}, '
          f' fmax {fmax}, max_steps {max_steps}, stds for rattle: {stds}')


    if not os.path.exists('xyzs'):
        os.makedirs('xyzs')
    if not os.path.exists('pictures'):
        os.makedirs('pictures')

    dft_confs = read(pj(db_path, 'dft_minima', dft_eq_xyz), ':')

    opt_wdir = 'xyzs/opt_xyzs'
    if not os.path.isdir(opt_wdir):
        os.makedirs(opt_wdir)

    std_optimisations(stds, dft_confs, gap_fname, dft, db_path, opt_wdir=opt_wdir, fmax=fmax, max_steps=max_steps, no_cores=no_cores)
    smi_optimisations(smiles, gap_fname, dft_confs, db_path, opt_wdir=opt_wdir, fmax=fmax, max_steps=max_steps, no_cores=no_cores)
    nm_optimisations(temps=temps, gap_fname=gap_fname, dft_confs=dft_confs, opt_wdir=opt_wdir,
                     db_path=db_path, fmax=fmax, max_steps=max_steps, no_cores=no_cores)

    if cleanup:
        print(f'Removing working dir {opt_wdir}.')
        shutil.rmtree(opt_wdir)


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

                start_at_fname = pj(db_path, f'gopt_test/starts/NM_starts_{conf_name}_{temp}K.xyz')
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

        make_kpca_picture(gap_fname, dft_confs, all_opt_ends_fnames, temp=temp)


def geo_opt_in_batches(gap_fname, start_fnames, finish_fnames, traj_fnames, steps, fmax, no_cores):


    for batch in util.grouper(zip(start_fnames, finish_fnames, traj_fnames), no_cores):

        batch = [b for b in batch if b is not None]
        bash_call = ''

        for start_fname, finish_fname, traj_fname in batch:

           bash_call += f"python /home/eg475/scripts/gopt_test/single_optimisation.py " \
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


def smi_optimisations(smiles, gap_fname, dft_confs, db_path, opt_wdir, fmax, max_steps, no_cores):

    all_opt_ends_fnames = []

    for smi in smiles:

        ends_fname = f'xyzs/opt_ends_{smi}.xyz'
        all_opt_ends_fnames.append(ends_fname)

        if not os.path.isfile(ends_fname):

            all_start_fnames = []
            all_finish_fnames = []
            all_traj_fnames = []

            start_at_fname = pj(db_path, f'gopt_test/starts/rdkit_starts_{smi}.xyz')
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

    make_kpca_picture(gap_fname, dft_confs, all_opt_ends_fnames, smiles=smiles)


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

                start_at_fname = pj(db_path, f'gopt_test/starts/starts_{conf_name}_{std}A_std.xyz')
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


        make_kpca_picture(gap_fname, dft_confs, all_opt_ends_fnames, std=std)



def make_kpca_picture(gap_fname, dft_eq_ats, all_opt_ends_fnames, std=None, smiles=None, temp=None):
    '''main function to do the kpca'''

    if std is not None:
        kpca_name = f'xyzs/kpca_{std}A_std.xyz'
    elif smiles is not None:
        kpca_name = f'xyzs/kpca_smiles.xyz'
    elif temp is not None:
        kpca_name = f'xyzs/kpca_{temp}K_nm.xyz'
    else:
        raise RuntimeError('give either smiles, temp or std please!')


    prepare_xyz_for_kpca(all_opt_ends_fnames, dft_eq_ats, kpca_name, std)

    util.do_kpca(kpca_name)

    end_pairs, dft_min = sort_kpca_atoms(kpca_name)
    plot_actual_kpca_plot(end_pairs, dft_min, gap_fname, std=std, smiles=smiles, temp=temp)


def prepare_xyz_for_kpca(all_opt_ends_fnames, dft_eq_ats, kpca_name, std):
    '''labels different datapoints and puts into a single xyz to do kpca on'''
    # make kpca dataset

    # Just set config_type to grab when plotting
    for at in dft_eq_ats:
        at.info['config_type'] = 'dft'

    # collect all atoms together
    opt_ends = []
    for ends_fname in all_opt_ends_fnames:
        opt_ends += read(ends_fname, ':')


    write(kpca_name,  opt_ends + dft_eq_ats, 'extxyz',
          write_results=False)

def sort_kpca_atoms(kpca_name):
    '''sorts atoms with kpca coords into different groups for plotting'''
    atoms = read(kpca_name, ':')

    dft_min = [at for at in atoms if 'dft' in at.info['config_type']]
    starts = [at for at in atoms if 'start' in at.info['config_type'] ]
    finishes = [at for at in atoms if 'finish' in at.info['config_type']]

    if len(starts) != len(finishes):
        raise Exception(f'Number of trajectory starts ({len(starts)}) differs from number of trajectory finishes ({len(finish)}')

    end_pairs = []
    for at_s, at_f in zip(starts, finishes):
        no_start = int(re.findall('\d+', at_s.info['config_type'])[-1])
        no_finish = int(re.findall('\d+', at_f.info['config_type'])[-1])

        if no_start != no_finish:
            raise Exception(f'Start index ({no_start}) doesn\'t match finish index ({no_finish}')

        pair = (at_s, at_f)
        end_pairs.append(pair)


    return  end_pairs, dft_min

def plot_actual_kpca_plot(end_pairs, dft_min, gap_fname, std=None, smiles=None, temp=None):
    '''plots the pca of three groups of points from what's in ther at.ino['pca_coord']'''

    gap_no = os.path.splitext(os.path.basename(gap_fname))[0]
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


    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.legend()
    if std:
        plt.title(f'kPCA of {gap_no} optimisation test, displacement std {std} Å')
        plt.tight_layout()
        plt.savefig(f'pictures/kpca_{gap_no}_{std}A_std.png', dpi=300)
    elif smiles:
        plt.title(
            f'kPCA of {gap_no} opt test, RDKit conformers')
        plt.tight_layout()
        plt.savefig(f'pictures/kpca_{gap_no}_rdkit_confs.png', dpi=300)
    elif temp:
        plt.title(f'kPCA of {gap_no} opt test, temperature for NM {temp} K')
        plt.tight_layout()
        plt.savefig(f'pictures/kpca_{gap_no}_{temp}K_nm.png', dpi=300)


if __name__ == '__main__':
    do_gap_geometry_optimisation_test()
