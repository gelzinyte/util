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

@click.command()
@click.option('--gap_fname', type=click.Path(exists=True), help='GAP xml to test.')
@click.option('--dft_eq_xyz', type=click.Path(exists=True), help='.xyz of DFT equilibrium structures')
@click.option('--no_runs', type=int, default=10, show_default=True, help='Number of runs  *per DFT reference structure per std* to do.')
@click.option('--fmax', type=float, default=0.01, show_default=True, help='Fmax for optimisation.')
@click.option('--max_steps', type=int, default=500, show_default=True,help='Maximum number of steps allowed in optimisation.')
@click.option('--stds', type=str, default='[0.1]', help='list of standard deviations to test')
@click.option('--quick', type=bool, default=False, help='if true, don\'t load GAP and don\'t do kpca')
def do_gap_geometry_optimisation_test(gap_fname, dft_eq_xyz, no_runs, stds,
                fmax, max_steps, quick):

    # set up
    stds = stds.strip('][').split(', ')
    stds = [float(std) for std in stds]


    print(f'Parameters: gap {gap_fname}, dft equilibrium fname {dft_eq_xyz}, '
          f'number of runs per structure {no_runs}, '
          f' fmax {fmax}, max_steps {max_steps}, stds for rattle: {stds}')

    if quick:
        print('Quick run: not loading GAP and not re-generating kpca')


    if not os.path.exists('xyzs'):
        os.makedirs('xyzs')

    if not quick:
        gap = Potential(param_filename=gap_fname)
    dft_confs = read(dft_eq_xyz, ':')


    for std in stds:

        ends_fname = f'all_opt_ends_{std}A_std'
        all_traj_ends = []

        for conf in dft_confs:
            for idx in range(no_runs):

                seed = int(time.time() * 1e7) % (2**32-1)
                at = conf.copy()
                traj_name = f'xyzs/{at.info["name"]}_{std}A_std_{idx}'

                if not os.path.isfile(f'{traj_name}.xyz'):
                    at.rattle(stdev=std, seed=seed)
                    tests.do_gap_optimisation(at, traj_name, fmax, max_steps, gap=gap)

                traj = read(f'{traj_name}.xyz', ':')
                start = traj[0]
                start.info['config_type'] = f'start_{idx}'
                finish = traj[-1]
                finish.info['config_type'] = f'finish_{idx}'

                all_traj_ends += [start, finish]

        write(f'{ends_fname}.xyz', all_traj_ends, 'extxyz', write_results=False)

        make_kpca_picture(gap_fname, dft_eq_xyz, ends_fname, std, quick)

    for dft_at in dft_confs:
        make_rmsd_vs_std_summary_plot(gap_fname, dft_at, stds)

def make_rmsd_vs_std_summary_plot(gap_fname, dft_at, stds):

    gap_no = int(re.findall('\d+', gap_fname)[0])

    struct_name = dft_at.info['name']
    plot_start_means = []
    plot_start_stds = []
    plot_finish_means = []
    plot_finish_stds = []
    for std in stds:
        ends = read(f'all_opt_ends_{std}A_std.xyz', ':')
        starts = [at for at in ends if 'start' in at.info['config_type'] and struct_name == at.info['name']]
        finishes = [at for at in ends if 'finish' in at.info['config_type'] and struct_name ==at.info['name']]

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


    #Actual plot
    fig = plt.figure(figsize=(10, 7))

    kw_args = {'fmt': 'X', 'ms': 6, 'capsize': 6, 'elinewidth': 1, 'capthick': 2, 'markeredgewidth': 2}
    plt.errorbar(stds, plot_start_means, yerr=plot_start_stds, c='tab:green', **kw_args)
    plt.plot(stds, plot_start_means, c='tab:green', linewidth=0.8, linestyle='--', label='Opt. traj. beginning')

    plt.errorbar(stds, plot_finish_means, yerr=plot_finish_stds, c='tab:orange', **kw_args)
    plt.plot(stds, plot_finish_means, c='tab:orange', linewidth=0.8, linestyle='--', label='Opt. traj. end')

    plt.title('Final structure\'s RMSD mean $\pm$ std.')
    plt.xlabel('Random displacement\'s standard deviation, Å')
    plt.ylabel('RMSD, Å')

    plt.legend()
    plt.savefig(f'GAP_{gap_no}_{struct_name}_test_summary.png', dpi=300)




def make_kpca_picture(gap_fname, dft_eq_xyz, ends_fname, std, quick):
    '''main function to do the kpca'''

    kpca_in_name = f'{ends_fname}_for_kpca.xyz'
    kpca_out_name = "ASAP-lowD-map.xyz"
    # kpca_out_name = f'xyzs/{ends_fname}_out_of_kpca.xyz'

    if not quick:
        prepare_xyz_for_kpca(ends_fname, gap_fname, dft_eq_xyz, kpca_in_name)
        do_kpca(kpca_in_name)

    ats_train, end_pairs, dft_min = sort_kpca_atoms(kpca_out_name)
    plot_actual_kpca_plot(ats_train, end_pairs, dft_min, gap_fname, std)

def do_kpca(kpca_in_name):
    gen_desc = f'asap gen_desc -f {kpca_in_name} --no-periodic soap'
    map = "asap map -f ASAP-desc.xyz -dm '[*]' pca"

    subprocess.run(gen_desc, shell=True)
    subprocess.run(map, shell=True)



def prepare_xyz_for_kpca(ends_fname, gap_fname, dft_eq_xyz, kpca_in_name):
    # make kpca dataset
    training_ats = ugap.atoms_from_gap(gap_fname, 'tmp_ats.xyz')
    training_ats = [at for at in training_ats if len(at)!=1]
    for at in training_ats:
        at.info['config_type'] = 'training'
    dft_eq_ats = read(dft_eq_xyz, ':')
    for at in dft_eq_ats:
        at.info['config_type'] = at.info['name']
    opt_ends = read(f'{ends_fname}.xyz', ':')
    write(kpca_in_name, training_ats + opt_ends + dft_eq_ats, 'extxyz',
          write_results=False)
    '''labels different datapoints and puts into a single xyz to do kpca on'''

def sort_kpca_atoms(kpca_out_name):
    '''sorts atoms with kpca coords into different groups for plotting'''
    atoms = read(kpca_out_name, ':')
    dft_min = []
    ats_train = []
    starts = []
    finishes = []
    end_pairs = []

    for at in atoms:
        if at.info['config_type'] == 'training':
            ats_train.append(at)

        elif at.info['name'] == at.info['config_type']:
            dft_min.append(at)

        elif 'start' in at.info['config_type']:
            starts.append(at)
        elif 'finish' in at.info['config_type']:
            finishes.append(at)
        else:
            raise RuntimeError(
                'Couldn\'t assign an atom to either training, '
                'dft equilibrium '
                'or gap optimisation start or finish')

    # group pairs of start and finish
    for start_at, finish_at in zip(starts, finishes):
        no_start = int(re.findall('\d+', start_at.info['config_type'])[0])
        no_finish = int(re.findall('\d+', finish_at.info['config_type'])[0])
        if no_start != no_finish:
            print(f'no_start: {no_start}')
            print(f'no_finish: {no_finish}')
            raise runtimeError(
                'start and finish atoms\' indices do not match')

        pair = (start_at, finish_at)
        end_pairs.append(pair)

    return ats_train, end_pairs, dft_min

def plot_actual_kpca_plot(ats_train, end_pairs, dft_min, gap_fname, std):
    '''plots the pca of three groups of points from what's in ther at.ino['pca_coord']'''

    gap_no = int(re.findall('\d+', gap_fname)[0])
    pca_dict_key = 'pca-d-10'

    pcax = 0
    pcay = 1

    marker_shapes = ['X', '^', 'o', 'D', 's', '*']
    dft_min_name_mareker_shape_dict = {}


    # suspicious_ats = []
    # for at in ats_train:
    #     if at.info[pca_dict_key][pcax] < -1 and at.info[pca_dict_key][pcay] > 2:
    #         suspicious_ats.append(at)
    # write('suspicious_ats.xyz', suspicious_ats, 'extxyz', write_results=False)


    fig = plt.figure(figsize=(10, 7))


    # training set
    xs_train = [at.info[pca_dict_key][pcax] for at in ats_train]
    ys_train = [at.info[pca_dict_key][pcay] for at in ats_train]
    plt.scatter(xs_train, ys_train, color='grey', marker='.', s=5, label='Training set', zorder=0)



    # dft minima
    for idx, at in enumerate(dft_min):


        dft_min_name = at.info['name']
        marker = marker_shapes[idx]
        dft_min_name_mareker_shape_dict[dft_min_name] = marker
        label = f'{dft_min_name} DFT minimum'

        x = at.info[pca_dict_key][pcax]
        y = at.info[pca_dict_key][pcay]

        plt.scatter(x, y, marker=marker, color='tab:red', s=80, linewidth=0.5,
                    linewidths=10, edgecolors='k', label=label, zorder=3)


    # optimisation ends
    for idx, (at_s, at_f) in enumerate(end_pairs):

        label_s = None
        label_f = None
        if idx == 0:
            label_s = 'Opt. start'
            label_f = 'Opt. end'

        xs = [at_s.info[pca_dict_key][pcax], at_f.info[pca_dict_key][pcax]]
        ys = [at_s.info[pca_dict_key][pcay], at_f.info[pca_dict_key][pcay]]

        parent_dft_min_name = at_s.info['name']
        marker = dft_min_name_mareker_shape_dict[parent_dft_min_name]


        plt.plot(xs, ys, c='k', linewidth=0.6, zorder=1)
        plt.scatter(xs[0], ys[0], color='tab:green', marker=marker, label=label_s, edgecolors='k', linewidth=0.4,  zorder=2)
        plt.scatter(xs[1], ys[1], color='tab:orange', marker=marker, edgecolors='k',label=label_f, linewidth=0.4, zorder=2)





    plt.title(f'kPCA of GAP {gap_no} optimisation test, displacement std {std} Å')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'GAP_{gap_no}_kpca_{std}A_std.png', dpi=300)



if __name__=='__main__':
    do_gap_geometry_optimisation_test()


