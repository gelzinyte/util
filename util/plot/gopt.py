import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from os.path import join as pj
import util
from ase.io import read
from quippy.descriptors import Descriptor




def gopt_plot_summary(ax, wdir, struct_names, start_label, task, all_dft_ats,
                      smiles=None, temps=None, stds=None, **end_kwargs):

    given = [entry is not None for entry in [smiles, temps, stds]]
    if given.count(True) != 1:
        raise RuntimeError(
            "Only one of 'smiles', 'temps' and 'stds' can be given")

    plot_start_means = []
    plot_finish_means = []
    plot_start_stds = []
    plot_finish_stds = []

    if smiles is not None:

        metric_start = []
        metric_finish = []

        for smi in smiles:
            ends = read(pj(wdir, 'xyzs', f'opt_ends_{smi}.xyz'), ':')

            out_st, out_fn = get_metric_distances(ends[0::2], ends[1::2],
                                                  all_dft_ats, task)
            metric_start += out_st
            metric_finish += out_fn

        plot_start_means.append(np.mean(metric_start))
        plot_finish_means.append(np.mean(metric_finish))
        plot_start_stds.append(np.std(metric_start))
        plot_finish_stds.append(np.std(metric_finish))

        xs = [1]

    if temps is not None:

        for temp in temps:
            metric_start = []
            metric_finish = []
            for struct_name in struct_names:
                starts = read(
                    pj(wdir, 'xyzs', f'starts_{struct_name}_{temp}K.xyz'),
                    ':')
                finishes = read(
                    pj(wdir, 'xyzs', f'finishes_{struct_name}_{temp}K.xyz'),
                    ':')

                out_st, out_fn = get_metric_distances(starts, finishes,
                                                      all_dft_ats, task)
                metric_start += out_st
                metric_finish += out_fn

            plot_start_means.append(np.mean(metric_start))
            plot_finish_means.append(np.mean(metric_finish))
            plot_start_stds.append(np.std(metric_start))
            plot_finish_stds.append(np.std(metric_finish))

        xs = temps

    if stds is not None:

        for std in stds:
            metric_start = []
            metric_finish = []
            for struct_name in struct_names:
                starts = read(
                    pj(wdir, 'xyzs', f'starts_{struct_name}_{std}A_std.xyz'),
                    ':')
                finishes = read(
                    pj(wdir, 'xyzs',
                       f'finishes_{struct_name}_{std}A_std.xyz'),
                    ':')

                out_st, out_fn = get_metric_distances(starts, finishes,
                                                      all_dft_ats, task)
                metric_start += out_st
                metric_finish += out_fn

            plot_start_means.append(np.mean(metric_start))
            plot_finish_means.append(np.mean(metric_finish))
            plot_start_stds.append(np.std(metric_start))
            plot_finish_stds.append(np.std(metric_finish))

        xs = stds

        # Define the plot format
    err_kwargs_base = {'fmt': 'X', 'ms': 6, 'capsize': 6, 'elinewidth': 1,
                       'capthick': 2, 'markeredgewidth': 2}
    err_kwargs = {**err_kwargs_base, **end_kwargs}
    err_kwargs['label'] = None

    # plot starts
    if smiles is not None:
        ax.scatter(xs, plot_start_means, c='k', s=200, marker='x')
    else:
        ax.plot(xs, plot_start_means, c='k', linewidth=0.6,
                label=start_label)
    #         print(plot_start_means)

    # plot ends
    ax.plot(xs, plot_finish_means, **end_kwargs)
    ax.errorbar(xs, plot_finish_means, yerr=plot_finish_stds, **err_kwargs)


def get_metric_distances(starts, finishes, all_dft_ats, task):
    metric_start = []
    metric_finish = []

    for at_s, at_f in zip(starts, finishes):

        no_start = int(re.findall('\d+', at_s.info['config_type'])[-1])
        no_finish = int(re.findall('\d+', at_f.info['config_type'])[-1])
        if no_start != no_finish:
            print(no_start, no_finish)
            raise RuntimeError('Start and finish numbers do not match')

        if 'rmsd' in task.lower():
            metric_start.append(min(
                [util.get_rmse(dft_at.positions, at_s.positions) for
                 dft_at in all_dft_ats]))
            metric_finish.append(min(
                [util.get_rmse(dft_at.positions, at_f.positions) for
                 dft_at in all_dft_ats]))

        elif 'soap' in task.lower():
            metric_start.append(
                min([util.soap_dist(dft_at, at_s) for dft_at in all_dft_ats]))
            metric_finish.append(
                min([util.soap_dist(dft_at, at_f) for dft_at in all_dft_ats]))

    return metric_start, metric_finish


def gopt_scatter_summary(ax, wdir, struct_names, start_label, task,
                         all_dft_ats, smiles=None, temps=None, stds=None,
                         **scatter_kwargs):
    # scatter everything with one set of settings and be done with it
    maxs = []
    mins = []

    start_coords = []
    finish_coords = []

    original_label = scatter_kwargs['label']
    scatter_kwargs['label'] = f'{original_label} RDKit'
    if not start_label:
        scatter_kwargs['label'] = original_label
    scatter_kwargs['marker'] = '*'
    scatter_kwargs['facecolors'] = 'none'
    scatter_kwargs['edgecolors'] = scatter_kwargs['color']

    # smiles
    for smi in smiles:
        ends = read(pj(wdir, 'xyzs', f'opt_ends_{smi}.xyz'), ':')
        out_st, out_fn = get_metric_distances(ends[0::2], ends[1::2],
                                              all_dft_ats, task)
        start_coords += out_st
        finish_coords += out_fn

    plt.scatter(start_coords, finish_coords, **scatter_kwargs)
    maxs.append(max(start_coords + finish_coords))
    mins.append(min(start_coords + finish_coords))

    scatter_kwargs['marker'] = 'o'
    scatter_kwargs['label'] = f'{original_label} Random'
    if not start_label:
        scatter_kwargs['label'] = None
    start_coords = []
    finish_coords = []

    for idx, struct_name in enumerate(struct_names):

        for std in stds:
            starts = read(
                pj(wdir, 'xyzs', f'starts_{struct_name}_{std}A_std.xyz'), ':')
            finishes = read(
                pj(wdir, 'xyzs', f'finishes_{struct_name}_{std}A_std.xyz'),
                ':')

            out_st, out_fn = get_metric_distances(starts, finishes,
                                                  all_dft_ats, task)
            start_coords += out_st
            finish_coords += out_fn

    plt.scatter(start_coords, finish_coords, **scatter_kwargs)
    maxs.append(max(start_coords + finish_coords))
    mins.append(min(start_coords + finish_coords))

    scatter_kwargs['label'] = f'{original_label} Normal modes'
    if not start_label:
        scatter_kwargs['label'] = None

    scatter_kwargs['marker'] = 'd'
    start_coords = []
    finish_coords = []

    for idx, struct_name in enumerate(struct_names):

        for temp in temps:
            starts = read(
                pj(wdir, 'xyzs', f'starts_{struct_name}_{temp}K.xyz'), ':')
            finishes = read(
                pj(wdir, 'xyzs', f'finishes_{struct_name}_{temp}K.xyz'),
                ':')

            out_st, out_fn = get_metric_distances(starts, finishes,
                                                  all_dft_ats, task)
            start_coords += out_st
            finish_coords += out_fn

    plt.scatter(start_coords, finish_coords, **scatter_kwargs)
    if len(start_coords + finish_coords)!=0:
        maxs.append(max(start_coords + finish_coords))
        mins.append(min(start_coords + finish_coords))

        return min(mins), max(maxs)
    return None, None

def compare(runs, task):
    print(f'task: {task}')

    cmap = mpl.cm.get_cmap('tab10')
    db_path = '/home/eg475/scripts/dft_minima'

    start_label = 'Opt. traj. start'

    # define plot layout
    if 'plot' in task:
        fig = plt.figure(figsize=(14, 5))
        no_temps = len(runs[0][3])
        no_stds = len(runs[0][4])
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, no_temps, no_stds])
        ax_smi = plt.subplot(gs[0])
        ax_nm = plt.subplot(gs[1], sharey=ax_smi)
        ax_rnd = plt.subplot(gs[2], sharey=ax_nm)
        axes = [ax_smi, ax_nm, ax_rnd]
    elif 'scatter' in task:
        fig = plt.figure(figsize=(14, 5))
        ax_sct = plt.gca()
        axes = [ax_sct]

    min_lims = []
    max_lims = []

    for idx, (
    run_name, dft_xyz, smiles, temps, stds, end_kwargs) in enumerate(runs):

        dft_ats = read(pj(db_path, dft_xyz), ':')
        struct_names = [at.info['name'] for at in dft_ats]

        if 'label' not in end_kwargs.keys():
            end_kwargs['label'] = run_name

        if 'plot' in task:
            if len(smiles) != 0:
                gopt_plot_summary(ax=ax_smi, wdir=run_name,
                                  struct_names=struct_names,
                                  start_label=start_label,
                                  task=task, all_dft_ats=dft_ats,
                                  smiles=smiles, **end_kwargs)
            if len(temps) != 0:
                gopt_plot_summary(ax=ax_nm, wdir=run_name,
                                  struct_names=struct_names,
                                  start_label=start_label,
                                  task=task, all_dft_ats=dft_ats, temps=temps,
                                  **end_kwargs)
            if len(stds) != 0:
                gopt_plot_summary(ax=ax_rnd, wdir=run_name,
                                  struct_names=struct_names,
                                  start_label=start_label,
                                  task=task, all_dft_ats=dft_ats, stds=stds,
                                  **end_kwargs)


        elif 'scatter' in task:
            min_lim, max_lim = gopt_scatter_summary(ax=ax_sct, wdir=run_name,
                                struct_names=struct_names, start_label=start_label,
                                task=task, all_dft_ats=dft_ats,
                                smiles=smiles, temps=temps, stds=stds,
                                                    **end_kwargs)
            min_lims.append(min_lim)
            max_lims.append(max_lim)

        if bool(start_label):
            start_label = None

    if 'scatter' in task:
        min_lims = [val for val in min_lims if val is not None]
        max_lims = [val for val in max_lims if val is not None]
        min_lim = min(min_lims)
        max_lim = max(max_lims)
        plt.plot([min_lim, max_lim], [min_lim, max_lim], linewidth=0.8,
                 color='k')

        if 'soap' in task.lower():
            plt.xlabel('Starting SOAP distance from closest DFT minimum')
            plt.ylabel('End SOAP distance from closest DFT minimum')

        elif 'rmsd' in task.lower():
            plt.xlabel('Starting RMSD wrt the closest DFT minimum')
            plt.ylabel('End RMSD wrt the closest DFT minimum')

    elif 'plot' in task:
        ax_rnd.set_xlabel('Random displacement size, Å')
        ax_nm.set_xlabel('NM displacement temperature, K')
        ax_smi.set_xlabel('RDKit')

        ax_rnd.set_title('Random displacements')
        ax_nm.set_title('Normal mode displacements')
        ax_smi.set_title('From RDKit')

        if 'soap' in task.lower():
            ax_smi.set_ylabel('SOAP distance (mean += std)')

        elif 'rmsd' in task.lower():
            ax_smi.set_ylabel('RMSD, Å')

    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(color='lightgrey')

    plt.suptitle(f'geometry optimisation test comparison')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.tight_layout()
    plt.show()

