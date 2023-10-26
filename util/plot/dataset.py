from ase.io import read, write
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib as mpl
from ase import Atoms
from ase import neighborlist
import util
import seaborn as sns
from collections import Counter
from pathlib import Path



def energy_by_idx(atoms, prop_prefix='dft_', title=None,
                  group_compounds=False,
                  isolated_atoms=None, info_label='config_type', dir='.',
                  cmap="tab10"):

    if title is None:
       title = 'energy per atom vs index'

    if isolated_atoms is None:
        isolated_atoms = [at for at in atoms if len(at) == 1]
    atoms = [at for at in atoms if len(at) != 1]

    data = {}
    for idx, at in enumerate(atoms):
        if info_label in at.info:
            cfg = at.info[info_label]
        else:
            cfg = "None"
        if cfg not in data.keys():
            data[cfg] = []
        at.info['dset_idx'] = idx
        data[cfg].append(at)

    if group_compounds:
        data = group_data(data)
        data = group_data(data)

    if cmap != "tab10":
        cmap = plt.get_cmap(cmap)
        colors = [cmap(idx) for idx in np.linspace(0, 1, len(data.keys()))]
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]
    

    plt.figure(figsize=(15, 4))
    global_idx = 0
    for idx, (cfg_type, configs) in enumerate(data.items()):

        atomization_energies_per_at = np.array([util.get_atomization_energy_per_at(at, isolated_atoms, prop_prefix+"energy")
                                            for at in configs if len(at) != 1])
        indices = np.array([at.info['dset_idx'] for at in configs if len(at) != 1])

        plt.scatter(indices, atomization_energies_per_at, label=cfg_type, s=8, marker='x', color=colors[idx])
        # plt.plot(range(global_idx, global_idx +len(atomization_energies_per_at)), atomization_energies_per_at, label=cfg_type)
        # global_idx += len(atomization_energies_per_at)

    plt.title(title)
    if len(data.keys()) < 11 or cmap != 'tab10':
        plt.legend()
    plt.xlabel('index in dataset')
    plt.ylabel(f'{prop_prefix} atomization energy / ev/atom')
    plt.grid(color='lightgrey', linestyle=':')

    plt.tight_layout()

    fig_name = title.replace(' ', '_')
    fig_name = Path(dir) / (fig_name + '.png')
    plt.savefig(fig_name, dpi=300)

def forces_by_idx(atoms, prop_prefix='dft_', title=None,
                  group_compounds=False, info_label='config_type',
                  dir='.', cmap='tab10'):

    if title is None:
        title = 'force components vs index'

    data = {}
    for idx, at in enumerate(atoms):
        if info_label in at.info:
            cfg = at.info[info_label]
        else:
            cfg = "None"
        if cfg == 'isolated_atom':
            continue
        if cfg not in data.keys():
            data[cfg] = []
        at.info['dset_idx'] = idx
        data[cfg].append(at)

    if group_compounds:
        data = group_data(data)
        data = group_data(data)

    if cmap != "tab10":
        cmap = plt.get_cmap(cmap)
        colors = [cmap(idx) for idx in np.linspace(0, 1, len(data.keys()))]
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]
    

    plt.figure(figsize=(15, 4))

    for c_idx, (cfg_type, configs) in enumerate(data.items()):
        xs = []
        ys = []
        for idx, at in enumerate(configs):
            if len(at) == 1:
                continue
            forces = list(at.arrays[f'{prop_prefix}forces'].flatten())
            at_idx = at.info['dset_idx']
            idc = [at_idx] * len(forces)
            xs += idc
            ys += forces

        plt.scatter(xs, ys, label=cfg_type, s=1, color=colors[c_idx])

    plt.title(title)
    if len(data.keys()) < 11 or cmap != 'tab10':
        plt.legend(bbox_to_anchor=(1, 1), markerscale=6)
    plt.xlabel('index in dataset')
    plt.ylabel(f'{prop_prefix} force component / eV/Ā')
    plt.grid(color='lightgrey', linestyle=':')

    plt.tight_layout()

    fig_name = title.replace(' ', '_')
    fig_name = Path(dir) / (fig_name + '.png')
    plt.savefig(fig_name, dpi=300)


def group_data(data):
    grouped_data = {}
    pat = re.compile(r'(?:^\d+_)?(.*)(?:_mol$|_rad\d+$)')
    for key, vals in data.items():
        out = pat.search(key)
        if out:
            root = out.groups()[0]
            if root not in grouped_data.keys():
                grouped_data[root] = []
            grouped_data[root] += vals

        else:
            if key not in grouped_data.keys():
                grouped_data[key] = []
            grouped_data[key] += vals

    data_out = {}
    pat = re.compile(r'(?:^\d+_)?(.*)')
    for key, vals in grouped_data.items():
        out = pat.search(key)
        if out:
            root = out.groups()[0]
            if root not in data_out.keys():
                data_out[root] = []
            data_out[root] += vals

        else:
            if key not in data_out.keys():
                data_out[key] = []
            data_out[key] += vals

    return data_out


def distances_distributions(atoms_in, group_compounds=True, extend_y=-0.7,
                            color='k', title=None, cutoff=6.0):
    """TODO: do distances within cutoff;
         Add labels to last x axis
         normalise y scale for everyone to be the same"""

    lw = 2

    if not title:
        title = 'interatomic_distances_distribution'

    data = {}
    for at in atoms_in:
        cfg = at.info['config_type']
        if cfg not in data.keys():
            data[cfg] = []
        data[cfg].append(at)

    if group_compounds:
        data = group_data(data)
        data = group_data(data)
    no_plots = len(data.keys())

    fig = plt.figure(figsize=(20, 1 * no_plots))
    gs = mpl.gridspec.GridSpec(no_plots, 1)
    axes_list = []
    up_limits = []
    for idx, (cfg, ats) in enumerate(data.items()):
        if cfg == 'isolated_atom':
            continue

        ax = plt.subplot(gs[idx])

        if cutoff is None:
            all_distances = all_pair_distances(ats)
        else:
            all_distances = pair_distances_within_cutoff(ats, cutoff=cutoff)

        sns.kdeplot(all_distances, label=cfg, fill=True, facecolor='white',
                    edgecolor=color, linewidth=lw)
        ax.axhline(y=0, color='k', lw=lw)
        lims = ax.get_xlim()
        up_limits.append(lims[1])
        axes_list.append(ax)

    new_lims = (extend_y, max(up_limits))
    for idx, (label, ax) in enumerate(zip(data.keys(), axes_list)):
        ax.text(new_lims[0], .02, label, size=18)
        ax.set_xlim(new_lims)
        if idx != len(axes_list) - 1:
            ax.set_axis_off()
        else:
            ax.yaxis.set_visible(False)
            ax.set_xlabel('atom-atom distance, Å')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.patch.set_visible(False)
            # for item in [fig, ax]:
            #     item.patch.set_visible(False)

    # ax.set_xlabel('interatomic distance')
    fig.subplots_adjust(hspace=-.5)
    # plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)



def pairwise_distances_hist(atoms, fig_name):


    distances_dict = util.distances_dict(atoms)
    num_pairs = len(distances_dict.keys())
    num_cols = 2
    num_rows = int(num_pairs/2)

    fig = plt.figure(figsize=(10, 3*num_rows))
    gs = mpl.gridspec.GridSpec(num_rows, num_cols)
    axes = [plt.subplot(g) for g in gs]

    for ax, (key, vals) in zip(axes, distances_dict.items()):

        if len(vals) == 0:
            continue

        ax.set_title(key)
        ax.grid(color='lightgrey', ls=':')
        ax.hist(vals, density=True, label=f'{min(vals):.3f}')
        ax.set_xlabel('distance, Å')
        ax.set_ylabel('density')
        ax.set_xlim(left=0)
        ax.legend(title="min distance, Å")

    plt.tight_layout()

    plt.savefig(fig_name, bbox_inches='tight')




def all_pair_distances(atoms_list):
    distances_dict = util.distances_dict(atoms_list)
    all_distances = []
    for key, vals in distances_dict.items():
        all_distances += list(vals)

    return all_distances


def pair_distances_within_cutoff(atoms_list, cutoff=6.0):
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]

    all_distances = np.array([])

    for atoms in atoms_list:

        cutoffs = [cutoff / 2] * len(atoms)
        neighbor_list = neighborlist.NeighborList(cutoffs,
                                                  self_interaction=False,
                                                  bothways=False,
                                                  skin=0)
        neighbor_list.update(atoms)

        for at in atoms:

            indices, _ = neighbor_list.get_neighbors(at.index)
            if len(indices) > 0:
                distances = atoms.get_distances(at.index, indices)
                if max(distances) > cutoff:
                    raise RuntimeError(
                        f'Return distance {max(distances)} > {cutoff}')
                all_distances = np.concatenate([all_distances, distances])

    return all_distances



