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



def energy_by_idx(atoms, prop_prefix='dft_', title=None,
                  group_compounds=False,
                  isolated_atoms=None, info_label='config_type'):
    """TODO: do binding energy per atom"""

    if title is None:
       title = 'energy per atom vs index'

    if isolated_atoms is None:
        isolated_atoms = [at for at in atoms if len(at) == 1]
    #
    # isolated_at_data = {}
    # for at in isolated_atoms:
    #     print(at)
    #     print(type(at))
    #     isolated_at_data[list(at.symbols)[0]] = at.info[f'{prop_prefix}energy']

    data = {}
    for idx, at in enumerate(atoms):
        cfg = at.info[info_label]
        if cfg not in data.keys():
            data[cfg] = []
        at.info['dset_idx'] = idx
        data[cfg].append(at)

    if group_compounds:
        data = group_data(data)
        data = group_data(data)


    plt.figure(figsize=(20, 6))
    global_idx = 0
    for cfg_type, configs in data.items():

        binding_energies_per_at = np.array([util.get_binding_energy_per_at(at, isolated_atoms, prop_prefix)
                                            for at in configs if len(at) != 1])
        indices = np.array([at.info['dset_idx'] for at in configs if len(at) != 1])

        plt.scatter(indices, binding_energies_per_at, label=cfg_type, s=8, marker='x')
        # plt.plot(range(global_idx, global_idx +len(binding_energies_per_at)), binding_energies_per_at, label=cfg_type)
        # global_idx += len(binding_energies_per_at)

    plt.title(title)
    if len(data.keys()) < 11:
        plt.legend()
    plt.xlabel('index in dataset')
    plt.ylabel('binding energy / ev/atom')
    plt.grid(color='lightgrey', linestyle=':')

    fig_name = title.replace(' ', '_')
    plt.tight_layout()
    plt.savefig(fig_name + '.png', dpi=300)

def forces_by_idx(atoms, prop_prefix='dft_', title=None,
                  group_compounds=False, info_label='config_type'):

    if title is None:
        title = 'force components vs index'

    data = {}
    for idx, at in enumerate(atoms):
        cfg = at.info[info_label]
        if cfg == 'isolated_atom':
            continue
        if cfg not in data.keys():
            data[cfg] = []
        at.info['dset_idx'] = idx
        data[cfg].append(at)

    if group_compounds:
        data = group_data(data)
        data = group_data(data)

    plt.figure(figsize=(20,6))

    for cfg_type, configs in data.items():
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

        plt.scatter(xs, ys, label=cfg_type, s=1)

    plt.title(title)
    if len(data.keys()) < 11:
        plt.legend(bbox_to_anchor=(1, 1), markerscale=6)
    plt.xlabel('index in dataset')
    plt.ylabel('force component / eV/Ā')
    plt.grid(color='lightgrey', linestyle=':')

    fig_name = title.replace(' ', '_')
    plt.tight_layout()
    plt.savefig(fig_name + '.png', dpi=300)


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


