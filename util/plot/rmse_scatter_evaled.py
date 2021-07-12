import warnings
from ase.io import read
import util
from util import ugap
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from util import dict_to_vals
from collections import OrderedDict
import numpy as np
import matplotlib as mpl


def prepare_data(ref_values, pred_values, labels):

    data = {}
    for ref_val, pred_val, label in zip(ref_values, pred_values, labels):

        if label not in data.keys():
            data[label] = {}
            data[label]['predicted'] = []
            data[label]['reference'] = []

        if isinstance(pred_val, int):
            data[label]['predicted'].append(pred_val)
            data[label]['reference'].append(ref_val)
        else:
            data[label]['predicted'] += list(pred_val.flatten())
            data[label]['reference'] += list(ref_val.flatten())

    for label in data.keys():
        pred_vals = data[label]['predicted']
        ref_vals = data[label]['reference']
        data[label]['predicted'] = np.array(pred_vals)
        data[label]['reference'] = np.array(ref_vals)

    return data

def read_energy(at, isolated_atoms, ref_prefix):
    return at.info[f'{ref_prefix}energy']

def scatter_plot(ref_energy_name,
                pred_energy_name,
                ref_force_name,
                pred_force_name, all_atoms,
                output_dir, prefix, color_info_name, isolated_atoms,
                energy_type, energy_shift):

    all_atoms = [at for at in all_atoms if len(at) != 1]

    marker_kwargs = {'marker':'x', 'alpha':0.5, 's':10}

    cmap = plt.get_cmap('tab10')
    colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]

    num_columns = 2
    if ref_force_name is None:
        num_columns = 1

    fig = plt.figure(figsize=(7*num_columns, 14))
    gs = gridspec.GridSpec(2, num_columns)
    axes = [plt.subplot(g) for g in gs]

    if ref_force_name is None:
        ax_e_err = axes[1]
        ax_e_corr = axes[0]
        ax_f_err = None
        ax_f_corr = None
    else:
        ax_e_err = axes[2]
        ax_e_corr = axes[0]
        ax_f_err = axes[3]
        ax_f_corr = axes[1]


    if color_info_name is not None:
        # TODO: label for no entry
        info_entries = [at.info[color_info_name] if color_info_name in
                                                    at.info.keys() else
                        'no info' for at in all_atoms]
    else:
        info_entries = ['no label' for at in all_atoms if len(at) != 1]

   ################### energy plots

    if energy_type == 'binding_energy':
        energy_getter_function = util.get_binding_energy_per_at
        y_energy_correlation_label = f'Predicted binding {pred_energy_name} / eV/at'
        x_energy_label = f'Binding {ref_energy_name} / eV/at'
        y_energy_error_label = f'absolute binding energy error / meV/at'
        energy_correlation_title =  'Binding energy correlation'
        energy_error_title = 'Binding energy error'

    elif energy_type == 'total_energy':
        energy_getter_function = read_energy
        y_energy_correlation_label = f'Predicted total {pred_energy_name} ' \
                                     f'/ eV'
        x_energy_label = f'Total {ref_energy_name} / eV'
        y_energy_error_label = f'absolute total energy error / meV'
        energy_correlation_title = 'Total energy correlation'
        energy_error_title = 'Total energy error'

    elif energy_type == 'mean_shifted_energy':
        energy_getter_function = read_energy
        y_energy_correlation_label = f'Predicted mean shifted total' \
                                     f' {pred_energy_name} / eV'
        x_energy_label = f'Mean shifted total {ref_energy_name} / eV'
        y_energy_error_label = f'absolute mean shifted total energy error / meV'
        energy_correlation_title = 'mean shifted total energy correlation'
        energy_error_title = 'mean shifted total energy error'
    else:
        raise ValueError(f'"energy_type" must be one of "binding_energy", '
                         f'"total_energy" or "mean_shifted_energy", '
                         f'not "{energy_type}". ')

    ref_prefix = ref_energy_name.replace('energy', '')
    pred_prefix = pred_energy_name.replace('energy', '')
    # TODO: change the util function
    ref_energies = [energy_getter_function(at, isolated_atoms,
                                                   ref_prefix)
                    for at in all_atoms if len(at) != 1]

    pred_energies = [energy_getter_function(at, isolated_atoms,
                                                   pred_prefix)
                    for at in all_atoms if len(at) != 1]

    if energy_shift:
        ref_energies = util.shift0(ref_energies, by=np.mean(ref_energies))
        pred_energies = util.shift0(pred_energies, by=np.mean(pred_energies))

        y_energy_correlation_label =  'Mean shifted ' + \
                                      y_energy_correlation_label
        x_energy_label = 'Mean shifted ' + x_energy_label
        y_energy_error_label = 'Mean shifted ' + y_energy_error_label
        energy_correlation_title = 'Mean shifted ' + energy_correlation_title
        energy_error_title = 'Mean shifted ' + energy_error_title


    all_plot_data = prepare_data(ref_values=ref_energies,
                             pred_values=pred_energies, labels=info_entries)

    ax_err = ax_e_err
    ax_corr = ax_e_corr

    for color, (label, data) in zip(colors, all_plot_data.items()):

        ref = data['reference']
        pred = data['predicted']

        rmse = util.get_rmse(ref, pred) * 1e3

        ax_corr.scatter(ref, pred, label=f'{label}: {rmse:.3f}', color=color,
                       zorder=2, **marker_kwargs)
        ax_err.scatter(ref, np.abs(ref - pred) * 1e3, **marker_kwargs,
                       color=color, zorder=2)
        ax_err.axhline(rmse, color=color, lw=0.8)


    ax_corr.legend(title=f' {color_info_name}: RMSE / meV/at')
    ax_corr.set_ylabel(y_energy_correlation_label)
    ax_err.set_ylabel(y_energy_error_label)
    ax_err.set_yscale('log')
    ax_err.set_title(energy_error_title)
    ax_corr.set_title(energy_correlation_title)

    xmin, xmax = ax_e_corr.get_xlim()
    extend_axis = 0.1
    xmin -= extend_axis
    xmax += extend_axis
    shade = 0.05
    ax_e_corr.fill_between([xmin, xmax], [xmin - shade, xmax - shade],
                           [xmin + shade, xmax + shade],
                           color='lightgrey', alpha=0.25, zorder=3)
    ax_e_err.fill_between([xmin, xmax], 0, shade*1e3, color='lightgrey',
                          alpha=0.5, zorder=0)

    for ax in [ax_e_corr, ax_e_err]:
        ax.set_xlabel(x_energy_label)
        ax.set_xlim(xmin, xmax)

    ######################### force plots


    if ref_force_name is not None:


        ref_forces = [at.arrays[ref_force_name] for at in all_atoms if len(at)
                      != 1]
        pred_forces = [at.arrays[pred_force_name] for at in all_atoms if len(at)
                      != 1]

        all_plot_data = prepare_data(ref_values=ref_forces,
                                     pred_values=pred_forces, labels=info_entries)

        ax_err = ax_f_err
        ax_corr = ax_f_corr

        for color, (label, data) in zip(colors, all_plot_data.items()):

            ref = data['reference']
            pred = data['predicted']

            rmse = util.get_rmse(ref, pred)*1e3

            ax_corr.scatter(ref, pred, label=f'{label}: {rmse:.3f}',
                            color=color,
                            **marker_kwargs)
            ax_err.scatter(ref, np.abs(ref - pred)*1e3, color=color,
            **marker_kwargs)
            ax_err.axhline(rmse, color=color, lw=0.8)

        ax_corr.legend(title=f'F component RMSE / meV/Å')
        ax_corr.set_ylabel(f'Predicted {pred_force_name} / eV/Å')
        ax_err.set_ylabel(f'absolute force component error / eV/Å')
        ax_err.set_yscale('log')
        ax_err.set_title('Force component error')
        ax_corr.set_title('Force component correlation')

        for ax in [ax_f_err, ax_f_corr]:
            ax.set_xlabel(f'{ref_force_name} / eV/Å')

    for ax in axes:
        ax.grid(color='lightgrey', ls=':')

    for ax in [ax_e_corr,ax_f_corr]:
        if ax is not None:
            left_lim, right_lim = ax.get_xlim()
            bottom_lim, top_lim = ax.get_ylim()
            lims = (min([left_lim, bottom_lim]), max(left_lim, top_lim))
            ax.plot(lims, lims, c='k', linewidth=0.8)



    if not prefix:
        prefix = os.path.basename(param_fname)
        prefix = os.path.splitext(prefix)[0]
    picture_fname = f'{prefix}_scatter.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.suptitle(prefix)
    plt.tight_layout()

    if output_dir:
        plt.savefig(picture_fname, dpi=300,
                    bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
