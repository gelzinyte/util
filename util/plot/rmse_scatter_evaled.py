import warnings
import logging
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

logger = logging.getLogger(__name__)

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


def scatter_plot(ref_energy_name,
                pred_energy_name,
                ref_force_name,
                pred_force_name, all_atoms,
                output_dir, prefix, color_info_name, isolated_atoms,
                energy_type, energy_shift=False,
                 no_legend=False,
                 error_type='rmse', 
                 skip_if_prop_not_present=False,
                 error_scatter_type='absolute'):

    # print(color_info_name)
    special_colors={
        "zinc-train":'k',
        "zinc-test":"tab:orange",
        "comp6": "tab:red",
        "ha22": "tab:green",
        "tyzack": "tab:purple",
        "validation": "purple",
        # "MACE": "#D62627",
        # "ACE": "#25D425",
        # "GAP": "#2525D4",
        # "MACE": "#D62627",
        # "ACE": "#1EA91E",
        # "GAP": "#1E1EA9"
        "MACE": "tab:red",
        "ACE": "tab:olive",
        "GAP": "tab:blue",
        "test": "tab:red",
        "COMP6.ANI-MD.ani_md_bench_10": "tab:brown",
        "ha22_hydrocarbons": "tab:green",
        "COMP6.GDB07to09": "tab:orange",
        "COMP6.GDB10to13": "tab:olive",
        "COMP6.DrugBank":"tab:cyan",
        "train": "k",
        "initial_train": "k",
        "extra_1": "tab:red",
        "extra_2": "tab:pink",
        "extra_3": "tab:brown",
        "ZINC-test": "tab:red",
        "train": "tab:blue",
        "ext-test": "tab:green",
        "rad": "tab:green", 
        "mol": "pink",
        "no label": "k"
    }
    # labels_order = ["zinc-train", "zinc-test", "comp6", "ha22", "tyzack"]
    # labels_order = ["zinc-test", "comp6", "ha22", "tyzack"]
    # labels_order = ["GAP", "ACE", "MACE"]


    errors_to_return = {"energy": {}, "forces": {}}

    if isolated_atoms is None and energy_type=='atomization_energy':
        isolated_atoms = [at for at in all_atoms if len(at) == 1]
    all_atoms = [at for at in all_atoms if len(at) != 1]


    assert error_type in ['rmse', 'mae']
    if error_type == 'rmse':
        error_function = util.get_rmse
        error_label = 'RMSE'
    elif error_type == 'mae':
        error_function = util.get_mae
        error_label = "MAE"


   ################### energy plots

    if energy_type == 'atomization_energy':
        energy_getter_function = util.get_atomization_energy_per_at
        y_energy_correlation_label = f'Predicted atomization {pred_energy_name} / eV/at'
        x_energy_label = f'Atomization {ref_energy_name} / eV/at'
        y_energy_error_label = f'atomization energy error / meV/at'
        energy_correlation_title =  'atomization energy correlation'
        energy_error_title = 'atomization energy error'
        e_error_units = 'meV/at'

    elif energy_type == 'total_energy':
        energy_getter_function = util.read_energy
        # y_energy_correlation_label = f'Predicted total {pred_energy_name} / eV'
        y_energy_correlation_label = f'Predicted BDE / eV'
        # y_energy_correlation_label = "Predicted atomization energy / eV/at"
        x_energy_label = f'Total {ref_energy_name} / eV'
        y_energy_error_label = f'Total energy error / meV'
        energy_correlation_title = 'Total energy correlation'
        # energy_correlation_title = "Bond Dissociation Energy correlation"
        energy_error_title = 'Total energy error'
        e_error_units = 'meV'

    elif energy_type == 'mean_shifted_energy':
        energy_getter_function = util.read_energy
        y_energy_correlation_label = f'Predicted mean shifted total' \
                                     f' {pred_energy_name} / eV'
        x_energy_label = f'Mean shifted total {ref_energy_name} / eV'
        y_energy_error_label = f'mean shifted total energy error / meV'
        energy_correlation_title = 'mean shifted total energy correlation'
        energy_error_title = 'mean shifted total energy error'
        e_error_units = 'meV'

    elif energy_type == "per_atom_energy":
        energy_getter_function = util.total_per_atom_energy
        y_energy_correlation_label = f'Predicted per atom total (not atomization)' \
                                     f' {pred_energy_name} / eV/at'
        x_energy_label = f'per atom total (not atomization) {ref_energy_name} / eV/at'
        y_energy_error_label = f'per atom total (not atomization) energy error / meV/at'
        energy_correlation_title = 'per atom total (not atomization) energy correlation'
        energy_error_title = 'per atom total (not atomization) energy error'
        e_error_units = 'meV/at'
    else:
        raise ValueError(f'"energy_type" must be one of "atomization_energy", '
                         f'"total_energy" or "mean_shifted_energy", '
                         f'not "{energy_type}". ')

    if error_scatter_type == 'absolute':
        y_energy_error_label = f'absolute {y_energy_error_label}'
        y_force_error_label = 'Force component error / meV/Å' 
        energy_error_title = f'Absolute {energy_error_title}'
        forces_error_title = "Force component error"
    elif error_scatter_type == 'signed':
        y_energy_error_label = f'Signed {y_energy_error_label}'
        y_force_error_label = 'Signed force component error / meV/Å' 
        energy_error_title = f'Signed {energy_error_title}'
        # forces_error_title = "Signed force component error"
        forces_error_title = "Force component error"
    else:
        raise ValueError(f'"error_scatter_type" must be one of "absolute" or "signed", not "{error_scatter_type}"')

    # ref_prefix = ref_energy_name.replace('energy', '')
    # pred_prefix = pred_energy_name.replace('energy', 'pred_prefix')

    ref_energies = []
    pred_energies = []
    info_entries = []

    number_of_skipped_configs=0
    for at in all_atoms:
        if len(at) == 1:
            continue

        if ref_energy_name not in at.info.keys() or pred_energy_name not in at.info.keys():
            if skip_if_prop_not_present:
                # logger.warn("did not found property in atoms, skipping")
                number_of_skipped_configs += 1
                continue
            else:
                print(at.info)
                raise RuntimeError(f"did not found property (either {ref_energy_name} or {pred_energy_name}) in atoms")

        ref_energies.append(energy_getter_function(at, isolated_atoms, ref_energy_name))
        pred_energies.append(energy_getter_function(at, isolated_atoms, pred_energy_name))

        info_entry = "no label"
        if color_info_name is not None:
            if color_info_name in at.info.keys():
                info_entry = at.info[color_info_name]
        info_entries.append(info_entry)

    if number_of_skipped_configs > 0: 
        logger.warn(f'skipped {number_of_skipped_configs} configs, because one of {ref_energy_name} or {pred_energy_name} was not found.')

    if energy_shift:
        # ref_energies = util.shift0(ref_energies, by=np.mean(ref_energies))
        # pred_energies = util.shift0(pred_energies, by=np.mean(pred_energies))

        y_energy_correlation_label =  'Mean shifted ' + \
                                      y_energy_correlation_label
        x_energy_label = 'Mean shifted ' + x_energy_label
        y_energy_error_label = 'Mean shifted ' + y_energy_error_label
        energy_correlation_title = 'Mean shifted ' + energy_correlation_title
        energy_error_title = 'Mean shifted ' + energy_error_title


    # print(info_entries)
    all_plot_data = prepare_data(ref_values=ref_energies,
                             pred_values=pred_energies, labels=info_entries)

    if energy_shift:
        for key, vals in all_plot_data.items():
            # import pdb; pdb.set_trace()
            pred_es = np.array(all_plot_data[key]["predicted"])
            all_plot_data[key]["predicted"] = np.array(util.shift0(pred_es, np.mean(pred_es)))

            ref_es = np.array(all_plot_data[key]["reference"])
            all_plot_data[key]["reference"] = np.array(util.shift0(ref_es, np.mean(ref_es)))



    n_colors = len(all_plot_data)
    if n_colors < 11:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(idx) for idx in np.linspace(0, 1, 10)]
    else:
        cmap = plt.get_cmap('jet')
        colors = [cmap(idx) for idx in np.linspace(0, 1, n_colors)]

    marker_kwargs = {'marker': 'o', 'alpha': 1, 's': 10, 'facecolors':'none'}
    # marker_kwargs = {'marker': 'x', 'alpha': 0.5, 's': 10}#, 'facecolors':'none'k

    num_columns = 2
    if ref_force_name is None:
        num_columns = 1

    if n_colors > 30:

        f_legend_kwargs = {'bbox_to_anchor':(1.04, 1),
                            'loc':'upper left' }

        e_legend_kwargs = {'bbox_to_anchor':(0, 1),
                           'loc':'upper right'}
        figsize=(( 7.5* num_columns, 15))
    else:
        f_legend_kwargs = {}
        e_legend_kwargs = {}
        figsize=(4.5 * num_columns, 9)


    fig = plt.figure(figsize=figsize)
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

    ax_err = ax_e_err
    ax_corr = ax_e_corr

    for color, (label, data) in zip(colors, all_plot_data.items()):
    # for label, data in all_plot_data.items():
    # for label in labels_order:
        # print(all_plot_data.keys())
        # print(len(all_plot_data[label]["predicted"]))
        # data = all_plot_data[label]
        if label in special_colors:
            color = special_colors[label]

        ref = data['reference']
        pred = data['predicted']

        error = error_function(ref, pred) * 1e3

        errors = (ref - pred) * 1e3
        if error_scatter_type == "absolute":
            errors = np.abs(errors)

        if error_scatter_type == "absolute":
            error_scatter_line = error
            error_scatter_line_label = f'{label}: {error:.3f}'
        elif error_scatter_type == 'signed':
            error_scatter_line = np.mean(errors) 
            error_scatter_line_label = f'{label} mean: {error_scatter_line:.3f} {e_error_units}' 
        
        errors_to_return["energy"][label] = errors

        ax_corr.scatter(ref, pred, label=f'{label}: {error:.3f}', 
                        color=color,
                        #edgecolors=color,
                       zorder=2, **marker_kwargs)
        print(f'mean ref: {np.mean(ref)} pred {np.mean(pred)}')
        ax_err.scatter(ref, errors, **marker_kwargs, 
                        color=color,
                        #edgecolors=color, 
                        # label =f'{label}: {error:.3f}' ,
                        zorder=2)
        ax_err.axhline(error_scatter_line, color=color, lw=0.8, label=error_scatter_line_label)

    if not no_legend:
        # just for now
        # e_error_units = "meV/at"
        ax_corr.legend(title=f' {color_info_name}: {error_label} / {e_error_units}',
                       **e_legend_kwargs)
        ax_err.legend(title=f' {color_info_name}: {error_label} / {e_error_units}',
                       **e_legend_kwargs)
        if error_scatter_type == 'signed':
            ax_err.legend(loc="lower right")
            ax_err.axhline(0, c='k', lw=0.8, ls='--')
   

    ax_corr.set_ylabel(y_energy_correlation_label)
    # ax_err.set_ylabel(y_energy_error_label)
    # ax_err.set_ylabel("MACE atomization energy error / meV/at")
    # ax_err.set_ylabel("Atomization energy error / meV/at")
    ax_err.set_ylabel("BDE error / meV/at")

    if error_scatter_type == "absolute": 
        ax_err.set_yscale('log')
        # ax_err.set_ylim(bottom=0)
    # ax_err.set_title(energy_error_title)
    # ax_err.set_title("Atomization energy error")
    ax_err.set_title("Bond Dissociation Energy error")
    ax_corr.set_title(energy_correlation_title)

    # ax_err.set_ylim((-65, 40))

    xmin, xmax = ax_e_corr.get_xlim()
    extend_axis = 0.1
    xmin -= extend_axis
    xmax += extend_axis
    shade = 0.005
    # ax_e_corr.fill_between([xmin, xmax], [xmin - shade, xmax - shade],
    #                        [xmin + shade, xmax + shade],
    #                        color='lightgrey', alpha=0.25, zorder=3)
    # ax_e_err.fill_between([xmin, xmax], 0, shade*1e3, color='lightgrey',
    #                       alpha=0.5, zorder=0)

    for ax in [ax_e_corr, ax_e_err]:
        # ax.set_xlabel(x_energy_label)
        # ax.set_xlabel("DFT atomization energy / eV/at")
        ax.set_xlabel("Mean-shifted DFT BDE / eV")
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
        # for label in labels_order:
            # data = all_plot_data[label]
            if label in special_colors:
                color = special_colors[label]

            ref = data['reference']
            pred = data['predicted']

            error = error_function(ref, pred)*1e3
            # errors = np.abs(ref - pred)*1e3
            errors = (ref - pred)*1e3
            if error_scatter_type == "absolute":
                errors = np.abs(errors)

            
            if error_scatter_type == "absolute":
                error_scatter_line = error 
                error_scatter_line_label = f'{label}: {error:.3f}'
            elif error_scatter_type == 'signed':
                error_scatter_line = np.mean(errors) 
                error_scatter_line_label = f'{label} mean: {error_scatter_line:.3f} {e_error_units}' 
            

            errors_to_return["forces"][label] = errors

            ax_corr.scatter(ref, pred, label=f'{label}: {error:.3f}',
                            color=color,
                            # edgecolors=color,
                            **marker_kwargs)
            ax_err.scatter(ref, errors , 
                            # label=f'{label}: {error:.3f}',
                            color=color,
                            # edgecolors=color,
                        **marker_kwargs)
            # print(error_scatter_line_label)
            ax_err.axhline(error_scatter_line, color=color, lw=0.8, label=error_scatter_line_label)

        if not no_legend:
            ax_corr.legend(title=f'{color_info_name} {error_label} / meV/Å',
                           **f_legend_kwargs)
            ax_err.legend(title=f'{color_info_name} {error_label} / meV/Å',
                           **f_legend_kwargs)
            if error_scatter_type == 'signed':
            #     ax_err.legend()  
                ax_err.axhline(0, color='k', lw=0.8, ls='--')


        ax_corr.set_ylabel(f'Predicted {pred_force_name} / eV/Å')
        ax_err.set_ylabel(y_force_error_label)

        if error_scatter_type == "absolute":
            ax_err.set_yscale('log')
            # ax_err.set_ylim(bottom=0)
            
        ax_err.set_title(forces_error_title)
        ax_corr.set_title('Force component correlation')

        for ax in [ax_f_err, ax_f_corr]:
            # ax.set_xlabel(f'{ref_force_name} / eV/Å')
            ax.set_xlabel(f'DFT force componenet / eV/Å')

    for ax in axes:
        ax.grid(color='lightgrey', ls=':')

    for ax in [ax_e_corr,ax_f_corr]:
        if ax is not None:
            left_lim, right_lim = ax.get_xlim()
            bottom_lim, top_lim = ax.get_ylim()
            lims = (min([left_lim, bottom_lim]), max(left_lim, top_lim))
            ax.plot(lims, lims, c='k', linewidth=0.8)

    if not prefix:
        prefix = ''
    picture_fname = f'{prefix}_{energy_type}_by_{color_info_name}_scatter.png'
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
        plt.close(fig)

    return errors_to_return

