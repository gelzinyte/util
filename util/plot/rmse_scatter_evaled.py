
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




def make_scatter_plots_from_evaluated_atoms(ref_energy_name, pred_energy_name, ref_force_name, pred_force_name, evaluated_train_fname, evaluated_test_fname,
               output_dir, prefix, by_config_type, force_by_element=True):

    train_ats = read(evaluated_train_fname, ':')
    if evaluated_test_fname:
        test_ats = read(evaluated_test_fname, ':')
    else:
        test_ats=None

    counts = list(set(np.hstack([np.array(list(util.get_counts(at).keys())) for at in train_ats])))
    no_unique_elements = len(counts)
    width = 14
    height = width * 0.5
    if force_by_element:
        height *= (no_unique_elements + 1)
        no_rows = no_unique_elements + 1
    else:
        no_rows = 2

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(no_rows, 2)
    ax = [plt.subplot(g) for g in gs]

    #main plot
    lgd = scatter_plot(train_ats=train_ats, ax=ax, test_ats=test_ats, by_config_type=by_config_type,
                       ref_energy_name=ref_energy_name, pred_energy_name=pred_energy_name,
                       ref_force_name=ref_force_name, pred_force_name=pred_force_name, force_by_element=force_by_element)

    if not prefix:
        prefix = os.path.basename(param_fname)
        prefix = os.path.splitext(prefix)[0]
    picture_fname = f'{prefix}_scatter.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.suptitle(prefix, y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 1.1])

    if output_dir:
        plt.savefig(picture_fname, dpi=300, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def scatter_plot(train_ats, ax, test_ats, by_config_type,
                       ref_energy_name, pred_energy_name,
                       ref_force_name, pred_force_name, force_by_element):

    test_set = False
    if test_ats:
        test_set = True

    train_ref_data = util.get_E_F_dict_evaled(train_ats, energy_name=ref_energy_name, force_name=ref_force_name)
    train_pred_data = util.get_E_F_dict_evaled(train_ats, energy_name=pred_energy_name, force_name=pred_force_name)



    if not force_by_element:
        tmp_no_f_sym_data = util.desymbolise_force_dict(train_ref_data['forces'])
        # print('returned keys:', tmp_no_f_sym_data.keys())
        train_ref_data['forces'].clear()
        train_ref_data['forces']['all elements'] = tmp_no_f_sym_data

        tmp_no_f_sym_data = util.desymbolise_force_dict(train_pred_data['forces'])
        train_pred_data['forces'].clear()
        train_pred_data['forces']['all elements'] = tmp_no_f_sym_data


    if test_set:
        test_ref_data = util.get_E_F_dict_evaled(test_ats, energy_name=ref_energy_name, force_name=ref_force_name)
        test_pred_data = util.get_E_F_dict_evaled(test_ats, energy_name=pred_energy_name, force_name=pred_force_name)

        if not force_by_element:
            tmp_no_f_sym_data = util.desymbolise_force_dict(test_ref_data['forces'])
            test_ref_data['forces'].clear()
            test_ref_data['forces']['all elements'] = tmp_no_f_sym_data

            tmp_no_f_sym_data = util.desymbolise_force_dict(test_pred_data['forces'])
            test_pred_data['forces'].clear()
            test_pred_data['forces']['all elements'] = tmp_no_f_sym_data

    #####################################################################
    # Energy plots
    ####################################################################

    #### error scatter
    train_ref_es = train_ref_data['energy']
    train_pred_es = train_pred_data['energy']

    this_ax = ax[0]
    if test_set:
        test_ref_es = test_ref_data['energy']
        test_pred_es = test_pred_data['energy']
        do_plot(test_ref_es, error_dict(test_pred_es, test_ref_es), this_ax,
                'Test', by_config_type)
    do_plot(train_ref_es, error_dict(train_pred_es, train_ref_es), this_ax, 'Training', by_config_type)
    this_ax.set_xlabel(f'{ref_energy_name} / eV/atom')
    this_ax.set_ylabel(f'|{pred_energy_name} - {ref_energy_name}| / eV/atom')
    this_ax.set_yscale('log')
    this_ax.set_title('Energy errors')

    #### predicted vs reference
    this_ax = ax[1]
    if test_set:
        do_plot(test_ref_es, test_pred_es, this_ax, 'Test', by_config_type)

    do_plot(train_ref_es, train_pred_es, this_ax, 'Training', by_config_type)

    left_lim, right_lim = this_ax.get_xlim()
    bottom_lim, top_lim = this_ax.get_ylim()
    lims = (min([left_lim, bottom_lim]), max(left_lim, top_lim))
    this_ax.plot(lims, lims, c='k', linewidth=0.8)

    this_ax.set_xlabel(f'{ref_energy_name} / eV/atom')
    this_ax.set_ylabel(f'{pred_energy_name} / eV/atom')
    this_ax.set_title('Energies')
    lgd = this_ax.legend(title='RMSE (meV/atom); RMSE/STD (%)', bbox_to_anchor=(1,1), loc="upper left")


    #####################################################################################
    # Force plots
    #####################################################################################


    for idx, sym in enumerate(train_ref_data['forces'].keys()):

        train_ref_fs = train_ref_data['forces'][sym]
        train_pred_fs = train_pred_data['forces'][sym]
        if test_set:
            test_ref_fs = test_ref_data['forces'][sym]
            test_pred_fs = test_pred_data['forces'][sym]

        # error
        this_ax = ax[2 * (idx + 1) ]
        # print(f'error: {sym}')

        if test_set:
            do_plot(test_ref_fs, error_dict(test_pred_fs, test_ref_fs), this_ax, 'Test', by_config_type)
        do_plot(train_ref_fs, error_dict(train_pred_fs, train_ref_fs), this_ax, 'Training', by_config_type)
        # print(f'element: {sym}, len(error_dict): {len(error_dict(train_pred_fs, train_ref_fs))}, len(fs): {len(train_pred_fs)}')
        # print(train_ref_fs.keys())
        this_ax.set_xlabel(f'{ref_force_name} / eV/Å')
        this_ax.set_ylabel(f'|{pred_force_name} - {ref_force_name}| / eV/Å')
        this_ax.set_yscale('log')
        this_ax.set_title(f'Force component errors on {sym}')

        this_ax = ax[2 * (idx + 1) + 1]
        # print(f'correlation')
        if test_set:
            do_plot(test_ref_fs, test_pred_fs, this_ax, 'Test', by_config_type)
        do_plot(train_ref_fs, train_pred_fs, this_ax, 'Training',
                    by_config_type)

        this_ax.set_xlabel(f'{ref_force_name} / eV/Å')
        this_ax.set_ylabel(f'{pred_force_name} / eV/Å')
        left_lim, right_lim = this_ax.get_xlim()
        bottom_lim, top_lim = this_ax.get_ylim()
        lims = (min([left_lim, bottom_lim]), max(left_lim, top_lim))
        this_ax.plot(lims, lims, c='k', linewidth=0.8)
        this_ax.set_title(f'Force components on {sym}')
        this_ax.legend(title='RMSE (meV/Å); RMSE/STD (%)', bbox_to_anchor=(1,1), loc="upper left" )
    
    return lgd



def error_dict(pred, ref):
    errors = OrderedDict()
    for pred_type, ref_type in zip(pred.keys(), ref.keys()):
        if pred_type != ref_type:
            raise ValueError('Reference and predicted config_types do not match')
        errors[pred_type] = abs(pred[pred_type] - ref[ref_type])
    return errors


def do_plot(ref_values, pred_values, ax, label, by_config_type=False):

    if not by_config_type:
        ref_vals = dict_to_vals(ref_values)
        pred_vals = dict_to_vals(pred_values)

        # print(f'plotting no of compounds: {len(ref_vals)}')
        # print(ref_vals)

        rmse = util.get_rmse(ref_vals, pred_vals)
        std = util.get_std(ref_vals, pred_vals)
        # TODO make formatting nicer
        performance = rmse / np.std(ref_vals) * 100
        print_label = f'{label:<8} {rmse*1000:.3f}; {performance:.1f} %'
        ax.scatter(ref_vals, pred_vals, label=print_label, s=3)

    else:
        n_groups = len(ref_values.keys())

        colors = np.arange(10)
        if label=='Training:':
            cmap = mpl.cm.get_cmap('tab10')
        elif label=='Test':
            cmap = mpl.cm.get_cmap('Dark2')
        else:
            print(f'label: {label}')
            cmap = mpl.cm.get_cmap('Dark2')


        for ref_config_type, pred_config_type, idx in zip(ref_values.keys(), pred_values.keys(), range(n_groups)):
            if ref_config_type != pred_config_type:
                raise ValueError('Reference and predicted config_types do not match')
            ref_vals = ref_values[ref_config_type]
            pred_vals = pred_values[pred_config_type]

            rmse = util.get_rmse(ref_vals, pred_vals)
            std = util.get_std(ref_vals, pred_vals)
            performance = rmse/np.std(ref_vals) * 100
            print_label = f'{ref_config_type}: {rmse*1000:.3f}, {performance:.1f} %'
            kws = {'marker': '.', 's':4, 'color': cmap(colors[idx % 10])}

            ax.scatter(ref_vals, pred_vals, label=print_label,  **kws)



