
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





def make_scatter_plots_from_file(param_fname, test_fname=None, output_dir=None, prefix=None,
                                 by_config_type=False, ref_name='dft'):

    test_ats = None
    if test_fname:
        test_ats = read(test_fname, index=':')

    make_scatter_plots(param_fname=param_fname,  test_ats=test_ats, output_dir=output_dir,
                       prefix=prefix, by_config_type=by_config_type, ref_name=ref_name)


def make_scatter_plots(param_fname, test_ats=None, output_dir=None, prefix=None, by_config_type=False, ref_name='dft'):

    train_ats = ugap.atoms_from_gap(param_fname)

    counts = util.get_counts(train_ats[0])
    no_unique_elements = len(counts.keys())
    width = 14
    height = width * 0.4
    height *= no_unique_elements

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(no_unique_elements+1, 2)
    ax = [plt.subplot(g) for g in gs]

    #main plot
    lgd = scatter_plot(param_fname=param_fname, train_ats=train_ats, ax=ax, test_ats=test_ats, by_config_type=by_config_type, ref_name=ref_name)

    if not prefix:
        prefix = os.path.basename(param_fname)
        prefix = os.path.splitext(prefix)[0]
    picture_fname = f'{prefix}_scatter.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.suptitle(prefix)
    plt.tight_layout(rect=[0, 0, 1, 1.1])

    if output_dir:
        plt.savefig(picture_fname, dpi=300, bbox_extra_artists=(lgd,),
                    bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def scatter_plot(param_fname, train_ats, ax, test_ats=None, by_config_type=False, ref_name='dft'):
    '''ax - axes list'''
    test_set = False
    if test_ats:
        test_set = True

    train_ref_data = util.get_E_F_dict(train_ats, calc_type=ref_name)
    train_pred_data = util.get_E_F_dict(train_ats, calc_type='gap', param_fname=param_fname)

    if test_set:
        test_ref_data = util.get_E_F_dict(test_ats, calc_type=ref_name)
        test_pred_data = util.get_E_F_dict(test_ats, calc_type='gap', param_fname=param_fname)

    #####################################################################
    # Energy plots
    ####################################################################

    #### error scatter
    train_ref_es = train_ref_data['energy']
    train_pred_es = train_pred_data['energy']

    this_ax = ax[0]
    do_plot(train_ref_es, error_dict(train_pred_es, train_ref_es), this_ax, 'Training', by_config_type)
    if test_set:
        test_ref_es = test_ref_data['energy']
        test_pred_es = test_pred_data['energy']
        do_plot(test_ref_es, error_dict(test_pred_es, test_ref_es), this_ax, 'Test', by_config_type)
    this_ax.set_xlabel(f'{ref_name.upper()} energy / eV/atom')
    this_ax.set_ylabel(f'|E$_{{GAP}}$ - E$_{{{ref_name.upper()}}}$| / eV/atom')
    this_ax.set_yscale('log')
    this_ax.set_title('Energy errors')

    #### predicted vs reference
    this_ax = ax[1]
    do_plot(train_ref_es, train_pred_es, this_ax, 'Training', by_config_type)

    if test_set:
        do_plot(test_ref_es, test_pred_es, this_ax, 'Test', by_config_type)

    left_lim, right_lim = this_ax.get_xlim()
    bottom_lim, top_lim = this_ax.get_ylim()
    lims = (min([left_lim, bottom_lim]), max(left_lim, top_lim))
    this_ax.plot(lims, lims, c='k', linewidth=0.8)

    this_ax.set_xlabel(f'{ref_name.upper()} energy / eV/atom')
    this_ax.set_ylabel(f'GAP energy / eV/atom')
    this_ax.set_title('Energies')
    lgd = this_ax.legend(title='RMSE  eV/atom', bbox_to_anchor=(1,1), loc="upper left")


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
        do_plot(train_ref_fs, error_dict(train_pred_fs, train_ref_fs), this_ax, 'Training', by_config_type)
        if test_set:
            do_plot(test_ref_fs, error_dict(test_pred_fs, test_ref_fs), this_ax, 'Test', by_config_type)
        this_ax.set_xlabel(f'{ref_name.upper()} force / eV/Å')
        this_ax.set_ylabel(f'|F$_{{GAP}}$ - F$_{{{ref_name.upper()}}}$| / eV/Å')
        this_ax.set_yscale('log')
        this_ax.set_title(f'Force component errors on {sym}')

        this_ax = ax[2 * (idx + 1) + 1]
        do_plot(train_ref_fs, train_pred_fs, this_ax, 'Training', by_config_type)

        if test_set:
            do_plot(test_ref_fs, test_pred_fs, this_ax, 'Test', by_config_type)

        this_ax.set_xlabel(f'{ref_name.upper()} force / eV/Å')
        this_ax.set_ylabel('GAP force / eV/Å')
        left_lim, right_lim = this_ax.get_xlim()
        bottom_lim, top_lim = this_ax.get_ylim()
        lims = (min([left_lim, bottom_lim]), max(left_lim, top_lim))
        this_ax.plot(lims, lims, c='k', linewidth=0.8)
        this_ax.set_title(f'Force components on {sym}')
        this_ax.legend(title='Set: RMSE, eV/Å', bbox_to_anchor=(1,1), loc="upper left" )
    
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

        rmse = util.get_rmse(ref_vals, pred_vals)
        std = util.get_std(ref_vals, pred_vals)
        # TODO make formatting nicer
        print_label = f'{label}: {rmse:.3f} $\pm$ {std:.3f}'
        ax.scatter(ref_vals, pred_vals, label=print_label, s=8, alpha=0.7)

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
            print_label = f'{ref_config_type}: {rmse:.3f}, {performance:.1f} %'
            kws = {'marker': '.', 's':4, 'color': cmap(colors[idx % 10])}

            ax.scatter(ref_vals, pred_vals, label=print_label,  **kws)



