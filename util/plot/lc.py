from util import ugap
import bde_summary
import util
from ase.io import read
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def learning_curves(training_set_sizes, prefix, error_measure='rmse', dft_bde_fname=None, dataset='both', force='overall'):

    print('plotting learning curves')

    ref_energy_name = 'dft_energy'
    pred_energy_name = 'gap_energy'
    ref_force_name = 'dft_forces'
    pred_force_name = 'gap_forces'
    evaluated_test_fnames = [f'xyzs/gap{size}_on_test.xyz' for size in training_set_sizes]
    evaluated_train_fnames = [f'xyzs/gap{size}_on_train.xyz' for size in training_set_sizes]

    _e_f_learning_curves(evaluated_test_fnames=evaluated_test_fnames, evaluated_train_fnames=evaluated_train_fnames, prefix=prefix,
                    ref_energy_name=ref_energy_name, pred_energy_name=pred_energy_name, ref_force_name=ref_force_name,
                    pred_force_name=pred_force_name, error_measure=error_measure,
                    dataset=dataset, force=force)


    gap_bde_fnames = [f'gap_bdes/gap{size}_optimised.xyz' for size in training_set_sizes]
    _bde_learning_curve(gap_bde_fnames, dft_bde_fname, prefix, training_set_sizes)

def _bde_learning_curve(gap_bde_fnames, dft_bde_fname, prefix, training_set_sizes):


    # sort into dictionary of my_dict[label] = learning curve data
    all_bde_errors = {}
    all_rmsds = {}
    for gap_bde_fname in gap_bde_fnames:

        labels, rmsds, bde_errors = get_bde_errors(gap_bde_fname, dft_bde_fname)

        for label, rmsd, error in zip(labels, rmsds, bde_errors):
            if label not in all_bde_errors.keys():
                all_bde_errors[label] = []
            if label not in all_rmsds.keys():
                all_rmsds[label] = []
            all_bde_errors[label].append(error)
            all_rmsds[label].append(rmsd)

    base_kw = {'marker':'x', 'alpha':0.6}

    plt.figure(figsize=(12,5))
    gs = mpl.gridspec.GridSpec(1, 2)
    ax_bde = plt.subplot(gs[0])
    ax_rmsd = plt.subplot(gs[1])

    for data, ax in zip([all_bde_errors, all_rmsds], [ax_bde, ax_rmsd]):

        if ax == ax_bde:
            ax.axhline(y=50, label='50 meV', color='k', ls='--', linewidth=0.6)

        for label, values in data.items():
            lkw = base_kw.copy()
            if label == 'mean':
                lkw['color'] = 'k'
                lkw['alpha'] = 1
            lkw['label'] = label
            ax.plot(training_set_sizes, values, **lkw)


        ax.grid(color='lightgrey', linestyle=':', which='both')
        ax.set_xlabel('Training set size (per config or total)')
        ax.set_yscale('log')
        ax.legend()
        ax.set_title(prefix)


    ax_bde.set_ylabel('BDE (radicals) or total energy (mol) absolute error / meV')
    ax_rmsd.set_ylabel('RMSD / Å')

    plt.tight_layout()

    plt.savefig(f'{prefix}_bde_lc.png', dpi=300)


def get_bde_errors(gap_bde_fname, dft_bde_fname):

    gap_ats = read(gap_bde_fname, ':')

    data = bde_summary.table(title=None, dft_fname=dft_bde_fname, gap_ats=gap_ats, precision=3, printing=False)
    data = data.drop([1])

    labels = list(data[data.columns[0]])
    rmsds = list(data[data.columns[-1]])
    bde_errs = list(data[data.columns[-2]])

    return labels, rmsds, bde_errs

def multi_gap_rmse_data(evaluated_test_ats_list, evaluated_train_ats_list, ref_energy_name,
                        pred_energy_name, ref_force_name, pred_force_name, error_measure):
    '''both ats_list are list of sets (lists) of atoms evaluated by gap with different training set size'''

    if error_measure == 'rmse_over_std':
        error_function = util.get_rmse_over_ref_std
    elif error_measure == 'rmse':
        error_function = util.get_rmse
    elif error_measure == 'mae':
        error_function = util.get_mae

    elements = ['C', 'H', 'O']

    err_train_dict = {}
    err_test_dict = {}
    for err_dict in [err_train_dict, err_test_dict]:
        err_dict['E'] = []
        err_dict['Ftotal'] = []
        for element in elements:
            err_dict[f'F{element}'] = []


    for idx, (test_ats, train_ats) in enumerate(zip(evaluated_test_ats_list, evaluated_train_ats_list)):

        for set_dict, set_ats in zip([err_train_dict, err_test_dict], [train_ats, test_ats]):

            # Training set
            E_F_dict_gap = util.get_E_F_dict_evaled(set_ats, energy_name=pred_energy_name, force_name=pred_force_name)
            E_F_dict_dft = util.get_E_F_dict_evaled(set_ats, energy_name=ref_energy_name, force_name=ref_force_name)


            ref_Es_for_all_config_types = util.dict_to_vals(E_F_dict_dft['energy'])
            pred_Es_for_all_config_types = util.dict_to_vals(E_F_dict_gap['energy'])
            # print(f'ref_Es_len, {len(ref_Es_for_all_config_types)}')
            # print(f'pred_Es_len, {len(pred_Es_for_all_config_types)}')
            set_dict['E'].append(error_function(pred_ar=pred_Es_for_all_config_types,
                                               ref_ar=ref_Es_for_all_config_types))
            all_pred_Fs = np.array([])
            all_ref_Fs = np.array([])
            # for element in elements:
            for element in E_F_dict_dft['forces'].keys():
                # concatenate all of the config_types into one
                ref_Fs_for_all_config_types = util.dict_to_vals(E_F_dict_dft['forces'][element])
                pred_Fs_for_all_config_types = util.dict_to_vals(E_F_dict_gap['forces'][element])
                # print(f'ref_Fs_len, {len(ref_Fs_for_all_config_types)}')
                # print(f'pred_Fs_len, {len(pred_Fs_for_all_config_types)}')

                set_dict[f'F{element}'].append(error_function(pred_ar=pred_Fs_for_all_config_types,
                                                                     ref_ar=ref_Fs_for_all_config_types))

                all_pred_Fs = np.append(all_pred_Fs, pred_Fs_for_all_config_types)
                all_ref_Fs = np.append(all_ref_Fs, ref_Fs_for_all_config_types)

            # print(f'totalF pred: {len(all_pred_Fs)}')
            # print(f'totalF ref: {len(all_ref_Fs)}')
            set_dict['Ftotal'].append(error_function(pred_ar=all_pred_Fs, ref_ar=all_ref_Fs))


    return err_train_dict, err_test_dict


def _e_f_learning_curves(evaluated_test_fnames,
                        evaluated_train_fnames, prefix, ref_energy_name, pred_energy_name,
                         ref_force_name, pred_force_name, error_measure,
                         force='by_element', dataset='test_only'):


    elements = ['C', 'H', 'O']

    error_measure = error_measure.lower()
    assert error_measure in ['rmse', 'mae', 'rmse_over_std']
    assert force in ['by_element', 'overall']
    assert dataset in ['test_only', 'both']

    evaluated_test_ats_list = [read(fname, ':') for fname in evaluated_test_fnames]
    evaluated_train_ats_list = [read(fname, ':') for fname in evaluated_train_fnames]

    xs = [len(ats) for ats in evaluated_train_ats_list]

    err_train_dict, err_test_dict = multi_gap_rmse_data(evaluated_test_ats_list, evaluated_train_ats_list, ref_energy_name, pred_energy_name,
                               ref_force_name, pred_force_name, error_measure)

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.gca()

    err_kwargs = {'fmt': 'X', 'ms': 6, 'capsize': 6, 'elinewidth': 1,
                  'capthick': 2, 'markeredgewidth': 2}

    color_dict = {'E':'tab:blue', 'FC':'tab:grey', 'FH':'tab:green', 'FO':'tab:red', 'Ftotal':'tab:orange'}

    label_dict = {'E':'Energy', 'Ftotal':'Force, overall'}
    for element in elements:
        label_dict[f'F{element}'] = f'Force on {element}'

    lkw = {'marker':'x'}

    prop_keys = ['Ftotal']
    if force == 'by_element':
        prop_keys += [f'F{element}' for element in elements]


    # plot energy stuff
    if dataset == 'both':
        ax1.plot(xs, err_train_dict['E'], color=color_dict['E'], linestyle=':',
                 label='Energy, Train', **lkw)
    ax1.plot(xs, err_test_dict['E'], color=color_dict['E'], linestyle='-',
             label='Energy, Test', **lkw)


    if error_measure == 'rmse_over_std':
        ax1.set_ylabel('Energy RMSE/STD, %')
    elif error_measure =='rmse':
        ax1.set_ylabel('Energy RMSE, eV/atom')
    elif 'error_measure' =='mae':
        ax1.set_ylabel('Energy MAE, eV/atom')

    ax2 = ax1.twinx()

    for prop in prop_keys:
        if dataset == 'both':
            ax2.plot(xs, err_train_dict[prop], color_dict[prop], linestyle=':', label=f'{label_dict[prop]}, Train', **lkw)
        ax2.plot(xs, err_test_dict[prop], color_dict[prop], linestyle='-', label=f'{label_dict[prop]}, Test', **lkw)

    if error_measure == 'rmse_over_std':
        ax2.set_ylabel('Force RMSE/STD, %')
    elif error_measure =='rmse':
        ax2.set_ylabel('Force RMSE, eV/Å')
    elif error_measure == 'mae':
        ax2.set_ylabel('Force MAE, eV/Å')

    ax1.legend()
    ax2.legend(bbox_to_anchor=(0, 0, 1, 0.85))
    ax2.set_xlabel('Training set size')
    ax1.set_xlabel('Training set size')
    ax2.set_yscale('log')
    ax1.set_yscale('log')
    plt.title(prefix)
    plt.tight_layout()

    if prefix is None:
        plt.show()
    else:
        plt.savefig(f'{prefix}_{error_measure}.png', dpi=300)

