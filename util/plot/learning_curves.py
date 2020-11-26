from util import ugap
import util
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np


def learning_curves(training_set_sizes, prefix, error_measure):

    print('plotting learning curves')

    ref_energy_name = 'dft_energy'
    pred_energy_name = 'energy'
    ref_force_name = 'dft_forces'
    pred_force_name = 'force'
    evaluated_test_fnames = [f'xyzs/gap{size}_on_test.xyz' for size in training_set_sizes]
    evaluated_train_fnames = [f'xyzs/gap{size}_on_train{size}.xyz' for size in training_set_sizes]

    _learning_curves(evaluated_test_fnames=evaluated_test_fnames, evaluated_train_fnames=evaluated_train_fnames, prefix=prefix,
                    ref_energy_name=ref_energy_name, pred_energy_name=pred_energy_name, ref_force_name=ref_force_name,
                    pred_force_name=pred_force_name, error_measure=error_measure)




def multi_gap_rmse_data(evaluated_test_ats_list, evaluated_train_ats_list, ref_energy_name, pred_energy_name, ref_force_name, pred_force_name, error_measure):
    '''both ats_list are list of sets (lists) of atoms evaluated by gap with different training set size'''

    config_type=None

    if error_measure == 'rmse_over_std':
        error_function = util.get_rmse_over_ref_std
    elif error_measure == 'rmse':
        error_function = util.get_rmse
    elif error_measure == 'mae':
        error_function = util.get_mae


    E_test_errors = []
    E_train_errors = []
    FC_test_errors = []
    FC_train_errors = []
    FH_test_errors = []
    FH_train_errors = []

    for idx, (test_ats, train_ats) in enumerate(zip(evaluated_test_ats_list, evaluated_train_ats_list)):

        # Training set
        E_F_dict_gap = util.get_E_F_dict_evaled(train_ats, energy_name=pred_energy_name, force_name=pred_force_name)
        E_F_dict_dft = util.get_E_F_dict_evaled(train_ats, energy_name=ref_energy_name, force_name=ref_force_name)

        if not config_type:
            config_type = list(E_F_dict_gap['energy'].keys())[0]

        # print(E_F_dict_gap['energy'].keys())

        E_train_errors.append(error_function(pred_ar=E_F_dict_gap['energy'][config_type],
                                           ref_ar=E_F_dict_dft['energy'][config_type]))
        FC_train_errors.append(error_function(pred_ar=E_F_dict_gap['forces']['C'][config_type],
                          ref_ar=E_F_dict_dft['forces']['C'][config_type]))
        FH_train_errors.append(error_function(pred_ar=E_F_dict_gap['forces']['H'][config_type],
                          ref_ar=E_F_dict_dft['forces']['H'][config_type]))

        # Testing set
        E_F_dict_gap = util.get_E_F_dict_evaled(test_ats, energy_name=pred_energy_name, force_name=pred_force_name)
        E_F_dict_dft = util.get_E_F_dict_evaled(test_ats, energy_name=ref_energy_name, force_name=ref_force_name)

        E_test_errors.append(error_function(pred_ar=E_F_dict_gap['energy'][config_type],
                                                       ref_ar=E_F_dict_dft['energy'][config_type]))
        FC_test_errors.append(error_function(pred_ar=E_F_dict_gap['forces']['C'][config_type],
                                                        ref_ar=E_F_dict_dft['forces']['C'][config_type]))
        FH_test_errors.append(error_function(pred_ar=E_F_dict_gap['forces']['H'][config_type],
                                                        ref_ar=E_F_dict_dft['forces']['H'][config_type]))


    return {
        'Ete': E_test_errors,
        'Etr': E_train_errors,
        'FCte': FC_test_errors,
        'FCtr': FC_train_errors,
        'FHte': FH_test_errors,
        'FHtr': FH_train_errors
    }



def _learning_curves(evaluated_test_fnames,
                        evaluated_train_fnames, prefix, ref_energy_name, pred_energy_name, ref_force_name, pred_force_name, error_measure):
    #         data = {Ermses:[np.array([gapx.a_rmse, gapx.b_rmse]),
    #                         np.array([...]), ...],
    #                 FCrmses: [],
    #                 FHrmses: []}
    #               len(Ermses), etc = len(xvalues)

    error_measure = error_measure.lower()
    assert error_measure in ['rmse', 'mae', 'rmse_over_std']

    evaluated_test_ats_list = [read(fname, ':') for fname in evaluated_test_fnames]
    evaluated_train_ats_list = [read(fname, ':') for fname in evaluated_train_fnames]

    #TODO dangerous I'm relying that evaluated train ats list is
    xs = [len(ats) for ats in evaluated_train_ats_list]

    data = multi_gap_rmse_data(evaluated_test_ats_list, evaluated_train_ats_list, ref_energy_name, pred_energy_name,
                               ref_force_name, pred_force_name, error_measure)

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.gca()

    err_kwargs = {'fmt': 'X', 'ms': 6, 'capsize': 6, 'elinewidth': 1,
                  'capthick': 2, 'markeredgewidth': 2}

    lkw = {'marker':'x'}

    color = 'tab:orange'
    # Train, Energy
    ax1.plot(xs, data['Etr'], color=color, linestyle='--',
             label='Energy, Training set', **lkw)

    # Test, Energy
    ax1.plot(xs, data['Ete'], color=color, linestyle='-',
             label='Energy, Testing set', **lkw)

    if error_measure == 'rmse_over_std':
        ax1.set_ylabel('Energy RMSE/STD, %')
    elif error_measure =='rmse':
        ax1.set_ylabel('Energy RMSE, eV/atom')
    elif 'error_measure' =='mae':
        ax1.set_ylabel('Energy MAE, eV/atom')

    ax2 = ax1.twinx()
    #     ax2 = fig.gca()

    color = 'tab:blue'
    # Train, FC
    ax2.plot(xs, data['FCtr'], color=color, linestyle='--',
             label='Force on C, Training set', **lkw)

    # Test, FC
    ax2.plot(xs, data['FCte'], color=color, linestyle='-',
             label='Force on C, Testing set', **lkw)

    color = 'tab:purple'
    # Train, FH
    ax2.plot(xs, data['FHtr'], color=color, linestyle='--',
             label='Force on H, Training set', **lkw)

    # Test, FH
    ax2.plot(xs, data['FHte'], color=color, linestyle='-',
             label='Force on H, Testing set', **lkw)

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

