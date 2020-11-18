from util import ugap
import util
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np


def learning_curves_100K(training_set_sizes):

    print('plotting learning curves')

    ref_energy_name = 'dft_energy'
    pred_energy_name = 'energy'
    ref_force_name = 'dft_forces'
    pred_force_name = 'force'
    evaluated_test_fnames = [f'xyzs/gap{size}_on_test.xyz' for size in training_set_sizes]
    evaluated_train_fnames = [f'xyzs/gap{size}_on_train{size}.xyz' for size in training_set_sizes]

    learning_curves(evaluated_test_fnames=evaluated_test_fnames, evaluated_train_fnames=evaluated_train_fnames, prefix='learning_curve_100K',
                    ref_energy_name=ref_energy_name, pred_energy_name=pred_energy_name, ref_force_name=ref_force_name, pred_force_name=pred_force_name)




def multi_gap_rmse_data(evaluated_test_ats_list, evaluated_train_ats_list, ref_energy_name, pred_energy_name, ref_force_name, pred_force_name):
    '''both ats_list are list of sets (lists) of atoms evaluated by gap with different training set size'''

    # TODO add option to pass as an argument
    config_type = '100 K'

    E_test_rmses = []
    E_train_rmses = []
    FC_test_rmses = []
    FC_train_rmses = []
    FH_test_rmses = []
    FH_train_rmses = []

    for idx, (test_ats, train_ats) in enumerate(zip(evaluated_test_ats_list, evaluated_train_ats_list)):

        print(idx)

        # Training set
        E_F_dict_gap = util.get_E_F_dict_evaled(train_ats, energy_name=pred_energy_name, force_name=pred_force_name)
        E_F_dict_dft = util.get_E_F_dict_evaled(train_ats, energy_name=ref_energy_name, force_name=ref_force_name)

        # print(E_F_dict_gap['energy'].keys())

        E_train_rmses.append(util.get_rmse(E_F_dict_gap['energy'][config_type],
                                           E_F_dict_dft['energy'][
                                               config_type]))
        FC_train_rmses.append(
            util.get_rmse(E_F_dict_gap['forces']['C'][config_type],
                          E_F_dict_dft['forces']['C'][config_type]))
        FH_train_rmses.append(
            util.get_rmse(E_F_dict_gap['forces']['H'][config_type],
                          E_F_dict_dft['forces']['H'][config_type]))

        # Testing set
        E_F_dict_gap = util.get_E_F_dict_evaled(test_ats, energy_name=pred_energy_name, force_name=pred_force_name)
        E_F_dict_dft = util.get_E_F_dict_evaled(test_ats, energy_name=ref_energy_name, force_name=ref_force_name)

        E_test_rmses.append(util.get_rmse(E_F_dict_gap['energy'][config_type],
                                          E_F_dict_dft['energy'][config_type]))
        FC_test_rmses.append(
            util.get_rmse(E_F_dict_gap['forces']['C'][config_type],
                          E_F_dict_dft['forces']['C'][config_type]))
        FH_test_rmses.append(
            util.get_rmse(E_F_dict_gap['forces']['H'][config_type],
                          E_F_dict_dft['forces']['H'][config_type]))

        # all_E_test_rmses.append(np.array(E_test_rmses))
        # all_E_train_rmses.append(np.array(E_train_rmses))
        # all_FC_test_rmses.append(np.array(FC_test_rmses))
        # all_FC_train_rmses.append(np.array(FC_train_rmses))
        # all_FH_test_rmses.append(np.array(FH_test_rmses))
        # all_FH_train_rmses.append(np.array(FH_train_rmses))

    return {
        'Ete': E_test_rmses,
        'Etr': E_train_rmses,
        'FCte': FC_test_rmses,
        'FCtr': FC_train_rmses,
        'FHte': FH_test_rmses,
        'FHtr': FH_train_rmses
    }



def learning_curves(evaluated_test_fnames,
                        evaluated_train_fnames, prefix, ref_energy_name, pred_energy_name, ref_force_name, pred_force_name):
    #         data = {Ermses:[np.array([gapx.a_rmse, gapx.b_rmse]),
    #                         np.array([...]), ...],
    #                 FCrmses: [],
    #                 FHrmses: []}
    #               len(Ermses), etc = len(xvalues)

    evaluated_test_ats_list = [read(fname, ':') for fname in evaluated_test_fnames]
    evaluated_train_ats_list = [read(fname, ':') for fname in evaluated_train_fnames]

    #TODO dangerous I'm relying that evaluated train ats list is
    xs = [len(ats) for ats in evaluated_train_ats_list]

    data = multi_gap_rmse_data(evaluated_test_ats_list, evaluated_train_ats_list, ref_energy_name, pred_energy_name, ref_force_name, pred_force_name)

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.gca()

    err_kwargs = {'fmt': 'X', 'ms': 6, 'capsize': 6, 'elinewidth': 1,
                  'capthick': 2, 'markeredgewidth': 2}

    color = 'tab:orange'
    # Train, Energy
    ax1.plot(xs, data['Etr'], color=color, linestyle='--',
             label='Energy, Training set')

    # Test, Energy
    ax1.plot(xs, data['Ete'], color=color, linestyle='-',
             label='Energy, Testing set')

    ax1.set_ylabel('Energy RMSE, eV/atom')

    ax2 = ax1.twinx()
    #     ax2 = fig.gca()

    color = 'tab:blue'
    # Train, FC
    ax2.plot(xs, data['FCtr'], color=color, linestyle='--',
             label='Force on C, Training set')

    # Test, FC
    ax2.plot(xs, data['FCte'], color=color, linestyle='-',
             label='Force on C, Testing set')

    color = 'tab:purple'
    # Train, FH
    ax2.plot(xs, data['FHtr'], color=color, linestyle='--',
             label='Force on H, Training set')

    # Test, FH
    ax2.plot(xs, data['FHte'], color=color, linestyle='-',
             label='Force on H, Testing set')

    ax2.set_ylabel('Force RMSE, eV/Å')

    ax1.legend()
    ax2.legend()
    ax2.set_xlabel('Training set size')
    ax1.set_xlabel('Training set size')
    ax2.set_xscale('log')
    plt.title(prefix)
    plt.tight_layout()

    if prefix is None:
        plt.show()
    else:
        plt.savefig(f'{prefix}.png', dpi=300)




