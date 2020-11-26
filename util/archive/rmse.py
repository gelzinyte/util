from util import ugap
import util
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np


def multi_gap_rmse_data(test_ats, gap_names):
    all_E_test_rmses = []
    all_E_train_rmses = []
    all_FC_test_rmses = []
    all_FC_train_rmses = []
    all_FH_test_rmses = []
    all_FH_train_rmses = []

    for gap_bunch in gap_names:

        E_test_rmses = []
        E_train_rmses = []
        FC_test_rmses = []
        FC_train_rmses = []
        FH_test_rmses = []
        FH_train_rmses = []

        for gap_fname in gap_bunch:
            print(gap_fname)

            # Training set
            train_ats = ugap.atoms_from_gap(gap_fname)
            E_F_dict_gap = util.get_E_F_dict(train_ats, 'gap', gap_fname)
            E_F_dict_dft = util.get_E_F_dict(train_ats, 'dft')

            E_train_rmses.append(util.get_rmse(E_F_dict_gap['energy']['none'],
                                               E_F_dict_dft['energy'][
                                                   'none']))
            FC_train_rmses.append(
                util.get_rmse(E_F_dict_gap['forces']['C']['none'],
                              E_F_dict_dft['forces']['C']['none']))
            FH_train_rmses.append(
                util.get_rmse(E_F_dict_gap['forces']['H']['none'],
                              E_F_dict_dft['forces']['H']['none']))

            # Testing set
            E_F_dict_gap = util.get_E_F_dict(test_ats, 'gap', gap_fname)
            E_F_dict_dft = util.get_E_F_dict(test_ats, 'dft')

            E_test_rmses.append(util.get_rmse(E_F_dict_gap['energy']['none'],
                                              E_F_dict_dft['energy']['none']))
            FC_test_rmses.append(
                util.get_rmse(E_F_dict_gap['forces']['C']['none'],
                              E_F_dict_dft['forces']['C']['none']))
            FH_test_rmses.append(
                util.get_rmse(E_F_dict_gap['forces']['H']['none'],
                              E_F_dict_dft['forces']['H']['none']))

        all_E_test_rmses.append(np.array(E_test_rmses))
        all_E_train_rmses.append(np.array(E_train_rmses))
        all_FC_test_rmses.append(np.array(FC_test_rmses))
        all_FC_train_rmses.append(np.array(FC_train_rmses))
        all_FH_test_rmses.append(np.array(FH_test_rmses))
        all_FH_train_rmses.append(np.array(FH_train_rmses))

    return {
        'Ete': all_E_test_rmses,
        'Etr': all_E_train_rmses,
        'FCte': all_FC_test_rmses,
        'FCtr': all_FC_train_rmses,
        'FHte': all_FH_test_rmses,
        'FHtr': all_FH_train_rmses
    }


def compare(gap_names, test_fname, xs, xlabel, title, pic_name=None):
    test_ats = read(test_fname, ':')

    #         len(gap_names) should be same as len(xvalues)
    #         data = list of np.arrays with rmses from each of the set of
    #         gaps with same parameters
    #         data = {Ermses:[np.array([gapx.a_rmse, gapx.b_rmse]),
    #         np.array([...]), ...],
    #                 FCrmses: [],
    # FHrmses: []}
    #         len(Ermses), etc = len(xvalues)

    data = multi_gap_rmse_data(test_ats, gap_names)

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.gca()

    err_kwargs = {'fmt': 'X', 'ms': 6, 'capsize': 6, 'elinewidth': 1,
                  'capthick': 2, 'markeredgewidth': 2}

    color = 'tab:orange'
    # Train, Energy
    means = [np.mean(arr) for arr in data['Etr']]
    stds = [np.std(arr) for arr in data['Etr']]
    ax1.plot(xs, means, color=color, linestyle='--',
             label='Energy, Training set')
    ax1.errorbar(xs, means, yerr=stds, color=color, **err_kwargs)

    # Test, Energy
    means = [np.mean(arr) for arr in data['Ete']]
    stds = [np.std(arr) for arr in data['Ete']]
    ax1.plot(xs, means, color=color, linestyle='-',
             label='Energy, Testing set')
    ax1.errorbar(xs, means, yerr=stds, color=color, **err_kwargs)

    ax1.set_ylabel('Energy RMSE, eV/atom')

    ax2 = ax1.twinx()
    #     ax2 = fig.gca()

    color = 'tab:blue'
    # Train, FC
    means = [np.mean(arr) for arr in data['FCtr']]
    stds = [np.std(arr) for arr in data['FCtr']]
    ax2.plot(xs, means, color=color, linestyle='--',
             label='Force on C, Training set')
    ax2.errorbar(xs, means, yerr=stds, color=color, **err_kwargs)
    print('C force train error means:', means)

    # Test, FC
    means = [np.mean(arr) for arr in data['FCte']]
    stds = [np.std(arr) for arr in data['FCte']]
    ax2.plot(xs, means, color=color, linestyle='-',
             label='Force on C, Testing set')
    ax2.errorbar(xs, means, yerr=stds, color=color, **err_kwargs)
    print('C force test error means:', means)

    color = 'tab:purple'
    # Train, FH
    means = [np.mean(arr) for arr in data['FHtr']]
    stds = [np.std(arr) for arr in data['FHtr']]
    ax2.plot(xs, means, color=color, linestyle='--',
             label='Force on H, Training set')
    ax2.errorbar(xs, means, yerr=stds, color=color, **err_kwargs)
    print('H force train error means:', means)

    # Test, FH
    means = [np.mean(arr) for arr in data['FHte']]
    stds = [np.std(arr) for arr in data['FHte']]
    ax2.plot(xs, means, color=color, linestyle='-',
             label='Force on H, Testing set')
    ax2.errorbar(xs, means, yerr=stds, color=color, **err_kwargs)
    print('H force test error means:', means)

    ax2.set_ylabel('Force RMSE, eV/Å')

    ax1.legend()
    ax2.legend()
    ax2.set_xlabel(xlabel)
    ax1.set_xlabel(xlabel)
    ax2.set_xscale('log')
    plt.title(title)
    plt.tight_layout()

    if pic_name is None:
        plt.show()
    else:
        plt.savefig(pic_name, dpi=300)


