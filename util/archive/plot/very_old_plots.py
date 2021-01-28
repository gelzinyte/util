def plot_heatmap(data_dict, ax, obs):
    df = pd.DataFrame.from_dict(data_dict)
    hmap = ax.pcolormesh(df, vmin=0)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(df.index)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(df.columns, rotation=90)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            color = 'red'
            text = ax.text(j + 0.5, i + 0.5, round(df.iat[i, j], 3), ha='center', color=color)
    cbar = plt.colorbar(hmap, ax=ax)
    if obs == 'Energy':
        units = 'eV/atom'
    elif obs == 'Force':
        units = 'eV/Å '
    cbar.ax.set_ylabel(f'{obs} RMSE, {units}', rotation=90, labelpad=6)
    ax.set_title(f'{obs} RMSE', fontsize=14 )



def rmse_heatmap(train_fname, gaps_dir='gaps', output_dir='pictures', prefix=None):

    train_ats = read(train_fname, index=':')
    dft_data = util.get_E_F_dict(train_ats, calc_type='dft')
    dft_data['forces'] = util.desymbolise_force_dict(dft_data['forces'])

    # TODO think this through better
    # max_gap_dset_no = 15
    # if len(dft_data['energy'].keys()) > max_gap_dset_no:
    #     print(f'more than {max_gap_dset_no} config types, taking the last {max_gap_dset_no}')
    #     dft_data = get_last_bunch(dft_data, bunch=max_gap_dset_no)

    E_rmses = dict()
    F_rmses = dict()

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = util.natural_sort(gap_fnames)

    # if len(gap_fnames) > max_gap_dset_no:
    #     print(f'more than {max_gap_dset_no} gaps, taking the last {max_gap_dset_no}')
    #     gap_fnames = gap_fnames[-max_gap_dset_no:]


    for gap_fname in tqdm(gap_fnames):
        gap_title = os.path.splitext(gap_fname)[0]
        gap_fname = os.path.join(gaps_dir, gap_fname)
        gap_data = util.get_E_F_dict(train_ats, calc_type='gap', param_fname=gap_fname)
        gap_data['forces'] = util.desymbolise_force_dict(gap_data['forces'])

        # if more than 20, take the last 20 only
        # if len(gap_data['energy'].keys()) > max_gap_dset_no:
        #     gap_data = get_last_bunch(gap_data, bunch=max_gap_dset_no)

        E_rmses[gap_title] = get_rmse_dict(obs='energy', dft_data=dft_data, gap_data=gap_data)
        F_rmses[gap_title] = get_rmse_dict(obs='forces', dft_data=dft_data, gap_data=gap_data)

    # Make plot
    # TODO save the pandas dataframe somewhere somehow

    N = len(gap_fnames)
    width = (N * 0.6 + 1.2) * 2
    height = N * 0.6
    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(1, 2)
    all_ax = [plt.subplot(g) for g in gs]

    plot_heatmap(E_rmses, all_ax[0], 'Energy')
    plot_heatmap(F_rmses, all_ax[1], 'Force')
    plt.tight_layout()

    if not prefix:
        prefix = 'summary'
    picture_fname = f'{prefix}_RMSE_heatmap.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.savefig(picture_fname, dpi=300)
    plt.close(fig)

def summary_scatter(gaps_dir, output_dir=None, prefix=None, ref_name='dft'):
    # not worked out and not really worth it.
    # gap fnames
    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = [os.path.join(gaps_dir, name) for name in gap_fnames]
    gap_fnames = util.natural_sort(gap_fnames)

    # get how many elements are present and make up axis of the right shape
    # assumes that last gap file has all of them
    tmp_ats = atoms_from_gap(gap_fnames[-1])[0]
    counts = get_counts(tmp_ats)
    no_unique_elements = len(counts.keys())
    width = 10
    height = width * 0.6
    height *= no_unique_elements
    plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(no_unique_elements + 1, 2)
    ax = [plt.subplot(g) for g in gs]

    for param_fname in gap_fnames:
        train_ats = atoms_from_gap(param_fname)
        scatter_plot(param_fname, train_ats, ax)



def get_train_test_rmse_dicts(gap_idx, dft_data, gap_data):
    no_dsets = len(dft_data['energy'].keys())

    training_rmses = dict()
    test_rmses = dict()

    e_train_dft = {}
    e_test_dft = {}
    e_train_gap = {}
    e_test_gap = {}

    for key1, key2 in zip(dft_data['energy'].keys(), gap_data['energy'].keys()):
        if key1 != key2:
            raise Exception(f'observations do not match. dft: {key1}, gap: {key2}')
        dset_no = int(re.search(r'\d+', key1).group())

        if dset_no <= gap_idx:
            e_train_dft[key1] = dft_data['energy'][key1]
            e_train_gap[key2] = gap_data['energy'][key2]
        else:
            e_test_dft[key1] = dft_data['energy'][key1]
            e_test_gap[key2] = gap_data['energy'][key2]

    training_rmses['energy'] = util.get_rmse(dict_to_vals(e_train_dft), dict_to_vals(e_train_gap))
    if gap_idx != no_dsets:
        test_rmses['energy'] = util.get_rmse(dict_to_vals(e_test_dft), dict_to_vals(e_test_gap))

    training_rmses['forces'] = {}
    test_rmses['forces'] = {}
    for sym1, sym2 in zip(dft_data['forces'].keys(), gap_data['forces'].keys()):
        if sym1 != sym2:
            raise Exception(f'observations do not match. dft: {sym1}, gap: {sym2}')

        f_train_dft = {}
        f_test_dft = {}
        f_train_gap = {}
        f_test_gap = {}
        for key1, key2 in zip(dft_data['forces'][sym1].keys(), gap_data['forces'][sym2].keys()):
            if key1 != key2:
                raise Exception(f'observations do not match. dft: {key1}, gap: {key2}')
            dset_no = int(re.search(r'\d+', key1).group())
            if dset_no <= gap_idx:
                f_train_dft[key1] = dft_data['forces'][sym1][key1]
                f_train_gap[key2] = gap_data['forces'][sym2][key2]
            else:
                f_test_dft[key1] = dft_data['forces'][sym1][key1]
                f_test_gap[key2] = gap_data['forces'][sym2][key2]
        training_rmses['forces'][sym1] = util.get_rmse(dict_to_vals(f_train_dft), dict_to_vals(f_train_gap))
        if gap_idx != no_dsets:
            test_rmses['forces'][sym2] = util.get_rmse(dict_to_vals(f_test_dft), dict_to_vals(f_test_gap))

    return (training_rmses, test_rmses)



def rmse_line_plots(train_fname, gaps_dir='gaps', output_dir='pictures', prefix=None):
    train_ats = read(train_fname, index=':')
    dft_data = util.get_E_F_dict(train_ats, calc_type='dft')

    all_train_rmses = {}
    all_test_rmses = {}

    gap_fnames = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames = util.natural_sort(gap_fnames)

    for idx, gap_fname in enumerate(tqdm(gap_fnames)):
        gap_title = os.path.splitext(gap_fname)[0]
        gap_fname = os.path.join(gaps_dir, gap_fname)
        gap_data = util.get_E_F_dict(train_ats, calc_type='gap', param_fname=gap_fname)

        train_rmses, test_rmses = get_train_test_rmse_dicts(gap_idx=idx + 1, dft_data=dft_data, gap_data=gap_data)

        all_train_rmses[gap_title] = train_rmses
        all_test_rmses[gap_title] = test_rmses

    fig = plt.figure(figsize=(12, 8))

    ax2 = plt.gca()
    syms = all_train_rmses[gap_title]['forces'].keys()
    cmap = mpl.cm.get_cmap('viridis')
    colors = np.linspace(0, 0.9, len(syms))
    for idx, sym in enumerate(syms):
        color = cmap(colors[idx])
        forces_train = [value['forces'][sym] for key, value in all_train_rmses.items()]
        ax2.plot(range(1, len(forces_train)+1), forces_train, linestyle='-', color=color, marker='x', \
                 label=f'on {sym}, training set')

        # plt.gca().annotate(int(no), xy=(pt['x'], pt['y']))

        forces_test = [value['forces'][sym] for key, value in all_test_rmses.items()]
        ax2.plot(range(1, len(forces_test)+1), forces_test, linestyle=':', color=color, marker='x', \
                label=f'on {sym}, testing set')

        ax2.annotate(forces_test[-1], xy=(len(forces_test)+1, forces_test[-1]))
        ax2.annotate(forces_train[-1], xy=(len(forces_train) + 1, forces_train[-1]))


    ax1 = ax2.twinx()
    energies_train = [value['energy'] for key, value in all_train_rmses.items()]
    ax1.plot(range(1, len(energies_train)+1), energies_train, linestyle='-', color='tab:red', marker='x', label='training set')

    energies_test = [value['energy'] for key, value in all_test_rmses.items() if 'energy' in value.keys()]
    ax1.plot(range(1, len(energies_test)+1), energies_test, linestyle=':', color='tab:red', marker='x', label='testing set')

    ax1.annotate(energies_test[-1], xy=(len(energies_test)+1, energies_test[-1]))
    ax1.annotate(energies_train[-1], xy=(len(energies_train)+1, energies_train[-1]))

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        formatter = mpl.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))
        ax.get_yaxis().set_minor_formatter(formatter)


    ax1.legend(title='Energy')
    ax2.legend(title='Force', loc='best', bbox_to_anchor=(0, 0, 1, 0.85))

    ax1.set_ylabel('Energy RMSE, eV/atom')
    ax2.set_ylabel('Force component RMSE, eV/Å')

    ax2.set_xlabel('Iteration')
    plt.title('RMSEs for GAP_i')

    if not prefix:
        prefix = 'summary'
    picture_fname = f'{prefix}_RMSEs.png'
    if output_dir:
        picture_fname = os.path.join(output_dir, picture_fname)

    plt.tight_layout()
    plt.savefig(picture_fname, dpi=300)
    plt.close(fig)



def kpca_plot(xyz_fname, pic_name, output_dir):
    atoms = read(xyz_fname, ':')
    cmap = mpl.cm.get_cmap('tab20')
    cmap10 = mpl.cm.get_cmap('tab10')
    color_idx = np.linspace(0, 1, 10)
    kpca_name = 'pca_d_10'

    # training set points
    xs_train = [at.info[kpca_name][0] for at in atoms if
                'iter' in at.info['config_type']]
    ys_train = [at.info[kpca_name][1] for at in atoms if
                'iter' in at.info['config_type']]


    optimised_ats = []
    non_optimised_ats = []
    dft_opt_ats = []
    for at in atoms:
        if 'iter' in at.info['config_type']:
            continue
        elif 'non_opt' in at.info['config_type']:
            non_optimised_ats.append(at)
        elif 'opt' in at.info['config_type']:
            optimised_ats.append(at)
        else:
            dft_opt_ats.append(at)

    dft_opt_ats_config_types = [at.info['config_type'] for at in dft_opt_ats]
    print(f'{len(xs_train)} training atoms, {len(non_optimised_ats)} non-optimised atoms, '
          f'{len(optimised_ats)} optimised atoms, {len(dft_opt_ats)} dft minima')
    print(f'DFT minima names:', dft_opt_ats_config_types)



    fig = plt.figure(figsize=(7, 4))

    # training_points
    colors_train_names = [at.info['config_type'] for at in atoms if 'iter' in at.info['config_type']]
    color_mapping_train = {'iter_1': cmap(0.05), 'iter_2': cmap(0.15), 'iter_3': cmap(0.25), 'iter_4': cmap(0.35), \
                       'iter_5': cmap(0.45), 'iter_6': cmap(0.55), 'iter_7':cmap(0.65), 'iter_8':cmap(0.75), 'iter_9':cmap(0.85), 'iter_10':cmap(0.95), \
                       'iter_11': cmap(0.05), 'iter_12': cmap(0.15), 'iter_13': cmap(0.25), 'iter_14': cmap(0.35), \
                       'iter_15': cmap(0.45), 'iter_16': cmap(0.55), 'iter_17': cmap(0.65), 'iter_18': cmap(0.75),\
                       'iter_19': cmap(0.85), 'iter_20': cmap(0.95), 'iter_21':cmap(0.05)}

    colors_train = [color_mapping_train[c] for c in colors_train_names]
    plt.scatter(xs_train, ys_train, color=colors_train, marker='.', label='training points')

    # non_optimised structures
    for idx, at in enumerate(non_optimised_ats):
        label=None
        if idx==0:
            label='to be optimised'
        color = cmap10(color_idx[idx+1])
        if idx==10:
            color='grey'
        plt.scatter(at.info[kpca_name][0], at.info[kpca_name][1], color=color,
                    label=label, marker='o', linewidth=0.5, s=50, linewidths=10, edgecolors='k')
        # print(f'{at.info["config_type"]} color idx: {idx+1}')

    # optimised structures
    for idx, at in enumerate(optimised_ats):
        label = None
        if idx == 1:
            label = 'optimised structures'

        plt.scatter(at.info[kpca_name][0], at.info[kpca_name][1],
                    color=cmap10(color_idx[idx]),
                    label=label, marker='X', linewidth=0.5, s=80,
                    linewidths=10, edgecolors='k')

        # print(f'{at.info["config_type"]} color idx: {idx}')

    # dft structures
    for idx, at in enumerate(dft_opt_ats):
        label=None
        if idx==1:
            label='dft equilibrium'

        plt.scatter(at.info[kpca_name][0], at.info[kpca_name][1],
                    color='k', marker='d', label=label)


        pcax = at.info[kpca_name][0]
        x_pos = pcax + 0.1 * np.abs(pcax)
        pcay = at.info[kpca_name[1]]
        y_pos = pcay + 0.1 * pcay
        plt.gca().annotate(at.info['config_type'], xy=(x_pos, y_pos), fontsize=8)



    plt.legend()
    plt.title(f'kPCA  on GAP_i training sets and GAP_i-optimised structures')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')

    if output_dir is not None:
        pic_name = os.path.join(output_dir, pic_name)

    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(f'{pic_name}.png', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def make_kpca_dset(training_set, all_opt_ats, non_opt_ats, dft_optg, xyz_fname):
    ## main training points
    #  expect to have at.info['config_type'] = 'dset_{i}'
    kpca_ats = read(training_set, ':')
    kpca_ats = [at for at in kpca_ats if len(at) != 1]

    ## all atoms from which dsets were derived. i.e. first guesses and opt_i
    # and expect to have at.info['config_type'] = 'opt_{i}'
    # confusingly opt_{0}
    gap_opt_ats = read(all_opt_ats, ':')
    if 'config_type' not in gap_opt_ats[0].info.keys():
        print('WARNING: gap-optimised atoms do not have config_types')

    # all optimisation starts
    # expect to have at.info['config_type'] = 'non_opt_{i}'
    non_opt_ats = read(non_opt_ats, ':')

    # all the relevant dft minima normally used for geometry optimisation test
    # expect at.info['name'] to specify which one it is
    dft_optg_atoms = []
    if dft_optg is not None:
        dft_optg_atoms = read(dft_optg, ':')
        for at in dft_optg_atoms:
            at.info['config_type'] = at.info['name']

    all_data =  kpca_ats + gap_opt_ats + non_opt_ats + dft_optg_atoms
    write(xyz_fname, all_data, 'extxyz', write_results=False)



def make_kpca_plots(training_set, all_opt_ats='xyzs/opt_all.xyz', non_opt_ats='xyzs/to_opt_all.xyz', \
                    dft_optg=None, xyz_fname='xyzs/for_kpca.xyz',  \
                    output_dir='pictures'):
    # arguments:
    #     training set    last training set that has all the appropriate config_types 'dset_{i}'
    #     all_opt_ats     optimised atoms from all iterations with config types 'opt_{i}'
    #     first_guess     first_guess that was used to get the first dataset
    #     dft_optg        structure optimised with dft
    #     param_fname     one of the gaps' filename with the appropriate command for training gap, optional

    # set up a dataset for kpca
    print('Making up dataset for kpca')
    make_kpca_dset(training_set, all_opt_ats, non_opt_ats, dft_optg, xyz_fname)

    util.do_kpca(xyz_fname)
    pic_name = 'kpca_default'

    kpca_plot(xyz_fname, pic_name, output_dir)



def opt_summary_plots(opt_all='xyzs/opt_all.xyz', dft_optg='molpro_optg/optimized.xyz', gaps_dir='gaps', \
                      output_dir='pictures'):
    dft_optg = read(dft_optg)
    dft_min = dft_optg.info['dft_energy']

    gap_fnames_all = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames_all = util.natural_sort(gap_fnames_all)

    atoms_all = read(opt_all, ':')

    group_size = 10

    for super_idx, (atoms, gap_fnames) in enumerate(zip(util.grouper(atoms_all, group_size), util.grouper(gap_fnames_all, group_size))):

        # remove 'None' padding from grouper
        atoms = [at for at in atoms if at is not None]
        gap_fnames = [gap_fname for gap_fname in gap_fnames if gap_fname is not None]

        dft_energies = [at.info['dft_energy']/len(at) - dft_min/len(at) for at in atoms]
        dft_fmaxs = [max(at.arrays['dft_forces'].flatten()) for at in atoms]

        fig1 = plt.figure(figsize=(12, 8))
        ax1 = plt.gca()

        fig2 = plt.figure(figsize=(12, 8))
        ax2 = plt.gca()

        N = len(atoms)
        cmap = mpl.cm.get_cmap('tab10')
        colors = np.linspace(0, 1, 10)


        for idx, gap_fname in enumerate(gap_fnames):

            absolute_idx = group_size*super_idx + idx

            gap_title = os.path.splitext(gap_fname)[0]
            gap_fname = os.path.join(gaps_dir, gap_fname)
            gap = Potential(param_filename=gap_fname)

            gap_energies_shifted = []
            gap_fmaxes = []
            for aa in atoms:
                at = aa.copy()
                at.set_calculator(gap)
                gap_energies_shifted.append((at.get_potential_energy()-dft_min)/len(at))
                gap_fmaxes.append(max(at.get_forces().flatten()))
            # gap_energies_shifted = [e - dft_min for e in gap_energies]

            c = cmap(colors[idx%10])

            E_label = f'GAP {absolute_idx + 1}'
            F_label = f'GAP {absolute_idx + 1}'


            if idx != 0:
                ax1.plot(range(group_size*super_idx+1, absolute_idx + 2), gap_fmaxes[:idx + 1], marker='x', label=F_label, color=c)
                ax2.plot(range(group_size*super_idx+1, absolute_idx + 2), np.absolute(gap_energies_shifted[:idx + 1]), marker='x', markersize=10, linestyle='-', label=E_label, color=c)

            if idx != N - 1:
                if idx != 0:
                    E_label = None
                    F_label = None


                ax1.plot(range(absolute_idx + 1, group_size*super_idx + len(gap_fmaxes) + 1), gap_fmaxes[idx:], marker='x', label=F_label, linestyle=':', color=c, alpha=0.7)
                ax2.plot(range(absolute_idx + 1, group_size*super_idx + len(gap_energies_shifted) + 1), np.absolute(gap_energies_shifted[idx:]), marker='x',
                         markersize=10, linestyle=':',  label=E_label, color=c, alpha=0.7)

            ax1.annotate(f'{gap_fmaxes[idx]:.4f}', xy=(absolute_idx + 1, gap_fmaxes[idx]))
            ax2.annotate(f'{np.absolute(gap_energies_shifted[idx]):.4f}', xy=(absolute_idx+1, np.absolute(gap_energies_shifted[idx])))

        ax1.plot(range(group_size*super_idx+1, group_size*super_idx + len(dft_fmaxs) + 1), dft_fmaxs, marker='+', markersize=10, label=f'DFT', color='k', linestyle='--')
        ax2.plot(range(group_size*super_idx+1, group_size*super_idx + len(dft_energies) + 1), np.absolute(dft_energies), marker='+', markersize=10, label=f'DFT', color='k', linestyle='--')

        ax1.annotate(f'{dft_fmaxs[idx]:.4f}', xy=(absolute_idx + 1, dft_fmaxs[idx]))
        ax2.annotate(f'{np.absolute(dft_energies[idx]):.4f}', xy=(absolute_idx + 1, np.absolute(dft_energies[idx])))

        for ax in [ax1, ax2]:
            ax.set_xlabel('iteration')
            ax.grid(which='both', c='lightgrey')
            ax.set_yscale('log')
            ax.legend(title='Evaluated with:', loc='upper left')
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        ax2.set_title(f'Energy error wrt DFT-OPTG structure on GAP_i-optimised structures {super_idx+1}')
        ax2.set_ylabel('|E - E$_{DFT\ OPTG}$|, eV/atom', fontsize=12)
        # fig2.tight_layout()

        ax1.set_title(f'Maximum force component on GAP_i-optimised structures {super_idx+1}')
        ax1.set_ylabel('Fmax, eV/Å', fontsize=12)
        # fig1.tight_layout()

    all_fig_nos = plt.get_fignums()
    fmax_fig_nos = all_fig_nos[0::2]
    e_fig_nos = all_fig_nos[1::2]

    for name_prfx, nos in zip(['opt_fmax_vs_iter', 'opt_energy_vs_iter'], [fmax_fig_nos, e_fig_nos]):

        all_ylim_upper = []
        all_ylim_lower = []
        for idx in nos:
            fig = plt.figure(idx)
            ylim = fig.get_axes()[0].get_ylim()
            all_ylim_upper.append(ylim[1])
            all_ylim_lower.append(ylim[0])

        lower_ylim = min(all_ylim_lower)
        upper_ylim = max(all_ylim_upper)

        for i, idx in enumerate(nos):
            fig = plt.figure(idx)
            fig.get_axes()[0].set_ylim(lower_ylim, upper_ylim)
            fig.tight_layout()

            name = f'{name_prfx}_{i+1}.png'
            if output_dir is not None:
                name = os.path.join(output_dir, name)

            plt.savefig(name, dpi=300)
            plt.close(fig)



        # fig2_name = f'opt_energy_vs_iter_{super_idx+1}.png'
        # fig1_name = f'opt_fmax_vs_iter_{super_idx+1}.png'
        # if output_dir:
        #     fig2_name = os.path.join(output_dir, fig2_name)
        #     fig1_name = os.path.join(output_dir, fig1_name)
        # fig2.savefig(fig2_name, dpi=300)
        # fig1.savefig(fig1_name, dpi=300)




def eval_plot(gaps_dir='gaps', first_guess='xyzs/first_guess.xyz', dft_optg='molpro_optg/optimized.xyz',
              dft_vib_name='dft_optg', \
              fmax=1e-3, steps=1000, output_dir='pictures'):


    print('\n---DFT vib modes\n')
    dft_atoms = read(dft_optg)
    vib_dft = Vibrations(dft_atoms, name=dft_vib_name)
    vib_dft.summary()

    gap_fnames_all = [f for f in os.listdir(gaps_dir) if 'gap' in f and 'xml' in f]
    gap_fnames_all = util.natural_sort(gap_fnames_all)

    group_size=10
    # group_size=3

    for super_idx, gap_fnames in enumerate(util.grouper(gap_fnames_all, group_size)):
        # if super_idx == 2:
        #     break

        gap_fnames = [gap_fname for gap_fname in gap_fnames if gap_fname is not None]

        plt.figure(figsize=(8, 5))
        for idx, gap_fname in enumerate(gap_fnames):

            absolute_idx = group_size * super_idx + idx

            gap_title = os.path.splitext(gap_fname)[0]
            gap_fname = os.path.join(gaps_dir, gap_fname)
            gap = Potential(param_filename=gap_fname)
            gap_no = int(re.findall(r'\d+', gap_title)[0])

            gap_optg_name = f'xyzs/{gap_title}_optg_for_NM.xyz'
            # might have done this for eval plots, check
            if not os.path.isfile(f'{gap_title}_optg.all.pckl'):
                if not os.path.isfile(f'xyzs/opt_at_{gap_no}.xyz'):
                    print(f'\n--- optimised structure (xyzs/opt_at_{gap_no}.xyz) not found, Optimising first_guess with {gap_title}\n')
                    atoms = read(first_guess)
                    atoms.set_calculator(gap)

                    # optimize
                    opt = PreconLBFGS(atoms, trajectory=f'xyzs/{gap_title}_optg_for_NM.traj')
                    opt.run(fmax=fmax, steps=steps)
                    write(gap_optg_name, atoms, 'extxyz', write_results=False)

                else:
                    print(f'\n---Loading structure optimised with {gap_title} previously')
                    atoms = read(f'xyzs/opt_at_{gap_no}.xyz')
                    atoms.set_calculator(gap)

                # get NM
                print(f'\n{gap_title} Normal Modes\n')
                vib_gap = Vibrations(atoms, name=f'{gap_title}_optg')
                atoms_gap = vib_gap.run()
                vib_gap.summary()
            else:
                print(f'\n---Found .all.pckl for {gap_title}, loading stuff\n')
                # either optimised now and saved in optg_for_NM.xyz or load previously optimised stuff
                try:
                    atoms = read(f'xyzs/opt_at_{gap_no}.xyz')
                except FileNotFoundError:
                    atoms = read(gap_optg_name)
                vib_gap = Vibrations(atoms, name=f'{gap_title}_optg')
                vib_gap.summary()

            evals = vib_gap.evals
            rmse = util.get_rmse(evals, vib_dft.evals)
            # plt.plot(range(len(evals)), evals, label=f'{gap_title}, RMSE: {rmse:.4f} eV$^2$')
            plt.scatter(vib_dft.evals, evals-vib_dft.evals, marker='x', label=f'{gap_title}, RMSE: {rmse:.4f} eV$^2$')

        # plt.plot(range(len(vib_dft.evals)), vib_dft.evals, label='DFT', linewidth=0.8, linestyle='--', color='k')

        ax = plt.gca()
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        plt.legend(loc='upper left')
        plt.grid(color='lightgrey')
        # plt.xlabel('#')
        # plt.ylabel('eigenvalue, eV$^2$')
        plt.xlabel('DFT eigenvalue, eV$^2$')
        plt.ylabel('GAP_i eval - DFT eval, eV$^2$')
        plt.title('Ordered Eigenvalues')
        plt.tight_layout()


    all_ylim_upper = []
    all_ylim_lower = []
    for idx in plt.get_fignums():
        fig = plt.figure(idx)
        ylim = fig.get_axes()[0].get_ylim()
        all_ylim_upper.append(ylim[1])
        all_ylim_lower.append(ylim[0])


    lower_ylim = min(all_ylim_lower)
    upper_ylim = max(all_ylim_upper)

    for idx in plt.get_fignums():
        fig = plt.figure(idx)
        fig.get_axes()[0].set_ylim(lower_ylim, upper_ylim)

        name = f'eval_plot_{idx}.png'
        if output_dir is not None:
            name = os.path.join(output_dir, name)

        plt.savefig(name, dpi=300)
        plt.close(fig)



def rmsd_plot(opt_all='xyzs/opt_all.xyz', dft_optg='molpro_optg/optimized.xyz', output_dir='pictures'):
    dft_optg_at = read(dft_optg)
    atoms = read(opt_all, ':')
    rmsd = [util.get_rmse(dft_optg_at.positions, at.positions) for at in atoms]

    fig = plt.figure(figsize=(8,5))
    plt.plot(range(1, len(rmsd)+1), rmsd, marker='x', markersize=10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.xlabel('Iteration')
    plt.ylabel('RMSD, Å')
    plt.grid(color='lightgrey', which='both')
    plt.title('Geometry optimisation, GAP i vs DFT')
    plt.yscale('log')
    plt.tight_layout()
    name = 'rmsd.png'
    if output_dir is not None:
        name = os.path.join(output_dir, name)
    plt.savefig(name, dpi=300)
    plt.close(fig)



