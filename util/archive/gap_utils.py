def sigma_check(descriptors, train_fname, esigmas, fsigmas, gaps_dir='gaps/sigma_test'):
    descriptors = deepcopy(descriptors)
    train_ats = read(train_fname, ':')

    ermses = {}
    frmses = {}

    if not os.path.exists(gaps_dir):
        os.makedirs(gaps_dir)

    for esig in esigmas:
        print(f'-energy sigma: {esig}')
        ermses[f'esig={esig}'] = {}
        frmses[f'esig={esig}'] = {}

        for fsig in fsigmas:
            print(f'--forces sigma: {fsig}')
            default_sigma = [esig, fsig, 0.0, 0.0]
            gap_fname = os.path.join(gaps_dir, f'gap_{esig}_{fsig}.xml')
            out_fname = os.path.join(gaps_dir, f'out_{esig}_{fsig}.txt')

            command = make_gap_command(gap_filename=gap_fname, training_filename=train_fname,
                                       descriptors_dict=descriptors, default_sigma=default_sigma,
                                       output_filename=out_fname, glue_fname='glue_orca.xml')

            print('command:\n', command)
            out = subprocess.run(command, shell=True)

            dft_data = util.get_E_F_dict(train_ats, calc_type='dft')
            gap_data = util.get_E_F_dict(train_ats, calc_type='gap', param_fname=gap_fname)

            dft_es = util.dict_to_vals(dft_data['energy'])
            gap_es = util.dict_to_vals(gap_data['energy'])
            ermses[f'esig={esig}'][f'fsig={fsig}'] = util.get_rmse(dft_es, gap_es)

            dft_fs = util.dict_to_vals(util.desymbolise_force_dict(dft_data['forces']))
            gap_fs = util.dict_to_vals(util.desymbolise_force_dict(gap_data['forces']))
            frmses[f'esig={esig}'][f'fsig={fsig}'] = util.get_rmse(dft_fs, gap_fs)

    fig = plt.figure(figsize=(14, 7))
    gs = mpl.gridspec.GridSpec(1, 2)
    all_ax = [plt.subplot(g) for g in gs]

    plot.plot_heatmap(ermses, all_ax[0], 'Energy')
    plot.plot_heatmap(frmses, all_ax[1], 'Force')
    plt.tight_layout()
    plt.show()

