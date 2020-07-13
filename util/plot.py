



def get_E_F_dict(atoms, calc_type, param_fname=None):

    data = dict()
    data['energy'] = OrderedDict()
    data['forces'] = OrderedDict()

    # select which energies and forces to extract
    if calc_type.upper() == 'GAP':
        if param_fname:
            gap = Potential(param_filename=param_fname)
        else:
            raise NameError('GAP filename is not given, but GAP energies requested.')

    else:
        if param_fname:
            print(f"WARNING: calc_type selected as {calc_type}, but gap filename is given, are you sure?")
        energy_name = f'{calc_type}_energy'
        if energy_name not in atoms[0].info.keys():
            print(f"WARNING: '{calc_type}_energy' not found, using 'energy', which might be anything")
            energy_name = 'energy'
        force_name = f'{calc_type}_forces'
        if force_name not in atoms[0].arrays.keys():
            print(f"WARNING: '{calc_type}_forces' not in found, using 'forces', which might be anything")
            force_name = 'forces'


    for atom in atoms:
        at = atom.copy()
        config_type='no_config_type'
        if 'config_type' in at.info.keys():
            config_type = at.info['config_type']


        if len(at) != 1:
            if calc_type.upper() == 'GAP':
                at.set_calculator(gap)
                energy = at.get_potential_energy()
                try:
                    data['energy'][config_type].append(energy)
                except KeyError:
                    data['energy'][config_type] = []
                    data['energy'][config_type].append(energy)

                forces = at.get_forces()
            else:
                try:
                    data['energy'][config_type].append(at.info[energy_name])
                except KeyError:
                    data['energy'][config_type] = []
                    data['energy'][config_type].append(at.info[energy_name])

                forces = at.arrays[force_name]


            sym_all = at.get_chemical_symbols()
            for j, sym in enumerate(sym_all):
                if sym not in data['forces'].keys():
                    data['forces'][sym] = OrderedDict()
                try:
                    data['forces'][sym][config_type].append(forces[j])
                except KeyError:
                    data['forces'][sym][config_type] = []
                    data['forces'][sym][config_type].append(forces[j])

    # TODO make it append np array in the loop
    for config_type, values in data['energy'].items():
        data['energy'][config_type] = np.array(values)
    for sym in data['forces'].keys():
        for config_type, values in data['forces'][sym].items():
            data['forces'][sym][config_type] = np.array(values)


    return data





