import os
from ase import neighborlist
import numpy as np
import warnings
import yaml


def assign_aromatic(inputs, outputs, elements_to_type, mult=1):

    fake_elements = {'C':'Ca', 'H':'He', 'O':'Os', 'Os':'Os'}
    fake_radii = {fake_elements['C']:0.76,
                  fake_elements['H']:0.31,
                  fake_elements['O']:0.66}


    bad_configs = []
    for idx, atoms in enumerate(inputs):

        if len(atoms) == 1:
            continue

        natural_cutoffs = neighborlist.natural_cutoffs(atoms,
                                                       mult=mult,
                                                       **fake_radii)

        neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                                  self_interaction=False,
                                                  bothways=True)
        _ = neighbor_list.update(atoms)


        symbols = np.array(atoms.symbols)
        for at in atoms:
            c_idx = None

            if 'H' in elements_to_type:
                if at.symbol == 'H':
                    h_idx = at.index

                    indices, offsets = neighbor_list.get_neighbors(h_idx)
                    if len(indices) != 1:
                        bad_configs.append(idx)
                        warnings.warn("Got more than one hydrogen neighbor")
                        continue

                    # find index of the atom H is bound to
                    h_neighbor_idx = indices[0]

                    if symbols[h_neighbor_idx] not in ['C', fake_elements['C']]:
                        continue

                    c_idx = [h_neighbor_idx]

            if 'C' in elements_to_type:
                if at.symbol == 'C':
                    c_idx = [at.index]

            if 'O' in elements_to_type:
                if at.symbol == 'O':
                    o_idx = at.index
                    indices, offsets = neighbor_list.get_neighbors(o_idx)
                    if len(indices) not in [1, 2]:
                        bad_configs.append(idx)
                        warnings.warn(f"Number of O neighbours is {len(indices)}")
                        continue

                    # find indices of carbon atoms O is bound to
                    c_idx = []
                    for neighbour_idx in indices:
                        if atoms[neighbour_idx].symbol in ['C', fake_elements['C']]:
                            c_idx.append(neighbour_idx)


            if c_idx is not None:
                for c_id in c_idx:

                    indices, offsets = neighbor_list.get_neighbors(c_id)

                    no_C_neighbours = len(indices)
                    if no_C_neighbours == 4:
                        continue

                    elif no_C_neighbours == 3:
                        at.symbol = fake_elements[at.symbol]
                        if at.symbol == fake_elements['O']:
                            continue


                    else:
                        print(at)
                        print(atoms.info)
                        print(f'conifg index {idx}')
                        warnings.warn(f'Got carbon with weird number of '
                                           f'neighbours: {no_C_neighbours}')
                        bad_configs.append(idx)
                        continue

        # atoms.arrays['atom_types'] = symbols
        outputs.write(atoms)

    print(f'bad_config idx ({len(bad_configs)}):', bad_configs)
    outputs.end_write()

def prepare_empty_atom_type_lookup_yamls(
        atom_type_dir='atom_type_lookup_fresh'):

    all_atoms = read('original_datasets.xyz', ':')

    config_type_numbers = {}
    for idx, at in enumerate(all_atoms):
        config_type = at.info['config_type']
        numbers = at.numbers
        if config_type not in config_type_numbers.keys():
            config_type_numbers[config_type] = numbers
        else:
            if not np.all(config_type_numbers[config_type] == numbers):
                raise RuntimeError('some configs have their atoms reordered')

    for config_type, numbers in config_type_numbers.items():

        # data = {config_type: yaml_data(numbers)}
        string = f'{config_type}:\n' + yaml_string(numbers)
        with open(os.path.join(atom_type_dir, config_type + '.yml'), 'w') as \
                file:
            file.write(string)

def yaml_string(numbers):
    string = ''
    for number in numbers:
        string += f'- {number}: {number}\n'
    return string


def yaml_data(numbers):
    yaml_list = []
    for number in numbers:
        yaml_list.append({str(number) : str(number)})
    return yaml_list

def atom_type(at,
              atom_type_dir='/home/eg475/scripts/source_files/atom_types'):

    config_type = at.info['config_type']
    orig_numbers, atom_typed_numbers = numbers_from_yaml(os.path.join(
        atom_type_dir, config_type+'.yml'), config_type)

    at.arrays['original_numbers'] = orig_numbers
    at.arrays['atom_typed_numbers'] = atom_typed_numbers
    at.numbers = atom_typed_numbers

    return at

def numbers_from_yaml(filename, config_type):
    with open(str(filename)) as yaml_file:
        data = yaml.safe_load(yaml_file)

    try:
        data = data[config_type]
    except KeyError:
        print(data.keys())
        raise

    orig_numbers = []
    atom_typed_numbers = []
    for num_pair in data:
        for key, val in num_pair.items():
            orig_numbers.append(key)
            atom_typed_numbers.append(val)

    return np.array(orig_numbers), np.array(atom_typed_numbers)








