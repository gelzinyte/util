import click
from ase.io import read, write
import random



def sample_configs(input_fn, output_fname, n_configs, sample_type, include_config_type, exclude_config_type):
    """takes a number of structures evenly from each config"""

    assert sample_type in ['total', 'per_config', 'all']
    assert include_config_type is None or exclude_config_type is None

    if include_config_type is not None:
        include_config_type = include_config_type.split()
    if exclude_config_type is not None:
        exclude_config_type = exclude_config_type.split()


    atoms_in = read(input_fn, ':')
    at_by_config = atoms_to_dict(atoms_in)

    print(f'total number of config_types: {len(at_by_config.keys())}')

    n_config_types = len(at_by_config.keys())

    if sample_type == 'all':

        ats_out = []
        if include_config_type is not None:
            for config_type in include_config_type:
                ats_out += at_by_config[config_type]
            counter = len(include_config_type)

        if exclude_config_type is not None:
            print(f'no of excluded config_types: {len(exclude_config_type)}')
            for config_type in exclude_config_type:
                del at_by_config[config_type]

            for at_list in at_by_config.values():
                ats_out += at_list
            counter = len(at_by_config.keys())

        print(f'number of included config_types: {counter}')
        write(output_fname, ats_out)
        return


    elif sample_type == 'per_config':
        config_sample_size = n_configs
    else:
        # sample_type == total
        config_sample_size = int(n_configs/n_config_types)


    print(f'No structures per config_type: {config_sample_size}')


    atoms_out = []
    for atoms_list in at_by_config.values():
        atoms_out += random.sample(atoms_list, config_sample_size)

    print(f'Dataset size: {len(atoms_out)}')

    write(output_fname, atoms_out)



def atoms_to_dict(atoms_in):

    atoms_dict = {}
    for at in atoms_in:

        if 'config_type' in at.info.keys():
            config_type = at.info['config_type']
        else:
            config_type = 'no_config_type'

        if config_type not in atoms_dict.keys():
            atoms_dict[config_type] = []


        atoms_dict[config_type].append(at)

    return atoms_dict

