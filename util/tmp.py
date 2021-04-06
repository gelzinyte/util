from ase.io import read, write


def process_config_info(fname_in, fname_out):

    ats = read(fname_in, ':')

    all_mol_or_rad_entries = []
    all_compound_entries = []

    for at in ats:

        cfg = at.info['config_type']
        words = cfg.split('_')

        mol_or_rad = words[-1]

        if 'mol' not in mol_or_rad and 'rad' not in mol_or_rad:
            raise RuntimeError(f'{mol_or_rad} isn\'t eiter molecule or radical')

        all_mol_or_rad_entries.append(mol_or_rad)
        at.info['mol_or_rad'] = mol_or_rad

        compound = '-'.join(words[:-1])
        all_compound_entries.append(compound)
        at.info['compound'] = compound

    write(fname_out, ats)
    print(f'all mol_or_rad entries: {set(all_mol_or_rad_entries)}')
    print(f' all compound entries: {set(all_compound_entries)}')


        


