from ase.io import read, write
import numpy as np
from ase import Atoms
from wfl.generate_configs import vib

v = vib.Vibrations



def convert_all(fname_in, fname_out, prop_prefix=None, info_to_keep=None,
                arrays_to_keep=None):

    ats_in = read(fname_in, ':')
    ats_out = []

    for at in ats_in:
        new_at =  convert_one(at, prop_prefix, info_to_keep, arrays_to_keep)
        ats_out.append(new_at)

    write(fname_out, ats_out)

def convert_one(at, prefix='', info_to_keep=None, arrays_to_keep=None):

    evals = at.info[f'{prefix}nm_eigenvalues']
    N_nm = len(at) * 3
    evecs = np.zeros((N_nm, N_nm))
    for idx in range(N_nm):
        evecs[idx] = at.arrays[f'{prefix}evec{idx}'].reshape(N_nm)

    at_new = Atoms(at.symbols, at.positions)

    if info_to_keep is not None:
        for key in info_to_keep.split():
            at_new.info[key] = at.info[key]

    if arrays_to_keep is not None:
        for key in arrays_to_keep.split():
            at_new.arrays[key] = at.arrays[key]

    masses = at.get_masses()

    frequencies = v.evals_to_freqs(evals)
    modes = v.evecs_to_modes(evecs, masses=masses)

    at_new.info[f'{prefix}normal_mode_frequencies'] = frequencies
    for idx, mode in enumerate(modes):
        at_new.arrays[f'{prefix}normal_mode_displacements_{idx}'] = mode

    return at_new






