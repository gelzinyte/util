import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms, Atom
from ase.io import read, write
try:
    from quippy.potential import Potential
except ModuleNotFoundError:
    pass

from util.plot import quick_dimer
from util import configs



def do_dimers(pred_calc, species, out_prefix, pred_prop_prefix, ref_isolated_ats=None, isolated_at_prop_prefix='dft_'):


    isolated_ats = parse_isolated_ats_e(pred_calc, ref_isolated_ats, pred_prop_prefix, out_prefix)

    if isolated_at_prop_prefix != pred_prop_prefix:
        for key, at in isolated_ats:
            error = (at.info[f'{pred_prop_prefix}energy'] - at.info[f'{isolated_at_prop_prefix}']) * 1e3
            print(f'{key} error: {error:.3f} meV')
    
    dimer_ats = make_dimer_ats(species) 

    get_energies(dimer_ats, pred_calc, pred_prop_prefix, out_prefix)

    quick_dimer.plot_dimers(
        dimer_ats=dimer_ats, 
        isolated_ats=isolated_ats,
        pred_prop_prefix=pred_prop_prefix,
        output_fn=out_prefix,
        isolated_at_prop_prefix=isolated_at_prop_prefix)



def get_energies(dimer_ats, pred_calc, pred_prop_prefix, out_prefix):

    for ats in dimer_ats: 
        # output_ats = []
        dimer_sym = ''.join(list(ats[0].symbols))
        output_fname = out_prefix + f'.{dimer_sym}.xyz'
        for at in ats:
            at.calc = pred_calc
            at.info[f'{pred_prop_prefix}energy'] = at.get_potential_energy()
            at.arrays[f'{pred_prop_prefix}forces'] = at.get_forces()
            # output_ats.append(at)
        write(output_fname, ats)


def make_dimer_ats(species):

    species.sort()
    dimer_symbols = []
    for sp1 in species:
        for sp2 in species:
            if f'{sp2}{sp1}' in dimer_symbols:
                continue
            dimer_symbols.append(f'{sp1}{sp2}')

    distances = np.concatenate([np.arange(0.1, 0.5, 0.05), np.arange(0.5, 1.0, 0.02), np.arange(1.0, 6.1, 0.05)])
    dimer_ats = []
    for dimer in dimer_symbols:
        dimer_ats[dimer] = []
        for d in distances:
            at = Atoms(dimer, positions=[(0, 0, 0), (0, 0, d)])
            dimer_ats.append(at)

    return dimer_ats

def parse_isolated_ats_e(pred_calc, ref_isolated_ats, pred_prop_prefix, out_prefix):
    for at in ref_isolated_ats:
        assert len(at) == 1
        at.info["at_symbol"] = list(at.symbols)[0]
        at.calc = pred_calc
        e = at.get_potential_energy()
        at.info[f'{pred_prop_prefix}energy'] = e

    write(out_prefix + '.isolated_at.xyz', ref_isolated_ats)

    return configs.into_dict_of_labels(ref_isolated_ats, info_label='at_symbol') 
        
