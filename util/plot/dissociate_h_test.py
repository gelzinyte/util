import matplotlib.pyplot as plt
from util import shift0 as sft
import numpy as np

def curves_from_all_atoms(ats_in, pred_prefix, rin=None, rout=None):

    isolated_at = [at for at in ats_in if len(at) == 1][0]

    data = {}
    for at in ats_in:
        if len(at) == 1:
            continue
        comp = at.info["compound"]
        if comp not in data.keys():
            data[comp] = {'diss_mol':[], 'iso_h':isolated_at, 'rad':None}
        if at.info["dissociation_test_config_type" ] == "mol":
            data[comp]['diss_mol'].append(at)
            continue
        elif at.info["dissociation_test_config_type"] == "rad":
            data[comp]['rad'] = at
            continue
        else:
            print(f"didn't assign {at.info['dissociation_test_config_type']}")


    for id, vals in data.items():
        print(id)
        c_idx = vals['rad'].info["dissociation_test_c_idx"]
        h_idx = vals['rad'].info["dissociation_test_h_idx"]
        title = f'{pred_prefix}dissociation_curve_{id}_C{c_idx}_H{h_idx}'
        plot_curve(diss_mol = vals["diss_mol"], iso_h = vals['iso_h'], rad=vals['rad'],
                   pred_prefix=pred_prefix, title=title, rin=rin, rout=rout)



def plot_curve(diss_mol, iso_h, rad, pred_prefix, title, rin=None, rout=None,
               bh_prefix='dft_hop_', dft_prefix='dft_'):

    c_idx = rad.info["dissociation_test_c_idx"]
    h_idx = rad.info["dissociation_test_h_idx"]

    distances = [at.get_distance(c_idx, h_idx) for at in diss_mol]
    pred_energies = [at.info[f'{pred_prefix}energy'] for at in diss_mol]

    dft_ref = iso_h.info[f'{dft_prefix}energy'] + np.min([rad.info[f'{bh_prefix}{idx}_energy'] for idx in range(10)])
    pred_ref = iso_h.info[f'{pred_prefix}energy'] + rad.info[f'{pred_prefix}energy']

    plt.figure(figsize=(6, 4))

    plt.plot(distances, sft(pred_energies, dft_ref))
    plt.axhline(dft_ref - dft_ref, lw=0.8, color='k', label=f"dft bhop reference: {dft_ref:.3f} eV")
    shifted_pred = pred_ref - dft_ref
    plt.axhline(shifted_pred, lw=0.8, color='tab:orange', ls='--',
                label=f'{pred_prefix} reference: {shifted_pred*1e3:.3f} meV or {shifted_pred*1e3/(len(rad)+1):.3f} meV/at')

    if rin is not None:
        plt.axvline(rin, color='k', lw=0.8, ls='--', label=f'rin: {rin:2f} Å')
    if rout is not None:
        plt.axvline(rout, color='k', lw=0.8, ls='--', label=f'rout: {rout:2f} Å')

    for at_idx, (dist, at)  in enumerate(zip(distances, diss_mol)):
        d = [dist] * 10
        e = [at.info[f'{bh_prefix}{idx}_energy'] for idx in range(10)]
        label = f'10 basin hops: {(min(e) - dft_ref)*1e3:.3f} meV' if at_idx == len(distances ) - 1 else None
        plt.scatter(d, sft(e, dft_ref), label=label, color='k', marker='x')

    plt.grid(color='lightgrey')
    plt.xlabel("C-H separation")
    plt.ylabel(f"Shifted energy, eV (w.r.t. {dft_ref:.2f})")
    plt.legend(title="(energy - reference) at max separation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title.replace(' ', '_')+'.pdf')

