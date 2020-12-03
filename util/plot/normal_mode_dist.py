from scipy import stats
from util.vibrations import Vibrations
from ase.io import read, write
import numpy as np
import math
from ase.units import kB
import matplotlib.pyplot as plt
import os
from ase.visualize import view
import seaborn as sns
from ase import units
import util
import importlib

def NM_distributions(dft_at, T1, T2, dft_out_fnames):

    vib = Vibrations(dft_at, name=dft_at.info['name'])

    n_at = 500
#     nms = np.arange(6, 3 * len(dft_at))
    nms = 'all'

    ensall100, NMsall100 = vib.multi_at_nm_displace(temp=T1, n_samples=n_at, nms=nms, return_harm_e=True)
    ensall400, NMsall400 = vib.multi_at_nm_displace(temp=T2, n_samples=n_at, nms=nms, return_harm_e=True)
    #


#     print(f'Mean_NM_energy/kB*T at {T1}K: { np.mean(ensall100)/(kB * T1)}')
#     print(f'Mean_NM_energy/kB*T at {T2}K: {np.mean(ensall400) /(kB * T2)}')

    dft_eq_e = dft_at.info['energy']
    up_x_lim = max(ensall100 + ensall400)
    n_bins = 200
    bins = np.linspace(0,up_x_lim, n_bins)
    temps = [T1, T2]
    alpha = 0.3
    density = True
    xs = np.linspace(0.0001, up_x_lim, 1000)


    plt.figure(figsize=(12,8))
    ax1 = plt.gca()



    kwargs = [
              {'label':f'DFT {T1}K', 'color':'tab:blue', 'histtype':'step'},
              {'label':f'DFT {T2}K', 'color':'tab:red', 'histtype':'step'}
             ]
    for out_fname, kw in zip(dft_out_fnames, kwargs):
        atoms = read(out_fname, ':')
        print(len(atoms))
        try:
            es = [at.info['energy'] - dft_eq_e for at in atoms]
        except KeyError:
            es = [at.info['dft_energy'] - dft_eq_e for at in atoms]
        ax1.hist(es, bins=bins, density=density, **kw)
#         ax1.hist(es, density=density, **kw)



    nm_kwargs = [
        {'label':f'Normal displacements {T1} K, {nms} NMs', 'color':'tab:blue', 'alpha':alpha},
        {'label':f'Normal displacements {T2} K, {nms} NMs', 'color':'tab:red', 'alpha':alpha}
             ]
    energies = [ensall100, ensall400]
    for es, kw in zip(energies, nm_kwargs):
        ax1.hist(es, bins=bins, density=density, **kw)



    tkwargs = [{'ls':'-', 'c':'b'},
              {'ls':'-', 'c':'r'}]

    for T, kw in zip(temps, tkwargs):
        a = 1/2 * (len(at) * 3 - 6)
        pdf = stats.gamma.pdf(xs, a=a, scale=kB*T)
        ax1.plot(xs, pdf, label=f'gamma(E, a={a}) T={T}', **kw)


    ax1.legend()


    ax1.set_xlabel('NM energy / DFT energy from equilibrium, eV ')
    ax1.set_ylabel('density')
    plt.title(dft_at.info["name"])
    plt.show()