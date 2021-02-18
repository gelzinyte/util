from wfl.generate_configs import vib
from ase.units import invcm
from matplotlib import gridspec
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def multi_evec_plot(ref_atoms, atoms2, prefix1, prefix2, labels2, suptitle):

    cmap = 'Reds'
    fontsize =14

    N = len(atoms2)

    plt.figure(figsize=( 6 *N, 5))

    grid = gridspec.GridSpec(1, N)

    for idx, (gs, at2, label2) in enumerate(zip(grid, atoms2, labels2)):

        ax = plt.subplot(gs)

        vib1 = vib.Vibrations(at2, prefix2) # x axis
        vib2 = vib.Vibrations(ref_atoms, prefix1) # y axis

        assert np.all(vib1.atoms.symbols == vib2.atoms.symbols)

        dot_products = dict()
        for i, ev1 in enumerate(vib1.evecs):
            dot_products[f'nm1_{i}'] = dict()
            for j, ev2 in enumerate(vib2.evecs):
                dot_products[f'nm1_{i}'][f'nm2_{j}'] = np.abs \
                    (np.dot(ev1, ev2))

        df = pd.DataFrame(dot_products)


        _ = ax.pcolormesh(df, vmin=0, vmax=1, cmap=cmap,
                             edgecolors='lightgrey', linewidth=0.01)

        if idx == 0:
            ax.set_ylabel('DFT modes', fontsize=fontsize)
        ax.set_xlabel(f'{label2} modes', fontsize=fontsize)

    plt.suptitle(suptitle, fontsize=fontsize+2)


