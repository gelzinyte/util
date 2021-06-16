import re
import os
import numpy as np
from ase.io import read, write
from wfl.calculators import orca
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


def read_orca_output(input_xyz, orca_label):

    at = read(input_xyz)
    calc = orca.ExtendedORCA()
    calc.label=orca_label
    calc.read_energy()
    at.info['dft_energy'] = calc.results['energy']
    try:
        calc.read_forces()
        at.arrays['dft_forces'] = calc.results['forces']
    except FileNotFoundError:
        pass

    return at

def orca_scf_plot(input_fname, fname='orca_scf_convergence.png'):

    with open(input_fname, 'r') as f:
        text = f.read()

    scf_column_names = ['Energy', 'Delta-E', 'Max-DP',
                        'RMS-DP', '[F, P]', 'Damp']

    pattern_scf_cycles = re.compile(
        # r"(?:SCF ITERATIONS\n"
        r"(?:SCF ITERATIONS\n"
        r"[-]+\n"
        r"ITER.*\n"
        r"(?:[\s*]+Starting incremental Fock matrix formation[\s*]+\n)?"
        r"([-\s\d.]+)"
        r"(?:[\s*]+Turning on DIIS[\s*]+\n)?"
        r"([-\s\d.]+)"
        r"[\s*]+Energy Check signals convergence[\s*]+\n"
        r")"
    )

    pattern_scf_step = re.compile(
        r"(?:\s*(\d+))\s+"     # cycle no
        r"([-0-9.]*)\s+"   # energy
        r"([-0-9.]*)\s+"   # Delta-E
        r"([-0-9.]*)\s+"   # Max-DP
        r"([-0-9.]*)\s+"   # RMS-DP
        r"([-0-9.]*)\s+"   # [F, P]
        r"([-0-9.]*)"    # Damp
    )

    scf_match = pattern_scf_cycles.search(text)

    if scf_match:

        scf_data = pd.DataFrame(columns=scf_column_names)

        for values_block in scf_match.groups():
            for values_line in values_block.split('\n'):

                step_match = pattern_scf_step.search(values_line)
                if step_match:
                    idx = int(step_match.groups()[0])
                    vals = np.array([float(num) for num in
                                    step_match.groups()[1:]])

                    scf_data.loc[idx] = vals

    else:
        raise RuntimeError('no SCF blocks found')

    fig = plt.figure(figsize=(10, 5))
    gs = mpl.gridspec.GridSpec(ncols=2, nrows=1)
    eax = fig.add_subplot(gs[0])
    conv_ax = fig.add_subplot(gs[1])
    scf_data.reset_index().plot(ax=eax, x='index', y=scf_column_names[0])
    scf_data.reset_index().plot(ax=conv_ax, x='index',
                                y=scf_column_names[1:], logy=True)

    for ax in [eax, conv_ax]:
        ax.grid(color='lightgrey', linestyle=':')
        ax.set_xlabel('SCF step')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    eax.title(input_fname)
    plt.savefig(fname, dpi=300)




