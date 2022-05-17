import re
import os
import numpy as np
from ase.io import read, write
from wfl.calculators import orca
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from ase.io.orca import read_geom_orcainp
from matplotlib import cm


def read_orca_output(orca_label, input_xyz, prop_prefix="dft_"):

    if input_xyz:
        at = read(input_xyz)
    else:
        at = read_geom_orcainp(orca_label + '.inp')
    calc = orca.ORCA()
    calc.label=orca_label
    calc.read_energy()
    at.info[f'{prop_prefix}energy'] = calc.results['energy']
    try:
        calc.read_forces()
        at.arrays[f'{prop_prefix}forces'] = calc.results['forces']
    except FileNotFoundError:
        pass

    return at

def dft_patterns():

    pattern_dft_cycles = re.compile(
        # r"(?:SCF ITERATIONS\n"
        r"(?:SCF ITERATIONS\n"
        r"[-]+\n"
        r"ITER.*\n"
        r"(?:[\s*]+Starting incremental Fock matrix formation[\s*]+\n)?"
        r"([-\s\d.]+)"
        r"(?:[\s*]+Turning on DIIS[\s*]+\n)?"
        r"([-\s\d.]+)"
        r"(?:[\s*]+ Restarting incremental Fock matrix formation [\s*]+\n)?"
        r"([-\s\d.+)?"
        r"(?:[\s*]+ Resetting DIIS [\s*]+\n)?"
        r"([-\s\d.]+)"
        r"(?:[\s*]+Energy Check signals convergence[\s*]+\n)?"
        r"(?:[\s*]+DIIS convergence achieved[\s*]+\n)?"
        r")"
    )


    pattern_dft_step = re.compile(
        r"(?:\s*(\d+))\s+"  # cycle no
        r"([-0-9.]*)\s+"  # energy
        r"([-0-9.]*)\s+"  # Delta-E
        r"([-0-9.]*)\s+"  # Max-DP
        r"([-0-9.]*)\s+"  # RMS-DP
        r"([-0-9.]*)\s+"  # [F, P]
        r"([-0-9.]*)"  # Damp
    )

    scf_column_names = ['Energy', 'Delta-E', 'Max-DP',
                        'RMS-DP', '[F, P]', 'Damp']

    return pattern_dft_cycles, pattern_dft_step, scf_column_names

def cc_patterns():

    pattern_cycles = re.compile(
         r"(RHF COUPLED CLUSTER ITERATIONS\n"
         r"[-]+\n\n"
         r"Number of amplitudes to be optimized\s+...\s*\d+\n\n"
         r"Iter.*\n"
        r"([-\s\d.]+)"
        r"(?:[\s*]+Turning on DIIS[\s*]+\n)?"
        r"([-\s\d.]+)"
        r"(?:[-\s*]+The Coupled-Cluster iterations have NOT converged[\s*-]+\n)?"
        r")"
    )


    pattern_step = re.compile(
        r"(?:\s*(\d+))\s+"  # cycle no
        r"([-0-9.]*)\s+"  # E(tot) 
        r"([-0-9.]*)\s+"  # E(Corr) 
        r"([-0-9.]*)\s+"  # Delta-E
        r"([-0-9.]*)\s+"  #Residual 
        r"(?:[-0-9.]*)"  # Time
    )

    column_names = ['E(tot)', 'E(Corr)', 'Delta-E', 'Residual']

    return pattern_cycles, pattern_step, column_names

def orca_scf_plot(input_fname, method='dft',  
    fname='orca_scf_convergence.png'):

    with open(input_fname, 'r') as f:
        text = f.read()

    if method == 'dft':
        pattern_cycle, pattern_step, scf_column_names = dft_patterns()
    elif method == 'cc':
        pattern_cycle, pattern_step, scf_column_names = cc_patterns()

    scf_match = pattern_cycle.search(text)


    if scf_match:

        scf_data = pd.DataFrame(columns=scf_column_names)

        for values_block in scf_match.groups():

            for values_line in values_block.split('\n'):

                step_match = pattern_step.search(values_line)
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

    eax.set_title(input_fname)
    plt.savefig(fname, dpi=300)

def print_populations(orca_out, xyz_in, orca_in, xyz_out, pop, cmap):
    # TODO make reading xyz from orca outptput file
    '''  NA   - Mulliken gross atomic population
  ZA   - Total nuclear charge
  QA   - Mulliken gross atomic charge
  VA   - Mayer's total valence
  BVA  - Mayer's bonded valence
  FA   - Mayer's free valence
'''

    if xyz_in:
        ats = read(xyz_in)
    elif orca_in:
        ats = read_geom_orcainp(orca_in)

    pop_data = get_pop_dict(orca_out)

    if not (np.array(list(ats.symbols)) == np.array(pop_data['elements'])).all():
        raise RuntimeError("atoms and populations' elements don't match")
    del pop_data['elements']

    for key, val in pop_data.items():
        ats.arrays[key] = np.array(val)

    ats = color_by_pop(ats, pop, cmap)


    write(xyz_out, ats)

def color_by_pop(ats, pop, cmap, vmin=-0.5, vmax=0.5):
    values = ats.arrays[pop]

    # norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values), clip=True)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap))

    colors = np.array([mapper.to_rgba(v)[0:3] for v in values])

    ats.arrays['Color'] = colors
    return ats



def get_pop_dict(in_fname):
    '''  NA   - Mulliken gross atomic population
  ZA   - Total nuclear charge
  QA   - Mulliken gross atomic charge
  VA   - Mayer's total valence
  BVA  - Mayer's bonded valence
  FA   - Mayer's free valence
'''

    d = {'elements':[], 'NA':[], 'QA':[], 'VA':[], 'BVA':[], 'FA':[]}
    # kw_to_idx = {'NA': 2, 'QA':4, 'VA':5, 'BVA':6, 'FA':7}

    with open(in_fname) as f:
        output = f.read().splitlines()

    for idx, line in enumerate(output):
        if 'MAYER POPULATION ANALYSIS' in line:
            start = idx + 11
        if 'Mayer bond orders larger than' in line:
            end = idx - 2
            break


    for line in output[start : end +1]:
        l = line.split()
        d['elements'].append(l[1])
        d['NA'].append(float(l[2]))
        d['QA'].append(float(l[4]))
        d['VA'].append(float(l[5]))
        d['BVA'].append(float(l[6]))
        d['FA'].append(float(l[7]))

    return d



