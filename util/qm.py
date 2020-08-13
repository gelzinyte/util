
from ase.calculators.orca import ORCA
from ase import Atom, Atoms
from ase.io import read, write
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import re


def plot_curve(dimer_name, multiplicities, orca_blocks, labels, distances, iso_at_mult):
    isolated_ats = isolated_at_data(dimer_name, multiplicities=iso_at_mult)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    min_e = 0

    for idx, (label, mult, block) in enumerate(
            zip(labels, multiplicities, orca_blocks)):
        dimer_ats = dimer_data(dimer_name, mult=mult, orca_block=block,
                               idx=idx, distances=distances)
        distances = [at.get_distance(0, 1) for at in dimer_ats]
        dft_energies = [at.info[f'dft_energy'] for at in dimer_ats]
        if min(dft_energies) < min_e:
            min_e = min(dft_energies)

        marker = '.'
        if idx % 2 == 0:
            marker = 'x'

        linestyle='-'
        if idx % 3 == 0:
            linestyle = '--'
        elif idx % 3 == 1:
            linestyle = '-.'

        plt.plot(distances, dft_energies, linewidth=1.5, marker=marker, alpha=0.5,  linestyle=linestyle,
                 label=f'{label}:{dft_energies[-1]:.2f}')

    iso_ats_es = [at.info['dft_energy'] for at in isolated_ats]
    hline_label = f'{dimer_name[0]} ({iso_ats_es[0]:.1f}) + {dimer_name[1]} ' \
                  f'({iso_ats_es[1]:.1f}) = {sum(iso_ats_es):.1f}'
    ax.axhline(y=sum(iso_ats_es), linestyle='-.', label=hline_label,
               color='k', linewidth=0.8)

    ax.set_xlabel('distance, Ang')
    ax.set_ylabel('energy, eV')
    ax.grid(color='lightgrey')
    plt.legend()
    plt.title(dimer_name)
    plt.tight_layout()
    ax.set_ylim(top=sum(iso_ats_es)+10,  bottom=min_e-5)
    plt.savefig(f'{dimer_name}_dissociation_curve.png', dpi=300)
    plt.show()


def isolated_at_data(dimer_name, multiplicities):
    _orca_command = '/home/eg475/programs/orca/orca_4_2_1_linux_x86' \
                    '-64_openmpi314/orca'

    isolated_fname = f'xyzs/isolated_{dimer_name}.xyz'
    if not os.path.isfile(isolated_fname):
        isolated_names = list(dimer_name)
        isolated_ats = []
        print('getting isolated atom energies')
        for mult, name in zip(multiplicities, isolated_names):

            at = Atoms(name, positions=[(0, 0, 0)])

            calc = ORCA(label=f"ORCA_{name}/orca_mult_{mult}",
                        orca_command=_orca_command,
                        charge=0,
                        mult=mult,
                        orcasimpleinput='engrad UKS BLYP 6-31G', orcablocks=
                        f"%scf Convergence tight \n SmearTemp 2000 \n "
                        f"maxiter 1000 "
                        f"\n end"
                        )

            at.set_calculator(calc)
            at.info['dft_energy'] = at.get_potential_energy()
            isolated_ats.append(at)
        write(isolated_fname, isolated_ats, 'extxyz', write_results=False)
    else:
        print('loading', isolated_fname)
        isolated_ats = read(isolated_fname, ':')
    return isolated_ats


def dimer_data(dimer_name, mult, orca_block, idx, distances=None):
    _orca_command = '/home/eg475/programs/orca/orca_4_2_1_linux_x86' \
                    '-64_openmpi314/orca'

    dimer_fname = f'xyzs/{dimer_name}_{idx}.xyz'
    orca_block_original = orca_block
    if not os.path.isfile(dimer_fname):

        if distances is None:
            distances = np.linspace(0.1, 6, 20)

        dimer_at = Atoms(dimer_name, positions=((0, 0, 0), (0, 0, 1)))
        print(f'getting {dimer_name} energies, idx={idx}')
        prev_dist = -1
        for dist in tqdm(distances):
            at_fname = f'xyzs/{dimer_name}_{idx}_{dist:.2f}.xyz'
            if not os.path.isfile(at_fname):
                at = dimer_at.copy()
                at.set_distance(0, 1, dist)

                if dist == distances[0]:
                    orca_simple_input = 'engrad UKS BLYP 6-31G'
                    orca_block = orca_block_original

                else:
                    orca_block = f'%moinp "orca_{idx}_dist_{prev_dist:.2f}.gbw"\n' + orca_block_original
                    orca_simple_input = 'engrad UKS BLYP 6-31G moread'

                calc = ORCA(
                    label=f"ORCA_{dimer_name}/orca_{idx}_dist_{dist:.2f}",
                    orca_command=_orca_command,
                    charge=0,
                    mult=mult,
                    task='gradient',
                    orcasimpleinput=orca_simple_input,
                    orcablocks=orca_block
                    )

                at.set_calculator(calc)
                at.info['dft_energy'] = at.get_potential_energy()
                at.arrays['dft_forces'] = at.get_forces()
                write(at_fname, at, 'extxyz', write_results=False)

            prev_dist = dist

        dimer_list = []
        for dist in distances:
            at = read(f'xyzs/{dimer_name}_{idx}_{dist:.2f}.xyz')
            dimer_list.append(at)
        write(dimer_fname, dimer_list, 'extxyz', write_results=False)
    else:
        print('loading', dimer_fname)
        dimer_list = read(dimer_fname, ':')

    return dimer_list


def orca_optg(orig_inp='ORCA/orca.inp', optg_inp='ORCA/orca_optg.inp', output='ORCA/orca_optg.out',
              orca_command='/home/eg475/programs/orca/orca_4_2_1_linux_x86-64_openmpi314/orca'):
    new_file = []
    with open(orig_inp, 'r') as f:
        for line in f:
            if '!' in line:
                line = line.rstrip() + ' Opt\n'
            new_file.append(line)

    new_file.append('XYZFile\n')

    with open('ORCA/orca_optg.inp', 'w') as f:
        for line in new_file:
            f.write(line)

    subprocess.run(f'{orca_command} {optg_inp} > {output}', shell=True)


def analyse_orca_scf(file):
    delta_E = []
    Max_DP = []
    RMS_DP = []
    F_P_comm = []
    search = False
    with open(file, 'r') as f:
        for line in f:
            if 'SCF ITERATIONS' in line:
                search = True
            if 'SCF NOT CONVERGED AFTER 1000 CYCLES' in line or 'Energy ' \
                                                                'Check ' \
                                                                'signals ' \
                                                                'convergence' in line:
                search = False

            if search == True:
                nos = re.findall('[- ]\d+.\d+', line)
                if len(nos) != 0 and len(nos) != 6:
                    print('weird number of numbers found')
                    return line

                if len(nos) == 6:
                    delta_E.append(float(nos[1].strip()))
                    Max_DP.append(float(nos[2].strip()))
                    RMS_DP.append(float(nos[3].strip()))
                    F_P_comm.append(float(nos[4].strip()))

    all_ax = []
    plt.figure()
    all_ax.append(plt.gca())
    plt.plot(range(len(delta_E)), np.abs(delta_E))
    plt.ylabel('|delta E|/Ha?')
    plt.title('Delta E')

    plt.figure()
    all_ax.append(plt.gca())
    plt.plot(range(len(Max_DP)), Max_DP)
    plt.ylabel('Max_DP')
    plt.title('Max_DP')

    plt.figure()
    all_ax.append(plt.gca())
    plt.plot(range(len(RMS_DP)), RMS_DP)
    plt.ylabel('RMS_DP')
    plt.title('RMS_DP')

    plt.figure()
    all_ax.append(plt.gca())
    plt.plot(range(len(F_P_comm)), F_P_comm)
    plt.ylabel('F_P_comm')
    plt.title('F_P_comm')

    for ax in all_ax:
        ax.set_yscale('log')
        ax.set_xlabel('iteration')
        ax.grid(color='lightgrey')