
from ase.calculators.orca import ORCA
from ase import Atom, Atoms
from ase.io import read, write
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import re
import sys
sys.path.append('/home/eg475/molpro_stuff/driver')
from molpro import Molpro
import molpro as mp
import util



def mp_optimise_at(at_fname, at_out_fname,  source_template):

    mp_command = '/opt/molpro/bin/molpro'

    base_name = os.path.splitext(at_fname)[0]
    # at_out_fname = base_name + '_optg_out.xyz'
    optg_template_fname = base_name + '_template.txt'

    make_optg_template(source_template=source_template,
                          optg_template=optg_template_fname,
                          input_fname=at_fname,
                          output_fname=at_out_fname)

    submission_script_fname = f'{base_name}.sh'
    job_name=f'{base_name}'
    output_fname = f'{base_name}.out'
    command = f'{mp_command} {optg_template_fname} -o {output_fname}'

    util.write_generic_submission_script(
        script_fname=submission_script_fname, job_name=job_name,
        command=command)

    subprocess.run(f'qsub {submission_script_fname}', shell=True)



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

    if not os.path.isdir('xyzs'):
        os.makedirs('xyzs')

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

    if not os.path.isdir('xyzs'):
        os.makedirs('xyzs')

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
                #at.arrays['dft_forces'] = at.get_forces()
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
            if 'SCF NOT CONVERGED AFTER' in line or 'Energy ' \
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


def isolated_at_molpro_data(dimer_name, is_at_template_paths):


    if not os.path.isdir('xyzs'):
        os.makedirs('xyzs')

    isolated_fname = f'xyzs/isolated_{dimer_name}.xyz'
    if not os.path.isfile(isolated_fname):

        print('getting isolated atom energies')



        isolated_names = list(dimer_name)
        isolated_ats = []

        for name, template_path in zip(isolated_names, is_at_template_paths):

            template_path = os.path.join(os.getcwd(), template_path)
            with open(template_path, 'r') as f:
                print('\nMolpro template:')
                print(template_path)
                for line in f.readlines():
                    print(line.rstrip())
                    if 'RKS' in line.upper():
                        calc_type = 'RKS'
                    elif 'UKS' in line.upper():
                        calc_type = 'UKS'
                print('\n')

            calc_args = {'template': template_path,
                         'molpro': '/opt/molpro/bin/molprop',
                         'energy_from': calc_type,
                         'extract_forces': False}

            at = Atoms(name, positions=[(0, 0, 0)])

            calc = Molpro(directory=f'MOLPRO_{name}', label='molpro_isolated_at', calc_args=calc_args)

            at.set_calculator(calc)
            at.info['dft_energy'] = at.get_potential_energy()
            isolated_ats.append(at)
        write(isolated_fname, isolated_ats, 'extxyz', write_results=False)
    else:
        print('loading', isolated_fname)
        isolated_ats = read(isolated_fname, ':')
    return isolated_ats



def dimer_molpro_data(dimer_name, template_path, idx, distances):



    if not os.path.isdir('xyzs'):
        os.makedirs('xyzs')


    dimer_fname = f'xyzs/{dimer_name}_{idx}.xyz'
    if not os.path.isfile(dimer_fname):

        print(f'getting {dimer_name} energies, idx={idx}')

        template_path = os.path.join(os.getcwd(), template_path)
        with open(template_path, 'r') as f:
            print('\nMolpro template:')
            print(template_path)
            for line in f.readlines():
                print(line.rstrip())
                if 'RKS' in line.upper():
                    calc_type = 'RKS'
                elif 'UKS' in line.upper():
                    calc_type = 'UKS'
            print('\n')

        calc_args = {'template': template_path,
                     'molpro': '/opt/molpro/bin/molprop',
                     'energy_from': calc_type,
                     'extract_forces': False}

        if distances is None:
            distances = np.linspace(0.1, 6, 20)

        dimer_at = Atoms(dimer_name, positions=((0, 0, -0.5), (0, 0, 0.5)))


        for dist in tqdm(distances):
            at_fname = f'xyzs/{dimer_name}_{idx}_{dist:.2f}.xyz'
            if not os.path.isfile(at_fname):
                at = dimer_at.copy()
                at.set_distance(0, 1, dist)

                label = f'molpro_{idx}_dist_{dist:.2f}'
                label = label.replace('.', '_')
                calc = Molpro(directory=f'MOLPRO_{dimer_name}', label=label, calc_args=calc_args)


                at.set_calculator(calc)
                at.info['dft_energy'] = at.get_potential_energy()
                # at.arrays['dft_forces'] = at.get_forces()
                write(at_fname, at, 'extxyz', write_results=False)


        dimer_list = []
        for dist in distances:
            at = read(f'xyzs/{dimer_name}_{idx}_{dist:.2f}.xyz')
            dimer_list.append(at)
        write(dimer_fname, dimer_list, 'extxyz', write_results=False)
    else:
        print('loading', dimer_fname)
        dimer_list = read(dimer_fname, ':')

    return dimer_list


def plot_molpro_curve(dimer_name, template_paths, labels, distances, is_at_template_paths):

    isolated_ats = isolated_at_molpro_data(dimer_name, is_at_template_paths)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    min_e = 0

    for idx, (label, template_path) in enumerate(
            zip(labels, template_paths)):

        dimer_ats = dimer_molpro_data(dimer_name, template_path,
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

def make_optg_template(source_template, optg_template, input_fname, output_fname):
    new_file = []
    with open(source_template, 'r') as f:
        for line in f.readlines():
            if 'force' in line:
                continue
            # overrides
            elif 'geom=' in line:
                # add full path?
                new_file.append(f'geom={input_fname}\n')
                continue
            new_file.append(line)

    new_file.append(
        f'optg, maxit=500\nput, xyz, {output_fname}\n')
    with open(optg_template, 'w') as f:
        for line in new_file:
            f.write(line)




def get_parallel_molpro_energies_forces(atoms, no_cores, mp_template='template_molpro.txt',
                                wdir='MOLPRO', energy_from='RKS', extract_forces=True):

    mp_path = '/opt/molpro/bin/molprop'
    dfile = mp.MolproDatafile(mp_template)

    if not os.path.isdir(wdir):
        os.makedirs(wdir)

    original_wdir = os.getcwd()
    os.chdir(wdir)

    all_atoms = []

    for at_batch in util.grouper(atoms, no_cores):

        at_batch = [at for at in at_batch if at is not None]
        bash_call = ''

        for idx, at in enumerate(at_batch):
            input_name = f'input_{idx}.xyz'
            template_name = f'template_{idx}.txt'
            output_name = f'output_{idx}'

            write(input_name, at)

            dfile['GEOM'] = [f'={input_name}']
            dfile.write(template_name)

            bash_call += f'{mp_path} {template_name} -o {output_name}.txt &\n'

            if os.path.isfile(f'{output_name}.txt'):
                os.remove(f'{output_name}.txt')

            if os.path.isfile(f'{output_name}.xml'):
                os.remove(f'{output_name}.xml')

        bash_call += 'wait \n'
        subprocess.run(bash_call, shell=True)

        for idx in range(len(at_batch)):
            at = mp.read_xml_output(f'output_{idx}.xml', energy_from=energy_from,
                                    extract_forces=extract_forces)
            all_atoms.append(at)
            # os.remove(f'output_{idx}.xml')
            # os.remove(f'output_{idx}.txt')

    os.chdir(original_wdir)
    return all_atoms