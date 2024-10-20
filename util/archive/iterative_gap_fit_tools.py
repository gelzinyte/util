
import os
from ase.io import read, write
import util
# from util import qm
from util import ugap
from util.vibrations import Vibrations
import subprocess
from quippy.potential import Potential
from copy import deepcopy
from ase.optimize.precon import PreconLBFGS
import matplotlib.pyplot as plt
import matplotlib as mpl
from util import shift0 as sft
from util import urdkit
import re
import time
import sys
sys.path.append('/home/eg475/molpro_stuff/driver')
import molpro as mp
import random
import string



def fit_gap(idx, descriptors, default_sigma, gap_fit_path=None, config_type_sigma=None):

    train_file = f'xyzs/dset_{idx}.xyz'
    gap_fname = f'gaps/gap_{idx}.xml'
    out_fname = f'gaps/out_{idx}.txt'

    desc = deepcopy(descriptors)

    command = ugap.make_gap_command(gap_filename=gap_fname, training_filename=train_file, descriptors_dict=desc,
                                  default_sigma=default_sigma, output_filename=out_fname, glue_fname=None,
                                    config_type_sigma=config_type_sigma, gap_fit_path=gap_fit_path)

    print(f'\n-------GAP {idx} command\n')
    print(command)
    stdout, stderr  = util.shell_stdouterr(command)

    print(f'---gap stdout:\n {stdout}')
    print(f'---gap stderr:\n {stderr}')


def get_no_cores(sub_script='sub.sh'):

    with open(sub_script, 'r') as f:
        for line in f:
            if '-pe smp' in line:
                no_cores = int(re.findall(r'\d+', line)[0])
                return no_cores


def remove_high_e_structs(atoms, upper_e_per_at):
    new_ats = []
    for at in atoms:
        if at.info['dft_energy'] / len(at) < upper_e_per_at:
           new_ats.append(at)
    print(f'Removed {len(atoms) - len(new_ats)} structures with energies higher than {upper_e_per_at} eV/atom.')
    return new_ats



def optimise_structure(iter_no, atoms, fmax=1e-2, steps=500):


    gap_name = f'gaps/gap_{iter_no}.xml'
    gap = Potential(param_filename=gap_name)
    # copy first guess so that first_guess stays non-optimised.
    guess = atoms.copy()
    guess.set_calculator(gap)
    print(f'Fmax = {fmax}')
    opt = PreconLBFGS(guess, trajectory=f'xyzs/optimisation_{iter_no}.traj')
    converged = opt.run(fmax=fmax, steps=steps)
    traj = read(f'xyzs/optimisation_{iter_no}.traj', ':')
    write(f'xyzs/optimisation_{iter_no}.xyz', traj, 'extxyz', write_results=False)
    return guess



def get_structures(source_atoms, n_dpoints, n_rattle_atoms, stdev):
    '''based on source atoms, this rattles the structures a number of times, which are then
       re-evaluated with DFT of choice'''
    atoms = []
    for source_at in source_atoms:
        for _ in range(n_dpoints):
            at = source_at.copy()
            at = util.rattle(at, stdev=stdev, natoms=n_rattle_atoms)
            atoms.append(at)
    return atoms


def extend_dset(iter_no, smiles, n_dpoints, n_rattle_atoms, stdev, wfl_command, orca_tmp_fname,
                stride=5, upper_energy_cutoff=None):

    new_dset_name = f'xyzs/dset_{iter_no+1}.xyz'

    # take every stride-th structure from trajectory, and always the last one
    source_atoms = []
    for smi in smiles:
        opt_traj_name = f'xyzs/optimisation_{iter_no}_{smi}'
        traj = read(opt_traj_name+'.xyz', ':')
        # skipping first structure, which is just first guess and reading every
        # stride-th plus the last structure
        source_atoms += traj[stride::stride]
        if len(traj) % stride != 1:
            source_atoms.append(traj[-1])

    atoms_to_compute = get_structures(source_atoms=source_atoms, n_dpoints=n_dpoints,
                        n_rattle_atoms=n_rattle_atoms, stdev=stdev)


    more_atoms = orca_par_data(atoms_in=atoms_to_compute, out_fname=orca_tmp_fname,
                               wfl_command=wfl_command, config_type=f'iter_{iter_no}')

    if upper_energy_cutoff is not None:
        more_atoms = remove_high_e_structs(more_atoms, upper_energy_cutoff)


    old_dset = read(f'xyzs/dset_{iter_no}.xyz', index=':')
    write(f'xyzs/more_atoms_{iter_no+1}.xyz', more_atoms, 'extxyz', write_results=False)
    new_dset = old_dset + more_atoms
    write(new_dset_name, new_dset, 'extxyz', write_results=False)






def do_opt(at, gap_fname, dft_calc, traj_name, dft_stride=5, fmax=0.01,
           steps=199):
    gap = Potential(param_filename=gap_fname)

    at.set_calculator(gap)
    opt = PreconLBFGS(at, use_armijo=False, trajectory=f'{traj_name}.traj')
    opt.run(fmax=fmax, steps=steps)
    traj = read(f'{traj_name}.traj', ':')
    for idx, trj_at in enumerate(traj):

        if idx % dft_stride == 0:
            print('dft_step:', idx)
            trj_at.set_calculator(dft_calc)
            try:
                trj_at.info['dft_energy'] = trj_at.get_potential_energy()
                trj_at.arrays['dft_forces'] = trj_at.get_forces()
            except Exception:
                print('couldn\'t get dft energy, skipping')

        trj_at.set_calculator(gap)
        trj_at.info['gap_energy'] = trj_at.get_potential_energy()
        trj_at.arrays['gap_forces'] = trj_at.get_forces()

    print('writing atoms for', traj_name)
    write(f'{traj_name}.xyz', traj, 'extxyz', write_results=False)




def get_structure_to_optimise(smi, seed, stdev=0.1):
    '''creates 3-D structure from smiles. if seed is not given, atoms are not rattled
    and basic knowledge (e.g. aromatic rings are flat) and experimental torsion angles
    aren't used. if seed is given, the conformers should be more accurate, but then
    the structures are rattled.'''

    useBasicKnowledge = True
    useExpTorsionAnglePrefs = True
    if seed is None:
        useBasicKnowledge = False
        useExpTorsionAnglePrefs = False

    urdkit.smi_to_xyz(smi, 'xyzs/mol_3d.xyz', useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge)

    at = read('xyzs/mol_3d.xyz')
    if seed is not None:
        at.rattle(stdev=stdev, seed=seed)
    at.info['smiles'] = smi
    return at



def orca_par_data(atoms_in, out_fname, wfl_command):
    '''calls workflow command to get orca energies '''

    in_fname = os.path.splitext(out_fname)[0] + '_to_eval.xyz'
    write(in_fname, atoms_in, 'extxyz', write_results=False)

    wfl_command += f' {in_fname}'
    print(f'Workflow command:\n{wfl_command}')

    stdout, stderr = util.shell_stdouterr(wfl_command)
    print(f'---wfl stdout\n{stdout}')
    print(f'---wfl stderr\n{stderr}')

    atoms_out = read(out_fname, ':')
    os.remove(in_fname)

    return atoms_out


def get_dft_energies(in_atoms, keep_files=False):
    '''get dft energies via wfl'''

    smearing = 2000
    n_wfn_hop = 1
    task = 'gradient'
    orcasimpleinput = 'UKS B3LYP def2-SV(P) def2/J D3BJ'

    # scratch_dir = '/scratch/eg475'
    scratch_dir = '/tmp/eg475'

    n_run_glob_op = 1
    kw_orca = f'smearing={smearing}'

    tmp_prefix='tmp_orca_'
    tmp_suffix='.xyz'
    tmp_dir = '.'

    # os_handle, tmp_fname = tempfile.mkstemp(suffix=tmp_suffix, prefix=tmp_prefix, dir=tmp_dir)

    tmp_fname = os.path.join(tmp_dir, tmp_prefix + util.rnd_string(8) + tmp_suffix)

    wfl_command = f'wfl -v ref-method orca-eval --output-file {tmp_fname} ' \
                  f'-tmp ' \
                  f'{scratch_dir} --keep-files {keep_files} --base-rundir ' \
                  f'orca_outputs_{util.rnd_string(8)} ' \
                  f'-nr {n_run_glob_op} -nh {n_wfn_hop} --kw "{kw_orca}" ' \
                  f'--orca-simple-input "{orcasimpleinput}"'


    atoms_out = orca_par_data(in_atoms, tmp_fname, wfl_command)

    if keep_files:
        print(f'keeping {os.path.basename(tmp_fname)}')
        if not os.path.isfile(os.path.basename(tmp_fname)):
            shutil.move(tmp_fname, '.')
            os.remove(tmp_fname)
    else:
        os.remove(tmp_fname)

    return atoms_out


def add_my_decorations(nm_data, at_info, del_ens_fs=True):
    '''prepares data to go into gap_fit'''

    for at in nm_data:
        at.cell = [40, 40, 40]
        at.info['dft_energy'] = at.info['energy']
        at.arrays['dft_forces'] = at.arrays['forces']
        if at_info:
            for key, value in at_info.items():
                at.info[key] = value

        if del_ens_fs:
            try:
                del at.info['energy']
            except:
                print('Could not delete "energy"')
            try:
                del at.arrays['forces']
            except:
                print('Could not delete "forces"')

    return nm_data
