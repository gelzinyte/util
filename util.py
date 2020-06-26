# General imports
import time, os, pickle, sys, re
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import zip_longest

# Packages for atoms and molecules
import ase
from ase.build import molecule
from ase.optimize.precon import PreconLBFGS
from ase.build import molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase import units
from ase.io import read, write
from ase.io.extxyz import key_val_dict_to_str
from ase.md.verlet import VelocityVerlet
from ase.constraints import FixBondLength
from ase.optimize.precon import PreconLBFGS
from ase.optimize import FIRE
from ase.neb import NEB
from ase.utils import  pickleload
from ase.utils import opencew
from ase.parallel import world
from ase.vibrations import Vibrations

from rdkit import Chem
from rdkit.Chem import AllChem

from quippy.potential import Potential

# My stuff
sys.path.append('/home/eg475/molpro_stuff/driver')
sys.path.append('/home/eg475/programs/my_scripts')
import gap_plots
from molpro import Molpro


# TODO deal with dftb potential
# TODO add default types in functions

def hello():
    print('Utils say hi')

####################################################################################
#
#  General small things
#
####################################################################################


def relax(at, calc, fmax=1e-3, steps=0):
    """Relaxes at with given calc with PreconLBFGS"""
    at.set_calculator(calc)
    opt = PreconLBFGS(at)
    opt.run(fmax=fmax, steps=steps)
    return at


def shift0(my_list, by=None):
    """shifts all values in list by first value or by value given"""
    if not by:
        by = my_list[0]
    return [x - by for x in my_list]


def get_counts(at):
    ''' Returns dictionary of chemical formula: dict[at_number] = count'''
    nos = at.get_atomic_numbers()
    unique, counts = np.unique(nos, return_counts=True)
    return dict(zip(unique, counts))


def get_rmse(ar1, ar2):
    sq_error = []
    for val1, val2 in zip(ar1, ar2):
        sq_error.append((val1-val2)**2)
    return np.sqrt(np.mean(sq_error))


def get_std(ar1, ar2):
    sq_error = []
    for val1, val2 in zip(ar1, ar2):
        sq_error.append((val1-val2)**2)
    return np.sqrt(np.var(sq_error))


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def dict_to_vals(my_dict):
    all_values = []
    for type, values in my_dict.items():
        all_values.append(values)
    all_values = np.concatenate(all_values)
    return all_values

def grouper(iterable, n, fillvalue=None):
    """groups a list/etc into chunks of n values, e.g. 
    grouper('ABCDEFG', 3, 'x') --> 'ABC' 'DEF' 'Gxx'
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def get_pair_dists(all_dists, at_nos, atno1, atno2):
    dists = []
    for i in range(len(all_dists[at_nos==atno1])):
        dists.append(np.array(all_dists[at_nos==atno1][i][at_nos==atno2]))

    dists = np.array(dists)
    if atno1==atno2:
        dists = np.triu(dists)
        dists = dists[dists!=0]
    else:
        dists = dists.flatten()
    return dists


def distances_dict(at_list):
    dist_hist = {}
    for at in at_list:
        all_dists = at.get_all_distances()

        at_nos = at.get_atomic_numbers()
        at_syms = np.array(at.get_chemical_symbols())

        formula = at.get_chemical_formula()
        unique_symbs = natural_sort(list(ase.formula.Formula(formula).count().keys()))
        for idx1, sym1 in enumerate(unique_symbs):
            for idx2, sym2 in enumerate(unique_symbs[idx1:]):
                label = sym1 + sym2

                atno1 = at_nos[at_syms == sym1][0]
                atno2 = at_nos[at_syms == sym2][0]

                if label not in dist_hist.keys():
                    dist_hist[label] = np.array([])

                distances = get_pair_dists(all_dists, at_nos, atno1, atno2)
                dist_hist[label] = np.concatenate([dist_hist[label], distances])
    return dist_hist


def rattle(at, stdev, natoms):
    '''natoms - how many atoms to rattle'''
    at = at.copy()
    pos = at.positions.copy()
    mask = np.ones(pos.shape).flatten()
    mask[:-natoms] = 0
    np.random.shuffle(mask)
    mask = mask.reshape(pos.shape)

    rng = np.random.RandomState()
    new_pos = pos + rng.normal(scale=stdev, size=pos.shape) * mask
    at.set_positions(new_pos)
    return at


def has_converged(template_path, molpro_out_path='MOLPRO/molpro.out'):
    with open(molpro_out_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'Final alpha occupancy' in line:
                final_iteration_no = int(re.search(r'\d+', lines[i - 2]).group())

    # print(final_iteration_no)
    maxit = 60  # the default
    with open(template_path, 'r') as f:
        for line in f:
            if 'maxit' in line:
                maxit = line.rstrip().split('maxit=')[1]  # take the non-default if present in the input
                break

    # print(maxit)
    if maxit == final_iteration_no:
        print(f'Final iteration no was found to be {maxit}, optimisation has not converged')
        return False
    else:
        return True

####################################################################################
#
# Nomral mode help
#
####################################################################################

class Vibrations(ase.vibrations.Vibrations):
    '''
    Little extention to ase.vibrations.Vibrations, here:
    https://wiki.fysik.dtu.dk/ase/ase/vibrations/modes.html#ase.vibrations.Vibrations    
    '''
    
    def run(self):
        '''Run the vibration calculations. Same as the ASE version without dipoles and 
        polarizability, but returns Atoms object with energies and forces and
        combines all .pickl files at the end of calculation. Atoms also have vib.name as 
        at.info['config_type'] and displacement type (e.g. '0x+') as at.info['displacement'].
        '''
        if os.path.isfile(f'{self.name}.all.pckl'):
            raise RuntimeError('Please remove or split the .all.pckl file')

        atoms = []
        for disp_name, at in self.iterdisplace(inplace=False):
            filename = f'{disp_name}.pckl'
            fd = opencew(filename)
            if fd is not None:
                at.set_calculator(self.calc)
                energy = at.get_potential_energy()
                forces = at.get_forces()
                at.info['energy'] = energy
                at.arrays['forces'] = forces
                vib_name, displ_name = disp_name.split('.')
                at.info['displacement'] = displ_name
                at.info['config_type'] = vib_name
                atoms.append(at.copy())
                if world.rank == 0:
                    pickle.dump(forces, fd, protocol=2)
                    sys.stdout.write(f'Writing {filename}\n')
                    fd.close()
                sys.stdout.flush()
        self.combine()
        return atoms
    
    @property
    def evals(self):
        if 'hnu' not in dir(self):
            self.read()
        return [np.real(val**2) for val in self.hnu]
    
    @property
    def evecs(self):
        if 'modes' not in dir(self):
            self.read()
        return self.modes
"""
# OUTDATED
"""
def my_vib_run(vib):
    
    if os.path.isfile(f'{vib.name}.all.pckl'):
        raise RuntimeError('Please remove or split the .all.pckl file')
        
    atoms = []
    for disp_name, at in vib.iterdisplace(inplace=False):
        filename = f'{disp_name}.pckl'
        fd = opencew(filename)
        if fd is not None:
            # forces = vib.calc.get_forceos(at)
            at.set_calculator(vib.calc)
            energy = at.get_potential_energy()
            forces = at.get_forces()
            at.info['energy'] = energy
            at.arrays['forces'] = forces
            vib_name, displ_name = disp_name.split('.')
            at.info['displacement'] = displ_name
            at.info['config_type'] = vib_name
            atoms.append(at.copy())
            if world.rank == 0:
                pickle.dump(forces, fd, protocol=2)
                sys.stdout.write(f'Writing {filename}\n')
                fd.close()
            sys.stdout.flush()
    write(f'{vib.name}.xyz', atoms, 'extxyz')
    return atoms
        
"""
# OUTDATED
"""
def get_ats_evals_evecs(opt_atoms, sub_name):
    vib = Vibrations(opt_atoms, name=sub_name)
    if not os.path.isfile(f'{vib.name}.all.pckl'):
        atoms = my_vib_run(vib)
        vib.combine()
    elif os.path.isfile(f'{vib.name}.xyz'):
        atoms = ase.io.read(f'{vib.name}.xyz', index=':')
    else:
        print("WARNING displaced forces.pckl are present, but not the corresponding atoms object.")

    vib.summary()
    evals = [np.real(val**2) for val in vib.hnu]
    evecs = vib.modes
    return atoms, evals, evecs


def eval_plot(evals_dft, evals_pred, name):
    plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(1,2)
    ax = plt.subplot(gs[0])
    ax.set_xlabel('Eigenvalue #')
    ax.set_ylabel('Eigenvalue value')
    ax.plot(range(len(evals_dft)), evals_dft, label='dft')
    ax.plot(range(len(evals_pred)), evals_pred, label='dftb')
    ax.legend()
    
    ax = plt.subplot(gs[1])
    ax = plt.gca()
    gap_plots.do_plot(evals_dft, evals_pred, ax, 'RMSE (eV$^2$)')
    for_limits = evals_pred + evals_dft
    flim = (min(for_limits) - 0.005, max(for_limits) + 0.005)
    ax.plot(flim, flim, c='k', linewidth=0.8)
    ax.legend()
    ax.set_xlabel('reference eigenvalues, eV$^2$')
    ax.set_ylabel('predicted eigenvalues, eV$^2$')
    plt.suptitle(f'Eigenvalues: {name}', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{name}_eigenvalue_plots.png', dpi=300)

    
def evec_plot(evals_dft, evecs_dft, evals_pred, evecs_pred, name, output_dir=None):
    mx = dict()
    for i, ev_dft in enumerate(evecs_dft):
        mx[f'dft_{i}'] = dict()
        for j, ev_pr in enumerate(evecs_pred):
            mx[f'dft_{i}'][f'pred_{j}'] = np.dot(ev_dft, ev_pr)
            
    df = pd.DataFrame(mx)
    N = len(evals_dft)
    if N>30:
        figsize=(0.25*N, 0.21*N)
    else:
        figsize=(10,8)
    fig, ax = plt.subplots(figsize=figsize)
    hmap = ax.pcolormesh(df, vmin=-1, vmax=1, cmap='bwr', edgecolors='lightgrey', linewidth=0.01)
    cbar = plt.colorbar(hmap)
    plt.yticks(np.arange(0.5, len(evals_pred), 1), [round(x,3) for x in evals_pred])
    plt.xticks(np.arange(0.5, len(evals_dft), 1), [round(x,3) for x in evals_dft], rotation=90)
    plt.xlabel('DFT eigenvalues')
    plt.ylabel('Predicted eigenvalues')
    plt.title(f'Dot products between eigenvectors for {name}', fontsize=14)
    plt.tight_layout()

    name = f'{name}_eigenvectors.png'
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        name = os.path.join(output_dir, name)

    plt.savefig(name, dpi=300)

    
def set_dft_vals(atoms):
    for at in atoms:
        at.info['dft_energy'] = at.info['energy']
        at.arrays['dft_forces'] = at.arrays['forces']
    return atoms


def nm_analysis(gap_filename, opt_atoms, name, directory=None):
    
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print('WARNING directory already exists')
        orig_dir = os.getcwd()
        os.chdir(directory)
    
    gap = Potential(param_filename=gap_filename)
    opt_atoms.set_calculator(gap)
    atoms_gap,  evals_gap, evecs_gap = get_ats_evals_evecs(opt_atoms, f'gap_{name}')
    
    # TODO think of how to better do athis
    template_path = '/home/eg475/reactions/main_molpro_template.txt'
    calc_args = { 'template' : template_path,
                  'molpro' : '/opt/molpro/bin/molprop',
                  'energy_from' : 'RKS',
                  # 'append_lines' : None,
                  # 'test_mode' : False,
                  # 'working_dir' : {/scratch-ssd/eg475/tmp},
                  'extract_forces' : True}

    with open(template_path, 'r') as f:
        print('Molpro template:')
        for line in f.readlines():
            print(line.rstrip())

    molpro = Molpro(calc_args = calc_args)
    
    opt_atoms.set_calculator(molpro)
    atoms_dft,  evals_dft, evecs_dft = get_ats_evals_evecs(opt_atoms, f'dft_{name}')
    atoms_dft = set_dft_vals(atoms_dft)
    
    # reconsider how to get away with namings
    gap_plots.make_scatter_plots(param_filename=gap_filename, train_ats=atoms_dft)
    evec_plot(evals_dft, evecs_dft, evals_gap, evecs_gap, name)
    eval_plot(evals_dft, evals_gap, name)

    if directory:
        os.chdir(orig_dir)
    
    return atoms_dft
    
   
    
####################################################################################
#
# RDKit help
#
####################################################################################


def get_xyz_str(mol):
    """For RDKit molecule, gets ase-compatable string with species and positions defined"""
    insert = 'Properties=species:S:1:pos:R:3'
    xyz = Chem.rdmolfiles.MolToXYZBlock(mol)
    xyz = xyz.split(sep='\n\n')
    return f'{xyz[0]}\n{insert}\n{xyz[1]}'


def write_rdkit_xyz(mol, filename):
    """Writes RDKit molecule to xyz file with filenam"""
    xyz_str = get_xyz_str(mol)
    with open(filename, 'w') as f:
        f.write(xyz_str)


def smi_to_xyz(smi, fname):
    """Converts smiles to 3D and writes to xyz file"""
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    out = AllChem.EmbedMolecule(mol)
    write_rdkit_xyz(mol, fname)


####################################################################################
#
#  GAP help
#
####################################################################################

def make_descr_str(descr_dict):
    """ given dictionary of {"name":"desc_name", "param":"value", ...} converts to string "desc_name param=value ...", suitable for GAP"""
    descr_dict = deepcopy(descr_dict)
    string = f"{descr_dict['name']} "
    del descr_dict['name']
    string += key_val_dict_to_str(descr_dict)
    return string


def make_config_type_sigma_str(dct):
    """Converts config_type sigma values to string to be included in GAP command"""
    '''dct should be {config_type,str:values,float list}'''
    string = ' config_type_kernel_regularisation={'

    for i, key in enumerate(dct.keys()):
        string += key
        for value in dct[key]:
            string += f':{value}'
        if i != len(dct) - 1:
            string += ':'

    string += '}'
    return string


def make_gap_command(gap_filename, training_filename, descriptors_dict, default_sigma, config_type_sigma=None, \
                     gap_path=None, output_filename=False, glue_fname=None):
    """Makes GAP command to be called in shell"""
    descriptor_strs = [make_descr_str(descriptors_dict[key]) for key in descriptors_dict.keys()]

    default_sigma = f'{{{default_sigma[0]} {default_sigma[1]} {default_sigma[2]} {default_sigma[3]}}}'

    if not gap_path:
        gap_path = 'gap_fit'

    if glue_fname:
        glue_command = f' core_param_file={glue_fname} core_ip_args={{IP Glue}}'
    else:
        glue_command = ''

    if not config_type_sigma:
        config_type_sigma = ''
    else:
        if type(config_type_sigma) != dict:
            raise TypeError('config_type_sigma should be a dictionary of config_type(str):values(list of floats)')
        config_type_sigma = make_config_type_sigma_str(config_type_sigma)

    gap_command = f'{gap_path} gp_file={gap_filename} atoms_filename={training_filename} force_parameter_name=forces sparse_separate_file=F default_sigma='
    gap_command += default_sigma + config_type_sigma + glue_command + ' gap={'

    for i, desc in enumerate(descriptor_strs):
        gap_command += desc
        if i != len(descriptor_strs) - 1:
            gap_command += ': '
        else:
            gap_command += '}'

    if output_filename:
        gap_command += f' > {output_filename}'

    return gap_command


def gap_gradient_test(gap_fname, start=-3, stop=-7):
    """Gradient test on methane molecule for given GAP"""
    methane = molecule('CH4')
    rattle_std = 0.01
    seed = np.mod(int(time.time() * 1000), 2 ** 32)
    methane.rattle(stdev=rattle_std, seed=seed)

    calc = Potential(param_filename=gap_fname)

    gradient_test(methane, calc, start, stop)


####################################################################################
#
#  Stuff for MD, NEB, related
#
####################################################################################


def make_end_images(sub, H_idx, separation):
    '''Makes end images for H abstraction. Quite a specific case. '''
    # dftb = Potential(args_str='TB DFTB', param_filename='/home/eg475/reactions/tightbind.parms.DFTB.mio-0-1.xml')
    
    # print('\n----Relaxing substrate and methanol\n')
    # sub.set_calculator(dftb)
    # opt = PreconLBFGS(sub)
    # opt.run(fmax=1e-3)

    methanol = molecule('CH3OH')
    # methanol.set_calculator(dftb)
    # opt = PreconLBFGS(methanol)
    # opt.run(fmax=1e-3)
    
    idx_shift = len(methanol)

    meth_O_idx = np.where(methanol.get_atomic_numbers()==8)[0][0]
    meth_C_idx = np.where(methanol.get_atomic_numbers()==6)[0][0]
    
    dists = sub.get_all_distances()[H_idx]
    sub_C_idx = np.argmin([d if d!=0 else np.inf for d in dists])
    
    OC = methanol.get_distance(meth_O_idx, meth_C_idx, vector=True)
    CH = sub.get_distance(sub_C_idx, H_idx, vector=True)

    methanol.rotate(OC, CH, center=methanol.positions[1])
    
    methanol.positions -=methanol.positions[meth_O_idx]
    sub.positions -= sub.positions[sub_C_idx]

    at = methanol + sub

    unit_dir = CH/np.linalg.norm(CH)
    at.positions[:idx_shift] += separation*unit_dir
    
    dists = at.get_all_distances()[meth_O_idx]
    tmp_H_idx = np.argmin([d if d!=0 else np.inf for d in dists])
    tmp_H_pos = at.positions[tmp_H_idx]
    del at[tmp_H_idx]
    
    at_init = at.copy()
    at_final = at.copy()
    at_final.positions[H_idx + idx_shift-1] = tmp_H_pos
    
    return at_init, at_final


def run_neb(neb, fname, steps_fire, fmax_fire, steps_lbfgs, fmax_lbfgs):
    "Runs NEB with fire/lbfgs for steps/fmax on given NEB object, returns NEB object"
    
    opt_fire = FIRE(neb, trajectory=f'structure_files/{fname}.traj')
    opt_lbfgs = PreconLBFGS(neb, precon=None, use_armijo=False, \
                            trajectory=f'structure_files/{fname}.traj')

    print('\n----NEB\n')
    if steps_fire:
        opt_fire.run(fmax=fmax_fire, steps=steps_fire)

    if steps_lbfgs:
        opt_lbfgs.run(fmax=fmax_lbfgs, steps=steps_lbfgs)
    return neb


def prepare_do_neb(atoms, calc_name, no_images, fname, steps_fire, fmax_fire, steps_lbfgs, fmax_lbfgs):
    """Sets up and NEB given end images"""
    images = [atoms[0]]
    images += [atoms[0].copy() for _ in range(no_images-2)]
    images += [atoms[1].copy()]
    
    neb = NEB(images)
    neb.interpolate()
    
    for i, image in enumerate(images):
        print(f'setting {i} image calculator')
        if calc_name=='dftb':
            image.set_calculator(Potential(args_str='TB DFTB', param_filename='/home/eg475/reactions/tightbind.parms.DFTB.mio-0-1.xml'))
        else:
            image.set_calculator(Potential(param_filename=calc_name))
        
    neb = run_neb(neb, fname, steps_fire, fmax_fire, steps_lbfgs, fmax_lbfgs)
    return neb


def do_md(sub, traj_name, H_idx, separation=4, temp=None):
    """Sets up MD for end images"""
    # TODO think if you need this or how to make this more usable/general

    dftb = Potential(args_str='TB DFTB', param_filename='/home/eg475/reactions/tightbind.parms.DFTB.mio-0-1.xml')

    if not temp:
        temp = 1000  # K
    time_step = 0.5  # fs
    no_steps = 1000

    print(f'\n----temperature: {temp} K, time step: {time_step} fs\n')

    print('\n----Relaxing substrate and methanol\n')
    sub.set_calculator(dftb)
    opt = PreconLBFGS(sub)
    opt.run(fmax=1e-3)

    methanol = molecule('CH3OH')
    methanol.set_calculator(dftb)
    opt = PreconLBFGS(methanol)
    opt.run(fmax=1e-3)

    at_init, at_final = make_end_images(sub, H_idx=H_idx, separation=separation)

    print('\n----reactants MD\n')
    do_md_run(at_init, temp, time_step, no_steps, f'{traj_name}_react')
    print('\n----products MD\n')
    do_md_run(at_final, temp, time_step, no_steps, f'{traj_name}_prod')



def do_md_run(atoms, temp, time_step, no_steps, traj_name, at_fix=None):
    """Sets up and runs MD"""
    # TODO add general potential
    #     print('\n----Relaxing initial MD structure\n')
    dftb = Potential(args_str='TB DFTB', param_filename='/home/eg475/reactions/tightbind.parms.DFTB.mio-0-1.xml')
    atoms.set_calculator(dftb)
    #     opt = PreconLBFGS(atoms, precon=None)
    #     opt.run(fmax=5e-3)

    MaxwellBoltzmannDistribution(atoms, temp * units.kB)
    Stationary(atoms)
    ZeroRotation(atoms)

    write(f'{traj_name}_1st_frame.xyz', atoms, 'extxyz')

    if at_fix:
        print(f"\n----Fixing bond between atoms {at_fix[0], at_fix[1]}\n")
        c = FixBondLength(at_fix[0], at_fix[1])
        atoms.set_constraint(c)

    dyn = VelocityVerlet(atoms, time_step * units.fs, trajectory=f'{traj_name}.traj')

    def print_energy(atoms=atoms):
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        print('Energies/eV: potential {:.3f}, kinetic {:.3f}, total {:.3f}; \
        temperature {:.1f} K'.format(epot, ekin, epot + ekin, ekin / 1.5 / units.kB))

    dyn.attach(print_energy, interval=50)
    print_energy(atoms)
    dyn.run(no_steps)
    traj = read(f'{traj_name}.traj', index=':')
    write(f'{traj_name}.xyz', traj, 'extxyz')


####################################################################################
#
# Miscellaneous
#
####################################################################################

def gradient_test(mol, calc, start=-3, stop=-7):
    """gradient test on general molecule with general calculator"""
    unit_d = np.random.random((len(mol), 3))
    unit_d = unit_d / (np.linalg.norm(unit_d))

    num = start - stop + 1
    epsilons = np.logspace(start, stop, num=num)

    mol.set_calculator(calc)
    f_analytical = np.vdot(mol.get_forces(), unit_d)

    print('{:<12s} {:<22s} {:<25s} {:<20s}'.format('', '', 'force', 'initial energy'))
    print('{:<12} {:<22} {:<25.14e} {:<20}'.format('', '', f_analytical, mol.get_potential_energy()))

    print('{:<12s} {:<22s} {:<25s} {:<20s}'.format('epsilon', 'ratio', 'energy gradient', 'displaced energy'))

    for eps in epsilons:
        atoms = mol.copy()
        atoms.set_calculator(calc)

        e0 = atoms.get_potential_energy()

        pos = atoms.get_positions().copy()
        pos += eps * unit_d
        atoms.set_positions(pos)

        e_displ = atoms.get_potential_energy()

        numerator = -(e_displ - e0)
        f_numerical = numerator / eps
        if abs(f_analytical) < 0.1:
            ratio = (1 + f_analytical) / (1 + f_numerical)
        else:
            ratio = f_analytical / f_numerical

        print('{:<12} {:<22} {:<25.14e} {:<20}'.format(eps, ratio, f_numerical, e_displ))


