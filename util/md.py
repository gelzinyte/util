import sys
import numpy as np
from quippy.potential import Potential

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io import read, write
from ase.md.verlet import VelocityVerlet
from ase.constraints import FixBondLength
# from ase.optimize.precon import PreconLBFGS
from ase.optimize import FIRE
from ase.neb import NEB
from ase import units
from ase.build import molecule


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
    # meth_C_idx = np.where(methanol.get_atomic_numbers()==6)[0][0]
    meth_OH_H_idx = 3

    dists = sub.get_all_distances()[H_idx]
    sub_C_idx = np.argmin([d if d!= 0 else np.inf for d in dists])

    # OC = methanol.get_distance(meth_O_idx, meth_C_idx, vector=True)
    HO = methanol.get_distance(meth_OH_H_idx, meth_O_idx, vector=True)
    CH = sub.get_distance(sub_C_idx, H_idx, vector=True)

    # methanol.rotate(OC, CH, center=methanol.positions[1])
    methanol.rotate(HO, CH, center=methanol.positions[1])

    # methanol.positions -= methanol.positions[meth_O_idx]
    methanol.positions -= methanol.positions[meth_OH_H_idx]
    sub.positions -= sub.positions[sub_C_idx]

    at = methanol + sub

    unit_dir = CH / np.linalg.norm(CH)
    at.positions[:idx_shift] += separation * unit_dir

    dists = at.get_all_distances()[meth_O_idx]
    tmp_H_idx = np.argmin([d if d != 0 else np.inf for d in dists])
    tmp_H_pos = at.positions[tmp_H_idx]
    del at[tmp_H_idx]

    at_init = at.copy()
    at_final = at.copy()
    at_final.positions[H_idx + idx_shift - 1] = tmp_H_pos

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
    images += [atoms[0].copy() for _ in range(no_images - 2)]
    images += [atoms[1].copy()]

    neb = NEB(images)
    neb.interpolate()

    for i, image in enumerate(images):
        print(f'setting {i} image calculator')
        if calc_name == 'dftb':
            image.set_calculator(
                Potential(args_str='TB DFTB', param_filename='/home/eg475/reactions/tightbind.parms.DFTB.mio-0-1.xml'))
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



def do_md_run(atoms, temp, time_step, no_steps, traj_name, calc=None, at_fix=None):
    """Sets up and runs MD"""
    if atoms.calc is None:
        if calc is None:
            print('Using DFTB as calculator')
            calc = Potential(args_str='TB DFTB', param_filename='/home/eg475/reactions/tightbind.parms.DFTB.mio-0-1.xml')
        atoms.set_calculator(calc)

    MaxwellBoltzmannDistribution(atoms, temp * units.kB)
    Stationary(atoms)
    ZeroRotation(atoms)


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

    return traj
