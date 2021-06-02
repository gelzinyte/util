'''some/a lot of things derived from ase.vibrations.vibrations'''

import ase
import ase.vibrations
from ase.units import kB
from ase.io.trajectory import Trajectory
import os
import numpy as np
from ase.utils import opencew, pickleload
import pickle
import sys
from ase.parallel import world
from ase import units
import shutil
import math
import os.path as op
from scipy import stats
from ase.io import read, write
import click
from ase.io import read, write
from itertools import zip_longest
import subprocess
from ase.optimize.precon import PreconLBFGS
import tempfile


from quippy.potential import Potential


class Vibrations:
    '''modified ase.vibrations.vibrations to go with the workflow'''

    def __init__(self, nm_fname):
        all_nm_atoms = read(nm_fname, ':')
        self.atoms = all_nm_atoms[0]
        N = len(self.atoms)
        self.N = N

        self.inverse_m = np.repeat(self.atoms.get_masses() ** -0.5, 3)

        self.evals = np.array([at.info['nm_eval'] for at in all_nm_atoms])
        self.evecs = np.zeros((3 * N, 3 * N))
        for idx, at in enumerate(all_nm_atoms):
            self.evecs[idx] = at.arrays['nm_evec'].reshape(3*N)

        energy_units = units._hbar * 1e10 / math.sqrt(units._e * units._amu)
        self.energies = np.array([energy_units * eigenval.astype(complex) ** 0.5 for eigenval in self.evals])


    def nm_displace(self, temp, nms='all'):
        no_atoms = len(self.atoms)
        if type(nms) == str:
            if nms == 'all':
                nms = np.arange(6, no_atoms * 3)

        n = len(nms)

        cov = np.eye(n) * units.kB * temp / (self.evals[nms])
        norm = stats.multivariate_normal(mean=np.zeros(n), cov=cov,
                                         allow_singular=True)
        alphas = norm.rvs()
        if len(nms) == 1:
            alphas = [alphas]

        individual_displacements = np.array(
            [aa * evec for aa, evec in zip(alphas, self.evecs[nms])])

        mass_wt_displs = individual_displacements.sum(axis=0)
        displacements = mass_wt_displs * self.inverse_m
        displacements = displacements.reshape(len(self.atoms), 3)

        # TODO this copies over stuff (e.g. energies, etc) from self.atoms, which might be inaccurate
        # just delete any "energy" or "forces" present, but should take care of everything else?
        at = self.atoms.copy()
        del_info_entries = ['energy', 'disp_direction']
        del_arrays_entries = ['forces']
        for entry in del_info_entries:
            if entry in at.info.keys():
                del at.info[entry]
        for entry in del_arrays_entries:
            if entry in at.arrays.keys():
                del at.arrays[entry]

        at.positions += displacements

        energy = sum([aa ** 2 * eigenval / 2  for aa, eigenval in
                      zip(alphas, self.evals[nms])])

        at.info['nm_energy'] = energy
        return at

    def multi_at_nm_displace(self, temp, n_samples, nms='all'):
        return [self.nm_displace(temp=temp, nms=nms) for _ in range(n_samples)]


    def summary(self):
        """Print a summary of the vibrational frequencies.
        """
        # conversion factor for eV to cm^-1
        s = 0.01 * units._e / units._c / units._hplanck


        print('---------------------\n')
        print('  #    meV     cm^-1\n')
        print('---------------------\n')
        for idx, en in enumerate(self.energies):
            if en.imag != 0:
                c = 'i'
                en = en.imag
            else:
                c = ' '
                en = en.real

            print(f'{idx:3d} {1000 * en:6.1f}{c} {s*en:7.1f}{c}')
        print('---------------------\n')
        print('Zero-point energy: %.3f eV\n' %
              self.get_zero_point_energy())

    def get_zero_point_energy(self):
        return 0.5 * sum(self.energies)


    def view_modes(self, prefix='nm', output_dir='NORMAL_MODES', nms='all', temp=300, nimages=32):
        '''writes out xyz files with oscillations along each of the normal modes'''

        if nms == 'all':
            nms = np.arange(len(self.atoms) * 3)
        elif type(nms) == int:
            nms = [nms]
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for nm in nms:
            mode = self.get_mode(nm) * math.sqrt(kB * temp / abs(self.energies[nm]))
            traj = os.path.join(output_dir, f'{prefix}_{nm}.xyz')
            if os.path.isfile(traj):
                os.remove(traj)
            for x in np.linspace(0, 2 * math.pi, nimages, endpoint=False):
                at = self.atoms.copy()
                at.positions += math.sin(x) * mode.reshape((self.N, 3))
                at.write(traj, append=True)


    def get_mode(self, n):
        # mode = np.zeros((len(self.atoms), 3))
        mode = self.evecs[n] * self.inverse_m
        return mode.reshape((-1, 3))


def get_normal_modes(in_fname, nm_out_fname, optimise=False, opt_fmax=1e-2, calc=None):
    '''  in_fname - .xyz of single structure
        calc
        optimise - whether to optimise the structure if it's not optimised
        opt_fmax - maximum force component in optimised structure'''

    atoms = read(in_fname)
    if optimise and calc is not None:
        atoms.set_calculator(calc)
        forces = atoms.get_forces()
        if max(forces.flatten()) > opt_fmax:
            atoms = do_optimisation(atoms, opt_fmax)


    displaced_ats = displace_at_in_xyz(atoms)
    displaced_ats = get_dft_energies(displaced_ats)
    save_normal_modes(nm_out_fname, displaced_ats)


def save_normal_modes(nm_out_fname, displaced_ats):

    delta = 0.01  # TODO either read out delta from eq_at.positions - non_eq_at.positions or deal with it somehow better
    eq_at = displaced_ats[0]
    n = 3 * len(eq_at)
    hessian = np.empty((n, n))
    masses = eq_at.get_masses()  # ase have a check for zero masses, but would I ever need one??
    inverse_m = np.repeat(masses ** -0.5, 3)

    for idx, (at_minus, at_plus) in enumerate(zip(displaced_ats[0::2], displaced_ats[1::2])):

        p_name = at_plus.info['disp_direction']
        m_name = at_minus.info ['disp_direction']


        if ('+' not in p_name) or ('-' not in m_name) or (p_name[:-1] != m_name[:-1]):
            raise ValueError(f'The displacements are not what I think they are, got: {p_name} and {m_name}')  # Any better Errors to raise?

        f_plus = at_plus.arrays['forces']
        f_minus = at_minus.arrays['forces']

        hessian[idx] = (f_minus-f_plus).ravel() / 4 / delta

    hessian += hessian.copy().T
    e_vals, e_vecs = np.linalg.eigh(np.array([inverse_m]).T * hessian * inverse_m)
    e_vecs = e_vecs.T


    ats_out = []
    for e_val, e_vec in zip(e_vals, e_vecs):
        at = eq_at.copy()
        at.info['nm_eval'] = e_val
        at.arrays['nm_evec'] = e_vec.reshape((len(eq_at), 3))
        del at.info['disp_direction']
        ats_out.append(at)

    write(nm_out_fname, ats_out, 'extxyz', write_results=False)


def displace_at_in_xyz(atoms):
    """
    displace each of the atoms along each of xyz backwards and forwards.
    """
    displaced_ats = []
    at_copy = atoms.copy()

    for disp_name, index, coordinate, displacement in displacements(atoms):
        at_copy = atoms.copy()
        at_copy.positions[index, coordinate] += displacement
        at_copy.info['disp_direction'] = disp_name
        displaced_ats.append(at_copy)

    return displaced_ats


def displacements(atoms):
    '''ase function'''
    delta = 0.01    # do this properly somehow
    indices = np.arange(len(atoms))
    for index in indices:
        for coord_idx, coord in enumerate('xyz'):
            for sign in [-1, 1]:
                if sign == -1:
                    sign_symbol = '-'
                else:
                    sign_symbol = '+'
                disp_name = f'{index}{coord}{sign_symbol}'
                displacement = sign * delta
                yield disp_name, index, coord_idx, displacement


def do_optimisation(atoms, fmax, steps=500):
    '''call the dft program to optimise directly'''
    opt = PreconLBFGS(atoms)
    opt.run(fmax=fmax, steps=steps)

    return atoms


#funs to generate datasets
@click.command()
@click.option('--dft_eq_fname', help='dft equilibrium filename')
@click.option('--out_dset_name', help='name of file with nm-displaced atoms')
@click.option('--temp', help='temperature at which to displace atoms')
@click.option('--n_samples', help='number of structures')
@click.option('--nms', default='all', show_default=True, help='which normal modes to displace along')
@click.option('--config_type', help='What config_type to set to the structures')
def nm_data_from_dft_eq(dft_eq_fname, out_dset_name, temp, n_samples, nms='all', config_type=None):
    name = os.path.basename(os.path.splitext(dft_eq_fname)[0])
    nm_fname = f'{name}.nm'
    get_normal_modes(in_fname=dft_eq_fname, nm_out_fname=nm_fname)
    nm_data_from_saved_nm_file(nm_fname, out_dset_name, temp, n_samples, nms, config_type)


def nm_data_from_smiles(smiles):
    '''Need to implement geometry optimisation with orca'''
    pass


def nm_data_from_saved_nm_file(nm_fname, out_dset_name, temp, n_samples, nms, config_type):
    vib = Vibrations(nm_fname)
    nm_data = vib.multi_at_nm_displace(temp=temp, n_samples=n_samples, nms=nms)
    nm_data = get_dft_energies(nm_data)
    nm_data = add_my_deccorations(nm_data, config_type)
    write(out_dset_name, nm_data, 'extxyz')


def add_my_deccorations(nm_data, config_type):

    for at in nm_data:
        at.cell=[40, 40, 40]
        at.info['dft_energy'] = at.info['energy']
        at.arrays['dft_forces'] = at.arrays['forces']
        if config_type:
            at.info['config_type'] = config_type

    return nm_data


### my functions that should be replaced in workflow
def shell_stdouterr(raw_command, cwd=None):
    """Abstracts the standard call of the commandline, when
    we are only interested in the stdout and stderr
    """
    stdout, stderr = subprocess.Popen(raw_command,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True,
                                      shell=True, cwd=cwd).communicate()
    return stdout.strip(), stderr.strip()


def orca_par_data(atoms_in, out_fname, wfl_command):
    '''calls workflow command to get orca energies and post-processes by
    assigning prefix as always and returns atoms'''

    in_fname = os.path.splitext(out_fname)[0] + '_to_eval.xyz'
    write(in_fname, atoms_in, 'extxyz', write_results=False)

    wfl_command += f' {in_fname}'
    print(f'Workflow command:\n{wfl_command}')

    stdout, stderr = shell_stdouterr(wfl_command)
    print(f'---wfl stdout\n{stdout}')
    print(f'---wfl stderr\n{stderr}')

    atoms_out = read(out_fname, ':')
    os.remove(in_fname)

    return atoms_out


def get_dft_energies(in_atoms, keep_tmp=True):
    '''get dft energies via wfl, etc
    very rudimental implementation, to be worked out'''

    smearing = 2000
    n_wfn_hop = 1
    task = 'gradient'
    orcasimpleinput = 'UKS B3LYP def2-SV(P) def2/J D3BJ'

    scratch_dir = '/scratch/eg475'

    n_run_glob_op = 1
    kw_orca = f'smearing={smearing}'

    tmp_prefix='tmp_orca_'
    tmp_suffix='.xyz'
    tmp_dir = '.'

    os_handle, tmp_fname = tempfile.mkstemp(suffix=tmp_suffix, prefix=tmp_prefix, dir=tmp_dir)

    wfl_command = f'wfl -v ref-method orca-eval --output-file {tmp_fname} ' \
                  f'-tmp ' \
                  f'{scratch_dir} --keep-files True --base-rundir ' \
                  f'orca_outputs ' \
                  f'-nr {n_run_glob_op} -nh {n_wfn_hop} --kw "{kw_orca}" ' \
                  f'--orca-simple-input "{orcasimpleinput}"'



    atoms_out = orca_par_data(in_atoms, tmp_fname, wfl_command)

    if keep_tmp:
        print(f'keeping {os.path.basename(tmp_fname)}')
        shutil.move(tmp_fname, '.')
    os.remove(tmp_fname)

    return atoms_out


def get_ase_calc_energies(atoms, calc):
    # dftb = Potential(args_str='TB DFTB', param_filename='tightbind.parms.DFTB.mio-0-1.xml')

    for at in atoms:
        calc.reset()
        at.set_calculator(calc)
        at.info['energy'] = at.get_potential_energy()
        at.arrays['forces'] = at.get_forces()

    return atoms


if __name__=='__main__':
    nm_data_from_dft_eq()




