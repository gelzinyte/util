
import ase
import ase.vibrations
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
                # vib_name, displ_name = disp_name.split('.')
                vib_name, displ_name = os.path.splitext(disp_name)
                displ_name = displ_name[1:]  # skips the '.' in the beginning of extensions
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

    def read(self, method='standard', direction='central'):
        ''' I'm only interested in and supporting 'standard' method and 'central' direction and nfree=2'''
        self.method = method.lower()
        self.direction = direction.lower()
        assert self.method in ['standard']
        assert self.direction in ['central']
        assert self.nfree == 2

        def load(fname, combined_data=None):
            if combined_data is None:
                with open(fname, 'rb') as fl:
                    f = pickleload(fl)
            else:
                try:
                    f = combined_data[op.basename(fname)]
                except KeyError:
                    f = combined_data[fname]  # Old version
            if not hasattr(f, 'shape') and not hasattr(f, 'keys'):
                # output from InfraRed
                return f[0]
            return f

        n = 3 * len(self.indices)
        H = np.empty((n, n))
        r = 0
        if op.isfile(self.name + '.all.pckl'):
            # Open the combined pickle-file
            combined_data = load(self.name + '.all.pckl')
        else:
            combined_data = None
        for a in self.indices:
            for i in 'xyz':
                name = '%s.%d%s' % (self.name, a, i)
                fminus = load(name + '-.pckl', combined_data)
                fplus = load(name + '+.pckl', combined_data)
                H[r] = .5 * (fminus - fplus)[self.indices].ravel()
                H[r] /= 2 * self.delta
                r += 1
        H += H.copy().T
        self.H = H
        m = self.atoms.get_masses()
        if 0 in [m[index] for index in self.indices]:
            raise RuntimeError('Zero mass encountered in one or more of '
                               'the vibrated atoms. Use Atoms.set_masses()'
                               ' to set all masses to non-zero values.')

        self.im = np.repeat(m[self.indices]**-0.5, 3)
        omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)
        self.modes = modes.T.copy()
        self._evals = omega2

        # Conversion factor:
        s = units._hbar * 1e10 / math.sqrt(units._e * units._amu)
        self.hnu = s * omega2.astype(complex)**0.5

    @property
    def evals(self):
        if '_evals' not in dir(self):
            self.read()
        return self._evals

    @property
    def evecs(self):
        if 'modes' not in dir(self):
            self.read()
        return self.modes

    def displace_one_nm(self, temp, n, direction):
        '''returns atoms displaced along normal mode n, in +/- direction (direction= pos/neg) for temperature temp'''

        if 'hnu' not in dir(self):
            self.read()

        mode = self.get_mode(n) * math.sqrt(temp * units.kB / abs(self.hnu[n]))
        p = self.atoms.positions.copy()

        if direction == 'pos':
            p += mode
        elif direction == 'neg':
            p -= mode
        else:
            raise RuntimeError('Set "direction" either as "pos" or "neg"')

        at = self.atoms.copy()
        at.set_positions(p)
        return at

    def displace_all_nms(self, temp):

        all_nms = []
        for n in range(len(self.atoms)*3):
            for dir in ['pos', 'neg']:

                at = self.displace_one_nm(temp, n, dir)
                all_nms.append(at)

        return all_nms


    def draw_nm_at_T(self, temp):
        n = len(self.atoms)*3 - 6
        # cov = np.eye(n) / self.evals[6:] * units.kB * temp / (6 * n)
        cov = np.eye(n) / self.evals[6:] * units.kB * temp
        # cov = np.eye(n) / self.evals[6:] * units.kB * temp / n

        norm = stats.multivariate_normal(mean=np.zeros(n), cov=cov,
                                         allow_singular=True)
        alphas = norm.rvs()
        individual_displacements = np.array(
            [aa * evec for aa, evec in zip(alphas, self.evecs)])

        mass_wt_displs = individual_displacements.sum(axis=0)
        displacements = mass_wt_displs * self.im
        displacements = displacements.reshape(len(self.atoms), 3)

        at = self.atoms.copy()
        p = at.positions.copy()
        p += displacements
        at.positions = p
        return at

    def draw_N_nm_at_T(self, temp, N):
        return [self.draw_nm_at_T(temp=temp) for _ in range(N)]