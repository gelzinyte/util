
import ase
import ase.vibrations
import os
import numpy as np
from ase.utils import opencew
import pickle
import sys
from ase.parallel import world



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