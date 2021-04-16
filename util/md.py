import ase
import os
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.md.langevin import Langevin
import numpy as np
from quippy.potential import Potential


def run_md(gap_filename, mol_filename):

    Temps = [350, 500, 650, 800, 950]
    np.random.seed(10)

    mol_name = os.path.splitext(mol_filename)[0]

    for Temp in Temps:
        
        calculator = Potential(param_filename=gap_filename)

        def write_frame():
            dyn.atoms.write('{}_gap_configs_T{}.xyz'.format(mol_name, Temp),
                            append=True)

        for initind in range(5):
            init_conf = read(mol_filename)
            init_conf.set_calculator(calculator)
            MaxwellBoltzmannDistribution(init_conf, Temp*units.kB)
            dyn = Langevin(init_conf, 0.3*units.fs, Temp*units.kB, 0.01)
            dyn.run(500)
            dyn.attach(write_frame, interval=200)
            dyn.run(2005)