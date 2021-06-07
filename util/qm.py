from ase.io import read, write
from wfl.calculators import orca

def read_orca_output(input_xyz, orca_label):

    at = read(input_xyz)
    calc = orca.ExtendedORCA()
    calc.label=orca_label
    calc.read_energy()
    at.info['dft_energy'] = calc.results['energy']
    try:
        calc.read_forces()
        at.arrays['dft_forces'] = calc.results['forces']
    except FileNotFoundError:
        pass

    return at





