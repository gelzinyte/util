from wfl.calculators.orca import ORCA
from ase.calculators.calculator import Calculator
from ase import Atoms


def read_xyz_from_output(output_fname):
    with open(output_fname, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
            first_line_idx = idx + 2
        if "CARTESIAN COORDINATES (A.U.)" in line:
            last_line_idx = idx - 3

    elements = []
    coords = []
    for line in lines[first_line_idx: last_line_idx +1]:
        line = line.split()
        elements.append(line[0])
        coords.append((line[1], line[2], line[3]))

    return Atoms(elements, positions=coords)
