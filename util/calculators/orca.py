from wfl.calculators.orca import ExtendedORCA
from ase.calculators.calculator import Calculator
from ase import Atoms

class PopORCA(ExtendedORCA):
    def __init__(self, restart=None,
                 ignore_bad_restart_file = Calculator._deprecated,
                 label='orca', atoms=None, **kwargs):
        super(PopORCA, self).__init__(restart, ignore_bad_restart_file,
                                      label, atoms, **kwargs)

    def read_results(self):
        """Reads all results"""

        if not self.is_converged():
            raise CalculationFailed("Wavefunction not fully converged")

        self.read_energy()
        self.read_forces()

        self.read_dipole()
        self.read_populations()
        if 'opt' in self.parameters.task:
            self.read_opt_atoms()
            self.read_trajectory()
        if 'freq' in self.parameters.task:
            self.read_frequencies()

    def read_populations(self):
        '''  reads results from ORCA's population analysis
        NA   - Mulliken gross atomic population
        ZA   - Total nuclear charge
        QA   - Mulliken gross atomic charge
        VA   - Mayer's total valence
        BVA  - Mayer's bonded valence
        FA   - Mayer's free valence
        '''

        full_name_dict = {"NA":"mulliken_gross_atomic_populations",
                          "ZA":"total_nuclear_charge",
                          "QA":"mulliken_gross_atomic_charge",
                          "VA":"mayers_total_valence",
                          "BVA":"mayers_bonded_valence",
                          "FA":"Mayers_free_valence"}

        with open(self.label + '.out', mode='r', encoding='utf-8') as fd:
            lines = fd.readlines()

        for idx, line in enumerate(lines):
            if 'MAYER POPULATION ANALYSIS' in line:
                start = idx + 11
            if 'Mayer bond orders larger than' in line:
                end = idx - 2
                break

        d = {'elements': [], 'NA': [], 'QA': [], 'VA': [], 'BVA': [],
             'FA': []}

        for line in lines[start: end + 1]:
            l = line.split()
            d['elements'].append(l[1])
            d['NA'].append(float(l[2]))
            d['QA'].append(float(l[4]))
            d['VA'].append(float(l[5]))
            d['BVA'].append(float(l[6]))
            d['FA'].append(float(l[7]))

        for key, vals in d.items():
            if key == 'elements':
                continue
            self.extra_results[full_name_dict[key]] = vals


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
