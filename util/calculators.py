import ase.calculators.calculator
from quippy.potential import Potential


class DifGAP(ase.calculators.calculator):

    implemented_properties = ['energy', 'forces']

    default_parameters = {
        'dftb_filename': '/home/eg475/scripts/source_files/tightbind.parms'
                       '.DFTB.mio-0-1.xml',
        'gap_filename': None
    }

    def __init__(self, atoms=None):
        """Calculator for getting target values from baseline method (DFTB)
        and a GAP model that predicts the difference between target and
        baseline values.
        """

        self.atom = None
        self.results = {}
        self.parameters = None

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if atoms is not None:
            atoms.calc = self
            if self.atoms is not None:
                # Atoms were read from file.  Update atoms:
                if not (equal(atoms.numbers, self.atoms.numbers) and
                        (atoms.pbc == self.atoms.pbc).all()):
                    raise CalculatorError('Atoms not compatible with file')
                atoms.positions = self.atoms.positions
                atoms.cell = self.atoms.cell

        self.set(**kwargs)

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()





