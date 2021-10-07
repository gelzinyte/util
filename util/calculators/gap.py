from ase.calculators.calculator import Calculator, all_changes
from wfl.utils.parallel import construct_calculator_picklesafe
from quippy.potential import Potential
from xtb.ase.calculator import XTB
from util.calculators import orca

class PopGAP(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(selfself, gap_filename, orca_kwargs, **kwargs):
        super().__init__(**kwargs)

        self.gap = Potential(param_filename)
        self.orca = orca.PopORCA()


    def calculate(selfself, atoms=None, properties='default',
                  system_changes=all_changes):

        if propertis == 'default':
            properties = self.implemented_properties


        Calculator.calculate(self, atoms=atoms, properties=properties,
                             system_changes=system_changes)

        atoms = atoms.copy