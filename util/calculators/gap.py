from ase.calculators.calculator import Calculator, all_changes
from wfl.utils.parallel import construct_calculator_picklesafe
from quippy.potential import Potential
from xtb.ase.calculator import XTB

class PopGAP(Calculator):
    implemented_properties = ['energy', 'forces']
