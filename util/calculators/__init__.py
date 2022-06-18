from ase.calculators.calculator import Calculator, all_changes
try:
    from quippy.potential import Potential
except ModuleNotFoundError:
    pass

try:
    pass
#    from xtb.ase.calculator import XTB
except ModuleNotFoundError:
    pass


class xtb2_plus_gap(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, gap_filename, **kwargs):

        super().__init__(**kwargs)

        self.gap = Potential(param_filename=gap_filename)
        self.xtb = XTB(method="GFN2-xTB")

    def calculate(self, atoms=None, properties="default", system_changes=all_changes):

        if properties == "default":
            properties = self.implemented_properties

        Calculator.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )

        atoms = self.atoms.copy()
        atoms.calc = self.gap
        gap_energy = atoms.get_potential_energy()
        gap_forces = atoms.get_forces()

        atoms = self.atoms.copy()
        atoms.calc = self.xtb
        xtb_energy = atoms.get_potential_energy()
        xtb_forces = atoms.get_forces()

        self.results["energy"] = gap_energy + xtb_energy
        self.results["forces"] = gap_forces + xtb_forces


def pyjulip_ace(ace_fname):
    import pyjulip
    return pyjulip.ACE1(ace_fname)
