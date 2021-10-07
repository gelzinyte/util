from ase.calculators.calculator import Calculator, all_changes
from wfl.utils.parallel import construct_calculator_picklesafe
from quippy.potential import Potential
from xtb.ase.calculator import XTB
from wfl.calculators import orca

class PopGAP(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, gap_filename, orca_kwargs=None, base_rundir=None,
                 dir_prefix='popGAP_ORCA_', output_prefix='dft_',
                 keep_files='default', **kwargs):
        super().__init__(**kwargs)

        self.gap = Potential(gap_filename)
        self.orca_kwargs = orca_kwargs
        self.base_rundir = base_rundir
        self.dir_prefix=dir_prefix
        self.output_prefix=output_prefix
        self.keep_files=keep_files


    def calculate(selfself, atoms=None, properties='default',
                  system_changes=all_changes):

        if propertis == 'default':
            properties = self.implemented_properties


        Calculator.calculate(self, atoms=atoms, properties=properties,
                             system_changes=system_changes)

        atoms = orca.evaluate_op(atoms=self.atoms.copy(),
                                 base_rundir=self.base_rundir,
                                 keep_files=self.keep_files,
                                 orca_kwargs=self.orca_kwargs,
                                 output_prefix=self.output_prefix,
                                 basin_hopping=False)

        atoms = self.prepare_local_q(atoms)

        atoms.calc = self.gap
        self.results['energy'] = atoms.get_potential_energy()
        self.results['forces'] = atoms.get_forces()


    @staticmethod
    def prepare_local_q(atoms,
                        input_arrays_name='mulliken_gross_atomic_charge'):
        """sets negative charge to be between 0 and 1 and positive between
        1 and 2 and assignes it to "local_q" arrays entry
        """

        in_vals = atoms.arrays[input_arrays_name]
        out_vals = in_vals + 1
        for in_val, out_val in zip(in_vals, out_vals):
            assert in_val < 1 and in_val > -1
            if in_val < 1:
                assert out_val > 0 and out_val < 1
            elif in_val > 1:
                assert out_val > 1 and out_val < 2
            elif in_val == 0:
                assert out_val == 1

        atoms.arrays['local_q'] = out_vals





