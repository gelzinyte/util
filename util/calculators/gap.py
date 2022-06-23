from ase.calculators.calculator import Calculator, all_changes
from wfl.utils.parallel import construct_calculator_picklesafe
from quippy.potential import Potential
try:
    from xtb.ase.calculator import XTB
except ModuleNotFoundError:
    pass

import logging
from wfl.calculators import orca

logger = logging.getLogger(__name__)

def at_wt_gap_calc(gap_fname, at_gaussian_weight=None):
    if at_gaussian_weight is None:
        logger.warn("no atom gaussian weight")
        calc_args = None
        add_arrays=None
    else:
        logger.warn(f"asking for at gaussian weight with name {at_gaussian_weight}")
        calc_args = f"atom_gaussian_weight_name={at_gaussian_weight}" 
        add_arrays = at_gaussian_weight

    gap = Potential(param_filename=gap_fname, add_arrays=add_arrays, 
                    calc_args=calc_args)
    





class PopGAP(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, gap_filename, orca_kwargs=None, workdir_root=None,
                 dir_prefix='popGAP_ORCA_', output_prefix='dft_',
                 keep_files='default', **kwargs):
        super().__init__(**kwargs)

        self.gap = Potential(param_filename=gap_filename,
                             add_arrays="atom_gaussian_weight",
             calc_args="atom_gaussian_weight_name=atom_gaussian_weight")

        self.orca_kwargs = orca_kwargs
        self.workdir_root = workdir_root
        self.dir_prefix=dir_prefix
        self.output_prefix=output_prefix
        self.keep_files=keep_files


    def calculate(self, atoms=None, properties='default',
                  system_changes=all_changes):

        if properties == 'default':
            properties = self.implemented_properties


        Calculator.calculate(self, atoms=atoms, properties=properties,
                             system_changes=system_changes)

        atoms = orca.evaluate_autopara_wrappable(atoms=self.atoms.copy(),
                                 workdir_root=self.workdir_root,
                                 keep_files=self.keep_files,
                                 orca_kwargs=self.orca_kwargs,
                                 output_prefix=self.output_prefix,
                                 basin_hopping=False)

        self.prepare_local_q(atoms)

        atoms.calc = self.gap
        self.results['energy'] = atoms.get_potential_energy()
        self.results['forces'] = atoms.get_forces()


    @staticmethod
    def prepare_local_q(atoms,
                        input_arrays_name='dft_mulliken_gross_atomic_charge'):
        """sets negative charge to be between 0 and 1 and positive between
        1 and 2 and assignes it to "local_q" arrays entry
        """

        in_vals = atoms.arrays[input_arrays_name]
        out_vals = in_vals + 1
        for in_val, out_val in zip(in_vals, out_vals):
            # print(in_val)
            assert in_val < 1 and in_val > -1
            if in_val < 0:
                assert out_val > 0 and out_val < 1
            elif in_val > 0:
                assert out_val > 1 and out_val < 2
            elif in_val == 0:
                assert out_val == 1

        atoms.arrays['atom_gaussian_weight'] = out_vals





