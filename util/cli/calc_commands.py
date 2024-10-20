import click
import yaml
import logging

from copy import deepcopy

from tqdm import tqdm

from ase.io import read, write

from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import normal_modes
from wfl.calculators import generic 
from wfl.autoparallelize.autoparainfo import AutoparaInfo
from util.calculators import pyjulip_ace
from wfl.autoparallelize.autoparainfo import AutoparaInfo

logger = logging.getLogger(__name__)


@click.command('xtb-normal-modes')
@click.argument('input-fname')
@click.option('-o', '--output-fname')
@click.option('--parallel-hessian', "parallel_hessian", flag_value=True,
              default=True)
@click.option('--parallel-atoms', "parallel_hessian", flag_value=False)
def xtb_normal_modes(input_fname, output_fname, parallel_hessian):

    from xtb.ase.calculator import XTB

    ConfigSet = ConfigSet(input_files=input_fname)
    OutputSpec = OutputSpec(output_files=output_fname)

    calc = (XTB, [], {'method':'GFN2-xTB'})

    prop_prefix = 'xtb2_'

    if parallel_hessian:
        normal_modes.generate_normal_modes_parallel_hessian(inputs=ConfigSet,
                                          outputs=OutputSpec,
                                          calculator=calc,
                                          prop_prefix=prop_prefix)
    else:
        normal_modes.generate_normal_modes_parallel_atoms(inputs=ConfigSet,
                                                 outputs=OutputSpec,
                                                 calculator=calc,
                                                 prop_prefix=prop_prefix,
                                                 num_inputs_per_python_subprocess=1)

@click.command('dftb')
@click.argument('input_fn')
@click.option('--output_fn', '-o')
@click.option('--prefix', '-p', default='dftb_')
def recalc_dftb_ef(input_fn, output_fn, prefix='dftb_'):

    from quippy.potential import Potential

    dftb = Potential('TB DFTB', param_filename='/home/eg475/scripts/source_files/tightbind.parms.DFTB.mio-0-1.xml')

    ats_in = read(input_fn, ':')
    ats_out = []
    for atoms in tqdm(ats_in):
        at = atoms.copy()
        at.calc = dftb
        at.info[f'{prefix}energy'] = at.get_potential_energy()
        at.arrays[f'{prefix}forces'] = at.get_forces()
        ats_out.append(at)

    write(output_fn, ats_out)



@click.command('xtb2')
@click.argument('input_fn')
@click.option('--output-fn', '-o')
@click.option('--prefix', '-p', default='xtb2_')
def calc_xtb2_ef(input_fn, output_fn, prefix):

    from xtb.ase.calculator import XTB

    xtb2 = XTB(method="GFN2-xTB")

    ats = read(input_fn, ':')
    for at in tqdm(ats):
        at.calc = xtb2
        at.info[f'{prefix}energy'] = at.get_potential_energy()
        at.arrays[f'{prefix}forces'] = at.get_forces()
    write(output_fn, ats)



@click.command('gap-plus-xtb2')
@click.argument('input-fname')
@click.option('--output-fname', '-o')
@click.option('--prefix', '-p', default='gap_plus_xtb2_')
@click.option('--gap-fname', '-g')
@click.option('--force', is_flag=True)
def evaluate_diff_calc(input_fname, output_fname, prefix, gap_fname, force):

    from util import calculators

    calculator = (calculators.xtb2_plus_gap, [], {'gap_filename': gap_fname})
    inputs = ConfigSet(input_files=input_fname)
    outputs = OutputSpec(output_files=output_fname, force=force)
    generic.run(inputs=inputs, outputs=outputs, calculator=calculator,
                properties=['energy', 'forces'], output_prefix=prefix)


@click.command('desc')
@click.argument('input_fname')
@click.option('--output-fname', '-o')
@click.option('--param-fname', '-p')
@click.option('--key', '-k', help='key for info/arays to store descriptor in')
@click.option('--local', is_flag=True, help='calculate local descriptor')
def calculate_descriptor(input_fname, output_fname, param_fname, key, local):

    import wfl.calc_descriptor

    with open(param_fname) as f:
        params = yaml.safe_load(f)

    if isinstance(params, list):
        # it's a descriptor dict
        descriptors = deepcopy(params)
    else:
        # means it's a gap_fit param
        descriptors = deepcopy(params.pop('_gap'))

    inputs = ConfigSet(input_files=input_fname)
    outputs = OutputSpec(output_files=output_fname)

    wfl.calc_descriptor.calc(inputs=inputs, outputs=outputs,
                             descs=descriptors, key=key,
                        local=local)


@click.command('ace')
@click.argument('input_fname')
@click.option('--output-fname', '-o', help='output filename')
@click.option('--ace-fname', '-a', help='ace json')
@click.option('--prop-prefix', '-p', default='ace_', show_default=True)
@click.option('--num_inputs_per_python_subprocess', '-c', default=100, show_default=True, help='num_inputs_per_python_subprocess for parallelisation')
def evaluate_ace(input_fname, output_fname, ace_fname, prop_prefix, num_inputs_per_python_subprocess):

    inputs = ConfigSet(input_fname)
    outputs = OutputSpec(output_fname)

    calc = (pyjulip_ace, [ace_fname], {})

    generic.run(inputs=inputs, outputs=outputs, calculator=calc, properties=["energy", "forces"],
                output_prefix=prop_prefix, autopara_info=AutoparaInfo(num_inputs_per_python_subprocess=num_inputs_per_python_subprocess))


@click.command('mace')
@click.argument('input_fname')
@click.option('--output-fname', '-o', help='output filename')
@click.option('--mace-fname', '-m', help='mace .cpu path')
@click.option('--prop-prefix', '-p', default='mace_', show_default=True)
@click.option('--num_inputs_per_python_subprocess', '-c', default=100, show_default=True, help='num_inputs_per_python_subprocess for parallelisation')
@click.option('--dtype', default='float64', show_default=True)
def evaluate_ace(input_fname, output_fname, mace_fname, prop_prefix, num_inputs_per_python_subprocess, dtype):

    from mace.calculators.mace import MACECalculator 

    inputs = ConfigSet(input_fname)
    outputs = OutputSpec(output_fname)

    calc = (MACECalculator, [], {"model_path":mace_fname, "default_dtype":dtype, "device":"cpu"})

    generic.run(inputs=inputs, outputs=outputs, calculator=calc, properties=["energy", "forces"],
                output_prefix=prop_prefix, autopara_info = AutoparaInfo(num_inputs_per_python_subprocess=num_inputs_per_python_subprocess))



