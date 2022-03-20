import click
import yaml
import logging

from copy import deepcopy

from tqdm import tqdm

from ase.io import read, write

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.generate_configs import vib

logger = logging.getLogger(__name__)


@click.command('xtb-normal-modes')
@click.argument('input-fname')
@click.option('-o', '--output-fname')
@click.option('--parallel-hessian', "parallel_hessian", flag_value=True,
              default=True)
@click.option('--parallel-atoms', "parallel_hessian", flag_value=False)
def xtb_normal_modes(input_fname, output_fname, parallel_hessian):

    from xtb.ase.calculator import XTB

    configset_in = ConfigSet_in(input_files=input_fname)
    configset_out = ConfigSet_out(output_files=output_fname)

    calc = (XTB, [], {'method':'GFN2-xTB'})

    prop_prefix = 'xtb2_'

    if parallel_hessian:
        vib.generate_normal_modes_parallel_hessian(inputs=configset_in,
                                          outputs=configset_out,
                                          calculator=calc,
                                          prop_prefix=prop_prefix)
    else:
        vib.generate_normal_modes_parallel_atoms(inputs=configset_in,
                                                 outputs=configset_out,
                                                 calculator=calc,
                                                 prop_prefix=prop_prefix,
                                                 chunksize=1)

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
    inputs = ConfigSet_in(input_files=input_fname)
    outputs = ConfigSet_out(output_files=output_fname, force=force)
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



    inputs = ConfigSet_in(input_files=input_fname)
    outputs = ConfigSet_out(output_files=output_fname)

    wfl.calc_descriptor.calc(inputs=inputs, outputs=outputs,
                             descs=descriptors, key=key,
                        local=local)


@click.command('ace')
@click.argument('input_fname')
@click.option('--output-fname', '-o', help='output filename')
@click.option('--ace-fname', '-a', help='ace json')
@click.option('--prop-prefix', '-p', default='ace_', show_default=True)
def evaluate_ace(input_fname, output_fname, ace_fname, prop_prefix):


    import ace

    inputs = ConfigSet_in(input_files=input_fname)
    outputs = ConfigSet_out(output_files=output_fname)
    # calc = (pyjulip.ACE, [ace_fname], {})

    # generic.run(inputs=inputs, outputs=outputs, calculator=calc,
    #             properties=['energy', 'forces'], output_prefix=prop_prefix)


    calc = ace.ACECalculator(jsonpath=ace_fname, ACE_version=1)
    logger.info('loaded up ace calculator')
    for at in tqdm(inputs):
        calc.reset()
        at.calc = calc
        at.calc.atoms = at
        at.info[f'{prop_prefix}energy'] = calc.get_potential_energy()
        at.arrays[f'{prop_prefix}forces'] = calc.get_forces()
        outputs.write(at)
    outputs.end_write()
