import click
from pathlib import Path
from util import mace
import logging
from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic

logger = logging.getLogger(__name__)


@click.command('to-cpu')
@click.argument('input-fnames', nargs=-1)
def convert_to_cpu(input_fnames):
    for fn_in in input_fnames:
        if not Path(fn_in).exists():
            logger.warn(f"did not find {fn_in}")
            continue
        fn_out = fn_in + '.cpu'
        mace.to_cpu(fn_in, fn_out)


@click.command('eval')
@click.option('--mace-fname', '-m')
@click.option('--pred-prop-prefix', '-p', default='mace_', show_default=True)
@click.option('--input-fn', '-i')
@click.option('--output-fn', '-o')
@click.option('--r-max', type=click.FLOAT)
@click.option('--at-num', type=click.FLOAT, multiple=True)
@click.option('--dtype', default="float64", show_default=True)
def eval_mace(mace_fname, pred_prop_prefix, input_fn, output_fn, r_max, at_num, dtype):
    from mace.calculators import mace
    calc = (mace.MACECalculator, [mace_fname], {"r_max": r_max, "device": "cpu", "atomic_numbers": at_num, "default_dtype": dtype})
    inputs = ConfigSet(input_files=input_fn)
    outputs = OutputSpec(output_files=output_fn)
    generic.run(
        inputs=inputs, 
        outputs=outputs,
        calculator=calc,
        properties = ["energy", "forces"],
        output_prefix=pred_prop_prefix)





