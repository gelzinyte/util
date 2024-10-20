import click
import os
from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize.autoparainfo import AutoparaInfo
from ase.io import read, write 
import numpy as np
from util.plot import dimer

@click.command('eval')
@click.option('--gap-fname', '-g')
@click.option('--at-gaussian-weight', help='key for atoms.array to use for weighting soap atom gaussians with')
@click.option('--pred-prop-prefix', '-p')
@click.option("--input-fn", '-i')
@click.option("--output-fn", '-o')
def eval_gap(gap_fname, at_gaussian_weight, pred_prop_prefix, input_fn, output_fn):
    ci = ConfigSet(input_fn)
    co = OutputSpec(output_fn)
    import util.calculators.gap
    gap_calc = util.calculators.gap.at_wt_gap_calc(gap_fname, at_gaussian_weight, type="initialiser")
    wfl_num_threads = int(os.environ.get("WFL_AUTOPARA_NPOOL", 1))
    n_ats = len(read(input_fn, ":"))
    num_inputs_per_python_subprocess = int(n_ats / wfl_num_threads) + 1
    print(num_inputs_per_python_subprocess)
    generic.calculate(
        inputs=ci, 
        outputs=co,
        calculator=gap_calc,
        properties = ["energy", "forces"],
        output_prefix=pred_prop_prefix, 
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess = num_inputs_per_python_subprocess))




@click.command('mem')
@click.option('--num-desc', '-nx', type=click.INT, help='number of descriptors', multiple=True)
@click.option('--num-der', '-ndx', type=click.INT, help='number of derrivatives of descriptors', multiple=True)
@click.option('--lmax', '-l', type=click.INT, help='l_max')
@click.option('--nmax', '-n', type=click.INT, help='n_max')
@click.option('--n-elements', '-el', type=click.INT, help='number of  elements')
@click.option('--n-prop', '-p', type=click.INT, multiple=True, help="number of target properties")
@click.option('--n-sparse', '-sp', type=click.INT, multiple=True, )
def estimate_mem(num_desc, num_der, lmax, nmax, n_elements, n_prop, n_sparse):

    GB = 1024**3

    # descriptors and gradients
    num_desc = np.sum(num_desc)
    num_der = np.sum(num_der)
    desc_dim = (lmax + 1) * (nmax * n_elements + 1) * (nmax * n_elements) / 2 
    mem_desc = (num_desc +  num_der) * desc_dim * 8 / GB # Gbytes

    # covariance matrices
    # mem_cov = np.sum(n_prop) * n_sparse * 8 / GB

    print(f'estimated memory: {mem_desc:.1f} GB')

@click.command('dimer')
@click.option('--gap-fname')
@click.option('--at-gaussian-weight', help='key for atoms.array to use for weighting soap atom gaussians with')
@click.option('--elements', '-el', help='elements for dimer curves', multiple=True)
@click.option('--out-prefix', '-p', help='prefix to save xyzs with potential to')
@click.option("--ref-isolated-at-fn")
@click.option('--isolated-at-prop-prefix', default='dft_', show_default=True, help="prefix to read from isolated atoms fname")
@click.option('--pred-prop-prefix')
def gap_dimer(gap_fname, at_gaussian_weight, elements, out_prefix, ref_isolated_at_fn, isolated_at_prop_prefix, pred_prop_prefix):
    import util.calculators.gap
    gap_calc = util.calculators.gap.at_wt_gap_calc(gap_fname, at_gaussian_weight)
    ref_isolated_ats = read(ref_isolated_at_fn, ":")
    dimer.do_dimers(
        pred_calc=gap_calc, 
        species=elements,
        out_prefix=out_prefix,
        pred_prop_prefix=pred_prop_prefix,
        ref_isolated_ats=ref_isolated_ats,
        isolated_at_prop_prefix=isolated_at_prop_prefix)


