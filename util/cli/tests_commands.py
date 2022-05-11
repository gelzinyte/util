import click
from ase.io import read
from pathlib import Path
from util.calculators import pyjulip_ace
import util.md.test
from expyre.resources import Resources
import wfl.pipeline.utils



@click.command("ace-md")
@click.option('--input-fname', '-i')
@click.option('--ace-fname', '-a')
@click.option('--steps', type=click.INT)
@click.option('--sample-interval', type=click.INT)
@click.option('--temp', '-t', type=click.INT)
@click.option('--pred-prop-prefix', '-p', default="ace_", show_default=True)
@click.option('--wdir', default='md_test', show_default=True)
@click.option('--info-label', default='graph_name', show_default=True)
def run_md(input_fname, ace_fname, temp, wdir, info_label, steps, sample_interval, pred_prop_prefix):

    calc = (pyjulip_ace, [ace_fname], {})
    ats = read(input_fname, ":")
    wdir = Path(wdir) / str(int(temp))

    util.md.test.run(
        workdir_root=wdir,
        in_ats=ats, 
        temp=temp,
        calc=calc,
        info_label=info_label, 
        steps=steps, 
        sampling_interval=sample_interval,
        pred_prop_prefix=pred_prop_prefix 
        )


