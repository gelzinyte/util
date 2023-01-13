import click
from util import data
from ase.io import read, write
from wfl.configset import OutputSpec


@click.command("zinc")
@click.argument("wget_fname")
@click.option('--output-label', '-l',
              help='label for all the output files ')
@click.option("--wdir", default='wdir')
@click.option("--elements", type=click.Choice(["CH", "CHO", "CHNOPS"]), help="elements to filter out")
def get_smiles_from_zinc(wget_fname, output_label, wdir, elements):
    from util import zinc
    zinc.main(wget_fname, output_label, wdir=wdir, elements=elements)

@click.command("read-ani")
@click.option('--hd5-fname', '-f')
@click.option('--label', '-l', help='label to prepend at.info["graph_name"] with')
@click.option('--elements-to-skip', '-s', multiple=True)
@click.option('--output-fn', '-o')
def read_ani(hd5_fname, label, elements_to_skip, output_fn):
    ats_out = data.read_ANI(hd5_fname, label=label, elements_to_skip=elements_to_skip)
    write(output_fn, ats_out)

