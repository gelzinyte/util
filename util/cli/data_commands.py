import click


@click.command("zinc")
@click.argument("wget_fname")
@click.option('--output-label', '-l',
              help='label for all the output files ')
@click.option("--wdir", default='wdir')
def get_smiles_from_zinc(wget_fname, output_label, wdir):
    from util.single_use import zinc
    zinc.main(wget_fname, output_label, wdir=wdir)
