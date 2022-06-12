import click
from util.plot import mace_loss


@click.command("plot-loss")
@click.option('--fig-name', default="train_summary.png", show_default=True, help='filename/prefix to save figure to')
@click.argument("in-fnames", nargs=-1)
def plot_loss(fig_name, in_fnames):
    mace_loss.plot_loss(fig_name, in_fnames)