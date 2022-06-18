import click
from util.plot import mace_loss 


@click.command("plot-loss")
@click.option('--fig-name', '-n', default="train_summary.png", show_default=True, help='filename/prefix to save figure to')
@click.option('--skip-first-n', '-s', type=click.INT, help='how many epochs to skip and not plot')
@click.option('--x-log-scale', is_flag=True, help="plot x in log scale")
@click.argument("in-fnames", nargs=-1)
def plot_loss(fig_name, in_fnames, skip_first_n, x_log_scale):
    mace_loss.plot_mace_train_summary(in_fnames=in_fnames, fig_name=fig_name, skip_first_n=skip_first_n, x_log_scale=x_log_scale)