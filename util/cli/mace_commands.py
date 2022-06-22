import click
from pathlib import Path
from util import mace

@click.command('to-cpu')
@click.argument('input-fnames', nargs=-1)
def convert_to_cpu(input_fnames):
    for fn_in in input_fnames:
        fn_out = fn_in + '.cpu'
        mace.to_cpu(fn_in, fn_out)

@click.command('dir-to-cpu')
@click.argument('dir_names', nargs=-1)
def dir_convert_to_cpu(dir_names):
    for dd in dir_names: 
        dd = Path(dd)
        for fns in dd.iterdir():
            # if fns.
            pass
        

    for fn_in in input_fnames:
        fn_out = fn_in + '.cpu'
        mace.to_cpu(fn_in, fn_out)



