import click
from util import plot
from util.plot import rmse_scatter_evaled
import os
from quippy.potential import Potential
from util import bde
import util
from ase.io import read, write
from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import orca
from ase.io.extxyz import key_val_str_to_dict


@click.group('util')
@click.pass_context
def cli(ctx):
    pass


@cli.group("bde")
@click.pass_context
def subcli_bde(ctx):
    pass


@cli.group('plot')
@click.pass_context
def subcli_plot(ctx):
    pass


@subcli_bde.command("multi")
@click.option('--dft_dir', '-d', type=click.Path())
@click.option('--gap_dir', '-g', type=click.Path())
@click.option('--gap_xml_fname', type=click.Path())
@click.option('--start_dir', '-s', type=click.Path())
@click.option('--dft_only', is_flag=True)
@click.option('--precision', '-p', type=int, default=3)
def multi_bde_summaries(dft_dir, gap_dir=None, gap_xml_fname=None, start_dir=None, dft_only=False,
                        precision=3):
    if gap_xml_fname is not None:
        calculator = (Potential, [], {'param_filename': gap_xml_fname})
    else:
        calculator = None

    bde.multi_bde_summaries(dft_dir=dft_dir, gap_dir=gap_dir, calculator=calculator,
                            start_dir=start_dir, dft_only=dft_only, precision=precision)


@subcli_bde.command('single')
@click.option('--dft_fname', '-d', type=click.Path(exists=True), help='dft reference fname')
@click.option('--gap_fname', '-g', type=click.Path(), help='gap bde filename')
@click.option('--gap_xml_fname', '-x', type=click.Path(exists=True),
              help='Gap.xml for optimising structures')
@click.option('--start_fname', '-s', type=click.Path(exists=True),
              help='structures to optimise with gap')
@click.option('--precision', '-p', type=click.INT, default=3,
              help='how many digits to print in the table')
@click.option('--printing', type=click.BOOL, default=True, help='whether to print the table')
def single_bde_summary(dft_fname, gap_fname=None, gap_xml_fname=None, start_fname=None, precision=3,
                       printing=True):
    """ BDE for single file"""

    if gap_xml_fname is not None:
        calculator = (Potential, [], {'param_filename': gap_xml_fname})
    else:
        calculator = None

    bde.bde_summary(dft_fname=dft_fname, gap_fname=gap_fname, calculator=calculator,
                    start_fname=start_fname, precision=precision, printing=printing)


@subcli_bde.command("bar-plot")
@click.option('--dft_dir', '-d', type=click.Path())
@click.option('--gap_dir', '-g', type=click.Path())
@click.option('--start_dir', '-s', type=click.Path())
@click.option('--gap_xml_fname', type=click.Path())
@click.option('--plot_title', '-t', type=click.STRING, default='bde_bar_plot')
@click.option('--output_dir', type=click.Path(), default='pictures')
def bde_bar_plot(dft_dir, gap_dir, start_dir=None, gap_xml_fname=None,
                 plot_title='bde_bar_plot', output_dir='pictures'):
    if gap_dir is not None:
        if not os.path.isdir(gap_dir):
            os.makedirs(gap_dir)

    if gap_xml_fname is not None:
        calculator = (Potential, [], {'param_filename': gap_xml_fname})
    else:
        calculator = None

    dft_basenames = util.natural_sort(os.listdir(dft_dir))
    dft_fnames = [os.path.join(dft_dir, fname) for fname in dft_basenames]
    gap_fnames = [os.path.join(gap_dir, basename.replace('optimised', 'gap_optimised')) for
                  basename in dft_basenames]

    if start_dir is not None:
        start_fnames = [os.path.join(start_dir, basename.replace('optimised', 'non_optimised')) for
                        basename in dft_basenames]
    else:
        start_fnames = [None for _ in dft_fnames]

    bde.bde_bar_plot(gap_fnames=gap_fnames, dft_fnames=dft_fnames, plot_title=plot_title,
                     start_fnames=start_fnames, calculator=calculator, output_dir=output_dir)


@subcli_bde.command("scatter")
@click.option('--dft_dir', '-d', type=click.Path())
@click.option('--gap_dir', '-g', type=click.Path())
@click.option('--start_dir', '-s', type=click.Path())
@click.option('--gap_xml_fname', type=click.Path())
@click.option('--plot_title', '-t', type=click.STRING, default='bde_bar_plot')
@click.option('--output_dir', type=click.Path(), default='pictures')
def bde_bar_plot(dft_dir, gap_dir, start_dir=None, gap_xml_fname=None,
                 plot_title='bde_bar_plot', output_dir='pictures'):
    if gap_dir is not None:
        if not os.path.isdir(gap_dir):
            os.makedirs(gap_dir)

    if gap_xml_fname is not None:
        calculator = (Potential, [], {'param_filename': gap_xml_fname})
    else:
        calculator = None

    dft_basenames = util.natural_sort(os.listdir(dft_dir))
    dft_fnames = [os.path.join(dft_dir, fname) for fname in dft_basenames]
    gap_fnames = [os.path.join(gap_dir, basename.replace('optimised', 'gap_optimised')) for
                  basename in dft_basenames]

    if start_dir is not None:
        start_fnames = [os.path.join(start_dir, basename.replace('optimised', 'non_optimised')) for
                        basename in dft_basenames]
    else:
        start_fnames = [None for _ in dft_fnames]

    bde.scatter_plot(gap_fnames=gap_fnames, dft_fnames=dft_fnames, plot_title=plot_title,
                     start_fnames=start_fnames, calculator=calculator, output_dir=output_dir)


@subcli_bde.command("from-mols")
@click.argument('mols_fname')
@click.option('--starts-dir', '-s', help='where to put start files')
@click.option('--dft-dir', '-d', help='optimised files destination')
@click.option('--calc-kwargs', help='orca kw arguments for optimisation')
@click.option('--h_to_remove', '-h', type=click.INT, help='h_idx to remove')
def multi_bdes(mols_fname, starts_dir, dft_dir, calc_kwargs, h_to_remove):
    """makes radicals and optimises them """

    start_fname_end = '_non_optimised.xyz'

    for dir_name in [starts_dir, dft_dir]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    atoms = read(mols_fname, ':')
    start_fnames = []
    dft_fnames = []
    for atom in atoms:

        start_basename = os.path.join(starts_dir, atom.info['config_type'])
        for idx in range(1, 10):
            if os.path.exists(start_basename + start_fname_end):
                start_basename += '_idx'
            else:
                start_fname = start_basename + start_fname_end
                break
        else:
            raise RuntimeError(f'All attempted fnames were taken, last one: {start_basename}')

        start_fnames.append(start_fname)
        dft_fname = os.path.basename(start_fname).replace('_non', '')
        dft_fnames.append(os.path.join(dft_dir, dft_fname))

        mol = atom.copy()
        mol.info['config_type'] += '_mol'
        rad = atom.copy()
        rad.info['config_type'] += f'_rad{h_to_remove}'
        del rad[h_to_remove]

        write(start_fname, [mol, rad])

    output_dict = {}
    for st, dft in zip(start_fnames, dft_fnames):
        output_dict[st] = dft

    # do optimisation
    inputs = ConfigSet_in(input_files=start_fnames)
    outputs = ConfigSet_out(output_files=output_dict)

    calc_kwargs = key_val_str_to_dict(calc_kwargs)
    calc_kwargs['orca_kwargs'] = key_val_str_to_dict(calc_kwargs['orca_kwargs'])
    if 'task' not in calc_kwargs['orca_kwargs'].keys():
        calc_kwargs['orca_kwargs']['task'] = 'optimise'

    print(f'orca_parameters: {calc_kwargs}')

    orca.evaluate(inputs=inputs, outputs=outputs, **calc_kwargs)






@subcli_plot.command('error-scatter')
@click.option('--ref_energy_name', '-re', type=str)
@click.option('--pred_energy_name', '-pe', type=str)
@click.option('--ref_force_name', '-rf', type=str)
@click.option('--pred_force_name', '-pf', type=str)
@click.option('--evaluated_train_fname', '-te', type=click.Path(exists=True))
@click.option('--evaluated_test_fname', '-tr', type=click.Path(exists=True))
@click.option('--output_dir', default='pictures', show_default=True, type=click.Path(),
              help='directory for figures. Create if not-existent')
@click.option('--prefix', '-p', help='prefix to label plots')
@click.option('--by_config_type', is_flag=True,
              help='if structures should be coloured by config_type in plots')
@click.option('--force_by_element', default=True, type=bool,
              help='whether to evaluate force on each element separately')
def make_plots(ref_energy_name, pred_energy_name, ref_force_name, pred_force_name,
               evaluated_train_fname, evaluated_test_fname,
               output_dir, prefix, by_config_type, force_by_element):
    """Makes energy and force scatter plots and dimer curves"""

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if evaluated_test_fname is None and by_config_type != False:
        by_config_type = True

    print('Scatter plotting')
    rmse_scatter_evaled.make_scatter_plots_from_evaluated_atoms(ref_energy_name=ref_energy_name,
                                                                pred_energy_name=pred_energy_name,
                                                                ref_force_name=ref_force_name,
                                                                pred_force_name=pred_force_name,
                                                                evaluated_train_fname=evaluated_train_fname,
                                                                evaluated_test_fname=evaluated_test_fname,
                                                                output_dir=output_dir,
                                                                prefix=prefix,
                                                                by_config_type=by_config_type,
                                                                force_by_element=force_by_element)
