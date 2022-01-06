import click
import logging

from ase.io import read, write

from quippy.potential import Potential

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import generic

logger = logging.getLogger(__name__)


@click.command('dissociate-h')
@click.argument('input_fname')
@click.option('--output-fname', '-o')
@click.option('--h-idx', '-h', type=click.INT,
              help='index of hidrogen to dissociate')
@click.option('--c-idx', '-c', type=click.INT,
              help='index of carbon in C-H to dissociate')
def do_dissociate_h(input_fname, output_fname, h_idx, c_idx):
    """Increases scan for C-H bond lengths"""
    from util.bde import dissociate_h

    atoms = read(input_fname, '0')
    frames = dissociate_h.dissociate_h(atoms, h_idx, c_idx)
    write(output_fname, frames)


@click.command('assign')
@click.option('--all-evaled-atoms-fname',
              help='All atoms with appropriate energies')
@click.option('--isolated-h-fname',
              help='H with appropriate energy. Can have other isolated atoms')
@click.option('--prop-prefix', help='prop prefix to get energies with')
@click.option('--dft-prefix', help='for getting dft_opt_mol_positions_has')
@click.option('--output-fname',
              help='output filename for radicals with bond dissociation '
                   'energies')
def assign_bdes(all_evaled_atoms_fname, isolated_h_fname, prop_prefix,
                dft_prefix, output_fname):

    import util.bde.table

    all_atoms = read(all_evaled_atoms_fname, ':')

    isolated_atoms = read(isolated_h_fname, ':')
    for at in isolated_atoms:
        if len(at) == 1:
            if list(at.symbols)[0] ==  'H':
                isolated_h_energy = at.info[f'{prop_prefix}energy']
                break
    else:
        raise RuntimeError('Haven\'t found isolated_h_energy')

    all_bde_ats = util.bde.table.assign_bde_info(all_atoms=all_atoms,
                                         h_energy=isolated_h_energy,
                                         prop_prefix=prop_prefix,
                                         dft_prop_prefix=dft_prefix)

    write(output_fname, all_bde_ats)



@click.command('generate')
@click.pass_context
@click.argument('dft-bde-file')
@click.option('--ip-fname', help='gap/ace fname')
@click.option('--iso-h-fname',
              help='if not None, evaluate isolated H energy with GAP'
                   'and save to given file.')
@click.option('--output-fname-prefix', '-o',
              help='prefix for main and all working files with all ip and dft properties')
@click.option('--dft-prop-prefix', default='dft_', show_default=True,
              help='label for all dft properties')
@click.option('--ip-prop-prefix', help='gap/ace property prefix')
@click.option('--wdir', default='bde_wdir', show_default=True,
              help='working directory for all interrim files')
@click.option('--calculator-name',
              type=click.Choice(["gap", "gap_plus_xtb2", 'ace', 'xtb2']),
                    default='gap')
@click.option('--chunksize', type=click.INT, default=10,
              help='chunksize for IP optimisation and re-evaluation.')
def generate_bdes(ctx, dft_bde_file, ip_fname, iso_h_fname, output_fname_prefix,
                      dft_prop_prefix, ip_prop_prefix, wdir,
                      calculator_name, chunksize):

    from quippy.potential import Potential
    import util.bde.generate
    from util import calculators

    logging.info(f'Generating {calculator_name} bdes from {dft_bde_file} files')

    if calculator_name == 'gap':
        calculator = (Potential, [], {'param_filename':ip_fname})
    elif calculator_name == 'gap_plus_xtb2':
        calculator = (calculators.xtb_plus_gap, [],
                      {'gap_filename':ip_fname})
    elif calculator_name == 'ace':
        calculator = (util.ace_constructor, [ip_fname], {})
    elif calculator_name == 'xtb2':
        from xtb.ase.calculator import XTB
        calculator = (XTB, [], {"method":"GFN2-xTB"})



    if iso_h_fname is not None and not os.path.isfile(iso_h_fname):
        # generate isolated atom stuff
        logger.info('generating isolated atom stuff for BDE')
        util.bde.generate.ip_isolated_h(calculator=calculator,
                                         dft_prop_prefix=dft_prop_prefix,
                                         ip_prop_prefix=ip_prop_prefix,
                                         output_fname=iso_h_fname,
                                         wdir=wdir)
    else:
        logger.info('not generating isolated atoms stuff')
    #
    # generate all molecule stuff
    logger.info('generating all the BDE stuff')
    util.bde.generate.everything(calculator=calculator,
                                 dft_bde_filename=dft_bde_file,
                                 output_fname_prefix=output_fname_prefix,
                                 dft_prop_prefix=dft_prop_prefix,
                                 ip_prop_prefix=ip_prop_prefix,
                                 wdir=wdir,
                                 chunksize=chunksize)




