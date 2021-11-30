import click
import util
from wfl.configset import ConfigSet_out
from ase.io import read, write

@click.command('assign-diff')
@click.argument('input-fn')
@click.option('--output-fn', '-o')
@click.option('--prop-prefix-1', '-p1', help='property prefix for first set of values')
@click.option('--prop-prefix-2', '-p2')
def assign_differences(input_fn, output_fn, prop_prefix_1, prop_prefix_2):

    ats = read(input_fn, ':')
    for at in ats:
        util.assign_differences(at, prop_prefix_1, prop_prefix_2)

    write(output_fn, ats)


@click.command('csv-to-mols-rads')
@click.argument('smiles-csv')
@click.option('--output-fname', '-o')
@click.option("--repeats", '-n', type=click.INT, default=1, help='number of '
                                                           'repeats for each smiles')
@click.option("--num-rads-per-mol", type=click.INT,
         help="how many radicals create for each molecule. Defaults to all.")
@click.option("--smiles-col", default="smiles")
@click.option("--name-col", default="zinc_id")
def smiles_to_molecules_and_rads(smiles_csv, repeats, output_fname,
                                 num_rads_per_mol, smiles_col, name_col):

    from util.iterations import tools as it

    outputs = ConfigSet_out(output_files=output_fname)

    it.make_structures(smiles_csv, iter_no=None, num_smi_repeat=repeats,
                       outputs=outputs, num_rads_per_mol=num_rads_per_mol,
                       smiles_col=smiles_col, name_col=name_col)


@click.command('csv-to-mols')
@click.argument('smiles-csv')
@click.option('--num-repeats', '-n', default=1, type=click.INT)
@click.option('--output-fname', '-o')
@click.option("--smiles-col", default="smiles", help="column name in csv")
@click.option("--name-col", default='zinc_id')
def smiles_to_molecules(smiles_csv, num_repeats, output_fname, smiles_col,
                        name_col):

    from util import configs

    outputs = ConfigSet_out(output_file=output_fname)
    configs.smiles_csv_to_molecules(smiles_csv,
        outputs, repeat=num_repeats, smiles_col=smiles_col, name_col=name_col)

@click.command('distribute')
@click.argument('in-fname')
@click.option('--num-tasks', '-n', default=8, type=click.INT, help='number of files to distribute across')
@click.option('--prefix', '-p', default='in_', help='prefix for individual files')
@click.option('--dir-prefix', help='prefix for directory to put file into')
def distribute_configs(in_fname, num_tasks, prefix, dir_prefix):

    from util import configs

    configs.batch_configs(in_fname=in_fname, num_tasks=num_tasks,
                              batch_in_fname_prefix=prefix, dir_prefix=dir_prefix)



@click.command('gather')
@click.option('--out-fname', '-o', help='output for gathered configs')
@click.option('--num-tasks', '-n', default=8, type=click.INT,
              help='number of files to gather configs from')
@click.option('--prefix', '-p', default='out_',
              help='prefix for individual files')
def gather_configs(out_fname, num_tasks, prefix):
    configs.collect_configs(out_fname=out_fname, num_tasks=num_tasks,
                              batch_out_fname_prefix=prefix)


@click.command('info-to-no')
@click.argument('fname_in')
@click.option('--info-key', '-i', help='atosm.info key to nubmer')
@click.option('--output', '-o', help='output filename')
def info_to_numbers(fname_in, info_key, output):

    ats = read(fname_in, ':')

    entries = list(set([at.info[info_key] for at in ats]))
    entries_dict = {}
    for idx, entry in enumerate(entries):
        entries_dict[entry] = idx

    for at in ats:
        at.info[info_key + '_no'] = entries_dict[at.info[info_key]]

    print(entries_dict)

    write(output, ats)
