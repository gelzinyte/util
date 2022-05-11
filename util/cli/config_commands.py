import click
import util
from wfl.configset import OutputSpec
from ase.io import read, write
from util import configs    
from util import qm

# @click.command("check-geometry")
# @click.option("--inputs", "-i")
# @click.option("--outputs", "-o")
# def check_geometry(inputs, outputs):
#     ats = read(inputs, ":")
#     results = configs.filter_insane_geometries(ats, mark_elements=True)
#     write(outputs, results["bad_geometries"])

@click.command("color-by-array")
@click.option('--input', '-i', help='intput xyz')
@click.option('--output', '-o', help='output xyz')
@click.option('--key', '-k', help='key for at.arrays dict')
@click.option('--cmap', '-c', default='seismic', show_default=True)
@click.option('--vmax', type=click.FLOAT, help='max (min is negative of vmax) value to set to the colormap')
def color_atoms_by_array(input, output, key, cmap, vmax):
    ats = read(input, ":")
    vmin = vmax * -1
    ats = [qm.color_by_pop(ats=at, pop=key, cmap=cmap, vmin=vmin, vmax=vmax) for at in ats]
    write(output, ats)

@click.command("remove-calc-results")
@click.argument("input-fname")
@click.option('--output-fname', '-o')
def remove_old_calc_results(input_fname, output_fname):
    """removes entries that have "energy" or "forces" in the label"""
    ats = read(input_fname, ':')
    for at in ats:
        util.remove_energy_force_containing_entries(at)
    write(output_fname, ats, write_results=False)



@click.command('hash')
@click.argument('input-fname')
@click.option('--output-fname', '-o')
@click.option('--prefix', '-p', help='prefix for info entries')
def hash_structures(input_fname, output_fname, prefix):
    """assigns positions/numbers hash to all structures in the file"""

    atoms = read(input_fname, ':')
    for at in atoms:
        at.info[f'{prefix}hash'] = configs.hash_atoms(at)

    write(output_fname, atoms)



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
         help="how many radicals create for each molecule. Defaults to all. "
              "Returns only molecules if zero.")
@click.option("--smiles-col", default="smiles")
@click.option("--name-col", default="zinc_id")
def smiles_to_molecules_and_rads(smiles_csv, repeats, output_fname,
                                 num_rads_per_mol, smiles_col, name_col):

    from util.iterations import tools as it

    outputs = OutputSpec(output_files=output_fname)

    it.make_structures(smiles_csv, num_smi_repeat=repeats,
                       outputs=outputs, num_rads_per_mol=num_rads_per_mol,
                       smiles_col=smiles_col, name_col=name_col)

#
# @click.command('csv-to-mols')
# @click.argument('smiles-csv')
# @click.option('--num-repeats', '-n', default=1, type=click.INT)
# @click.option('--output-fname', '-o')
# @click.option("--smiles-col", default="smiles", help="column name in csv")
# @click.option("--name-col", default='zinc_id')
# def smiles_to_molecules(smiles_csv, num_repeats, output_fname, smiles_col,
                        # name_col):
#
#     from util import configs
#
#     outputs = OutputSpec(output_files=output_fname)
#     configs.smiles_csv_to_molecules(smiles_csv,
#         outputs, repeat=num_repeats, smiles_col=smiles_col, name_col=name_col)

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
@click.option('--num-tasks', '-n', type=click.INT,
              help='number of files to gather configs from')
@click.option('--prefix', '-p', default='out_', show_default=True,
              help='prefix for individual files')
@click.option("--dir-prefix", default='job_', show_default=True)
def gather_configs(out_fname, num_tasks, prefix, dir_prefix):
    from util import configs
    configs.collect_configs(out_fname=out_fname, num_tasks=num_tasks,
                              batch_out_fname_prefix=prefix,
                            dir_prefix=dir_prefix)


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


@click.command('check-geometry')
@click.option('--input', "-i")
@click.option('--output', '-o')
def check_geometry(input, output):
    ats = read(input, ":")
    for at in ats:
        out = configs.check_geometry(at, mult=1.2, mark_elements=True, skin=0)
        if out == False:
            print(at.info)
    write(output, ats)


@click.command("summary")
@click.argument("input")
@click.option("--info", '-i', default='config_type', help="name of atoms.info entry to group compounds by")
def configs_summary(input, info):
    ats = read(input, ":")
    dd = configs.into_dict_of_labels(ats, info)
    for key, entries in dd.items():
        print(f"{key}: {len(entries)}")