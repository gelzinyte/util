import click
import warnings
import logging

@click.group('util')
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def cli(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    # ignore calculator writing warnings
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.extxyz")

    warnings.filterwarnings("ignore", category=FutureWarning,
                            module="ase.calculators.calculator")

    if verbose:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')

@cli.group("gap")
def subcli_gap():
    pass
from util.cli.gap_commands import estimate_mem, gap_dimer, eval_gap
subcli_gap.add_command(estimate_mem)
subcli_gap.add_command(gap_dimer)
subcli_gap.add_command(eval_gap)


@cli.group("mace")
def subcli_mace():
    pass
from util.cli.mace_commands import convert_to_cpu, eval_mace
subcli_mace.add_command(convert_to_cpu)
subcli_mace.add_command(eval_mace)

@cli.group("misc")
@click.pass_context
def subcli_misc(ctx):
    pass
from util.cli.single_use_commands import run_md, test_aces, grab_first
subcli_misc.add_command(run_md)
subcli_misc.add_command(test_aces)
subcli_misc.add_command(grab_first)



@cli.group("blobs")
@click.pass_context
def subcli_blob(ctx):
    """density plotting and the like"""
    pass
from util.cli.blob_commands import vmd_plots, integrate_densities, combine_sidewise, \
    add_titles, combine_all, combine_int_dens, symmetrise
subcli_blob.add_command(vmd_plots)
subcli_blob.add_command(integrate_densities)
subcli_blob.add_command(combine_sidewise)
subcli_blob.add_command(add_titles)
subcli_blob.add_command(combine_all)
subcli_blob.add_command(combine_int_dens)
subcli_blob.add_command(symmetrise)


@cli.group('ip')
@click.pass_context
def subcli_ip(ctx):
    """Stuff for making or calling gaps"""
    pass
from util.cli.ip_commands import evaluate_ip, fit #eval_h, fit
subcli_ip.add_command(evaluate_ip)
# subcli_ip.add_command(eval_h)
subcli_ip.add_command(fit)



@cli.group("bde")
@click.pass_context
def subcli_bde(ctx):
    """related to handling bond dissociation energies"""
    pass
from util.cli.bde_commands import do_dissociate_h, assign_bdes, generate_bdes, \
    bde_table
subcli_bde.add_command(do_dissociate_h)
subcli_bde.add_command(assign_bdes)
subcli_bde.add_command(generate_bdes)
subcli_bde.add_command(bde_table)



@cli.group("plot")
def subcli_plot():
    """Plots and tables for analysing results"""
    pass
from util.cli.plot_commands import plot_dataset, plot_error_table, scatter,  \
                distance_autocorrelation, plot_mols, plot_dimer, dissociate, \
                ace_2b, plot_ard_scores, md_test_summary, plot_quick_dimer, plot_mace_loss
subcli_plot.add_command(plot_dataset)
subcli_plot.add_command(plot_error_table)
subcli_plot.add_command(scatter)
subcli_plot.add_command(distance_autocorrelation)
subcli_plot.add_command(plot_mols)
subcli_plot.add_command(plot_dimer)
subcli_plot.add_command(dissociate)
subcli_plot.add_command(ace_2b)
subcli_plot.add_command(plot_ard_scores)
subcli_plot.add_command(md_test_summary)
subcli_plot.add_command(plot_mace_loss)
subcli_plot.add_command(plot_quick_dimer)


@cli.group("track")
def subcli_track():
    """Tracking memory"""
    pass
from util.cli.track_commands import memory
subcli_track.add_command(memory)



@cli.group("data")
def subcli_data():
    """gathering and otherwise workign with data"""
    pass
from util.cli.data_commands import get_smiles_from_zinc
subcli_data.add_command(get_smiles_from_zinc)


@cli.group("configs")
def subcli_configs():
    """operations on atomic configurations"""
    pass
from util.cli.config_commands import assign_differences, \
    smiles_to_molecules_and_rads, distribute_configs, \
    gather_configs, info_to_numbers, hash_structures, \
    remove_old_calc_results, check_geometry, configs_summary, color_atoms_by_array
# smiles_to_molecules
subcli_configs.add_command(assign_differences)
subcli_configs.add_command(smiles_to_molecules_and_rads)
# subcli_configs.add_command(smiles_to_molecules)
subcli_configs.add_command(distribute_configs)
subcli_configs.add_command(gather_configs)
subcli_configs.add_command(info_to_numbers)
subcli_configs.add_command(hash_structures)
subcli_configs.add_command(remove_old_calc_results)
subcli_configs.add_command(check_geometry)
subcli_configs.add_command(configs_summary)
subcli_configs.add_command(color_atoms_by_array)


@cli.group("calc")
def subcli_calc():
    """Calculating things (apart from gap)"""
    pass
from util.cli.calc_commands import xtb_normal_modes, recalc_dftb_ef,  \
    calc_xtb2_ef, evaluate_diff_calc, calculate_descriptor, \
    evaluate_ace
subcli_calc.add_command(xtb_normal_modes)
subcli_calc.add_command(recalc_dftb_ef)
subcli_calc.add_command(calc_xtb2_ef)
subcli_calc.add_command(evaluate_diff_calc)
subcli_calc.add_command(calculate_descriptor)
subcli_calc.add_command(evaluate_ace)


@cli.group("jobs")
def subcli_jobs():
    """submiting jobs"""
    pass
from util.cli.jobs_commands import sub_from_pattern
subcli_jobs.add_command(sub_from_pattern)


@cli.group("qm")
def subcli_qm():
    """stuff to do with quantum mechanics codes"""
    pass
from util.cli.qm_commands import read_orca_stuff, calculate, \
    plot_scf_convergence_graph
subcli_qm.add_command(read_orca_stuff)
subcli_qm.add_command(calculate)
subcli_qm.add_command(plot_scf_convergence_graph)



if __name__=="__main__":
    cli()
