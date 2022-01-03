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
                            format='%(asctime)s %(message)s')



@cli.group('gap')
@click.pass_context
def subcli_gap(ctx):
    """Stuff for making or calling gaps"""
    pass
from util.cli.gap_commands import evaluate_gap, eval_h, fit
subcli_gap.add_command(evaluate_gap)
subcli_gap.add_command(eval_h)
subcli_gap.add_command(fit)



@cli.group("bde")
@click.pass_context
def subcli_bde(ctx):
    """related to handling bond dissociation energies"""
    pass
from util.cli.bde_commands import do_dissociate_h, assign_bdes, generate_bdes
subcli_bde.add_command(do_dissociate_h)
subcli_bde.add_command(assign_bdes)
subcli_bde.add_command(generate_bdes)



@cli.group("plot")
def subcli_plot():
    """Plots and tables for analysing results"""
    pass
from util.cli.plot_commands import plot_dataset, plot_error_table, scatter,  \
                distance_autocorrelation, plot_mols, plot_dimer, dissociate
subcli_plot.add_command(plot_dataset)
subcli_plot.add_command(plot_error_table)
subcli_plot.add_command(scatter)
subcli_plot.add_command(distance_autocorrelation)
subcli_plot.add_command(plot_mols)
subcli_plot.add_command(plot_dimer)
subcli_plot.add_command(dissociate)


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
    gather_configs, info_to_numbers, hash_structures
# smiles_to_molecules
subcli_configs.add_command(assign_differences)
subcli_configs.add_command(smiles_to_molecules_and_rads)
# subcli_configs.add_command(smiles_to_molecules)
subcli_configs.add_command(distribute_configs)
subcli_configs.add_command(gather_configs)
subcli_configs.add_command(info_to_numbers)
subcli_configs.add_command(hash_structures)


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
