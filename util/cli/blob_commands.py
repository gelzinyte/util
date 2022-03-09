from util import blobs
from util.blobs import combine_plots
from pathlib import Path
import subprocess
import click
import util


@click.command("vmd-plots")
@click.option("--density-dir", help="directory with all the density files")
def vmd_plots(density_dir):
    blobs.vmd.main(density_dir)


@click.command("integrate")
@click.option("--density-dir", help="directory with all the density files")
def integrate_densities(density_dir):
    script_path = Path(util.__file__).parent /  "blobs/scripts/integrate_densities.jl"
    command = f"julia {script_path} --density-dir {density_dir} "
    subprocess.run(command, shell=True)

@click.command("combine-sideways")
@click.option("--cross-section-dir", help="dir with cross-section figs")
@click.option("--isosurface-dir")
@click.option("--integrated-dens-dir")
@click.option("--output-dir")
def combine_sidewise(cross_section_dir, isosurface_dir, integrated_dens_dir, output_dir):
    combine_plots.sideways(cross_section_dir=cross_section_dir, isosurface_dir=isosurface_dir, integrated_dens_dir=integrated_dens_dir, output_dir=output_dir)


@click.command("add-titles")
@click.option("--input-dir")
@click.option("--output-dir")
def add_titles(input_dir, output_dir):
    combine_plots.add_titles(input_dir, output_dir)


@click.command("combine-all")
@click.option("--input-dir")
@click.option("--output-dir")
def combine_all(input_dir, output_dir):
    combine_plots.all(input_dir, output_dir)

@click.command("combine-int-only")
@click.option("--input-dir")
@click.option("--output-dir")
def combine_int_dens(input_dir, output_dir):
    combine_plots.int_dens_only(input_dir, output_dir)




@click.command("single-per-config")
@click.option("--xyz-dir")
@click.option("--orca-template")
@click.option("--sub-template")
@click.option("--output-dir")
@click.option("--submit", is_flag=True)
def symmetrise(xyz_dir, orca_template, sub_template, output_dir, submit):
    from util import cc_results    

    cc_results.inp_and_template_per_geometry(xyz_dir, orca_template, sub_template, output_dir, submit)

