import click
from pathlib import Path
from util.single_use import md_test
from wfl.configset import ConfigSet, OutputSpec
from ase.io import read, write
from wfl.generate import md
from util import configs

try:
    import ace
except ModuleNotFoundError:
    pass


@click.command("grab-first")
@click.option("--input-fname", '-i')
@click.option("--output-fname", '-o')
@click.option("--labels", '-l', multiple=True)
def grab_first(input_fname, output_fname, labels):
    ats = read(input_fname, ":")
    all_trajs = configs.into_dict_of_labels(ats, "graph_name")
    ats_out = [traj[0] for name, traj in all_trajs.items()]
    if len(labels) != 0:
        ats_out = [at for at in ats_out if at.info["graph_name"] in labels]
    write(output_fname, ats_out)


@click.command("md-test")
@click.option("--sub-template")
@click.option("--input-fname")
@click.option('--aces-dir')
@click.option('--ace-fname', help='alternative to "aces-dir"')
@click.option('--output-dir', default='md_trajs')
@click.option('--temps', '-t', type=click.FLOAT, multiple=True, default=[300, 500, 800])
def test_aces(sub_template, input_fname, aces_dir, ace_fname, output_dir, temps):
    md_test.main(sub_template, input_fname, aces_dir, ace_fname, temps)


@click.command('md')
@click.option('--ace-fname', '-a', help='json for ace')
@click.option('--xyz', '-x', help='xyz with structure')
@click.option('--temp', '-t', type=click.FLOAT, help='temp to run md at')
@click.option('--output', '-o')
@click.option("--pred-prop-prefix", '-p')
@click.option("--steps", type=click.INT, default=1000000)
@click.option("--sampling-interval", type=click.INT, default=500)
def run_md(ace_fname, xyz, temp, output, pred_prop_prefix, steps, sampling_interval):

    at = read(xyz, ":")
    assert len(at) == 1

    ci = ConfigSet(input_files=xyz)
    co = OutputSpec(
        output_files=output, 
        force=True, 
        all_or_none=True,
        set_tags={
            "md_temp": temp,
            "ace_name": Path(ace_fname).name 
        })

    traj_fname = output.replace(".out.", ".traj.")

    # calc = ace.ACECalculator(jsonpath=ace_fname, ACE_version=1)
    from pyjulip import ACE1
    calc = ACE1(ace_fname)

    md_params = {
        "steps": steps,
        "dt": 0.5,  # fs
        "temperature": temp,  # K
        "temperature_tau": 500,  # fs, somewhat quicker than recommended (???)
        "traj_step_interval": sampling_interval,
        "results_prefix": pred_prop_prefix,
        "reuse_momenta": True}

    md.sample(
        inputs=ci,
        outputs=co,
        calculator=calc,
        verbose=True,
        num_python_subprocesses=None,
        traj_fname=traj_fname,
        **md_params)


