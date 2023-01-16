import click
import os
from pathlib import Path
from util.single_use import md_test
from wfl.configset import ConfigSet, OutputSpec
from ase.io import read, write
from wfl.generate import md
from util import configs
import logging
from util.configs import check_geometry

logger = logging.getLogger(__name__)

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


@click.command('no-wfl-md')
@click.option('--mace-fname', '-m')
@click.option('--in-fname', '-x')
@click.option('--in-dir')
@click.option('--temp', '-t', type=click.FLOAT)
@click.option('--output-dir')
@click.option('--pred-prop-prefix', '-p', default='mace_')
@click.option('--steps', type=click.INT)
@click.option('--sampling-interval', type=click.INT)
def run_md_no_wfl_autopara(mace_fname, in_dir, in_fname, temp, output_dir,  pred_prop_prefix, 
    steps, sampling_interval):

    logger.info(f"\n mace: {mace_fname} \n in_dir: {in_dir} \n in_fname: {in_fname} \n temp {temp} K \n output {output_dir}\n pred-prop-prefix {pred_prop_prefix}\n steps {steps}\n sampling interval {sampling_interval}")
    
    from mace.calculators.mace import MACECalculator 
    calc = MACECalculator(model_path=mace_fname, default_dtype="float64", device="cpu")

    atoms = read(Path(in_dir) / in_fname, ":")
    assert len(atoms) == 1
    atoms[0].info[f'md_start_hash'] = configs.hash_atoms(atoms[0])

    from wfl.generate.md import _sample_autopara_wrappable as run_md

    output_dir = Path(output_dir)
    in_fname = Path(in_fname)

    ok_at_output_dir = output_dir / "ok_output" / str(in_fname.parent) 
    ok_at_output_dir.mkdir(exist_ok=True)
    ok_at_fn = ok_at_output_dir / str(in_fname.name).replace(".xyz", ".sample.xyz") 

    bad_traj_output_dir = output_dir / "failed_traj" / str(in_fname.parent) 
    bad_traj_output_dir.mkdir(exist_ok=True)
    bad_traj_fn = bad_traj_output_dir / str(in_fname.name).replace('.xyz', ".bad_traj.xyz")

    running_traj_dir = output_dir / "running_traj"  / str(in_fname.parent)
    running_traj_dir.mkdir(exist_ok=True)
    running_traj_fn = running_traj_dir / str(in_fname.name).replace('.xyz', ".in_progress_traj.xyz")


    md_params = {
        "steps": steps,
        "dt": 0.5,  # fs
        "temperature": temp,  # K
        "temperature_tau": 500,  # fs, somewhat quicker than recommended (???)
        "traj_step_interval": sampling_interval,
        "results_prefix": pred_prop_prefix,
        "update_config_type": False}

    traj = run_md(
        atoms = atoms,
        calculator = calc, 
        traj_fname = running_traj_fn,
        **md_params)

    assert len(traj) == 1
    traj = traj[0]

    traj_ok = [check_geometry(at) for at in traj]
    failed_count = sum([1 for check in traj_ok if check is False])

    if failed_count < 5: 
        write(ok_at_fn, traj[-1])
    else:
        write(bad_traj_fn, traj)
    os.remove(running_traj_fn)
