import util
from pathlib import Path
import subprocess


def plot_ace_2b(ace_fname, plot_type, cc_in=None, ch_in=None, hh_in=None):
    script_path = Path(util.__file__).parent / "scripts/2b.jl"
    command = f"julia {script_path} -p {ace_fname} -t {plot_type}"
    if cc_in is not None:
        command += f" --cc-in {cc_in}"
    if ch_in is not None:
        command += f" --ch-in {ch_in}"
    if hh_in is not None:
        command += f" --hh-in {hh_in}"

    subprocess.run(command, shell=True)