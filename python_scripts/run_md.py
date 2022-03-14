from ase.io import read, write
import os
import subprocess
from pathlib import Path

sub_template = Path("/data/eg475/heap_of_ch/iterative_fits/2_mols_only/1.1_mols_only/md_runs/sub_template.sh")

input_configs = read("/data/eg475/heap_of_ch/iterative_fits/2_mols_only/prep_configs_for_md/validation.rdkit.xtb2_md.dft.mols_only.for_md.with_momenta.10_configs.xyz", ":")

aces_dir = Path("/data/eg475/heap_of_ch/iterative_fits/2_mols_only/1.1_mols_only/md_runs/aces_for_md")

outputs_dir = Path("md_trajs")
outputs_dir.mkdir(exist_ok=True)

temps = [300, 500, 800]

home_dir = os.getcwd()

for temp in temps:

    for ace_fname in aces_dir.iterdir():
        if "ace" not in ace_fname.name:
            continue

        ace_label = ace_fname.stem
        ace_fname = ace_fname.resolve()

        for at in input_configs:
            at_name = at.info["graph_name"]
            at_dir = outputs_dir / at_name
            at_dir.mkdir(exist_ok=True)

            temp_dir = at_dir / temp
            temp_dir.mkdir(exist_ok=True)

            job_dir = temp_dir / ace_label 
            job_dir.mkdir(exist_ok=True)

            label = f"{at_name}.{temp}.{ace_fname}"

            at_fname = job_dir /  (at_name + '.xyz')
            out_fname = job_dir / (at_name + ".out.xyz")

            write(at_fname, at)

            command = f"util misc md -a {ace_fname} -x {at_fname} -t {temp} -o {out_fname} -p {ace_label}"

            os.chdir(job_dir)

            with open(sub_template, "r") as f:
                sub_text = f.read()

            sub_text = sub_text.replace("<command>", command)
            sub_text = sub_text.replace("<label>", label)
            
            with open(sub_template, "w") as f:
                f.write(sub_text)

            subprocess.run("qsub sub.sh", shell=True)

            os.chdir(home_dir)

