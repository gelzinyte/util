import os
import shutil
import subprocess
from pathlib import Path


def calculate(
    calculation_stem,
    compound_wdir,
    orca_template_fname,
    sub_template_fname,
    mult,
    fname,
    submit,
):

    make_move_calc_files(
        calculation_stem, orca_template_fname, sub_template_fname, mult, fname
    )

    if submit:
        subprocess.run("qsub sub.sh", shell=True)


def main(
    structures_dir,
    uks_orca_template_fname,
    cc_orca_template_fname,
    sub_template_fname,
    output_dir,
    task,
    submit=False,
    method=None,
    spin=None
):

    print(task)

    if spin is None:
        multiplicities = [1, 3]
        multiplicities_names = ["singlet", "triplet"]
    elif spin=="singlet":
        multiplicities = [1]
        multiplicities_names = ["singlet"]
    elif spin=="triplet":
        multiplicities = [3]
        multiplicities_names = ["triplet"]
    else:
        raise RuntimeError("wrong multiplicity")


    if method is None:
        methods = ["uks_cc-pvdz", "dlpno-ccsd_cc-pvdz"]
        template_fnames = [uks_orca_template_fname, cc_orca_template_fname]
    elif method == "uks_cc-pvdz":
        methods = [method]
        template_fnames = [uks_orca_template_fname]
    elif method == "dlpno-ccsd_cc-pvdz":
        methods = [method]
        template_fnames = [cc_orca_template_fname]
    else:
        raise RuntimeError("wrong method")


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    home_dir = os.getcwd()

    xyz_filenames = get_xyz_files_from_dir(structures_dir)

    for fname in xyz_filenames:

        compound_stem = os.path.basename(os.path.splitext(fname)[0])
        handle_dirs(compound_stem, output_dir)
        compound_wdir = os.getcwd()

        for mult, mult_name in zip(multiplicities, multiplicities_names):
            for orca_template_fname, method in zip(template_fnames, methods):

                calculation_stem = compound_stem + f".{method}.{mult_name}"
                handle_dirs(calculation_stem, compound_wdir)

                if task == "calculate":
                    calculate(
                        calculation_stem,
                        compound_wdir,
                        orca_template_fname,
                        sub_template_fname,
                        mult,
                        fname,
                        submit,
                    )

                elif task == "density_plots":
                    plot_densities(calculation_stem, method)

                os.chdir(compound_wdir)

        os.chdir(home_dir)


def plot_densities(calculation_stem, method):

    grid = (5000, 50, 50)

    plot_scf = (
        f'printf "5\n7\n4\n{grid[0]} {grid[1]} {grid[2]}\n1\n2\ny\n10\n1\n3\ny\n10\n11" | '
        f"orca_plot "
        f"{calculation_stem}.gbw -i"
    )

    plot_mdci = (
        f'printf "5\n7\n4\n{grid[0]} {grid[1]} {grid[2]}\n1\n7\ny\n10\n1\n8\ny\n10\n11" | '
        f"orca_plot "
        f"{calculation_stem}.gbw -i"
    )

    subprocess.run(plot_scf, shell=True)

    if method == "uks_cc-pvdz":
        shutil.move(
            f"{calculation_stem}.eldens.cube", f"{calculation_stem}.uks_scf.eldens.{grid[0]}x{grid[1]}x{grid[2]}.cube"
        )
        shutil.move(
            f"{calculation_stem}.spindens.cube",
            f"{calculation_stem}.uks_scf.spindens.{grid[0]}x{grid[1]}x{grid[2]}.cube",
        )

    elif method == "dlpno-ccsd_cc-pvdz":

        shutil.move(
            f"{calculation_stem}.eldens.cube", f"{calculation_stem}.uhf_scf.eldens.{grid[0]}x{grid[1]}x{grid[2]}.cube"
        )
        shutil.move(
            f"{calculation_stem}.spindens.cube",
            f"{calculation_stem}.uhf_scf.spindens.{grid[0]}x{grid[1]}x{grid[2]}.cube",
        )

        subprocess.run(plot_mdci, shell=True)

        shutil.move(
            f"{calculation_stem}.eldens.cube", f"{calculation_stem}.cc_unrelaxed.eldens.{grid[0]}x{grid[1]}x{grid[2]}.cube"
        )
        shutil.move(
            f"{calculation_stem}.spindens.cube",
            f"{calculation_stem}.cc_unrelaxed.spindens.{grid[0]}x{grid[1]}x{grid[2]}.cube",
        )


def handle_dirs(stem, output_dir):
    structures_output_dir = os.path.join(output_dir, stem)
    if not os.path.exists(structures_output_dir):
        os.makedirs(structures_output_dir)
    os.chdir(structures_output_dir)


def make_move_calc_files(
    calculation_stem, orca_template_fname, sub_template_fname, multiplicity, xyz_fname
):

    input_xyz_fname = calculation_stem + ".xyz"
    # print(f'xyz_fname: {xyz_fname}')
    # print(f'input_xyz_fname: {input_xyz_fname}')
    shutil.copy(xyz_fname, input_xyz_fname)

    make_orca_input_file(
        orca_template_fname, multiplicity, calculation_stem
    )

    make_sub_file(sub_template_fname, calculation_stem)


def make_sub_file(sub_template_fname, calculation_stem):

    with open(sub_template_fname, "r") as f:
        sub_text = f.read()

    sub_text = sub_text.replace("<stem>", calculation_stem)

    with open("sub.sh", "w") as f:
        f.write(sub_text)


def make_orca_input_file(
    orca_template_fname, multiplicity, calculation_stem
):

    with open(orca_template_fname, "r") as f:
        orca_text = f.read()

    orca_text = orca_text.replace("<multiplicity>", str(multiplicity))
    # orca_text = orca_text.replace("<input_filename>", input_xyz_fname)
    orca_text = orca_text.replace("<orca_label>", calculation_stem)

    input_orca_fname = calculation_stem + ".inp"
    with open(input_orca_fname, "w") as f:
        f.write(orca_text)


def inp_and_template_per_geometry(xyz_dir, orca_template, sub_template, output_dir, submit=False):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    orca_template = Path(orca_template).resolve()
    sub_template = Path(sub_template).resolve()

    xyz_filenames = get_xyz_files_from_dir(Path(xyz_dir).resolve())

    homedir = os.getcwd()

    for fname in xyz_filenames:
        fname = Path(fname)
        compound_stem = fname.stem
        handle_dirs(compound_stem, output_dir)        

        if "singlet" in compound_stem:
            multiplicity=1
        elif "triplet" in compound_stem:
            multiplicity=3

        make_move_calc_files(
            calculation_stem=compound_stem, 
            orca_template_fname=orca_template, 
            sub_template_fname=sub_template,
            multiplicity=multiplicity,
            xyz_fname=fname
        )

        if submit:
            subprocess.run("qsub sub.sh", shell=True)

        os.chdir(homedir)


def get_xyz_files_from_dir(dir_fname):
    return [os.path.join(dir_fname, dir_name) \
        for dir_name in os.listdir(dir_fname) \
        if "xyz" in dir_name]

