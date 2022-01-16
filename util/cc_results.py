import os
import shutil
import subprocess


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
):

    print(task)

    multiplicities = [1, 3]
    multiplicities_names = ["singlet", "triplet"]
    methods = ["uks_cc-pvdz", "dlpno-ccsd_cc-pvdz"]
    template_fnames = [uks_orca_template_fname, cc_orca_template_fname]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    home_dir = os.getcwd()

    xyz_filenames = [
        os.path.join(structures_dir, dir_name)
        for dir_name in os.listdir(structures_dir)
        if "xyz" in dir_name
    ]

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

    plot_scf = (
        f'printf "5\n7\n4\n200\n1\n2\ny\n10\n1\n3\ny\n10\n11" | '
        f"orca_plot "
        f"{calculation_stem}.gbw -i"
    )

    plot_mdci = (
        f'printf "5\n7\n4\n200\n1\n7\ny\n10\n1\n8\ny\n10\n11" | '
        f"orca_plot "
        f"{calculation_stem}.gbw -i"
    )

    subprocess.run(plot_scf, shell=True)

    if method == "uks_cc-pvdz":
        shutil.move(
            f"{calculation_stem}.eldens.cube", f"{calculation_stem}.uks_scf.eldens.cube"
        )
        shutil.move(
            f"{calculation_stem}.spindens.cube",
            f"{calculation_stem}.uks_scf.spindens.cube",
        )

    elif method == "dlpno-ccsd_cc-pvdz":

        shutil.move(
            f"{calculation_stem}.eldens.cube", f"{calculation_stem}.uhf_scf.eldens.cube"
        )
        shutil.move(
            f"{calculation_stem}.spindens.cube",
            f"{calculation_stem}.uhf_scf.spindens.cube",
        )

        subprocess.run(plot_mdci, shell=True)

        shutil.move(
            f"{calculation_stem}.eldens.cube", f"{calculation_stem}.mdci.eldens.cube"
        )
        shutil.move(
            f"{calculation_stem}.spindens.cube",
            f"{calculation_stem}.mdci.spindens.cube",
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
    shutil.copy(xyz_fname, input_xyz_fname)

    make_orca_input_file(
        orca_template_fname, multiplicity, input_xyz_fname, calculation_stem
    )

    make_sub_file(sub_template_fname, calculation_stem)


def make_sub_file(sub_template_fname, calculation_stem):

    with open(sub_template_fname, "r") as f:
        sub_text = f.read()

    sub_text = sub_text.replace("<stem>", calculation_stem)

    with open("sub.sh", "w") as f:
        f.write(sub_text)


def make_orca_input_file(
    orca_template_fname, multiplicity, input_xyz_fname, calculation_stem
):

    with open(orca_template_fname, "r") as f:
        orca_text = f.read()

    orca_text = orca_text.replace("<multiplicity>", str(multiplicity))
    orca_text = orca_text.replace("<input_filename>", input_xyz_fname)

    input_orca_fname = calculation_stem + ".inp"
    with open(input_orca_fname, "w") as f:
        f.write(orca_text)


if __name__ == "__main__":
    main()
