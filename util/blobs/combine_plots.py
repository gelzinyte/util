import os
import re
import subprocess
import click
from tqdm import tqdm

def sideways(cross_section_dir, isosurface_dir, integrated_dens_dir, output_dir):
    
    # cross_section_dir = 'plots_crossection'
    # isosurface_dir = 'plots_isosurface'
    # integrated_dens_dir = 'plots_integrated_densities'
    # output_dir = 'combined_plots_per_density'

    tmp_fname = 'tmp_plot.tga'


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    plot_names = [os.path.splitext(fname)[0] for fname in os.listdir(
        cross_section_dir)]

    fnames_cross = [os.path.join(cross_section_dir, fname+'.tga') for fname in
               plot_names]

    fnames_iso = [os.path.join(isosurface_dir, fname+'.tga') for fname in
               plot_names]

    fnames_int = [os.path.join(integrated_dens_dir, fname+'.pdf') for fname in
                  plot_names]


    for fname_cross, fname_iso, fname_int in tqdm(zip(fnames_cross,
    fnames_iso, fnames_int)):

        fname_joined = os.path.join(output_dir, os.path.splitext(
            os.path.basename(fname_cross))[0])

        command =  f'convert +append {fname_iso} {fname_cross} {fname_int} ' \
                   f'{fname_joined}.tga'

        subprocess.run(command, shell=True)


def all(pdf_dir, output_dir):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fnames = [os.path.join(pdf_dir, fname) for fname in os.listdir(pdf_dir)]

    compounds = ['HC30H', 'HC31H', 'HC32H', 'HC33H', 'H2C30H2_flat',
                 'H2C30H2_angle', 'H2C31H2_flat', 'H2C31H2_angle']

    opts = ['uDFT_triplet_optimised', 'uDFT_singlet_optimised']

    mult = ['singlet', 'triplet']
    dens_methods = ['uDFT_density', 'uHF_density', 'CC_unrelaxed_density']

    for compound in compounds:
        for opt in opts:
            prefix = f'{compound}.{opt}'
            fnames_to_join = [fname for fname in fnames if prefix in
                                                               fname]
            fnames_to_join = natural_sort(fnames_to_join)

            eldens = [fname for fname in fnames_to_join if 'electron_density' in fname]
            spindens = [fname for fname in fnames_to_join if 'spin_density' in
                        fname]

            exec = 'convert '

            for dens in [spindens, eldens]:
                for m in mult:
                    for dens_m in dens_methods:
                        for fname in dens:
                            key = f'{m}.{dens_m}'
                            if key in fname:
                                exec += fname + ' '

            exec += os.path.join(output_dir, prefix + '.pdf')

            subprocess.run(exec, shell=True)

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def add_titles(input_dir, output_dir):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    filenames = [os.path.join(input_dir, fname) for fname in os.listdir(
        input_dir)]

    for filename in tqdm(filenames):

        base = os.path.basename(os.path.splitext(filename)[0])
        target_fname = os.path.join(output_dir, base + '.pdf')

        exec = f"convert {filename} -gravity North -draw 'scale 4,4 text 0," \
               f"10 {base}' {target_fname}"

        subprocess.run(exec, shell=True)


def int_dens_only(integrated_dens_dir, output_dir):
    
    print(f'Output_dir: {output_dir}')
    print(f'integrated_dens_dir: {integrated_dens_dir}')
    

    compounds = ['H2C30H2_angle', 'HC30H', 'HC31H', 'HC32H', 'HC33H', 'H2C30H2_flat',
                  'H2C31H2_flat', 'H2C31H2_angle']

    opts = ['uDFT_triplet_optimised', 'uDFT_singlet_optimised']

    mult = ['singlet', 'triplet']
    dens_methods = ['uHF_density', 'uDFT_density', 'CC_unrelaxed_density']

    all_fnames = [os.path.join(integrated_dens_dir, fname) for fname in os.listdir(integrated_dens_dir)]

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # first combine length-wise then side-wise
    for comp in compounds:
        for opt in opts:
            relevant_fnames = [fname for fname in all_fnames if comp in fname and opt in fname]

            counter = 0
            main_exec = "convert "


            for dens_type in ["spin_density", "electron_density"]:
                for spin in [".singlet.", ".triplet."]:

                    exec = "convert +append "
                        
                    row_fnames = [fname for fname in relevant_fnames if spin in fname and dens_type in fname]
                    # print(row_fnames)

                    for method in dens_methods:
                        for fname in row_fnames:
                            if method in fname:
                                exec += fname + ' '
            
                    counter += 1
                    intermediate_fname = f"tmp{counter}.tga"
                    main_exec += intermediate_fname + " "
                    exec += intermediate_fname
                    # print(exec)
                    subprocess.run(exec, shell=True)
                    # return

            out_fname = os.path.join(output_dir, f"{comp}.{opt}.pdf")
            main_exec +=  f"{out_fname}"
            print(main_exec)
            subprocess.run(main_exec, shell=True)
            # return




    