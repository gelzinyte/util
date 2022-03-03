import os
import re
import subprocess
import click
from tqdm import tqdm

#@click.command()
#@click.option('--cross_section_dir', show_default=True,
#              default='/Users/elena/code/carbyne_comulene'
#                      '/vmd_cross_sections')
#@click.option('--isosurface_dir', show_default=True, 
#              default='/Users/elena/code/carbyne_comulene/vmd_isosurfaces')
#@click.option('--output_dir', show_default=True,
#              default='/Users/elena/code/carbyne_comulene'
#                      '/combined_density_and_cross_section')
#def main(cross_section_dir, isosurface_dir, output_dir):
def main():
    
    # cross_section_dir ='/Users/elena/code/carbyne_comulene/plots_crossection'
    # isosurface_dir = '/Users/elena/code/carbyne_comulene/plots_isosurface'
    # integrated_dens_dir = '/Users/elena/code/carbyne_comulene/plots_integrated_densities'
    # output_dir = '/Users/elena/code/carbyne_comulene' \
    #              '/combined_plots_per_density'

    cross_section_dir = 'plots_crossection'
    isosurface_dir = 'plots_isosurface'
    integrated_dens_dir = 'plots_integrated_densities'
    output_dir = 'combined_plots_per_density'

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



if __name__=='__main__':
    main()
