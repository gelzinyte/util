import os
import re
import subprocess
import click

@click.command()
@click.option('--pdf_dir', show_default=True,
              default='/Users/elena/code/carbyne_comulene/pdf_plots')
@click.option('--output_dir', show_default=True,
              default='/Users/elena/code/carbyne_comulene/combined_pdf_plots')
def main(pdf_dir, output_dir):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    fnames = [os.path.join(pdf_dir, fname) for fname in os.listdir(pdf_dir)]

    compounds = ['HC30H', 'HC31H', 'HC32H', 'HC33H', 'H2C30H2_flat',
                 'H2C30H2_angle', 'H2C31H2_flat', 'H2C31H2_angle']

    opts = ['triplet_opt', 'singlet_opt']

    mult = ['singlet', 'triplet']
    dens_methods = ['uks_scf', 'uhf_scf', 'mdci']

    for compound in compounds:
        for opt in opts:
            prefix = f'{compound}.{opt}'
            fnames_to_join = [fname for fname in fnames if prefix in
                                                               fname]
            fnames_to_join = natural_sort(fnames_to_join)

            eldens = [fname for fname in fnames_to_join if 'eldens' in fname]
            spindens = [fname for fname in fnames_to_join if 'spindens' in
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



if __name__=='__main__':
    main()
