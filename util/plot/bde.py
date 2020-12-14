import click
import os
from ase.io import read, write
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import util

@click.command()
@click.option('--dft_bde_dir', '-d', type=click.Path(), default='/home/eg475/data/bde_files/dft', help='where to look for dft-optimised mols and rads')
@click.option('--start_bde_dir', '-s', type=click.Path(), default='/home/eg475/data/bde_files/starts', help='where to look for non-optimised mols and rads')
@click.option('--gap_train_bde_dir', '-g', type=click.Path(),  help='where to look for gap-optimised mols and rads in training set')
@click.option('--prefix', '-p', type=str, default='bde', help='to save the pic')
@click.option('--gap_test_bde_dir', type=click.Path(),  help='where to look for gap-optimised mols and rads in test set')
@click.option('--measure', '-m', type=str, help='how to measure similarity of structures')
def bde_summary(dft_bde_dir, start_bde_dir, gap_train_bde_dir, prefix, gap_test_bde_dir, measure):

    assert measure in ['rmsd', 'soap', 'energy']

    dft_fnames = [os.path.join(dft_bde_dir, name) for name in os.listdir(dft_bde_dir)]
    dft_fnames = util.natural_sort(dft_fnames)

    mins = []
    maxs = []

    cidx_list = np.linspace(0, 1, len(dft_fnames))
    cmap = mpl.cm.get_cmap('tab10')
    plt.figure = plt.figure(figsize=(10, 5))


    for cidx, dft_fname in zip(cidx_list, dft_fnames):

        basename = os.path.basename(dft_fname)
        start_fname = os.path.join(start_bde_dir, basename.replace('_optimised.xyz', '_non-optimised.xyz'))

        scatter_kwargs = {}
        scatter_kwargs['label'] = basename.replace('_optimised.xyz', '')

        train_gap_bde_fname = os.path.join(gap_train_bde_dir, basename.replace('_optimised.xyz', '_gap_optimised.xyz'))
        test_gap_bde_fname = None
        if gap_test_bde_dir is not None:
            test_gap_bde_fname = os.path.join(gap_test_bde_dir, basename.replace
                                                  ('_optimised.xyz', '_gap_optimised.xyz'))

        # check if filenames exist (and choose color)
        if test_gap_bde_fname is not None:
            if os.path.exists(train_gap_bde_fname) and os.path.exists(test_gap_bde_fname):
                raise RuntimeError('Gap bde files exist in training and test directories')
        if os.path.exists(train_gap_bde_fname):
            gap_bde_fname = train_gap_bde_fname
            scatter_kwargs['color'] = cmap(cidx)
        elif gap_test_bde_dir is not None:
            if os.path.exists(test_gap_bde_fname):
                gap_bde_fname = test_gap_bde_fname
                scatter_kwargs['color'] = 'tab:red'
        else:
            print \
                (f"Didn't find {basename} optimised with gap neither in training, not in training set, skipping")
            continue


        dft_ats = read(dft_fname, ':')
        gap_ats = read(gap_bde_fname, ':')
        start_ats = read(start_fname, ':')

        if len(gap_ats)!= len(start_ats):
            raise RuntimeWarning(f"Could not optimise all {start_fname} structures with GAP")
            # print(f"Could not optimise all {start_fname} structures with GAP")
            # continue

        if measure == 'rmsd':
            start_coord = [util.get_rmse(dft_at.positions, start_at.positions) for dft_at, start_at in zip(dft_ats, start_ats) if len(dft_at ) != 1]
            end_coord = [util.get_rmse(dft_at.positions, gap_at.positions) for dft_at, gap_at in zip(dft_ats, gap_ats) if len(dft_at ) !=1]
        elif measure == 'soap':
            start_coord = [util.soap_dist(dft_at, start_at) for dft_at, start_at in zip(dft_ats, start_ats) if len(dft_at ) != 1]
            end_coord = [util.soap_dist(dft_at, gap_at) for dft_at, gap_at in zip(dft_ats, gap_ats) if len(dft_at ) !=1 ]
        else:
            # print(gap_bde_fname)
            # print('start_coord')
            # print(dft_ats[0].info)
            start_coord = util.get_bdes(bde_ats=dft_ats, e_key='dft_energy')
            # print('end_coord')
            # print(gap_ats[0].info)
            end_coord = util.get_bdes(bde_ats=gap_ats, e_key='gap_energy')

        mins.append(min(start_coord + end_coord))
        maxs.append(max(start_coord + end_coord))
        plt.scatter(start_coord, end_coord, **scatter_kwargs)

    min_lim = min(mins)
    max_lim = max(maxs)
    plt.plot([min_lim, max_lim], [min_lim, max_lim], linewidth=0.8,  color='k')

    if measure in ['soap', 'rmsd']:
        # plt.xlim(1e-4, 1)
        # plt.ylim(1e-4, 1)
        plt.xscale('log')
        plt.yscale('log')

    plt.grid(color='lightgrey')
    plt.legend()

    plt.title(f'GAP vs DFT optimised molecules and radicals')

    if measure == 'soap':
        plt.xlabel('Non-optimised structure SOAP distance corresponding DFT eq. ')
        plt.ylabel('GAP-optimised structure SOAP distance from corresponding DFT eq. ')

    elif measure == 'rmsd':
        plt.xlabel('Non-optimised structure RMSD corresponding DFT eq., Å ')
        plt.ylabel('GAP-optimised structure RMSD from corresponding DFT eq., Å ')

    else:
        plt.xlabel('DFT Bond Dissociation Energy, eV')
        plt.ylabel('GAP Bond Dissociation Energy, eV')

    plt.tight_layout()
    if not os.path.isdir('pictures'):
        os.makedirs('pictures')
    plt.savefig(f'pictures/{prefix}_{measure}.png', dpi=300)



if __name__ == '__main__':
    bde_summary()

