import nm
from wfl.generate_configs import vib
import click
from ase.io import write
from ase  import Atoms



@click.command()
@click.argument("old_fnames", nargs=-1)
@click.option("--new_fname", '-o', help='target file for new-type normal modes')
@click.option("--config_types", '-i', help='string of config type entries corresponding to the old fnames')
def old_to_new(old_fnames, new_fname, config_types=None):

    if config_types is not None:
        config_types = config_types.split()
    else:
        config_types = [None for _ in old_fnames]

    new_atoms = []

    for old_fname, config_type in zip(old_fnames, config_types):

        vib_old = nm.Vibrations(old_fname)
        # vib_old.summary()
        old_at = vib_old.atoms.copy()
        N = len(old_at)

        new_at = Atoms(vib_old.atoms.symbols, positions=vib_old.atoms.positions)
        # new_at.info['energy'] = old_at.info['dft_energy']
        # new_at.arrays['forces'] = old_at.arrays['dft_forces']

        if config_type is not None:
            new_at.info['config_type'] = config_type

        eigenvectors = vib_old.evecs
        eigenvalues = vib_old.evals

        new_at.info['nm_eigenvalues'] = eigenvalues

        for idx, evec in enumerate(eigenvectors):
            new_at.arrays[f'evec{idx}'] = evec.reshape((N, 3))


        new_atoms.append(new_at)

        new_vib = vib.Vibrations(new_at)
        # print(f'-'*50)
        # new_vib.summary()

        assert (new_vib._get_mode(8) == vib_old.get_mode(8)).all()


    write(new_fname, new_atoms)


if __name__ == '__main__':
    old_to_new()