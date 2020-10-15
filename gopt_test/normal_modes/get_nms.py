from ase.io import read, write
from util.vib import Vibrations
from util import itools
import pickle
from ase.utils import opencew
import click
import os


@click.command()
@click.option('--dft_at_fname')
def calculate_NMs(dft_at_fname):
    ###
    # orca stuff
    ###

    atoms = read(dft_at_fname, ':')
    for at in atoms:
        dft_name = at.info['name']

        scratch_dir = '/scratch/eg475'
        no_cores = os.environ['OMP_NUM_THREADS']
        smearing = 2000
        maxiter = 200
        n_wfn_hop = 1
        task = 'gradient'
        orcasimpleinput = 'UKS BLYP 6-31G slowconv'
        orcablocks = f"%scf Convergence tight \n SmearTemp {smearing} \n maxiter " \
                     f"{maxiter} end \n"

        n_run_glob_op = 1
        kw_orca = f'n_hop={n_wfn_hop} smearing={smearing} maxiter={maxiter}'

        orca_tmp_fname = f'orca_tmp.xyz'
        wfl_command = f'wfl -v ref-method orca-eval -o {orca_tmp_fname} -tmp ' \
                      f'{scratch_dir} ' \
                      f'-n {n_run_glob_op} -p {no_cores} --kw "{kw_orca}" ' \
                      f'--orca-simple-input "{orcasimpleinput}"'




        vib = Vibrations(at, name=dft_name)
        displaced_ats = []
        names = []
        for name, at in vib.iterdisplace():
            displaced_ats.append(at)
            names.append(name)

        if not os.path.isfile(orca_tmp_fname):
            displaced_ats_out = itools.orca_par_data(atoms_in = displaced_ats,
                                                     out_fname=orca_tmp_fname,
                                                     wfl_command=wfl_command)
        else:
            print('found atoms file with dft energies, not recalculating')
            displaced_ats_out = read(orca_tmp_fname, ':')

        for name, at in zip(names, displaced_ats_out):
            forces = at.arrays['force']
            filename = name + '.pckl'
            fd = opencew(filename)
            if fd is not None:
                pickle.dump(forces, fd, protocol=2)
                fd.close()

        vib.combine()
        vib.summary()

if __name__=='__main__':
    calculate_NMs()
