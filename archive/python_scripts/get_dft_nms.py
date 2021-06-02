from ase.io import read, write
from util.vibrations import Vibrations
from util import itools
import pickle
from ase.utils import opencew
import click
import os
import shutil


@click.command()
@click.option('--dft_at_fname')
@click.option('--wdir')
def calculate_NMs(dft_at_fname, wdir):
    ###
    # orca stuff
    ###

    if wdir:
        if not os.path.isdir(wdir):
            os.makedirs(wdir)
        cwdir = os.getcwd()
        os.chdir(wdir)


    atoms = read(dft_at_fname, ':')
    for at in atoms:
        dft_name = at.info['name']

        scratch_dir = '/scratch/eg475'
        if not os.path.isdir(scratch_dir):
            os.makedirs(scratch_dir)
            
        no_cores = os.environ['OMP_NUM_THREADS']
        smearing = 2000
        maxiter = 200
        n_wfn_hop = 1
        task = 'gradient'
        orcasimpleinput = 'UKS B3LYP def2-SV(P) def2/J D3BJ'
        orcablocks = f"%scf Convergence tight \n SmearTemp {smearing} \n maxiter " \
                     f"{maxiter} end \n"
        calc_rundir = 'ORCA_outputs'

        n_run_glob_op = 1
        kw_orca = f'smearing={smearing}'

        orca_tmp_fname = f'orca_tmp.xyz'
        wfl_command = f'wfl -v ref-method orca-eval --output-file {orca_tmp_fname} -tmp ' \
                      f'{scratch_dir} ' \
                      f'-nr {n_run_glob_op} -nh {n_wfn_hop} --base-rundir {calc_rundir} --keep-files True --kw "{kw_orca}" ' \
                      f'--orca-simple-input "{orcasimpleinput}"'


        vib = Vibrations(at, name=dft_name)
        print(f'getting vibrations for: dft_name: {dft_name}; at.info[name]: {at.info["name"]}')
        
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
            forces = at.arrays['dft_forces']
            filename = name + '.pckl'
            fd = opencew(filename)
            if fd is not None:
                pickle.dump(forces, fd, protocol=2)
                fd.close()

        vib.combine()
        vib.summary()

    if wdir:
        shutil.copy(f'{dft_name}.all.pckl', cwdir)
        os.chdir(cwdir)

if __name__=='__main__':
    calculate_NMs()
