from ase.calculators.orca import ORCA
from ase.io import read, write

iso_ats = read('isolated_atoms_orca.xyz', ':')

smearing = 2000
maxiter = 200
n_wfn_hop = 1
task = 'energy'
orcasimpleinput = 'UKS B3LYP def2-SV(P) def2/J D3BJ'
orcablocks =  f"%scf Convergence tight \n SmearTemp {smearing} \n maxiter {maxiter} end \n"
              # f'%pal nprocs {no_cores} end'

print(orcasimpleinput)

mult=1
charge=0

orca_command = '/home/eg475/programs/orca/orca_4_2_1_linux_x86' \
               '-64_openmpi314/orca'

# for small non-parallel calculations

out_ats = []
for at in iso_ats:

    if at.symbols=='C' or at.symbols=='O':
        mult=1
    elif at.symbols=='H':
        mult=2

    calc = ORCA(label="ORCA/orca",
        orca_command=orca_command,
        charge=charge,
        mult=mult,
        task=task,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks
        )



    at.set_calculator(calc)
    energy = at.get_potential_energy()
    print(f'old energy: {at.info["dft_energy"]}\nnew_energy: {energy}')
    at.info['dft_energy'] = energy
    out_ats.append(at)

write('new_isolated_atoms_orca.xyz', out_ats, 'extxyz', write_results=False)
