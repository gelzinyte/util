import numpy as np

from ase.io import read, write
from ase import Atoms

from wfl.calculators.orca import ORCA
from wfl import ConfigSet, OutputSpec
from wfl.autoparallelize.autoparainfo import AutoparaInfo
from wfl.calculators import generic

from mace.calculators import mace

import util
from util import configs
from util.util_config import Config
from util import opt
from util.radicals import abstract_sp3_hydrogen_atoms
from util.bde.table import get_bde

# -----------------------------------------
# filenames and things 
# -----------------------------------------

ir = "in" # input root
mace_fn = "/rds/user/eg475/hpc-work/mace_fits/7.cho_fits/2.iterations/test_bde_generation/1.fit_mace/checkpoints/mace_run-123.model.cpu"

mace_opt_mols_rads_fn = f'{ir}.mace_all.mace_opt.xyz'
dft_opt_mols_rads_fn = f'{ir}.dft_all.dft_opt.xyz'
mace_reopt_mols_rads_fn = f'{ir}.dft_all.dft_opt.mace_reopt.xyz'
dft_reopt_mols_rads_fn = f'{ir}.mace_all.mace_opt.dft_reopt.xyz'
isolated_H_fn = "isolated_H.mace.dft.xyz"

# -----------------------------------------
# set up
# -----------------------------------------

# remote_info
remote_info = {}
remote_info["mace_opt"] = None
remote_info["dft_opt"] = None
remote_info["mace_eval"] = None
remote_info["dft_eval"] = None

# orca calc 
orca_kwargs = util.default_orca_params()
single_point_orca = (ORCA, [], orca_kwargs)
geometry_opt_orca = (ORCA, [], {**{"task":"opt"}, **orca_kwargs})

# mace calc
mace_calc = (mace.MACECalculator, [], {"model_path":mace_fn, "device":"cpu"})


# check that geometry type is set 
# and add a hash
in_fname = f"{ir}.xyz"
in_ats = read(in_fname, ":")
for at in in_ats:
    at.info["geometry_type"] = "rdkit_start"
    at.info["bde_initial_hash"] = configs.hash_atoms(at)
write(in_fname, in_ats)

# -----------------------------------------
# helper functions 
# -----------------------------------------

def set_tags(outputspec, key, val):
    assert outputspec.single_file
    fn = outputspec.files[0]
    ats = read(fn, ":")
    for at in ats:
        at.info[key] = val
    write(fn, ats)
    return ConfigSet(fn)


# -----------------------------------------
# geometry-optimise molecules
# -----------------------------------------


# mace optimise 
inputs = ConfigSet(f'{ir}.xyz')
outputs = OutputSpec(f'{ir}.mols.mace_opt.xyz')
mace_opt_mols = opt.optimise(
    inputs=inputs, 
    outputs=outputs,
    calculator = mace_calc, 
    output_prefix= "mace_",
    autopara_info = AutoparaInfo(
        num_inputs_per_python_subprocess=8,
        remote_info=remote_info["mace_opt"])
) 
set_tags(outputs, "geometry_type", "mace_opt")


# orca optimise
inputs = ConfigSet(f'{ir}.xyz')
outputs = OutputSpec(f'{ir}.mols.dft_opt.xyz')
dft_opt_mols = generic.run(
    inputs=inputs, 
    outputs=outputs,
    calculator=geometry_opt_orca,
    properties=["energy", "forces"],
    output_prefix='dft_',
    autopara_info = AutoparaInfo(
        remote_info=remote_info["orca_opt"],
        num_inputs_per_python_subprocess=1)
)
set_tags(outputs, "geometry_type", "dft_opt")



# -----------------------------------------
# make radicals 
# -----------------------------------------

# mace
outputs = OutputSpec(f"{ir}.mace_all.xyz")
mace_mols_rads = abstract_sp3_hydrogen_atoms(
    inputs = mace_opt_mols,
    outputs=outputs,
    label_config_type=True, 
    return_mol = True
)

# dft
outputs = OutputSpec(f"{ir}.dft_all.xyz")
dft_mols_rads = abstract_sp3_hydrogen_atoms(
    inputs = dft_opt_mols,
    outputs=outputs,
    label_config_type=True, 
    return_mol = True
)


# -----------------------------------------
# geometry-optimise molecules and radicals
# -----------------------------------------

# mace optimise 
outputs = OutputSpec(mace_opt_mols_rads_fn)
mace_opt_mols = opt.optimise(
    inputs=mace_mols_rads, 
    outputs=outputs,
    calculator = mace_calc, 
    output_prefix= "mace_",
    autopara_info = AutoparaInfo(
        num_inputs_per_python_subprocess=8,
        remote_info=remote_info["mace_opt"])
) 
set_tags(outputs, "geometry_type", "mace_opt")


# orca optimise
outputs = OutputSpec(dft_opt_mols_rads_fn)
dft_opt_mols = generic.run(
    inputs=dft_mols_rads, 
    outputs=outputs,
    calculator=geometry_opt_orca,
    properties=["energy", "forces"],
    output_prefix='dft_',
    autopara_info = AutoparaInfo(
        remote_info=remote_info["orca_opt"],
        num_inputs_per_python_subprocess=1)
)
set_tags(outputs, "geometry_type", "dft_opt")



# -----------------------------------------
# re-optimise with mace and dft 
# -----------------------------------------

# mace optimise 
outputs = OutputSpec(mace_reopt_mols_rads_fn)
mace_opt_mols = opt.optimise(
    inputs=dft_opt_mols, 
    outputs=outputs,
    calculator = mace_calc, 
    output_prefix= "mace_",
    autopara_info = AutoparaInfo(
        num_inputs_per_python_subprocess=8,
        remote_info=remote_info["mace_opt"])
) 
set_tags(outputs, "geometry_type", "mace_reopt")


# orca optimise
outputs = OutputSpec(dft_reopt_mols_rads_fn)
dft_opt_mols = generic.run(
    inputs=dft_mols_rads, 
    outputs=outputs,
    calculator=geometry_opt_orca,
    properties=["energy", "forces"],
    output_prefix='dft_',
    autopara_info = AutoparaInfo(
        remote_info=remote_info["orca_opt"],
        num_inputs_per_python_subprocess=1)
)
set_tags(outputs, "geometry_type", "dft_reopt")


# -----------------------------------------
# freshly re-evaluate dft and mace 
# -----------------------------------------

# clean old results
inputs = ConfigSet([
    mace_opt_mols_rads_fn,
    dft_opt_mols_rads_fn,
    mace_reopt_mols_rads_fn,
    dft_reopt_mols_rads_fn,
])

outputs = OutputSpec({
    mace_opt_mols_rads_fn: mace_opt_mols_rads_fn.replace(".xyz", ".cleaned.xyz"),
    dft_opt_mols_rads_fn: dft_opt_mols_rads_fn.replace(".xyz", ".cleaned.xyz"),
    mace_reopt_mols_rads_fn: mace_reopt_mols_rads_fn.replace(".xyz", ".cleaned.xyz"),
    dft_reopt_mols_rads_fn: dft_reopt_mols_rads_fn.replace(".xyz", ".cleaned.xyz"),
})

configs.clean_calc_results(inputs, outputs)

# relabel
mace_opt_mols_rads_fn = mace_opt_mols_rads_fn.replace(".xyz", ".cleaned.xyz"),
dft_opt_mols_rads_fn = dft_opt_mols_rads_fn.replace(".xyz", ".cleaned.xyz"),
mace_reopt_mols_rads_fn = mace_reopt_mols_rads_fn.replace(".xyz", ".cleaned.xyz"),
dft_reopt_mols_rads_fn = dft_reopt_mols_rads_fn.replace(".xyz", ".cleaned.xyz"),

# aside - make isolated h
isolated_h_fn = "isolated_h.xyz"
write(isolated_h_fn, Atoms("H", position=[0, 0, 0]))


# eval with mace
inputs = ConfigSet([
    mace_opt_mols_rads_fn,
    dft_opt_mols_rads_fn,
    mace_reopt_mols_rads_fn,
    dft_reopt_mols_rads_fn,
    isolated_h_fn
])

outputs = OutputSpec({
    mace_opt_mols_rads_fn: mace_opt_mols_rads_fn.replace(".xyz", ".mace.xyz"),
    dft_opt_mols_rads_fn: dft_opt_mols_rads_fn.replace(".xyz", ".mace.xyz"),
    mace_reopt_mols_rads_fn: mace_reopt_mols_rads_fn.replace(".xyz", ".mace.xyz"),
    dft_reopt_mols_rads_fn: dft_reopt_mols_rads_fn.replace(".xyz", ".mace.xyz"),
    isolated_h_fn: isolated_h_fn.replace(".xyz", ".mace.xyz")
})

generic.run(
    inputs=inputs, 
    outputs=outputs,
    calculator=mace_calc,
    properties=["energy"],
    output_prefix='mace_',
    autopara_info = AutoparaInfo(
        remote_info=remote_info["mace_eval"],
        num_inputs_per_python_subprocess=1)
)

# relabel
mace_opt_mols_rads_fn = mace_opt_mols_rads_fn.replace(".xyz", ".mace.xyz"),
dft_opt_mols_rads_fn = dft_opt_mols_rads_fn.replace(".xyz", ".mace.xyz"),
mace_reopt_mols_rads_fn = mace_reopt_mols_rads_fn.replace(".xyz", ".mace.xyz"),
dft_reopt_mols_rads_fn = dft_reopt_mols_rads_fn.replace(".xyz", ".mace.xyz"),
isolated_h_fn = isolated_h_fn.replace(".xyz", "mace.xyz")


# eval with dft
inputs = ConfigSet([
    mace_opt_mols_rads_fn,
    dft_opt_mols_rads_fn,
    mace_reopt_mols_rads_fn,
    dft_reopt_mols_rads_fn,
    isolated_h_fn
])

outputs = OutputSpec({
    mace_opt_mols_rads_fn: mace_opt_mols_rads_fn.replace(".xyz", ".dft.xyz"),
    dft_opt_mols_rads_fn: dft_opt_mols_rads_fn.replace(".xyz", ".dft.xyz"),
    mace_reopt_mols_rads_fn: mace_reopt_mols_rads_fn.replace(".xyz", ".dft.xyz"),
    dft_reopt_mols_rads_fn: dft_reopt_mols_rads_fn.replace(".xyz", ".dft.xyz"),
    isolated_h_fn: isolated_h_fn.replace(".xyz", ".mace.xyz")
})

generic.run(
    inputs=inputs, 
    outputs=outputs,
    calculator=single_point_orca,
    properties=["energy"],
    output_prefix='dft_',
    autopara_info = AutoparaInfo(
        remote_info=remote_info["dft_eval"],
        num_inputs_per_python_subprocess=1)
)

# relabel
mace_opt_mols_rads_fn = mace_opt_mols_rads_fn.replace(".xyz", ".dft.xyz"),
dft_opt_mols_rads_fn = dft_opt_mols_rads_fn.replace(".xyz", ".dft.xyz"),
mace_reopt_mols_rads_fn = mace_reopt_mols_rads_fn.replace(".xyz", ".dft.xyz"),
dft_reopt_mols_rads_fn = dft_reopt_mols_rads_fn.replace(".xyz", ".dft.xyz"),
isolated_h_fn = isolated_h_fn.replace(".xyz", "dft.xyz")


# -----------------------------------------
# collect data
# -----------------------------------------

def assign_bde_entries(dest_mols, all_bde_ats, bde_array_label, isolated_h_fn, prop_prefix):

    isolated_H = read(isolated_h_fn)
    isolated_h_energy = isolated_H.info[f"{prop_prefix}energy"]

    ats = configs.into_dict_of_labels(all_bde_ats, "bde_initial_hash")

    for ref in dest_mols:

        bdes = np.empty(len(ref))
        bdes.fill(np.nan)

        bde_ats = ats[ref.info["bde_initial_hash"]]
        bde_ats = configs.into_dict_of_labels(bde_ats, "mol_or_rad")

        assert len(bde_ats["mol"]) == 1
        mol = bde_ats["mol"][0]
        mol_energy = mol.info[f"{prop_prefix}energy"]

        for rad in bde_ats["rad"]:
            rad_no = rad["rad_num"]
            rad_energy = rad.info[f"{prop_prefix}energy"]
            bde = get_bde(
                mol_energy = mol_energy,
                rad_energy = rad_energy,
                isolated_h_energy=isolated_h_energy)
            bdes[rad_no] = bde

        ref.arrays[bde_array_label] = bdes

    return dest_mols



ref_geometries = mace_opt_mols_rads_fn 
ats = read(ref_geometries, ":")
ref_mols = configs.into_dict_of_labels(ats, "mol_or_rad")["mol"]
ref_mols = [util.remove_energy_force_containing_entries(at) for at in ats]


geom_fnames = [mace_opt_mols_rads_fn, dft_opt_mols_rads_fn, mace_reopt_mols_rads_fn, dft_reopt_mols_rads_fn]
geom_labels = ["mace_opt", "dft_opt", "mace_reopt", "dft_reopt"]

for geom_fn, geom_label in zip(geom_fnames, geom_labels):
    for prop_prefix in ["mace_", "dft_"]:
        bde_array_label = f"{geom_label}_{prop_prefix}bde"

        ref_mols = assign_bde_entries(
            dest_mols=ref_mols, 
            all_bde_ats=read(geom_fn, ":"),
            bde_array_label = bde_array_label,
            isolated_h_fn = isolated_h_fn,
            prop_prefix = prop_prefix)


write("summary.xyz", ref_mols)


