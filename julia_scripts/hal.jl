using JuLIP
using HAL
using IPFitting
using ACE1
using LinearAlgebra
BLAS.set_num_threads(32)

ENV["GKSwstype"] = "nul"
ENV["MKL_NUM_THREADS"]=32

al = IPFitting.Data.read_xyz("../MoNbTaWTi_convert.xyz", energy_key="energy", force_key="forces", virial_key="virial")

Vref = OneBody(:Nb => -1647.1951, :Ta => -8431.6730, :W => -9248.4461, :Mo => -1926.1786, :Ti => -1585.9720)

start_configs = IPFitting.Data.read_xyz("../MoNbTaWTi_convert.xyz", energy_key="energy", force_key="forces", virial_key="virial")

run_info = Dict(
    "HAL_iters" => 1, # num MD runs per start config 
    #"var" => true,  # bias for variance, not std (var standard)
    #"refit" => 1, # every how many configs to refit ACE
    "ncoms" => 32,  # num committee members; num of threads 
    "brrtol" => 1e-3, 
    "Rshift" => 0.0, # shift inner cutoff 
    "nsteps" => 5000, # max md steps
    "default" => Dict(
        "swap" => false,  # swap atoms
        "vol" => false,  # MC vol
        "baro_thermo" => false,  # false just thermo
        "minR" => 0.6,  # min distance before count clash
        "volstep" => 50,  # every vlostep do swap (irrelevant here)
        "swapstep" => 50, 
        "temp_dict" => Dict(5000 => 300), # first 5000 steps, temp 2000 
        "rτ" => 3e-2, #  biasing, 3%
        "μ" => 1e-1,  # barostat timescale, irrelevant
        "Pr0" => 0.0,  # set pressure 
        "dt" => 1.0,  # timestep 
        "γ" => 0.2,  # langevan timescale
        "maxp" => 0.2,  # max uncertainty 
        "Freg" => 0.2))  # rel force uncertainty regulariser (epsilon)


weights = Dict(
    "default" => Dict("E" => 1.0, "F" => 1.0 , "V" => 1.0 ),
    )

# calc_settings = Dict(
#     "calculator" => "CASTEP",
#     "_castep_command" => "/usr/bin/mpirun -n 32 /opt/womble/castep/19.11/castep.mpi",
#     "_castep_pp_path" => "/home/casv2/CASTEP/convert_MoNbTaWTi/_CASTEP",
#     "_directory" => "./_CASTEP",
#     "cut_off_energy" => 500,
#     "calculate_stress" => true,
#     "smearing_width" => 0.1,
#     "finite_basis_corr" => "automatic",
#     "mixing_scheme" => "Pulay",
#     "write_checkpoint" => "none",
#     "kpoint_mp_spacing" => "0.05")

calc_settings = Dict(
    "calculator" => "ORCA",
    "label" => "orca",
    "orca_command" => "/opt/womble/orca/orca_4_2_1_linux_x86-64_openmpi314/orca",
    "charge" => 0,
    "mult" => 1,
    "task" => "gradient",
    "orcasimpleinput" => "RKS wB97X 6-31G(d) Grid6 FinalGrid6 NormalSCF",
    "orcablocks" => "")


r0 = 0.20*(rnn(:Mo) + rnn(:Nb) + rnn(:Ta) + rnn(:Ti) + rnn(:W))

Bsite = rpi_basis(species = [:Mo, :Nb, :Ta, :W, :Ti],
                  N = 2,       # correlation order = body-order - 1
                  maxdeg = 14,  # polynomial degree
                  r0 = r0,     # estimate for NN distance
                  rin = 1.0,
                  rcut = 5.5,   # domain for radial basis (cf documentation)
                  pin = 2)                     # require smooth inner cutoff

Bpair = pair_basis(species = [:Mo, :Nb, :Ta, :W, :Ti],
                   r0 = r0,
                   maxdeg = 18,
                   rcut = 7.0,
                   rin = 0.0,
                   pin = 0 )   # pin = 0 means no inner cutoff

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

al_HAL = HAL.RUN.run_HAL(Vref, weights, al, start_configs, run_info, calc_settings, B)