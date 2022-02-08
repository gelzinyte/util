using JuLIP
using HAL
using IPFitting
using ACE
using LinearAlgebra
BLAS.set_num_threads(16)

ENV["GKSwstype"] = "nul"
ENV["MKL_NUM_THREADS"]= 16

al = IPFitting.Data.read_xyz("/data/eg475/heap_of_ch/hal/0_run/train.dft.xyz", energy_key="energy", force_key="forces")

Vref = OneBody(:C => -1028.4321581399468, :H => -13.547478676206193)

start_configs = IPFitting.Data.read_xyz("/data/eg475/heap_of_ch/hal/0_run/hal_start_configs.xyz", energy_key="energy", force_key="forces")[43]
start_configs = [start_configs]

run_info = Dict(
    "HAL_iters" => 10, # num MD runs per start config 
    #"var" => true,  # bias for variance, not std (var standard)
    #"refit" => 1, # every how many configs to refit ACE
    "ncoms" => 16,  # num committee members; num of threads 
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
        "rτ" => 1e-2, #  biasing, 3%
        "μ" => 1e-1,  # barostat timescale, irrelevant
        "Pr0" => 0.0,  # set pressure 
        "dt" => 0.2,  # timestep 
        "γ" => 0.2,  # langevan timescale
        "maxp" => 0.1,  # max uncertainty 
        "Freg" => 0.2))  # rel force uncertainty regulariser (epsilon)


weights = Dict(
    "default" => Dict("E" => 1.0, "F" => 1.0 , "V" => 1.0 ),
    )

calc_settings = Dict(
    "calculator" => "ORCA",
    "label" => "orca",
    "orca_command" => "/opt/womble/orca/orca_5.0.0/orca",
    "charge" => 0,
    "mult" => 1,
    "task" => "gradient",
    "orcasimpleinput" => " UKS B3LYP def2-SV(P) def2/J D3BJ NOAUTOSTART",
    "orcablocks" => "%scf Convergence Tight SmearTemp 5000 end")


r0 = 1.1
species = [:C, :H]

Bsite = rpi_basis(species = species,
                  N = 3,       # correlation order = body-order - 1
                  maxdeg = 18,  # polynomial degree
                  r0 = r0,     # estimate for NN distance
                  rin = 0.6,
                  rcut = 4.4,   # domain for radial basis (cf documentation)
                  pin = 2)                     # require smooth inner cutoff

Bpair = pair_basis(species = species,
                   r0 = r0,
                   maxdeg = 18,
                   rcut = 5.5,
                   rin = 0.0,
                   pin = 0 )   # pin = 0 means no inner cutoff

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

al_HAL = HAL.RUN.run_HAL(Vref, weights, al, start_configs, run_info, calc_settings, B)
