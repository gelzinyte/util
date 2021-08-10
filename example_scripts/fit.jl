using Pkg 
Pkg.activate("/home/eg475/ace")

using IPFitting, ACE, JuLIP
using JuLIP.MLIPs: combine, SumIP

r0 = 1.1        # typical lengthscale for distance transform 
r0_2b = 1.1
N = 3           # correlation order = body-order - 1
deg_pair = 4

#degrees
Dd = Dict( "default" => 20,
           1 => 12, #16,
           2 => 10,  #14,
           3 => 8)  #12)

#n weights
Dn = Dict( "default" => 1.0 )
#l weights
Dl = Dict( "default" => 1.5 ) 

Deg = ACE.RPI.SparsePSHDegreeM(Dn, Dl, Dd)

E0_H = -13.547478676206193
E0_C = -1028.4321581399468
E0_O = -2040.0734198446598

Vref = OneBody(:C => E0_C, :H => E0_H, :O => E0_O)

weights = Dict(
        "nothing" => Dict("E" => 30.0, "F" => 1.0 ));                   # 

println("Training set")
al_train = IPFitting.Data.read_xyz("./train.xyz", 
                               energy_key="dft_energy", force_key="dft_forces");

println("Testing set")
al_test = IPFitting.Data.read_xyz("./test.xyz", 
                               energy_key="dft_energy", force_key="dft_forces");

println("dft-optimised")
al_dft_opt = IPFitting.Data.read_xyz("./dft_opt.xyz", 
                               energy_key="dft_opt_dft_energy", force_key="dft_opt_dft_forces");


# construction of a basic basis for site energies
Bsite = rpi_basis(species = [:C, :H, :O],
                   N = N,
                   r0 = r0,
                   D = Deg,
                   rin = 1.03, rcut = 4.0*r0,   # domain for radial basis (cf documentation)
                   maxdeg = 1.0, #maxdeg increases the entire basis size;
                   pin = 2)     # require smooth inner cutoff

# pair potential basis
Bpair = pair_basis(species = [:C, :H, :O], r0 = r0_2b, maxdeg = deg_pair,
                   rcut = 5.0 * r0_2b, rin = 0.0,
                   pin = 0 )   # pin = 0 means no inner cutoff


B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

@show length(B)

dbname = "ethanol"

dB = LsqDB(dbname, B, al_train);

IP, lsqinfo = IPFitting.Lsq.lsqfit(dB, Vref=Vref,
             solver=(:lsqr, (0.5, 1e-6)),               # w/ laplacian :itlsq (ridge strength; laplacian reg)
             asmerrs=true, weights=weights)

println("Training set")
rmse_table(lsqinfo["errors"])
save_dict("ethanol_test.json",
           Dict("IP" => write_dict(IP), "info" => lsqinfo))


println("Test set")
add_fits_serial!(IP, al_test, fitkey="IP")
rmse_, rmserel_ = rmse(al_test; fitkey="IP");
rmse_table(rmse_, rmserel_)


println("DFT-opt set")
add_fits_serial!(IP, al_dft_opt, fitkey="IP")
rmse_, rmserel_ = rmse(al_dft_opt; fitkey="IP");
rmse_table(rmse_, rmserel_)



