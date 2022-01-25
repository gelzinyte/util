using Pkg
Pkg.activate("/home/eg475/ace")

using IPFitting, ACE, JuLIP, Plots

using ACE: z2i, i2z, order
using LinearAlgebra
using ACE.RPI: get_maxn
using ACE.Transforms: multitransform

r0 = 1.1        # typical lengthscale for distance transform 
r0_2b = 1.1
N = 3           # correlation order = body-order - 1
deg_pair = 4

zH = AtomicNumber(:H);
zC = AtomicNumber(:C);


E0_H = -13.547478676206193
E0_C = -1028.4321581399468

Vref = OneBody(:C => E0_C, :H => E0_H)

println("Training set")
train_set = IPFitting.Data.read_xyz("/home/eg475/scripts/tests/files/tiny_gap.train_set.xyz", 
                               energy_key="dft_energy", force_key="dft_forces");


#n weights
Dn = Dict( "default" => 1.0 )
#l weights
Dl = Dict( "default" => 1.5 ) 



Dd = Dict( "default" => 10,
			1=> 10, 
			2=> 10, 
			(3, zC) => 6, 
			(3, zH) => 3) 


Deg = ACE.RPI.SparsePSHDegreeM(Dn, Dl, Dd)

rcut_site = 4.0*r0
rcut_pair = 5.0*r0_2b



maxdeg = 1.0
pcut=2
pin=2
constants = false
species=[:C, :H]

# keep the transforms as 
trans_general = PolyTransform(2, r0)
transforms = Dict(
		(:C, :C) => trans_general, 
		(:C, :H) => trans_general, 
		(:H, :H) => trans_general)

cutoffs = Dict(
		(:C, :C) => (0.8, rcut_site), 
		(:C, :H) => (1.6, rcut_site), 
		(:H, :H) => (1.0, rcut_site))

trans = multitransform(transforms, cutoffs=cutoffs)

rbasis = transformed_jacobi(get_maxn(Deg, maxdeg, species), trans; pcut=pcut, pin=pin)
basis1p = BasicPSH1pBasis(rbasis; species=species, D=Deg)
Bsite=RPIBasis(basis1p, N, Deg, maxdeg, constants)


#
# pair potential basis
#

pair_trans = PolyTransform(2, r0_2b)
pair_rin = 0.0
rbasis = transformed_jacobi(deg_pair, pair_trans, rcut_pair, pair_rin; pcut=2, pin=0)
Bpair = PolyPairBasis(rbasis, species)

B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

@show length(B)
dbname = "ace_old_script"
@show dbname
@show length(Bpair)
@show length(Bsite)

if isfile("$(dbname)_kron.h5")
		@info("loading LSQ data")
		dB = LsqDB(dbname)
else
		@info("constructing LSQ system")
		dB = LsqDB(dbname, B, train_set);
end

ard_tol = 1e-1
ard_threshold_lambda = 100


fit_name = "$(dbname).json"
@show fit_name

#solver=(:ard, ard_tol, ard_threshold_lambda)
solver=(:rrqr, 1e-5)

IP, lsqinfo = IPFitting.Lsq.lsqfit(dB, Vref=Vref,
		solver=solver)


save_dict(fit_name,
		Dict("IP" => write_dict(IP), "info" => lsqinfo))



println("Train set")
add_fits_serial!(IP, train_set, fitkey="IP")
rmse_, rmserel_ = rmse(train_set; fitkey="IP");
rmse_table(rmse_, rmserel_)

d = load_dict(fit_name)
println("norm(c): $(norm(d["info"]["c"]))")

# println("lml_score: $(d["info"]["score"])")

println("finished execution")
