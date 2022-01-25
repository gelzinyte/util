using Pkg
Pkg.activate("/home/eg475/ace")

using IPFitting, ArgParse, ACE, LinearAlgebra


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin 
        "--param-fname", "-p"
            help="ace.json filename"
    end
	return parse_args(s)
end


function main()

	ip_fname = parse_commandline()["param-fname"]

    println("Training set")
    train_set = IPFitting.Data.read_xyz("/data/eg475/heap_of_ch/train.dft.xyz", 
                                energy_key="dft_energy", force_key="dft_forces");


    println("Test set")
    test_set = IPFitting.Data.read_xyz("/data/eg475/heap_of_ch/test.dft.xyz", 
                                energy_key="dft_energy", force_key="dft_forces");

    IP = read_dict(load_dict(ip_fname)["IP"])

    println("Train set")
    add_fits_serial!(IP, train_set, fitkey="IP")
    rmse_, rmserel_ = rmse(train_set; fitkey="IP");
    rmse_table(rmse_, rmserel_)

    println("Test set")
    add_fits_serial!(IP, test_set, fitkey="IP")
    rmse_, rmserel_ = rmse(test_set; fitkey="IP");
    rmse_table(rmse_, rmserel_)

    d = load_dict(ip_fname);

	@show(d["info"])

end

main()
