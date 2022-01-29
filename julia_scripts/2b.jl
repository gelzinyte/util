using IPFitting, ACE, Plots, ArgParse 


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin 
        "--param-fname", "-p"
            help="ace.json filename"
        "--fname", "-t"
            help="fname for plot"
    end
	return parse_args(s)
end

function main()
    args = parse_commandline()
	ip_fname = args["param-fname"]
    fname = args["fname"]
    # fname = replace(ip_fname, ".json"=>"")


    IP = read_dict(load_dict(ip_fname)["IP"])

    R = [r for r in 0.1:0.01:8]
    p = plot()

    elements = [:C, :H]
    symbols = Dict(:C=>"C", :H=>"H")

    done = String[]
    for i1 in 1:2
        for i2 in i1:2

            el1 = elements[i1]
            el2 = elements[i2]
            label  = symbols[el1]*symbols[el2]

            E = [(dimer_energy(IP.components[2], r, AtomicNumber(el1), AtomicNumber(el2))) for r in R];
            plot!(p, R, E, label=label)
        end
    end
    ylims!(p, (-10.0, 10.0))
    ylabel!(p, "ev/atom")
    xlabel!(p, "separation, Å")
    title!("2b", titlefontsize=8)
    savefig(fname)


end

function dimer_energy(IP, r::Float64, spec1, spec2)
    X = [[0.0,0.0,0.0], [0.0, 0.0, r]]
    C = [[100.0,0.0,0.0], [0.0, 100.0, 0.0],[0.0, 0.0, 100.0] ]
    at = Atoms(X, [[0.0,0.0,0.0], [0.0, 0.0, 0.0]], [0.0, 0.0], [spec1, spec2], C, false)
    return energy(IP, at)
 end

main()
