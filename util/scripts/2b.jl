using IPFitting, ACE1, Plots, ArgParse 
ENV["GKSwstype"] = "nul"


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin 
        "--param-fname", "-p"
            help="ace.json filename"
        "--type", "-t"
            help = "either \"2b\" or \"full\" for which dimer curves to plot"
        "--cc-in"
            help = "inner cutoff for cc"
            arg_type = Float32
        "--ch-in"
            help = "inner cutoff for ch"
            arg_type = Float32
        "--hh-in"
            help = "inner cutoff for hh"
            arg_type = Float32
    end
	return parse_args(s)
end

function main()
    params = parse_commandline()
	ip_fname = params["param-fname"]
    type = params["type"]
    title = replace(ip_fname, ".json"=>"") 
    if type == "2b"
        title *= "_2b_ACE_only"
    elseif type == "full"
        title *= "_full_ACE_on_dimer"
    end

    IP = read_dict(load_dict(ip_fname)["IP"])

    R = [r for r in 0.1:0.01:8]
    p = plot()

    elements = [:C, :H]
    symbols = Dict(:C=>"C", :H=>"H")

    isolated_energies = Dict(
        :C => -1028.4321581399468,
        :H => -13.547478676206193)

    inner_cutoffs = Dict(
        (:C, :C) => params["cc-in"],
        (:C, :H) => params["ch-in"],
        (:H, :H) => params["hh-in"])

    colors = Dict(
        (:C, :C) => "blue",
        (:C, :H) => "orange",
        (:H, :H) => "green")

    for i1 in 1:2
        for i2 in i1:2

            el1 = elements[i1]
            el2 = elements[i2]
            label  = symbols[el1]*symbols[el2]
            # color = colors[(el1, el2)]
            color = i1 + i2 - 1

            if type == "2b"
                E = [(dimer_energy_2b(IP.components[2], r, AtomicNumber(el1), AtomicNumber(el2))) for r in R];
            elseif type == "full"
                E = [(dimer_energy_full(IP, r, AtomicNumber(el1), AtomicNumber(el2))) for r in R];
                shift = isolated_energies[el1] + isolated_energies[el2]
                label = label * " wrt $(round(shift, digits=2)) eV"
            end

            inner_cutoff = inner_cutoffs[(el1, el2)]
            if !isnothing(inner_cutoff)
                vline!([inner_cutoff], color=color, label=nothing, linestyle=:dash)
                label = label * " inner cutoff $(inner_cutoff)"
            end

            plot!(p, R, E, label=label, color=color)

        end
    end

    ylims!(p, (-10.0, 10.0))
    ylabel!(p, "ev/atom")
    xlabel!(p, "separation, Å")
    title!(title, titlefontsize=8)
    savefig(title*".2b.pdf")


end

function dimer_energy_full(IP, r::Float64, spec1, spec2)
    isolated_energies = Dict(
        AtomicNumber(:C) => -1028.4321581399468,
        AtomicNumber(:H) =>-13.547478676206193)
    energy = dimer_energy_2b(IP, r, spec1, spec2)
    return energy - isolated_energies[spec1] - isolated_energies[spec2]

end

function dimer_energy_2b(IP, r::Float64, spec1, spec2)
    X = [[0.0,0.0,0.0], [0.0, 0.0, r]]
    C = [[100.0,0.0,0.0], [0.0, 100.0, 0.0],[0.0, 0.0, 100.0] ]
    at = Atoms(X, [[0.0,0.0,0.0], [0.0, 0.0, 0.0]], [0.0, 0.0], [spec1, spec2], C, false)
    return energy(IP, at)
 end

main()
