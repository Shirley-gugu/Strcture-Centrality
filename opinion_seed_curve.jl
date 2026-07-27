include("src/graph.jl")
include("src/forest.jl")
include("src/rw.jl")
include("src/FoIter.jl")
include("src/iter.jl")
include("opt.jl")
include("max.jl")
include("src/mis_adapter.jl")

using LinearAlgebra
using Printf

const DATASET = get(ENV, "DATASET", "Wiki-Vote.txt")
const RESULT_DIR = get(ENV, "RESULT_DIR", "results")
const KS = [10, 100, 1000, 10000]

function csv_escape(value)
    text = string(value)
    if occursin(",", text) || occursin("\"", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_curve_csv(path, rows)
    open(path, "w") do io
        println(
            io,
            "dataset,method,time,requested_k,k,initial_value,predicted_gain,true_gain,final_value",
        )
        for row in rows
            println(
                io,
                join(
                    csv_escape.([
                        row.dataset,
                        row.method,
                        row.time,
                        row.requested_k,
                        row.k,
                        row.initial_value,
                        row.predicted_gain,
                        row.true_gain,
                        row.final_value,
                    ]),
                    ",",
                ),
            )
        end
    end
end

function latex_escape(text)
    return replace(
        string(text),
        "_" => "\\_",
        "%" => "\\%",
        "&" => "\\&",
        "#" => "\\#",
    )
end

function write_curve_tex(path, rows, methods, requested_ks)
    x_values = vcat(0, collect(requested_ks))
    symbolic_coords = join(x_values, ",")

    open(path, "w") do io
        println(io, "\\begin{tikzpicture}")

        println(
            io,
            "\\begin{axis}[",
            "width=0.95\\linewidth,",
            "height=0.62\\linewidth,",
            "xlabel={\$k\$},",
            "ylabel={Aggregate equilibrium opinion},",
            "legend pos=north west,",
            "legend style={font=\\small},",
            "grid=both,",
            "symbolic x coords={$(symbolic_coords)},",
            "xtick=data,",
            "mark size=2pt",
            "]",
        )

        for method in methods
            coords = String[]

            for requested_k in x_values
                row_index = findfirst(
                    row ->
                        row.method == method &&
                        row.requested_k == requested_k,
                    rows,
                )

                row_index === nothing && error(
                    "Missing result for method=$(method), k=$(requested_k)",
                )

                row = rows[row_index]

                push!(
                    coords,
                    @sprintf(
                        "(%d,%.12g)",
                        row.requested_k,
                        row.final_value,
                    ),
                )
            end

            println(
                io,
                "\\addplot+[thick,mark=*] coordinates {",
                join(coords, " "),
                "};",
            )

            println(
                io,
                "\\addlegendentry{",
                latex_escape(method),
                "}",
            )
        end

        println(io, "\\end{axis}")
        println(io, "\\end{tikzpicture}")
    end
end

function main()
    mkpath(RESULT_DIR)

    G = read_data(DATASET)
    s = initial_opinion_vector(
        G.n;
        distribution = :uniform,
        seed = 20260718,
    )

    ref_time = @elapsed reference_rho = Iter_exact_residual(
        G;
        tol = 1e-12,
        maxiter = 20_000,
    )
    forest_time = @elapsed forest_rho = forestZ(G, 1000)
    forest_col_time = @elapsed forest_col_rho = forestColumnSums_parallel(G, 1000)
    pwfc_time = @elapsed pwfc_rho = PW_FC(G, 100)
    mis_rho, mis_time = globalInfApprox(G, 1e-3)

    method_results = [
        (name = "Iter", rho = reference_rho, time = ref_time),
        (name = "Forest", rho = forest_rho, time = forest_time),
        (name = "ForestColumnSums", rho = forest_col_rho, time = forest_col_time),
        (name = "PW_FC", rho = pwfc_rho, time = pwfc_time),
        (name = "MIS", rho = mis_rho, time = mis_time),
    ]

    rows = NamedTuple[]
    for result in method_results
        curve = opinion_seed_k_curve(reference_rho, s, result.rho, KS)
        for point in curve
            push!(
                rows,
                (
                    dataset = DATASET,
                    method = result.name,
                    time = result.time,
                    requested_k = point.requested_k,
                    k = point.k,
                    initial_value = point.initial_value,
                    predicted_gain = point.predicted_gain,
                    true_gain = point.true_gain,
                    final_value = point.final_value,
                ),
            )
        end
    end

    csv_path = joinpath(RESULT_DIR, "opinion_seed_curve.csv")
    tex_path = joinpath(RESULT_DIR, "opinion_seed_curve.tex")
    write_curve_csv(csv_path, rows)
    write_curve_tex(tex_path, rows, [r.name for r in method_results], KS)

    println("wrote ", csv_path)
    println("wrote ", tex_path)
end

"""
Compute the aggregate-opinion curve obtained by setting selected s_i to 1.

true_rho:
    Accurate structural centrality used to evaluate the actual gain.

estimated_rho:
    Structural centrality estimated by a method and used to rank nodes.

The ranking score is estimated_rho[i] * (1 - s[i]).
"""
function opinion_seed_k_curve(
    true_rho::AbstractVector,
    s::AbstractVector,
    estimated_rho::AbstractVector,
    requested_ks,
)
    true_rho = vec(Float64.(true_rho))
    s = vec(Float64.(s))
    estimated_rho = vec(Float64.(estimated_rho))

    n = length(s)

    length(true_rho) == n ||
        throw(DimensionMismatch(
            "length(true_rho)=$(length(true_rho)), but length(s)=$n",
        ))

    length(estimated_rho) == n ||
        throw(DimensionMismatch(
            "length(estimated_rho)=$(length(estimated_rho)), but length(s)=$n",
        ))

    all(isfinite, true_rho) ||
        throw(ArgumentError("true_rho contains NaN or Inf"))

    all(isfinite, estimated_rho) ||
        throw(ArgumentError("estimated_rho contains NaN or Inf"))

    all(isfinite, s) ||
        throw(ArgumentError("s contains NaN or Inf"))

    # Initial aggregate equilibrium opinion: rho' * s
    initial_value = dot(true_rho, s)

    # The estimated marginal gain produced by setting s_i to 1
    predicted_marginal_gain = estimated_rho .* (1.0 .- s)

    # Rank nodes once. The top-k sets are nested.
    order = sortperm(predicted_marginal_gain; rev=true)

    curve = NamedTuple[]

    # Initial point, corresponding to no intervention
    push!(
        curve,
        (
            requested_k = 0,
            k = 0,
            initial_value = initial_value,
            predicted_gain = 0.0,
            true_gain = 0.0,
            final_value = initial_value,
        ),
    )

    for requested_k in requested_ks
        requested_k >= 0 ||
            throw(ArgumentError("k must be nonnegative, got $requested_k"))

        # Prevent k from exceeding the number of nodes
        k = min(Int(requested_k), n)

        if k == 0
            selected = Int[]
        else
            selected = order[1:k]
        end

        predicted_gain =
            isempty(selected) ? 0.0 :
            sum(predicted_marginal_gain[selected])

        # Evaluate the selected nodes using accurate reference_rho
        true_gain =
            isempty(selected) ? 0.0 :
            sum(true_rho[selected] .* (1.0 .- s[selected]))

        final_value = initial_value + true_gain

        push!(
            curve,
            (
                requested_k = Int(requested_k),
                k = k,
                initial_value = initial_value,
                predicted_gain = predicted_gain,
                true_gain = true_gain,
                final_value = final_value,
            ),
        )
    end

    return curve
end

main()
