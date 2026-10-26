using LinearAlgebra
using Random
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "forest.jl"))

const AUDIT_SEED = 20261021
const SAMPLE_COUNT = 300_000

struct AuditGraph
    n::Int
    src::Vector{Int}
    dst::Vector{Int}
    weight::Vector{Float64}
end

function validate_graph(G::AuditGraph)
    length(G.src) == length(G.dst) == length(G.weight) || error("edge arrays differ")
    all(1 .<= G.src .<= G.n) || error("invalid source")
    all(1 .<= G.dst .<= G.n) || error("invalid destination")
    all(G.weight .> 0) || error("forest weights must be positive")
end

function matrices(G::AuditGraph, gamma::Vector{Float64})
    validate_graph(G)
    A = zeros(Float64, G.n, G.n)
    for e in eachindex(G.src)
        A[G.src[e], G.dst[e]] += G.weight[e]
    end
    d = vec(sum(A, dims=2))
    Gamma = Diagonal(gamma)
    Hstate = (Gamma + Diagonal(d) - A) \ Matrix(Gamma)
    return A, d, Hstate
end

"""Enumerate edge-record forests. Self-loops are invalid one-edge cycles;
parallel records remain distinct and therefore contribute their weights additively."""
function enumerate_root_distribution(G::AuditGraph, gamma::Vector{Float64})
    options = Vector{Vector{Int}}(undef, G.n)
    for u in 1:G.n
        options[u] = [0; [e for e in eachindex(G.src)
                         if G.src[e] == u && G.dst[e] != u]]
    end
    choice = zeros(Int, G.n)
    mass = zeros(Float64, G.n, G.n)
    total = Ref(0.0)

    function accept_choice()
        roots = zeros(Int, G.n)
        for start in 1:G.n
            seen = falses(G.n)
            u = start
            while choice[u] != 0
                seen[u] && return
                seen[u] = true
                u = G.dst[choice[u]]
            end
            roots[start] = u
        end
        weight = 1.0
        for u in 1:G.n
            edge = choice[u]
            weight *= edge == 0 ? gamma[u] : G.weight[edge]
        end
        total[] += weight
        for v in 1:G.n
            mass[v, roots[v]] += weight
        end
    end

    function visit(u::Int)
        if u > G.n
            accept_choice()
            return
        end
        for option in options[u]
            choice[u] = option
            visit(u + 1)
        end
    end
    visit(1)
    total[] > 0 || error("no valid rooted forest")
    return mass ./ total[], total[]
end

function outgoing_records(G::AuditGraph)
    records = [Int[] for _ in 1:G.n]
    d = zeros(Float64, G.n)
    for e in eachindex(G.src)
        push!(records[G.src[e]], e)
        d[G.src[e]] += G.weight[e]
    end
    return records, d
end

function weighted_choice(rng, G, records, d, u)
    z = rand(rng) * (gamma_current[u] + d[u])
    z < gamma_current[u] && return 0
    z -= gamma_current[u]
    for e in records[u]
        z -= G.weight[e]
        z < 0 && return e
    end
    return last(records[u])
end

# Kept global only to avoid allocating a closure in the high-repetition audit.
gamma_current = Float64[]

"""Wilson-style killed loop-erased walk sampler for weighted directed forests."""
function sample_weighted_roots(G::AuditGraph, gamma::Vector{Float64}, rng)
    global gamma_current = gamma
    records, d = outgoing_records(G)
    rooted = falses(G.n)
    root = zeros(Int, G.n)
    path = Int[]
    position = zeros(Int, G.n)
    for start in 1:G.n
        rooted[start] && continue
        empty!(path)
        fill!(position, 0)
        u = start
        terminal = 0
        while !rooted[u]
            if position[u] != 0
                cut = position[u]
                for q in cut:length(path)
                    position[path[q]] = 0
                end
                resize!(path, cut - 1)
            end
            push!(path, u)
            position[u] = length(path)
            edge = weighted_choice(rng, G, records, d, u)
            if edge == 0
                terminal = u
                break
            end
            u = G.dst[edge]
        end
        terminal == 0 && (terminal = root[u])
        for v in path
            rooted[v] = true
            root[v] = terminal
        end
    end
    return root
end

"""Legacy convention: weighted degree in killing, uniform outgoing record."""
function sample_legacy_roots(G::AuditGraph, gamma::Vector{Float64}, rng)
    records, d = outgoing_records(G)
    rooted = falses(G.n); root = zeros(Int, G.n); nxt = zeros(Int, G.n)
    for start in 1:G.n
        u = start
        while !rooted[u]
            if rand(rng) * (gamma[u] + d[u]) < gamma[u] || isempty(records[u])
                rooted[u] = true; root[u] = u; break
            end
            edge = rand(rng, records[u])
            nxt[u] = G.dst[edge]
            u = nxt[u]
        end
        terminal = root[u]; u = start
        while !rooted[u]
            rooted[u] = true; root[u] = terminal; u = nxt[u]
        end
    end
    return root
end

function empirical_distribution(G, gamma; samples=SAMPLE_COUNT, legacy=false)
    counts = zeros(Int, G.n, G.n)
    rng = Xoshiro(AUDIT_SEED + (legacy ? 991 : 0) + G.n)
    production_graph = Graph(
        G.n,
        length(G.src),
        copy(G.src),
        copy(G.dst),
        Float32.(G.weight),
    )
    for _ in 1:samples
        roots = legacy ? sample_legacy_roots(G, gamma, rng) :
                         sample_weighted_root_map(production_graph, gamma; rng=rng)
        for v in 1:G.n
            counts[v, roots[v]] += 1
        end
    end
    return counts ./ samples
end

function block_step(A, d, gamma, X, W)
    scale = gamma ./ (gamma .+ d)
    return scale .* (W + A' * (X ./ gamma))
end

function cases()
    return [
        ("directed_cycle", AuditGraph(3, [1,2,3], [2,3,1], [1.2,0.7,2.0]), [0.3,1.1,2.2]),
        ("sink_and_zero_outdegree", AuditGraph(4, [1,1,2,3], [2,4,4,4], [0.4,1.8,2.1,0.6]), [0.8,0.2,1.7,0.5]),
        ("weighted_branching", AuditGraph(5, [1,1,2,2,3,4,5], [2,3,3,5,4,2,1],
                                           [0.1,3.0,0.8,1.9,2.2,0.4,1.3]),
                                  [0.4,1.8,0.6,2.4,0.9]),
        ("self_loops_and_parallel", AuditGraph(4, [1,1,1,2,2,3,3,4], [1,2,2,2,3,1,4,4],
                                                [4.0,0.3,1.7,2.5,0.8,1.2,2.1,3.3]),
                                      [0.7,1.4,0.2,2.0]),
        ("mixed_six_node", AuditGraph(6, [1,1,2,3,3,4,5,5,6], [2,6,3,1,4,5,4,6,2],
                                       [0.2,1.4,2.0,0.9,0.3,1.7,0.5,2.4,1.1]),
                              [0.1,0.6,1.5,0.4,2.2,0.8]),
    ]
end

function loader_convention_audit()
    path, io = mktemp()
    try
        write(io, "10 10 9.0\n10 20 0.25\n10 20 7.0\n20 10 2.0\n")
        close(io)
        G = read_data(path; remap_ids=true)
        return (m=G.m, n=G.n, selfloops=count(G.u .== G.v),
                parallel_records=count((G.u .== 1) .& (G.v .== 2)),
                weights=Tuple(G.w))
    finally
        isopen(io) && close(io)
        rm(path; force=true)
    end
end

function run_audit()
    outdir = joinpath(@__DIR__, "..", "results", "weighted_forest_correctness")
    mkpath(outdir)
    rows = NamedTuple[]
    tolerance = max(0.006, 7sqrt(0.25 / SAMPLE_COUNT))
    for (name, G, gamma) in cases()
        exact_enum, total_mass = enumerate_root_distribution(G, gamma)
        A, d, Hstate = matrices(G, gamma)
        matrix_error = maximum(abs.(exact_enum - Hstate))
        empirical = empirical_distribution(G, gamma)
        sampling_error = maximum(abs.(empirical - exact_enum))
        legacy = empirical_distribution(G, gamma; legacy=true)
        legacy_error = maximum(abs.(legacy - exact_enum))

        W = hcat(collect(1.0:G.n) ./ sum(1:G.n), reverse(collect(1.0:G.n)) ./ sum(1:G.n))
        Hinfluence = Hstate' * W
        H0 = empirical' * W
        unbiased_error = maximum(abs.(H0 - Hinfluence))
        propagated = H0
        for _ in 1:3
            propagated = block_step(A, d, gamma, propagated, W)
        end
        C = Diagonal(gamma ./ (gamma .+ d)) * A' * Diagonal(1.0 ./ gamma)
        identity_rhs = C^3 * (H0 - Hinfluence)
        identity_error = maximum(abs.((propagated - Hinfluence) - identity_rhs))
        pass = matrix_error < 1e-11 && sampling_error <= tolerance &&
               unbiased_error <= tolerance && identity_error < 1e-11
        push!(rows, (case_name=name, n=G.n, edge_records=length(G.src),
                    total_forest_mass=total_mass, matrix_max_error=matrix_error,
                    sampler_max_error=sampling_error, shared_unbiased_max_error=unbiased_error,
                    K3_identity_max_error=identity_error, legacy_max_error=legacy_error,
                    tolerance=tolerance, pass=pass))
        @printf("%-28s matrix %.3e sampler %.3e legacy %.3e identity %.3e %s\n",
                name, matrix_error, sampling_error, legacy_error, identity_error,
                pass ? "PASS" : "FAIL")
    end
    loader = loader_convention_audit()
    csv_path = joinpath(outdir, "audit_results.csv")
    open(csv_path, "w") do io
        println(io, join(string.(propertynames(first(rows))), ','))
        for row in rows
            println(io, join(string.(values(row)), ','))
        end
    end
    open(joinpath(outdir, "loader_conventions.txt"), "w") do io
        for key in propertynames(loader)
            println(io, key, "=", getproperty(loader, key))
        end
    end
    all(row.pass for row in rows) || error("weighted forest correctness gate failed")
    loader == (m=4, n=2, selfloops=1, parallel_records=2,
               weights=(9.0f0, 0.25f0, 7.0f0, 2.0f0)) ||
        error("loader convention changed: $loader")
    maximum(row.legacy_max_error for row in rows) > 5tolerance ||
        error("weighted counterexample did not distinguish the legacy sampler")
    println("ALL PRODUCTION WEIGHTED-SAMPLER TESTS PASSED")
    println("LOADER CONVENTION: retains self-loop and duplicate records; preserves third-column weights")
end

run_audit()
