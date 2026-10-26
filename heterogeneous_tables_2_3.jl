"""
Recompute manuscript Tables 2 and 3 under the heterogeneous FJ equilibrium

    Omega = (Gamma + L)^(-1) Gamma,

using a fixed moderate clipped-lognormal stubbornness vector per dataset.

The full experiment keeps the manuscript budgets:
  * Table 2: 20 repetitions; Forest/RPF/PF S=1000; IFR S=10, K=100.
  * Table 3: 30 repetitions on TwitterLists, web-Stanford, and web-Google.

The driver checkpoints after every repetition and can be rerun safely.
"""

using LinearAlgebra
using Printf
using Random
using Serialization
using SparseArrays
using Statistics
using Base.Threads

include(joinpath(@__DIR__, "..", "..", "src", "graph.jl"))

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_OUTPUT = joinpath(PROJECT_ROOT, "results", "heterogeneous_tables_2_3_20261024")
const DEFAULT_DATASETS = [
    "Wiki-Vote.txt",
    "out.dblp-cite",
    "Twitterlists",
    "web-Stanford.txt",
    "web-Google.txt",
    "soc-pokec-relationships.txt",
    "out.flickr-growth",
    "wiki-topcats.txt",
]
const VARIANCE_DATASETS = Set(["Twitterlists", "web-Stanford.txt", "web-Google.txt"])
const GAMMA_BASE_SEED = 20261019
const FOREST_BASE_SEED = 20261024
const GAMMA_MU = -0.125
const GAMMA_SIGMA = 0.5
const GAMMA_LOWER = 0.1
const GAMMA_UPPER = 10.0

safe_slug(value::AbstractString) = replace(value, r"[^A-Za-z0-9._-]+" => "_")

struct PreparedGraph
    graph::Graph
    degree::Vector{Float64}
    rowptr::Vector{Int}
    destinations::Vector{Int}
    weights::Vector{Float64}
    adjacency_transpose::Transpose{Float64, SparseMatrixCSC{Float64, Int}}
end

function prepare_graph(G::Graph)
    degree = degree_vector(G)
    counts = zeros(Int, G.n)
    @inbounds for edge in 1:G.m
        counts[G.u[edge]] += 1
    end
    rowptr = Vector{Int}(undef, G.n + 1)
    rowptr[1] = 1
    @inbounds for u in 1:G.n
        rowptr[u + 1] = rowptr[u] + counts[u]
    end
    cursor = copy(rowptr[1:end-1])
    destinations = Vector{Int}(undef, G.m)
    weights = Vector{Float64}(undef, G.m)
    @inbounds for edge in 1:G.m
        u = G.u[edge]
        position = cursor[u]
        destinations[position] = G.v[edge]
        weights[position] = abs(Float64(G.w[edge]))
        cursor[u] += 1
    end
    adjacency = SparseMatrixCSC{Float64, Int}(adjacency_matrix_directed(G))
    return PreparedGraph(G, degree, rowptr, destinations, weights, transpose(adjacency))
end

function gamma_vector(n::Int, dataset_index::Int)
    seed = GAMMA_BASE_SEED + dataset_index * 10_000_019
    rng = Xoshiro(seed)
    gamma = exp.(GAMMA_MU .+ GAMMA_SIGMA .* randn(rng, n))
    clamp!(gamma, GAMMA_LOWER, GAMMA_UPPER)
    return gamma, seed
end

@inline function weighted_destination(prepared::PreparedGraph, u::Int, rng::AbstractRNG)
    first_edge = prepared.rowptr[u]
    last_edge = prepared.rowptr[u + 1] - 1
    first_edge <= last_edge || return u
    threshold = rand(rng) * prepared.degree[u]
    @inbounds for edge in first_edge:last_edge
        threshold -= prepared.weights[edge]
        threshold < 0.0 && return prepared.destinations[edge]
    end
    return prepared.destinations[last_edge]
end

function weighted_forest_estimate(
    prepared::PreparedGraph,
    gamma::AbstractVector,
    sample_count::Int;
    seed::Int,
)
    sample_count > 0 || throw(ArgumentError("sample_count must be positive"))
    n = prepared.graph.n
    length(gamma) == n || throw(DimensionMismatch("gamma length must equal n"))
    active_threads = min(nthreads(), sample_count)
    partial = zeros(Float64, n, active_threads)

    @threads :static for worker in 1:active_threads
        rng = Xoshiro(seed + worker * 10_000_019)
        in_forest = falses(n)
        next_vertex = zeros(Int, n)
        root = zeros(Int, n)
        root_count = zeros(Int, n)
        local_sum = @view partial[:, worker]
        local_samples = div(sample_count, active_threads) +
            (worker <= sample_count % active_threads ? 1 : 0)

        for _ in 1:local_samples
            fill!(in_forest, false)
            fill!(root_count, 0)
            @inbounds for source in 1:n
                u = source
                while !in_forest[u]
                    if rand(rng) * (gamma[u] + prepared.degree[u]) < gamma[u]
                        in_forest[u] = true
                        root[u] = u
                        root_count[u] += 1
                        break
                    end
                    next_vertex[u] = weighted_destination(prepared, u, rng)
                    u = next_vertex[u]
                end
                terminal_root = root[u]
                u = source
                while !in_forest[u]
                    in_forest[u] = true
                    root[u] = terminal_root
                    root_count[terminal_root] += 1
                    u = next_vertex[u]
                end
            end
            @inbounds @simd for i in 1:n
                local_sum[i] += root_count[i]
            end
        end
    end
    estimate = vec(sum(partial; dims = 2))
    estimate ./= sample_count
    return estimate
end

function refinement_workspace(prepared::PreparedGraph, gamma::AbstractVector)
    scale = gamma ./ (gamma .+ prepared.degree)
    inverse_gamma = 1.0 ./ gamma
    return scale, inverse_gamma
end

function refine(
    prepared::PreparedGraph,
    gamma::AbstractVector,
    initial::AbstractVector,
    rounds::Int,
)
    rounds >= 0 || throw(ArgumentError("rounds must be nonnegative"))
    scale, inverse_gamma = refinement_workspace(prepared, gamma)
    current = Vector{Float64}(initial)
    next_value = similar(current)
    scaled = similar(current)
    product = similar(current)
    for _ in 1:rounds
        @. scaled = current * inverse_gamma
        mul!(product, prepared.adjacency_transpose, scaled)
        @. next_value = scale * (1.0 + product)
        current, next_value = next_value, current
    end
    return current
end

function heterogeneous_iter(
    prepared::PreparedGraph,
    gamma::AbstractVector;
    tolerance::Float64,
    max_iterations::Int,
    step_tolerance::Union{Nothing, Float64} = nothing,
)
    scale, inverse_gamma = refinement_workspace(prepared, gamma)
    current = zeros(Float64, prepared.graph.n)
    next_value = similar(current)
    scaled = similar(current)
    product = similar(current)
    residual = similar(current)
    rhs_norm = sqrt(Float64(prepared.graph.n))
    last_residual = Inf

    for iteration in 1:max_iterations
        @. scaled = current * inverse_gamma
        mul!(product, prepared.adjacency_transpose, scaled)
        @. next_value = scale * (1.0 + product)

        if step_tolerance !== nothing
            relative_step = norm(next_value - current) /
                max(norm(next_value), eps(Float64))
            if relative_step <= step_tolerance
                return copy(next_value), iteration, relative_step
            end
        else
            @. scaled = next_value * inverse_gamma
            mul!(product, prepared.adjacency_transpose, scaled)
            @. residual = 1.0 + product -
                ((gamma + prepared.degree) * inverse_gamma) * next_value
            last_residual = norm(residual) / rhs_norm
            if last_residual <= tolerance
                return copy(next_value), iteration, last_residual
            end
        end
        current, next_value = next_value, current
    end
    return copy(current), max_iterations, last_residual
end

mutable struct VarianceAccumulator
    count::Int
    mean::Vector{Float64}
    m2_total::Float64
end

VarianceAccumulator(n::Int) = VarianceAccumulator(0, zeros(Float64, n), 0.0)

function update!(accumulator::VarianceAccumulator, observation::AbstractVector)
    accumulator.count += 1
    count = accumulator.count
    increment = 0.0
    @inbounds @simd for i in eachindex(observation, accumulator.mean)
        delta = observation[i] - accumulator.mean[i]
        accumulator.mean[i] += delta / count
        increment += delta * (observation[i] - accumulator.mean[i])
    end
    accumulator.m2_total += increment
    return accumulator
end

function variance_metrics(accumulator::VarianceAccumulator, reference::AbstractVector)
    accumulator.count >= 2 || error("at least two observations are required")
    total_variance = accumulator.m2_total / (accumulator.count - 1)
    relative_std = sqrt(total_variance /
        max(sum(abs2, accumulator.mean), eps(Float64)))
    squared_bias = sum(abs2, accumulator.mean .- reference)
    relative_rmse = sqrt((total_variance + squared_bias) /
        max(sum(abs2, reference), eps(Float64)))
    return total_variance, relative_std, relative_rmse
end

function load_graph_from_existing_cache(dataset::AbstractString)
    cache_path = joinpath(PROJECT_ROOT, "results", "rpf_sample_efficiency", "cache",
        string(safe_slug(dataset), ".graph.jls"))
    source = abspath(graph_data_path(dataset))
    if isfile(cache_path)
        cached = open(deserialize, cache_path)
        cache_version_ok = !hasproperty(cached, :format_version) ||
            cached.format_version == GRAPH_CACHE_FORMAT_VERSION
        cache_shape_ok = hasproperty(cached, :graph) &&
            cached.graph isa Graph
        if cache_version_ok && cache_shape_ok &&
           cached.source_size == filesize(source) &&
           cached.source_mtime == mtime(source) &&
           cached.remap_ids == true && cached.id_offset == 0
            return cached.graph, true
        end
    end
    return read_data(dataset), false
end

function reference_path(output_dir, dataset)
    joinpath(output_dir, "reference_$(safe_slug(dataset)).jls")
end

function state_path(output_dir, dataset)
    joinpath(output_dir, "state_$(safe_slug(dataset)).jls")
end

function load_or_compute_reference(prepared, gamma, gamma_seed, dataset, output_dir;
    tolerance, max_iterations)
    path = reference_path(output_dir, dataset)
    metadata = (dataset = dataset, n = prepared.graph.n, m = prepared.graph.m,
        gamma_seed = gamma_seed, gamma_mu = GAMMA_MU, gamma_sigma = GAMMA_SIGMA,
        gamma_lower = GAMMA_LOWER, gamma_upper = GAMMA_UPPER,
        tolerance = tolerance, max_iterations = max_iterations)
    if isfile(path)
        cached = open(deserialize, path)
        if cached.metadata == metadata
            return cached.reference, cached.elapsed, cached.iterations,
                cached.residual, true
        end
    end
    reference = Float64[]
    iterations = 0
    residual = Inf
    elapsed = @elapsed begin
        reference, iterations, residual = heterogeneous_iter(prepared, gamma;
            tolerance = tolerance, max_iterations = max_iterations)
    end
    open(path, "w") do io
        serialize(io, (metadata = metadata, reference = reference,
            elapsed = elapsed, iterations = iterations, residual = residual))
    end
    return reference, elapsed, iterations, residual, false
end

function empty_state(dataset, n, variance_enabled)
    accumulators = variance_enabled ? Dict(
        :Forest => VarianceAccumulator(n),
        :RPF => VarianceAccumulator(n),
        :IFR => VarianceAccumulator(n),
    ) : Dict{Symbol, VarianceAccumulator}()
    return (dataset = dataset, completed = 0, rows = NamedTuple[],
        accumulators = accumulators)
end

function save_state(path, state)
    open(path, "w") do io
        serialize(io, state)
    end
end

function push_trial!(rows, dataset, method, repetition, elapsed, estimate, reference)
    relative_error = norm(estimate - reference) /
        max(norm(reference), eps(Float64))
    push!(rows, (dataset = dataset, method = method, repetition = repetition,
        time = elapsed, relative_error = relative_error))
end

function run_dataset(dataset, dataset_index, config)
    println("[", dataset_index, "/", length(config.datasets), "] loading ", dataset)
    G, cache_hit = load_graph_from_existing_cache(dataset)
    println("  graph n=", G.n, " m=", G.m, " cache=", cache_hit)
    prepared = prepare_graph(G)
    gamma, gamma_seed = gamma_vector(G.n, dataset_index)
    println("  gamma seed=", gamma_seed, " min=", minimum(gamma),
        " mean=", mean(gamma), " max=", maximum(gamma))

    reference, reference_time, reference_iterations, reference_residual,
        reference_cache_hit = load_or_compute_reference(
            prepared, gamma, gamma_seed, dataset, config.output_dir;
            tolerance = config.reference_tolerance,
            max_iterations = config.reference_max_iterations)
    reference_residual <= max(100 * config.reference_tolerance, 1e-9) ||
        error("reference did not converge for $dataset: residual=$reference_residual")
    println("  reference cache=", reference_cache_hit,
        " iterations=", reference_iterations,
        " residual=", @sprintf("%.3e", reference_residual),
        " time=", @sprintf("%.4f", reference_time), " s")

    variance_enabled = dataset in config.variance_datasets
    required_repetitions = variance_enabled ?
        max(config.table2_repetitions, config.table3_repetitions) :
        config.table2_repetitions
    path = state_path(config.output_dir, dataset)
    state = isfile(path) ? open(deserialize, path) :
        empty_state(dataset, G.n, variance_enabled)
    state.dataset == dataset || error("checkpoint dataset mismatch at $path")

    # Untimed compilation/warm-up on the loaded graph.
    if state.completed == 0
        warm = weighted_forest_estimate(prepared, gamma, 1;
            seed = FOREST_BASE_SEED)
        refine(prepared, gamma, warm, 1)
        heterogeneous_iter(prepared, gamma; tolerance = 0.0,
            max_iterations = 1, step_tolerance = config.mis_epsilon)
    end

    rows = state.rows
    accumulators = state.accumulators
    for repetition in (state.completed + 1):required_repetitions
        shared_seed = FOREST_BASE_SEED + dataset_index * 1_000_003 +
            repetition * 10_000_019
        forest = Float64[]
        forest_time = @elapsed forest = weighted_forest_estimate(
            prepared, gamma, config.forest_samples; seed = shared_seed)
        rpf = Float64[]
        refinement_time = @elapsed rpf = refine(prepared, gamma, forest, 1)

        if repetition <= config.table2_repetitions
            push_trial!(rows, dataset, :Forest, repetition, forest_time,
                forest, reference)
            push_trial!(rows, dataset, :RPF, repetition,
                forest_time + refinement_time, rpf, reference)
        end

        ifr_initial = Float64[]
        ifr_sampling_time = @elapsed ifr_initial = weighted_forest_estimate(
            prepared, gamma, config.ifr_samples;
            seed = shared_seed + 300_000_007)
        ifr = Float64[]
        ifr_refinement_time = @elapsed ifr = refine(
            prepared, gamma, ifr_initial, config.ifr_rounds)
        if repetition <= config.table2_repetitions
            push_trial!(rows, dataset, :IFR, repetition,
                ifr_sampling_time + ifr_refinement_time, ifr, reference)
        end

        if repetition <= config.table2_repetitions
            pf = Float64[]
            pf_time = @elapsed pf = weighted_forest_estimate(
                prepared, gamma, config.pf_samples;
                seed = shared_seed + 600_000_011)
            push_trial!(rows, dataset, :PF, repetition, pf_time, pf, reference)
        end

        if variance_enabled && repetition <= config.table3_repetitions
            update!(accumulators[:Forest], forest)
            update!(accumulators[:RPF], rpf)
            update!(accumulators[:IFR], ifr)
        end

        state = (dataset = dataset, completed = repetition, rows = rows,
            accumulators = accumulators)
        save_state(path, state)
        println("  repetition ", repetition, "/", required_repetitions,
            " Forest=", @sprintf("%.3fs", forest_time),
            " RPF=", @sprintf("%.3fs", forest_time + refinement_time),
            " IFR=", @sprintf("%.3fs", ifr_sampling_time + ifr_refinement_time))
        flush(stdout)
    end

    if !any(row -> row.method == :MIS, rows)
        mis = Float64[]
        mis_iterations = 0
        mis_step = Inf
        mis_time = @elapsed begin
            mis, mis_iterations, mis_step = heterogeneous_iter(
                prepared, gamma; tolerance = 0.0,
                max_iterations = config.mis_max_iterations,
                step_tolerance = config.mis_epsilon)
        end
        push_trial!(rows, dataset, :MIS, 0, mis_time, mis, reference)
        println("  MIS iterations=", mis_iterations,
            " relative_step=", @sprintf("%.3e", mis_step))
    end
    if !any(row -> row.method == :Iter, rows)
        push!(rows, (dataset = dataset, method = :Iter, repetition = 0,
            time = reference_time, relative_error = 0.0))
    end
    save_state(path, (dataset = dataset, completed = required_repetitions,
        rows = rows, accumulators = accumulators))

    variance_rows = NamedTuple[]
    if variance_enabled
        for method in (:Forest, :RPF, :IFR)
            total_variance, relative_std, relative_rmse =
                variance_metrics(accumulators[method], reference)
            push!(variance_rows, (dataset = dataset, method = method,
                repetitions = accumulators[method].count,
                total_variance = total_variance,
                relative_std = relative_std,
                relative_rmse = relative_rmse))
        end
    end
    graph_row = (dataset = dataset, n = G.n, m = G.m,
        gamma_seed = gamma_seed, gamma_min = minimum(gamma),
        gamma_mean = mean(gamma), gamma_max = maximum(gamma),
        reference_time = reference_time,
        reference_iterations = reference_iterations,
        reference_residual = reference_residual)
    return rows, variance_rows, graph_row
end

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join((getproperty(row, field) for field in header), ','))
        end
    end
end

function summarize_table2(rows)
    summary = NamedTuple[]
    for dataset in unique(row.dataset for row in rows)
        for method in (:Iter, :Forest, :RPF, :MIS, :PF, :IFR)
            selected = filter(row -> row.dataset == dataset && row.method == method, rows)
            isempty(selected) && continue
            times = [row.time for row in selected]
            errors = [row.relative_error for row in selected]
            push!(summary, (dataset = dataset, method = method,
                repetitions = length(selected), mean_time = mean(times),
                std_time = length(times) > 1 ? std(times) : 0.0,
                mean_relative_error = mean(errors),
                std_relative_error = length(errors) > 1 ? std(errors) : 0.0))
        end
    end
    return summary
end

function write_metadata(path, config, graph_rows)
    open(path, "w") do io
        println(io, "# Heterogeneous Tables 2--3 experiment")
        println(io, "")
        println(io, "- Equilibrium: `Omega = (Gamma + L)^(-1) Gamma`.")
        println(io, "- Gamma: `clip(LogNormal(-0.125, 0.5), 0.1, 10)`.")
        println(io, "- Gamma base seed: `", GAMMA_BASE_SEED, "`; dataset seed adds `index * 10000019`.")
        println(io, "- Forest base seed: `", FOREST_BASE_SEED, "`.")
        println(io, "- Table 2 repetitions: `", config.table2_repetitions, "`.")
        println(io, "- Table 3 repetitions: `", config.table3_repetitions, "`.")
        println(io, "- Forest/RPF/PF samples: `", config.forest_samples, "`.")
        println(io, "- IFR: `S=", config.ifr_samples, ", K=", config.ifr_rounds, "`.")
        println(io, "- MIS relative-step tolerance: `", config.mis_epsilon, "`.")
        println(io, "- Iter relative residual tolerance: `", config.reference_tolerance, "`.")
        println(io, "- Julia threads: `", nthreads(), "`.")
        println(io, "")
        println(io, "| dataset | n | m | gamma min | gamma mean | gamma max | reference iterations | residual |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in graph_rows
            println(io, "| ", row.dataset, " | ", row.n, " | ", row.m,
                " | ", @sprintf("%.6g", row.gamma_min),
                " | ", @sprintf("%.6g", row.gamma_mean),
                " | ", @sprintf("%.6g", row.gamma_max),
                " | ", row.reference_iterations,
                " | ", @sprintf("%.3e", row.reference_residual), " |")
        end
    end
end

function run_experiment(config)
    mkpath(config.output_dir)
    all_rows = NamedTuple[]
    variance_rows = NamedTuple[]
    graph_rows = NamedTuple[]
    for (index, dataset) in enumerate(config.datasets)
        rows, variances, graph_row = run_dataset(dataset, index, config)
        append!(all_rows, rows)
        append!(variance_rows, variances)
        push!(graph_rows, graph_row)
        write_csv(joinpath(config.output_dir, "table2_trials.csv"),
            (:dataset, :method, :repetition, :time, :relative_error), all_rows)
        write_csv(joinpath(config.output_dir, "table2_summary.csv"),
            (:dataset, :method, :repetitions, :mean_time, :std_time,
             :mean_relative_error, :std_relative_error),
            summarize_table2(all_rows))
        write_csv(joinpath(config.output_dir, "table3_variance.csv"),
            (:dataset, :method, :repetitions, :total_variance,
             :relative_std, :relative_rmse), variance_rows)
        write_csv(joinpath(config.output_dir, "dataset_metadata.csv"),
            (:dataset, :n, :m, :gamma_seed, :gamma_min, :gamma_mean,
             :gamma_max, :reference_time, :reference_iterations,
             :reference_residual), graph_rows)
        write_metadata(joinpath(config.output_dir, "README.md"), config, graph_rows)
        GC.gc()
    end
    println("Experiment completed: ", config.output_dir)
end

function configuration(args)
    smoke = "--smoke" in args
    output_dir = smoke ?
        joinpath(PROJECT_ROOT, "results", "heterogeneous_tables_2_3_smoke") :
        DEFAULT_OUTPUT
    return (
        datasets = smoke ? ["test.txt"] : DEFAULT_DATASETS,
        variance_datasets = smoke ? Set(["test.txt"]) : VARIANCE_DATASETS,
        output_dir = output_dir,
        table2_repetitions = smoke ? 2 : 20,
        table3_repetitions = smoke ? 3 : 30,
        forest_samples = smoke ? 20 : 1000,
        pf_samples = smoke ? 20 : 1000,
        ifr_samples = smoke ? 5 : 10,
        ifr_rounds = smoke ? 3 : 100,
        mis_epsilon = 1e-3,
        mis_max_iterations = 20_000,
        reference_tolerance = 1e-12,
        reference_max_iterations = 20_000,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    config = configuration(ARGS)
    if "--smoke" in ARGS
        G = read_data("test.txt")
        prepared = prepare_graph(G)
        gamma, _ = gamma_vector(G.n, 1)
        reference, _, _ = heterogeneous_iter(prepared, gamma;
            tolerance = 1e-12, max_iterations = 20_000)
        adjacency = adjacency_matrix_directed(G)
        dense_reference = Diagonal(gamma) *
            ((Diagonal(gamma + prepared.degree) - transpose(adjacency)) \
             ones(G.n))
        dense_error = norm(reference - dense_reference) / norm(dense_reference)
        dense_error <= 1e-10 || error("heterogeneous fixed-point self-check failed")
        println("self-check dense relative error=", @sprintf("%.3e", dense_error),
            " sum error=", @sprintf("%.3e", abs(sum(reference) - G.n)))
    end
    run_experiment(config)
end
