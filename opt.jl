using LinearAlgebra
using Random
using SparseArrays

function directed_system_matrix(G::Graph)
    A = adjacency_matrix_directed(G)
    d = degree_vector(G)
    return spdiagm(0 => 1.0 .+ d) - A
end

function add_directed_edges(G::Graph, edges; weight::Float32 = Float32(1.0))
    new_u = copy(G.u)
    new_v = copy(G.v)
    new_w = copy(G.w)

    for (u, v) in edges
        push!(new_u, Int(u))
        push!(new_v, Int(v))
        push!(new_w, weight)
    end

    return Graph(G.n, length(new_u), new_u, new_v, new_w)
end

function existing_directed_edges(G::Graph)
    edges = Set{Tuple{Int, Int}}()
    for edge_idx in 1:G.m
        push!(edges, (G.u[edge_idx], G.v[edge_idx]))
    end
    return edges
end

function random_directed_candidates(
    G::Graph,
    candidate_num::Int;
    seed::Union{Nothing, Int} = nothing,
)
    candidate_num >= 0 ||
        throw(ArgumentError("candidate_num must be nonnegative"))

    if seed !== nothing
        Random.seed!(seed)
    end

    max_candidates = G.n * (G.n - 1) - G.m
    candidate_num <= max_candidates ||
        throw(ArgumentError("candidate_num exceeds the number of missing directed edges"))

    blocked = existing_directed_edges(G)
    candidates = Set{Tuple{Int, Int}}()

    while length(candidates) < candidate_num
        u = rand(1:G.n)
        v = rand(1:G.n)
        if u != v && !((u, v) in blocked) && !((u, v) in candidates)
            push!(candidates, (u, v))
        end
    end

    return collect(candidates)
end

function opinion_sum_from_rho(rho::AbstractVector, s::AbstractVector)
    length(rho) == length(s) ||
        throw(DimensionMismatch("rho and s must have the same length"))
    return dot(rho, s)
end

function initial_opinion_vector(
    n::Int;
    distribution::Symbol = :uniform,
    seed::Union{Nothing, Int} = nothing,
    normal_mean::Float64 = 0.5,
    normal_std::Float64 = 0.15,
    normalize_exponential::Bool = true,
)
    n >= 0 ||
        throw(ArgumentError("n must be nonnegative"))

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    if distribution == :uniform
        return rand(rng, n)
    elseif distribution == :normal
        return clamp.(normal_mean .+ normal_std .* randn(rng, n), 0.0, 1.0)
    elseif distribution == :exponential
        s = randexp(rng, n)
        if normalize_exponential && !isempty(s)
            scale = maximum(s)
            if scale > 0.0
                s ./= scale
            end
        end
        return s
    else
        throw(ArgumentError("unknown opinion distribution: $distribution"))
    end
end

function select_opinion_seed_nodes(
    rho::AbstractVector,
    s::AbstractVector,
    k::Int;
    candidates = eachindex(s),
    target_value::Float64 = 1.0,
    require_positive_gain::Bool = true,
)
    length(rho) == length(s) ||
        throw(DimensionMismatch("rho and s must have the same length"))
    k >= 0 ||
        throw(ArgumentError("k must be nonnegative"))

    scored = Tuple{Float64, Int}[]
    for node in candidates
        gain = rho[node] * (target_value - s[node])
        if !require_positive_gain || gain > 0.0
            push!(scored, (Float64(gain), Int(node)))
        end
    end

    sort!(scored; by = x -> x[1], rev = true)
    take = min(k, length(scored))
    selected = [scored[i][2] for i in 1:take]
    gains = [scored[i][1] for i in 1:take]

    return (
        selected_nodes = selected,
        gains = gains,
        total_gain = sum(gains),
        initial_value = opinion_sum_from_rho(rho, s),
        final_value = opinion_sum_from_rho(rho, s) + sum(gains),
    )
end

function set_opinion_nodes_to_value(
    s::AbstractVector,
    nodes;
    target_value::Float64 = 1.0,
)
    new_s = Vector{Float64}(s)
    for node in nodes
        new_s[node] = target_value
    end
    return new_s
end

function select_opinion_minimization_nodes(
    rho::AbstractVector,
    s::AbstractVector,
    k::Int;
    candidates = eachindex(s),
    target_value::Float64 = 0.0,
    require_positive_decrease::Bool = true,
)
    length(rho) == length(s) ||
        throw(DimensionMismatch("rho and s must have the same length"))
    k >= 0 ||
        throw(ArgumentError("k must be nonnegative"))

    scored = Tuple{Float64, Int}[]
    for node in candidates
        decrease = rho[node] * (s[node] - target_value)
        if !require_positive_decrease || decrease > 0.0
            push!(scored, (Float64(decrease), Int(node)))
        end
    end

    sort!(scored; by = x -> x[1], rev = true)
    take = min(k, length(scored))
    selected = [scored[i][2] for i in 1:take]
    decreases = [scored[i][1] for i in 1:take]

    initial_value = opinion_sum_from_rho(rho, s)
    return (
        selected_nodes = selected,
        decreases = decreases,
        total_decrease = sum(decreases),
        initial_value = initial_value,
        final_value = initial_value - sum(decreases),
    )
end

function opinion_minimization_curve(
    reference_rho::AbstractVector,
    s::AbstractVector,
    selected_nodes;
    target_value::Float64 = 0.0,
)
    values = [opinion_sum_from_rho(reference_rho, s)]
    current = values[1]

    for node in selected_nodes
        current -= reference_rho[node] * (s[node] - target_value)
        push!(values, current)
    end

    return values
end

function opinion_minimization_comparison_from_rhos(
    reference_rho::AbstractVector,
    s::AbstractVector,
    method_results,
    k::Int;
    target_value::Float64 = 0.0,
)
    base_value = opinion_sum_from_rho(reference_rho, s)
    reference_selection = select_opinion_minimization_nodes(
        reference_rho,
        s,
        k;
        target_value = target_value,
    )
    reference_nodes = Set(reference_selection.selected_nodes)
    reference_decrease = reference_selection.total_decrease
    rows = NamedTuple[]

    for result in method_results
        name = result.name
        rho = result.rho
        elapsed = haskey(result, :time) ? result.time : NaN

        selection = select_opinion_minimization_nodes(
            rho,
            s,
            k;
            target_value = target_value,
        )

        true_decreases =
            reference_rho[selection.selected_nodes] .*
            (s[selection.selected_nodes] .- target_value)
        true_decrease = sum(true_decreases)
        final_s = set_opinion_nodes_to_value(
            s,
            selection.selected_nodes;
            target_value = target_value,
        )
        true_curve = opinion_minimization_curve(
            reference_rho,
            s,
            selection.selected_nodes;
            target_value = target_value,
        )

        push!(
            rows,
            (
                name = name,
                time = elapsed,
                rho_error = norm(reference_rho - rho) / norm(reference_rho),
                predicted_decrease = selection.total_decrease,
                true_decrease = true_decrease,
                regret = reference_decrease - true_decrease,
                relative_decrease_to_reference =
                    true_decrease / max(abs(reference_decrease), eps(Float64)),
                decrease_ratio = true_decrease / max(abs(base_value), eps(Float64)),
                final_value = opinion_sum_from_rho(reference_rho, final_s),
                overlap_with_reference =
                    length(intersect(Set(selection.selected_nodes), reference_nodes)),
                selected_nodes = selection.selected_nodes,
                value_curve = true_curve,
            ),
        )
    end

    return rows
end

function opinion_seed_comparison_from_rhos(
    reference_rho::AbstractVector,
    s::AbstractVector,
    method_results,
    k::Int;
    target_value::Float64 = 1.0,
)
    base_value = opinion_sum_from_rho(reference_rho, s)
    reference_selection = select_opinion_seed_nodes(
        reference_rho,
        s,
        k;
        target_value = target_value,
    )
    reference_nodes = Set(reference_selection.selected_nodes)
    reference_gain = reference_selection.total_gain
    rows = NamedTuple[]

    for result in method_results
        name = result.name
        rho = result.rho
        elapsed = haskey(result, :time) ? result.time : NaN

        selection = select_opinion_seed_nodes(
            rho,
            s,
            k;
            target_value = target_value,
        )

        true_gains =
            reference_rho[selection.selected_nodes] .*
            (target_value .- s[selection.selected_nodes])
        true_gain = sum(true_gains)
        final_s = set_opinion_nodes_to_value(
            s,
            selection.selected_nodes;
            target_value = target_value,
        )

        push!(
            rows,
            (
                name = name,
                time = elapsed,
                rho_error = norm(reference_rho - rho) / norm(reference_rho),
                predicted_gain = selection.total_gain,
                true_gain = true_gain,
                regret = reference_gain - true_gain,
                relative_gain_to_reference =
                    true_gain / max(abs(reference_gain), eps(Float64)),
                gain_ratio = true_gain / max(abs(base_value), eps(Float64)),
                final_value = opinion_sum_from_rho(reference_rho, final_s),
                overlap_with_reference =
                    length(intersect(Set(selection.selected_nodes), reference_nodes)),
                selected_nodes = selection.selected_nodes,
            ),
        )
    end

    return rows
end

function opinion_seed_k_curve(
    reference_rho::AbstractVector,
    s::AbstractVector,
    selector_rho::AbstractVector,
    ks;
    target_value::Float64 = 1.0,
)
    length(reference_rho) == length(s) ||
        throw(DimensionMismatch("reference_rho and s must have the same length"))
    length(selector_rho) == length(s) ||
        throw(DimensionMismatch("selector_rho and s must have the same length"))

    n = length(s)
    requested_ks = Int[]
    push!(requested_ks, 0)
    for k in ks
        k >= 0 || throw(ArgumentError("k must be nonnegative"))
        push!(requested_ks, Int(k))
    end
    requested_ks = unique(requested_ks)

    base_value = opinion_sum_from_rho(reference_rho, s)
    predicted_scores = selector_rho .* (target_value .- s)
    order = sortperm(predicted_scores; rev = true)
    true_gains =
        reference_rho[order] .* (target_value .- s[order])
    predicted_gains = predicted_scores[order]
    cumulative_true_gain = vcat(0.0, cumsum(true_gains))
    cumulative_predicted_gain = vcat(0.0, cumsum(predicted_gains))

    rows = NamedTuple[]
    for requested_k in requested_ks
        actual_k = min(requested_k, n)
        push!(
            rows,
            (
                requested_k = requested_k,
                k = actual_k,
                initial_value = base_value,
                predicted_gain = cumulative_predicted_gain[actual_k + 1],
                true_gain = cumulative_true_gain[actual_k + 1],
                final_value = base_value + cumulative_true_gain[actual_k + 1],
                selected_nodes = order[1:actual_k],
            ),
        )
    end

    return rows
end

function fj_opinion_directed(G::Graph, s::AbstractVector)
    length(s) == G.n ||
        throw(DimensionMismatch("s must have length G.n"))
    return directed_system_matrix(G) \ Vector{Float64}(s)
end

function structural_centrality_directed(
    G::Graph;
    method::Symbol = :iter,
    tol::Float64 = 1e-2,
    maxiter::Int = 20_000,
    sample_num::Int = 10,
)
    if method == :iter
        return Iter_exact_residual(G; tol = tol, maxiter = maxiter)
    elseif method == :pw_fc
        rho, _, _ = PW_FC_tol(
            G;
            sample_num = sample_num,
            tol = tol,
            maxiter = maxiter,
        )
        return rho
    elseif method == :direct
        M = directed_system_matrix(G)
        return transpose(M) \ ones(Float64, G.n)
    else
        throw(ArgumentError("unknown method: $method"))
    end
end

function greedy_opinion_sum_directed(
    G::Graph,
    s::AbstractVector,
    candidates,
    k::Int;
    require_positive_gain::Bool = false,
)
    length(s) == G.n ||
        throw(DimensionMismatch("s must have length G.n"))
    k >= 0 ||
        throw(ArgumentError("k must be nonnegative"))
    k <= length(candidates) ||
        throw(ArgumentError("k must not exceed the number of candidate edges"))

    M = directed_system_matrix(G)
    W = inv(Matrix(M))
    opinion = W * Vector{Float64}(s)
    rho = transpose(W) * ones(Float64, G.n)

    selected = Tuple{Int, Int}[]
    selected_indices = Int[]
    gains = Float64[]
    values = [sum(opinion)]
    chosen = falses(length(candidates))

    for _ in 1:k
        best_idx = 0
        best_gain = -Inf

        for idx in eachindex(candidates)
            if chosen[idx]
                continue
            end

            u, v = candidates[idx]
            denom = 1.0 + W[u, u] - W[v, u]
            if abs(denom) <= eps(Float64)
                continue
            end

            gain = rho[u] * (opinion[u] - opinion[v]) / denom
            if gain > best_gain
                best_gain = gain
                best_idx = idx
            end
        end

        if best_idx == 0
            break
        end

        if require_positive_gain && best_gain <= 0.0
            break
        end

        chosen[best_idx] = true
        push!(selected_indices, best_idx)

        u, v = candidates[best_idx]
        push!(selected, (Int(u), Int(v)))
        push!(gains, best_gain)

        denom = 1.0 + W[u, u] - W[v, u]
        col_u = copy(view(W, :, u))
        rowdiff = copy(view(W, u, :)) .- copy(view(W, v, :))

        W .-= (col_u * transpose(rowdiff)) ./ denom
        opinion .-= col_u .* ((opinion[u] - opinion[v]) / denom)
        rho .-= rowdiff .* (rho[u] / denom)

        push!(values, values[end] - best_gain)
    end

    return (
        selected_edges = selected,
        selected_indices = selected_indices,
        gains = gains,
        values = values,
        final_value = values[end],
        final_rho = rho,
        final_opinion = opinion,
        graph = add_directed_edges(G, selected),
    )
end
