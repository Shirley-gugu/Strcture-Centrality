using Random
using LinearAlgebra
using SparseArrays

"""
Precompute graph information shared by forest sampling
and iterative refinement.
"""
function prepare_PWFC(G::Graph)
    d = degree_vector(G)
    nbr, _ = neighbor_weight(G)

    inv_diag = 1.0 ./ (1.0 .+ d)
    offdiag_weight = 1.0 ./ (2.0 .+ d)

    A = adjacency_matrix_directed(G)
    At = transpose(A)

    return (
        d = d,
        nbr = nbr,
        inv_diag = inv_diag,
        offdiag_weight = offdiag_weight,
        At = At,
    )
end

function forestColumnSums(
    G::Graph,
    sample_num::Int,
    prep;
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    sample_num > 0 ||
        throw(ArgumentError("sample_num must be positive"))

    n = G.n

    d = prep.d
    nbr = prep.nbr
    diag_weight = prep.inv_diag
    offdiag_weight = prep.offdiag_weight

    in_forests = Vector{Bool}(undef, n)
    nxt = Vector{Int}(undef, n)
    root = Vector{Int}(undef, n)

    sample_weight = 1.0 / sample_num

    # Constant term of w_ii.
    colsum = copy(diag_weight)

    root_count = zeros(Int, n)
    closed_count = zeros(Int, n)
    root_is_neighbor = Vector{Bool}(undef, n)

    for _ in 1:sample_num

        fill!(in_forests, false)

        # =====================================================
        # Step 1: Generate a random rooted spanning forest
        # =====================================================
        for src in 1:n

            u = src

            while !in_forests[u]

                # Stop with probability 1 / (1 + d[u]).
                if rand(rng) * (d[u] + 1) < 1.0
                    in_forests[u] = true
                    root[u] = u
                    break
                end

                nxt[u] = rand(rng, nbr[u])
                u = nxt[u]
            end

            r = root[u]
            u = src

            while !in_forests[u]
                in_forests[u] = true
                root[u] = r
                u = nxt[u]
            end
        end

        # =====================================================
        # Step 2: Count nodes associated with each root
        # =====================================================
        fill!(root_count, 0)

        @inbounds for j in 1:n
            root_count[root[j]] += 1
        end

        # =====================================================
        # Step 3: Aggregate counts over closed neighborhoods
        # =====================================================
        copyto!(closed_count, root_count)
        fill!(root_is_neighbor, false)

        @inbounds for edge_idx in 1:G.m
            u = G.u[edge_idx]
            v = G.v[edge_idx]

            closed_count[v] += root_count[u]

            if root[v] == u
                root_is_neighbor[v] = true
            end
        end

        # =====================================================
        # Step 4: Update all column-sum estimates
        # =====================================================
        @inbounds for i in 1:n

            r = root[i]

            if root_is_neighbor[i]
                colsum[i] += diag_weight[i] * sample_weight
            end

            root_is_closed = (r == i || root_is_neighbor[i])

            offdiag_count = closed_count[i]

            if root_is_closed
                offdiag_count -= 1
            end

            colsum[i] +=
                offdiag_count *
                offdiag_weight[i] *
                sample_weight
        end
    end

    return colsum
end

function forestColumnSums(
    G::Graph,
    sample_num::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    prep = prepare_PWFC(G)
    return forestColumnSums(G, sample_num, prep; rng = rng)
end

"""
Given a forest estimate z0, compute the adjacent iterative
estimators z^(k) and z^(k+1).
"""
function adjacent_iterates(
    z0::AbstractVector,
    At,
    inv_diag::AbstractVector,
    k::Int,
)
    k >= 0 ||
        throw(ArgumentError("k must be nonnegative"))

    z = Vector{Float64}(z0)
    next_z = similar(z)

    # Compute z^(k).
    for _ in 1:k
        mul!(next_z, At, z)

        @. next_z = (1.0 + next_z) * inv_diag

        z, next_z = next_z, z
    end

    z_k = copy(z)

    # Compute z^(k+1).
    mul!(next_z, At, z)
    @. next_z = (1.0 + next_z) * inv_diag

    z_k1 = copy(next_z)

    return z_k, z_k1
end

function refine_iterate!(
    z::AbstractVector,
    At,
    inv_diag::AbstractVector;
    maxiter::Int = 100,
    tol::Float64 = 1e-10,
)
    maxiter >= 0 ||
        throw(ArgumentError("maxiter must be nonnegative"))

    next_z = similar(z)
    last_relchange = Inf

    for iter in 1:maxiter
        mul!(next_z, At, z)
        @. next_z = (1.0 + next_z) * inv_diag

        last_relchange =
            norm(next_z - z) / max(norm(next_z), eps(Float64))

        z, next_z = next_z, z

        if last_relchange <= tol
            return copy(z), iter, last_relchange
        end
    end

    return copy(z), maxiter, last_relchange
end

function choose_adjacent_k(
    z0::AbstractVector,
    At,
    inv_diag::AbstractVector;
    maxiter::Int = 100,
    tol::Float64 = 1e-8,
)
    maxiter >= 1 ||
        throw(ArgumentError("maxiter must be at least 1"))

    z = Vector{Float64}(z0)
    next_z = similar(z)
    last_relchange = Inf

    for k in 0:(maxiter - 1)
        mul!(next_z, At, z)
        @. next_z = (1.0 + next_z) * inv_diag

        last_relchange =
            norm(next_z - z) / max(norm(next_z), eps(Float64))

        if last_relchange <= tol
            return k, (
                selected_k = k,
                relchange = last_relchange,
                hit_tolerance = true,
            )
        end

        z, next_z = next_z, z
    end

    return maxiter - 1, (
        selected_k = maxiter - 1,
        relchange = last_relchange,
        hit_tolerance = false,
    )
end

"""
Estimate the variance-optimal affine-combination parameter

    z_ext = alpha * z_k + (1 - alpha) * z_k1

by minimizing either the total componentwise variance or a pilot
mean-squared error against a further-refined fixed-point reference.

The pilot samples must be independent of the samples used
to construct the final estimator.
"""
function estimate_adjacent_alpha(
    G::Graph,
    prep,
    k::Int;
    pilot_reps::Int = 30,
    pilot_sample_num::Int = 1,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    alpha_bounds = nothing,
    objective::Symbol = :mse,
    reference_extra_iters::Int = 100,
    reference_tol::Float64 = 1e-10,
)
    pilot_reps >= 2 ||
        throw(ArgumentError("pilot_reps must be at least 2"))

    pilot_sample_num > 0 ||
        throw(ArgumentError("pilot_sample_num must be positive"))

    objective in (:variance, :mse) ||
        throw(ArgumentError("objective must be :variance or :mse"))

    reference_extra_iters >= 0 ||
        throw(ArgumentError("reference_extra_iters must be nonnegative"))

    n = G.n

    mean_k = zeros(Float64, n)
    mean_k1 = zeros(Float64, n)

    # These quantities accumulate the trace of the covariance
    # and cross-covariance matrices.
    M2_k = 0.0
    M2_k1 = 0.0
    C_k_k1 = 0.0
    mse_numerator = 0.0
    mse_denominator = 0.0
    reference_iters_total = 0
    reference_relchange_max = 0.0

    for rep in 1:pilot_reps

        z0 = forestColumnSums(
            G,
            pilot_sample_num,
            prep;
            rng = rng,
        )

        z_k, z_k1 = adjacent_iterates(
            z0,
            prep.At,
            prep.inv_diag,
            k,
        )

        if objective == :mse
            z_ref, ref_iters, ref_relchange = refine_iterate!(
                copy(z_k1),
                prep.At,
                prep.inv_diag;
                maxiter = reference_extra_iters,
                tol = reference_tol,
            )

            reference_iters_total += ref_iters
            reference_relchange_max =
                max(reference_relchange_max, ref_relchange)
        else
            z_ref = nothing
        end

        inv_rep = 1.0 / rep

        # Vector-valued Welford update.
        @inbounds for i in 1:n

            delta_k = z_k[i] - mean_k[i]
            delta_k1 = z_k1[i] - mean_k1[i]

            mean_k[i] += delta_k * inv_rep
            mean_k1[i] += delta_k1 * inv_rep

            M2_k += delta_k * (z_k[i] - mean_k[i])
            M2_k1 += delta_k1 * (z_k1[i] - mean_k1[i])

            # Online cross-covariance update.
            C_k_k1 += delta_k * (z_k1[i] - mean_k1[i])

            if objective == :mse
                control = z_k[i] - z_k1[i]
                mse_numerator += control * (z_ref[i] - z_k1[i])
                mse_denominator += control * control
            end
        end
    end

    V_k = M2_k / (pilot_reps - 1)
    V_k1 = M2_k1 / (pilot_reps - 1)
    C = C_k_k1 / (pilot_reps - 1)

    # The denominator equals the total variance of z_k - z_k1.
    denominator = V_k + V_k1 - 2.0 * C

    tolerance =
        100.0 * eps(Float64) *
        max(V_k + V_k1, 1.0)

    degenerate = denominator <= tolerance

    if objective == :mse
        mse_tolerance =
            100.0 * eps(Float64) *
            max(mse_denominator, 1.0)

        if mse_denominator <= mse_tolerance
            alpha_raw = 0.0
        else
            alpha_raw = mse_numerator / mse_denominator
        end
    elseif degenerate
        # z_k and z_k1 are essentially identical.
        # Use the later iterate.
        alpha_raw = 0.0
    else
        alpha_raw = (V_k1 - C) / denominator
    end

    alpha = alpha_raw

    # Optional stabilization.
    #
    # alpha_bounds = (0.0, 1.0) gives the optimal convex combination.
    # alpha_bounds = (-5.0, 5.0) gives clipped extrapolation.
    if alpha_bounds !== nothing
        lower, upper = alpha_bounds

        lower <= upper ||
            throw(ArgumentError(
                "alpha_bounds must satisfy lower <= upper"
            ))

        alpha = clamp(alpha, lower, upper)
    end

    combined_variance =
        alpha^2 * V_k +
        (1.0 - alpha)^2 * V_k1 +
        2.0 * alpha * (1.0 - alpha) * C

    ratio_to_k1 =
        V_k1 > 0.0 ? combined_variance / V_k1 : NaN

    diagnostics = (
        V_k = V_k,
        V_k1 = V_k1,
        C_k_k1 = C,
        denominator = denominator,
        objective = objective,
        alpha_raw = alpha_raw,
        alpha_used = alpha,
        estimated_combined_variance = combined_variance,
        variance_ratio_to_k1 = ratio_to_k1,
        mse_numerator = mse_numerator,
        mse_denominator = mse_denominator,
        average_reference_iters =
            objective == :mse ?
            reference_iters_total / pilot_reps :
            0.0,
        max_reference_relchange = reference_relchange_max,
        degenerate = degenerate,
    )

    return alpha, diagnostics
end

"""
Forest sampling + iterative refinement +
pilot-optimized extrapolation of adjacent iterates.

By default this minimizes a pilot squared error against a further
refined fixed-point reference. Pass objective = :variance to recover
the original variance-only coefficient.

Returns:
    estimate:
        variance-reduced structural centrality estimate

    alpha:
        estimated extrapolation parameter

    diagnostics:
        pilot estimates of variances and covariance
"""
function PW_FC_VOE(
    G::Graph,
    k::Int;
    sample_num::Int = 10,
    pilot_reps::Int = 30,
    pilot_sample_num::Int = 1,
    seed::Integer = 20260714,
    alpha_bounds = (-5.0, 5.0),
    objective::Symbol = :mse,
    reference_extra_iters::Int = 100,
    reference_tol::Float64 = 1e-10,
    fixed_alpha = nothing,
)
    k >= 0 ||
        throw(ArgumentError("k must be nonnegative"))

    sample_num > 0 ||
        throw(ArgumentError("sample_num must be positive"))

    prep = prepare_PWFC(G)

    if fixed_alpha === nothing
        # Separate random streams ensure that alpha is estimated
        # independently of the final forest estimator.
        pilot_rng = MersenneTwister(seed)

        alpha, diagnostics = estimate_adjacent_alpha(
            G,
            prep,
            k;
            pilot_reps = pilot_reps,
            pilot_sample_num = pilot_sample_num,
            rng = pilot_rng,
            alpha_bounds = alpha_bounds,
            objective = objective,
            reference_extra_iters = reference_extra_iters,
            reference_tol = reference_tol,
        )
    else
        alpha = Float64(fixed_alpha)

        diagnostics = (
            objective = :fixed_alpha,
            alpha_raw = alpha,
            alpha_used = alpha,
            alpha_fixed = true,
            k = k,
        )
    end

    main_rng = MersenneTwister(seed + 1)

    # Independent main forest estimate.
    z0 = forestColumnSums(
        G,
        sample_num,
        prep;
        rng = main_rng,
    )

    z_k, z_k1 = adjacent_iterates(
        z0,
        prep.At,
        prep.inv_diag,
        k,
    )

    estimate = similar(z_k)

    # Equivalent to:
    #
    # estimate = alpha * z_k + (1 - alpha) * z_k1
    #
    # This form emphasizes the zero-mean control variate
    # z_k - z_k1.
    @. estimate = z_k1 + alpha * (z_k - z_k1)

    return (
        estimate = estimate,
        alpha = alpha,
        diagnostics = diagnostics,
    )
end

function PW_FC_VOE(
    G::Graph,
    ::Nothing;
    sample_num::Int = 10,
    pilot_reps::Int = 30,
    pilot_sample_num::Int = 1,
    seed::Integer = 20260714,
    alpha_bounds = (-5.0, 5.0),
    objective::Symbol = :mse,
    reference_extra_iters::Int = 100,
    reference_tol::Float64 = 1e-10,
    auto_k_sample_num::Int = sample_num,
    auto_k_maxiter::Int = 100,
    auto_k_tol::Float64 = 1e-8,
    alpha_max_k::Int = 50,
    skip_alpha_on_auto_k_failure::Bool = true,
)
    auto_k_sample_num > 0 ||
        throw(ArgumentError("auto_k_sample_num must be positive"))

    alpha_max_k >= 0 ||
        throw(ArgumentError("alpha_max_k must be nonnegative"))

    prep = prepare_PWFC(G)
    auto_rng = MersenneTwister(seed - 1)
    z0 = forestColumnSums(
        G,
        auto_k_sample_num,
        prep;
        rng = auto_rng,
    )

    selected_k, auto_k_diagnostics = choose_adjacent_k(
        z0,
        prep.At,
        prep.inv_diag;
        maxiter = auto_k_maxiter,
        tol = auto_k_tol,
    )

    should_skip_alpha =
        selected_k > alpha_max_k ||
        (
            skip_alpha_on_auto_k_failure &&
            !auto_k_diagnostics.hit_tolerance
        )

    if should_skip_alpha
        main_rng = MersenneTwister(seed + 1)
        z0_main = forestColumnSums(
            G,
            sample_num,
            prep;
            rng = main_rng,
        )

        _, z_k1 = adjacent_iterates(
            z0_main,
            prep.At,
            prep.inv_diag,
            selected_k,
        )

        return (
            estimate = z_k1,
            alpha = 0.0,
            diagnostics = (
                objective = objective,
                alpha_raw = 0.0,
                alpha_used = 0.0,
                alpha_skipped = true,
                alpha_skip_reason =
                    selected_k > alpha_max_k ?
                    :selected_k_exceeds_alpha_max_k :
                    :auto_k_did_not_hit_tolerance,
                auto_k_selected = selected_k,
                auto_k_relchange = auto_k_diagnostics.relchange,
                auto_k_hit_tolerance =
                    auto_k_diagnostics.hit_tolerance,
            ),
        )
    end

    result = PW_FC_VOE(
        G,
        selected_k;
        sample_num = sample_num,
        pilot_reps = pilot_reps,
        pilot_sample_num = pilot_sample_num,
        seed = seed,
        alpha_bounds = alpha_bounds,
        objective = objective,
        reference_extra_iters = reference_extra_iters,
        reference_tol = reference_tol,
    )

    return (
        estimate = result.estimate,
        alpha = result.alpha,
        diagnostics = merge(
            result.diagnostics,
            (
                auto_k_selected = selected_k,
                auto_k_relchange = auto_k_diagnostics.relchange,
                auto_k_hit_tolerance =
                    auto_k_diagnostics.hit_tolerance,
            ),
        ),
    )
end

function PW_FC_VOE(G::Graph; kwargs...)
    return PW_FC_VOE(G, nothing; kwargs...)
end
