include("graph.jl")

using Statistics

function estimate_sampling_variance(
    estimator::Function,
    repetitions::Int;
    reference = nothing,
)
    repetitions >= 2 ||
        throw(ArgumentError("repetitions must be at least 2"))
    estimator(0)
    first_estimate = Vector{Float64}(undef, 0)
    first_time = @elapsed first_estimate = Vector{Float64}(estimator(1))
    node_mean = copy(first_estimate)
    node_m2 = zeros(Float64, length(first_estimate))
    elapsed_times = zeros(Float64, repetitions)
    elapsed_times[1] = first_time

    for repetition in 2:repetitions
        estimate = Vector{Float64}(undef, length(first_estimate))
        elapsed_times[repetition] = @elapsed estimate .= estimator(repetition)
        delta = estimate .- node_mean
        node_mean .+= delta ./ repetition
        node_m2 .+= delta .* (estimate .- node_mean)
    end

    node_variance = node_m2 ./ (repetitions - 1)
    total_variance = sum(node_variance)
    relative_standard_deviation = sqrt(
        total_variance / max(sum(abs2, node_mean), eps(Float64)),
    )

    if reference === nothing
        squared_bias = missing
        mse = missing
        relative_rmse = missing
    else
        reference_vector = Vector{Float64}(reference)
        length(reference_vector) == length(node_mean) ||
            throw(DimensionMismatch("reference length must match estimate length"))
        squared_bias = sum(abs2, node_mean .- reference_vector)
        mse = total_variance + squared_bias
        relative_rmse = sqrt(
            mse / max(sum(abs2, reference_vector), eps(Float64)),
        )
    end

    return (
        mean = node_mean,
        node_variance = node_variance,
        total_variance = total_variance,
        mean_node_variance = mean(node_variance),
        maximum_node_variance = maximum(node_variance),
        relative_standard_deviation = relative_standard_deviation,
        squared_bias = squared_bias,
        mse = mse,
        relative_rmse = relative_rmse,
        mean_time = mean(elapsed_times),
        time_standard_deviation = std(elapsed_times),
        repetitions = repetitions,
    )
end

function compare_sampling_variance(
    G::Graph;
    sample_num::Int = 1000,
    pw_iterations::Int = 100,
    repetitions::Int = 30,
    seed::Int = 1,
    reference = nothing,
    print_results::Bool = true,
)
    sample_num > 0 || throw(ArgumentError("sample_num must be positive"))
    pw_iterations >= 0 ||
        throw(ArgumentError("pw_iterations must be nonnegative"))

    repetition_seed(repetition) = seed + repetition * 10_000_019
    estimators = (
        forestZ = repetition -> forestZ_parallel(
            G, sample_num; seed = repetition_seed(repetition),
        ),
        forestZiter = repetition -> forestZ_iter1_in_forest_parallel(
            G, sample_num; seed = repetition_seed(repetition),
        ),
        PW_FC = repetition -> PW_FC(
            G, pw_iterations;
            sample_num = sample_num,
            seed = repetition_seed(repetition),
        ),
    )

    results = map(
        estimator -> estimate_sampling_variance(
            estimator, repetitions; reference = reference,
        ),
        estimators,
    )
    print_results && print_variance_comparison(results)
    return results
end

function grid_search_pw_fc(
    G::Graph;
    sample_nums = [1, 2, 5, 10, 20, 50, 100],
    iteration_nums = [0, 1, 2, 5, 10, 20, 50, 100, 200],
    repetitions::Int = 20,
    seed::Int = 1,
    reference = nothing,
    reference_tol::Float64 = 1e-12,
    reference_maxiter::Int = 20_000,
    error_tol = nothing,
    print_results::Bool = true,
)
    repetitions >= 2 ||
        throw(ArgumentError("repetitions must be at least 2"))
    isempty(sample_nums) && throw(ArgumentError("sample_nums must not be empty"))
    isempty(iteration_nums) &&
        throw(ArgumentError("iteration_nums must not be empty"))
    all(sample_num -> sample_num > 0, sample_nums) ||
        throw(ArgumentError("all sample_nums must be positive"))
    all(iteration_num -> iteration_num >= 0, iteration_nums) ||
        throw(ArgumentError("all iteration_nums must be nonnegative"))
    error_tol === nothing || error_tol > 0 ||
        throw(ArgumentError("error_tol must be positive"))

    reference_vector = reference === nothing ?
        Iter_exact_residual(
            G; tol = reference_tol, maxiter = reference_maxiter,
        ) : Vector{Float64}(reference)
    length(reference_vector) == G.n ||
        throw(DimensionMismatch("reference length must match G.n"))
    reference_norm = max(norm(reference_vector), eps(Float64))

    PW(G, forestZ_parallel(G, first(sample_nums); seed = seed),
        first(iteration_nums))

    rows = NamedTuple[]
    for sample_num in sample_nums
        for iteration_num in iteration_nums
            errors = zeros(Float64, repetitions)
            sampling_times = zeros(Float64, repetitions)
            iteration_times = zeros(Float64, repetitions)

            for repetition in 1:repetitions
                repetition_seed = seed + repetition * 10_000_019
                initial_estimate = Vector{Float64}(undef, G.n)
                sampling_times[repetition] = @elapsed begin
                    initial_estimate .= forestZ_parallel(
                        G, sample_num; seed = repetition_seed,
                    )
                end

                estimate = Vector{Float64}(undef, G.n)
                iteration_times[repetition] = @elapsed begin
                    estimate .= PW(G, initial_estimate, iteration_num)
                end
                errors[repetition] =
                    norm(estimate - reference_vector) / reference_norm
            end

            error_mean = mean(errors)
            error_standard_deviation = std(errors)
            confidence_half_width =
                1.96 * error_standard_deviation / sqrt(repetitions)
            sampling_time_mean = mean(sampling_times)
            iteration_time_mean = mean(iteration_times)
            total_times = sampling_times .+ iteration_times

            push!(rows, (
                K = Int(sample_num),
                T = Int(iteration_num),
                repetitions = repetitions,
                relative_l2_error_mean = error_mean,
                relative_l2_error_std = error_standard_deviation,
                relative_l2_error_ci95_lower =
                    max(0.0, error_mean - confidence_half_width),
                relative_l2_error_ci95_upper =
                    error_mean + confidence_half_width,
                sampling_time_mean = sampling_time_mean,
                sampling_time_std = std(sampling_times),
                iteration_time_mean = iteration_time_mean,
                iteration_time_std = std(iteration_times),
                total_time_mean = mean(total_times),
                total_time_std = std(total_times),
            ))
        end
    end

    feasible_rows = error_tol === nothing ? rows : filter(
        row -> row.relative_l2_error_ci95_upper <= error_tol, rows,
    )
    best = isempty(feasible_rows) ? nothing :
        argmin(row -> row.total_time_mean, feasible_rows)

    print_results && print_pw_fc_grid(rows; best = best)
    return (reference = reference_vector, rows = rows, best = best)
end

function print_pw_fc_grid(rows; best = nothing)
    println(rpad("K", 7), rpad("T", 7), lpad("error mean", 14),
        lpad("error std", 14), lpad("95% CI", 27),
        lpad("sample (s)", 14), lpad("iterate (s)", 14),
        lpad("total (s)", 14))
    for row in rows
        confidence_interval = string(
            "[", round(row.relative_l2_error_ci95_lower; sigdigits = 5),
            ", ", round(row.relative_l2_error_ci95_upper; sigdigits = 5), "]",
        )
        println(rpad(string(row.K), 7), rpad(string(row.T), 7),
            lpad(string(round(row.relative_l2_error_mean; sigdigits = 6)), 14),
            lpad(string(round(row.relative_l2_error_std; sigdigits = 6)), 14),
            lpad(confidence_interval, 27),
            lpad(string(round(row.sampling_time_mean; sigdigits = 6)), 14),
            lpad(string(round(row.iteration_time_mean; sigdigits = 6)), 14),
            lpad(string(round(row.total_time_mean; sigdigits = 6)), 14))
    end
    best === nothing || println(
        "best feasible combination: K=", best.K, ", T=", best.T,
        ", error upper bound=", best.relative_l2_error_ci95_upper,
        ", mean time=", best.total_time_mean, " s",
    )
    return nothing
end

function measure_error_time(
    estimator::Function,
    reference::AbstractVector,
    repetitions::Int,
)
    repetitions >= 2 ||
        throw(ArgumentError("repetitions must be at least 2"))

    estimator(0)
    reference_norm = max(norm(reference), eps(Float64))
    errors = zeros(Float64, repetitions)
    elapsed_times = zeros(Float64, repetitions)

    for repetition in 1:repetitions
        estimate = Vector{Float64}(undef, length(reference))
        elapsed_times[repetition] = @elapsed begin
            estimate .= estimator(repetition)
        end
        errors[repetition] = norm(estimate - reference) / reference_norm
    end

    error_mean = mean(errors)
    error_std = std(errors)
    confidence_half_width = 1.96 * error_std / sqrt(repetitions)
    return (
        error_mean = error_mean,
        error_std = error_std,
        error_ci95_lower = max(0.0, error_mean - confidence_half_width),
        error_ci95_upper = error_mean + confidence_half_width,
        time_mean = mean(elapsed_times),
        time_std = std(elapsed_times),
    )
end

function compare_error_time(
    G::Graph;
    sample_nums = [1, 2, 5, 10, 20, 50, 100],
    pw_iterations = [0, 1, 2, 5, 10, 20, 50, 100, 200],
    global_tolerances = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6],
    repetitions::Int = 20,
    seed::Int = 1,
    reference = nothing,
    reference_tol::Float64 = 1e-12,
    reference_maxiter::Int = 20_000,
    global_maxiter::Int = 20_000,
    pd_undirected::Bool = false,
    pd_alist = 1:G.n,
    csv_path = nothing,
    print_results::Bool = true,
)
    repetitions >= 2 ||
        throw(ArgumentError("repetitions must be at least 2"))
    isempty(sample_nums) && throw(ArgumentError("sample_nums must not be empty"))
    isempty(pw_iterations) &&
        throw(ArgumentError("pw_iterations must not be empty"))
    isempty(global_tolerances) &&
        throw(ArgumentError("global_tolerances must not be empty"))
    all(value -> value > 0, sample_nums) ||
        throw(ArgumentError("all sample_nums must be positive"))
    all(value -> value >= 0, pw_iterations) ||
        throw(ArgumentError("all pw_iterations must be nonnegative"))
    all(value -> value > 0, global_tolerances) ||
        throw(ArgumentError("all global_tolerances must be positive"))

    reference_vector = reference === nothing ?
        Iter_exact_residual(
            G; tol = reference_tol, maxiter = reference_maxiter,
        ) : Vector{Float64}(reference)
    length(reference_vector) == G.n ||
        throw(DimensionMismatch("reference length must match G.n"))
    rows = NamedTuple[]
    repetition_seed(repetition) = seed + repetition * 10_000_019

    function add_row(method, K, T, tolerance, statistics)
        push!(rows, (
            method = method,
            K = K,
            T = T,
            tolerance = tolerance,
            repetitions = repetitions,
            relative_l2_error_mean = statistics.error_mean,
            relative_l2_error_std = statistics.error_std,
            relative_l2_error_ci95_lower = statistics.error_ci95_lower,
            relative_l2_error_ci95_upper = statistics.error_ci95_upper,
            time_mean = statistics.time_mean,
            time_std = statistics.time_std,
        ))
    end

    for sample_num in sample_nums
        forest_statistics = measure_error_time(
            repetition -> forestZ_parallel(
                G, sample_num; seed = repetition_seed(repetition),
            ),
            reference_vector,
            repetitions,
        )
        add_row(:forestZ_parallel, Int(sample_num), missing, missing,
            forest_statistics)

        forest_iter_statistics = measure_error_time(
            repetition -> forestZ_iter1_in_forest_parallel(
                G, sample_num; seed = repetition_seed(repetition),
            ),
            reference_vector,
            repetitions,
        )
        add_row(:forestZ_iter1_in_forest_parallel, Int(sample_num), 1,
            missing, forest_iter_statistics)

        pd_statistics = measure_error_time(
            repetition -> PD_structural_centrality(
                G,
                sample_num,
                pd_alist;
                undirected = pd_undirected,
                seed = repetition_seed(repetition),
            ),
            reference_vector,
            repetitions,
        )
        add_row(:PD_structural_centrality, Int(sample_num), missing, missing,
            pd_statistics)

        for iteration_num in pw_iterations
            pw_statistics = measure_error_time(
                repetition -> PW_FC(
                    G,
                    Int(iteration_num);
                    sample_num = Int(sample_num),
                    seed = repetition_seed(repetition),
                ),
                reference_vector,
                repetitions,
            )
            add_row(:PW_FC, Int(sample_num), Int(iteration_num), missing,
                pw_statistics)
        end
    end

    for tolerance in global_tolerances
        global_statistics = measure_error_time(
            _ -> first(globalInfApprox(
                G, Float64(tolerance); maxiter = global_maxiter,
            )),
            reference_vector,
            repetitions,
        )
        add_row(:globalInfApprox, missing, missing, Float64(tolerance),
            global_statistics)
    end

    csv_path === nothing || save_error_time_csv(csv_path, rows)
    print_results && print_error_time_comparison(rows)
    return (reference = reference_vector, rows = rows)
end

function save_error_time_csv(path, rows)
    open(path, "w") do io
        println(io,
            "method,K,T,tolerance,repetitions,relative_l2_error_mean,",
            "relative_l2_error_std,relative_l2_error_ci95_lower,",
            "relative_l2_error_ci95_upper,time_mean,time_std",
        )
        for row in rows
            values = (
                row.method, row.K, row.T, row.tolerance, row.repetitions,
                row.relative_l2_error_mean, row.relative_l2_error_std,
                row.relative_l2_error_ci95_lower,
                row.relative_l2_error_ci95_upper,
                row.time_mean, row.time_std,
            )
            println(io, join((ismissing(value) ? "" : value for value in values), ","))
        end
    end
    return path
end

function print_error_time_comparison(rows)
    println(rpad("method", 36), lpad("K", 7), lpad("T", 7),
        lpad("tolerance", 13), lpad("error mean", 14),
        lpad("error std", 14), lpad("time (s)", 14))
    for row in rows
        println(rpad(string(row.method), 36),
            lpad(ismissing(row.K) ? "-" : string(row.K), 7),
            lpad(ismissing(row.T) ? "-" : string(row.T), 7),
            lpad(ismissing(row.tolerance) ? "-" :
                string(row.tolerance), 13),
            lpad(string(round(row.relative_l2_error_mean; sigdigits = 6)), 14),
            lpad(string(round(row.relative_l2_error_std; sigdigits = 6)), 14),
            lpad(string(round(row.time_mean; sigdigits = 6)), 14))
    end
    return nothing
end

function ablation_pw_fc(
    G::Graph;
    sample_nums = [1, 5, 10, 50, 100],
    iteration_nums = [1, 5, 10, 50, 100],
    repetitions::Int = 20,
    seed::Int = 1,
    reference = nothing,
    reference_tol::Float64 = 1e-12,
    reference_maxiter::Int = 20_000,
    print_results::Bool = true,
)
    isempty(sample_nums) && throw(ArgumentError("sample_nums must not be empty"))
    isempty(iteration_nums) &&
        throw(ArgumentError("iteration_nums must not be empty"))
    all(K -> K > 0, sample_nums) ||
        throw(ArgumentError("sample_nums must be positive"))
    all(T -> T > 0, iteration_nums) ||
        throw(ArgumentError("iteration_nums must be positive"))

    reference_vector = reference === nothing ?
        Iter_exact_residual(
            G; tol = reference_tol, maxiter = reference_maxiter,
        ) : Vector{Float64}(reference)
    length(reference_vector) == G.n ||
        throw(DimensionMismatch("reference length must match G.n"))
    rows = NamedTuple[]
    repetition_seed(repetition) = seed + repetition * 10_000_019

    function add_ablation_row(variant, K, T, statistics)
        push!(rows, (
            variant = variant,
            K = K,
            T = T,
            repetitions = repetitions,
            relative_l2_error_mean = statistics.error_mean,
            relative_l2_error_std = statistics.error_std,
            relative_l2_error_ci95_lower = statistics.error_ci95_lower,
            relative_l2_error_ci95_upper = statistics.error_ci95_upper,
            time_mean = statistics.time_mean,
            time_std = statistics.time_std,
        ))
    end

    for K in sample_nums
        forest_statistics = measure_error_time(
            repetition -> forestZ_parallel(
                G, K; seed = repetition_seed(repetition),
            ),
            reference_vector,
            repetitions,
        )
        add_ablation_row(:K_positive_T_zero, Int(K), 0, forest_statistics)

        for T in iteration_nums
            deterministic_statistics = measure_error_time(
                _ -> PW(G, zeros(Float64, G.n), T),
                reference_vector,
                repetitions,
            )
            add_ablation_row(:K_zero_T_positive, 0, Int(T),
                deterministic_statistics)

            combined_statistics = measure_error_time(
                repetition -> PW_FC(
                    G,
                    T;
                    sample_num = K,
                    seed = repetition_seed(repetition),
                ),
                reference_vector,
                repetitions,
            )
            add_ablation_row(:K_positive_T_positive, Int(K), Int(T),
                combined_statistics)
        end
    end

    summary = NamedTuple[]
    for K in sample_nums, T in iteration_nums
        forest_row = first(filter(
            row -> row.variant == :K_positive_T_zero && row.K == K, rows,
        ))
        deterministic_row = first(filter(
            row -> row.variant == :K_zero_T_positive && row.T == T, rows,
        ))
        combined_row = only(filter(
            row -> row.variant == :K_positive_T_positive &&
                row.K == K && row.T == T,
            rows,
        ))
        push!(summary, (
            K = Int(K),
            T = Int(T),
            forest_error = forest_row.relative_l2_error_mean,
            deterministic_error = deterministic_row.relative_l2_error_mean,
            combined_error = combined_row.relative_l2_error_mean,
            forest_time = forest_row.time_mean,
            deterministic_time = deterministic_row.time_mean,
            combined_time = combined_row.time_mean,
            error_improvement_vs_forest =
                1.0 - combined_row.relative_l2_error_mean /
                    max(forest_row.relative_l2_error_mean, eps(Float64)),
            error_improvement_vs_deterministic =
                1.0 - combined_row.relative_l2_error_mean /
                    max(deterministic_row.relative_l2_error_mean, eps(Float64)),
        ))
    end

    print_results && print_ablation_summary(summary)
    return (reference = reference_vector, rows = rows, summary = summary)
end

function print_ablation_summary(summary)
    println(rpad("K", 7), rpad("T", 7), lpad("Forest(K,0)", 18),
        lpad("Iter(0,T)", 18), lpad("PW_FC(K,T)", 18),
        lpad("combined time", 16), lpad("vs Forest", 14),
        lpad("vs Iter", 14))
    for row in summary
        println(rpad(string(row.K), 7), rpad(string(row.T), 7),
            lpad(string(round(row.forest_error; sigdigits = 6)), 18),
            lpad(string(round(row.deterministic_error; sigdigits = 6)), 18),
            lpad(string(round(row.combined_error; sigdigits = 6)), 18),
            lpad(string(round(row.combined_time; sigdigits = 6)), 16),
            lpad(string(round(row.error_improvement_vs_forest; sigdigits = 5)), 14),
            lpad(string(round(row.error_improvement_vs_deterministic; sigdigits = 5)), 14))
    end
    return nothing
end

function print_variance_comparison(results::NamedTuple)
    println(rpad("method", 14), lpad("total variance", 18),
        lpad("relative std", 16), lpad("relative RMSE", 16),
        lpad("mean time (s)", 16))

    for method in keys(results)
        result = getproperty(results, method)
        relative_rmse = ismissing(result.relative_rmse) ? "-" :
            string(round(result.relative_rmse; sigdigits = 6))
        println(rpad(string(method), 14),
            lpad(string(round(result.total_variance; sigdigits = 6)), 18),
            lpad(string(round(result.relative_standard_deviation; sigdigits = 6)), 16),
            lpad(relative_rmse, 16),
            lpad(string(round(result.mean_time; sigdigits = 6)), 16))
    end
    return nothing
end

