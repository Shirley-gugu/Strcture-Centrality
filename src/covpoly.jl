using Random
using LinearAlgebra
using SparseArrays
using Base.Threads: @threads, nthreads


# ============================================================
# 1. 预计算图结构和并行森林采样工作空间
# ============================================================

"""
    prepare_pwfc(G)

预计算森林采样和迭代中重复使用的数据，包括：

- 度向量；
- 邻居表；
- 对角权重；
- 转置邻接矩阵；
- 多线程森林采样工作空间。

返回的 `prep` 可在多次 pilot 采样和 main 采样中重复使用，
避免重复分配大型数组。

注意：同一个 `prep` 不应被多个外层任务同时调用。
"""
function prepare_pwfc(G::Graph)

    n = G.n
    d = degree_vector(G)
    nbr, _ = neighbor_weight(G)

    inv_diag = 1.0 ./ (1.0 .+ d)
    offdiag_weight = 1.0 ./ (2.0 .+ d)

    A = adjacency_matrix_directed(G)

    # 复制转置矩阵，避免每次 mul! 使用惰性 transpose wrapper
    At = copy(transpose(A))

    thread_num = nthreads()

    # 每个线程累计自己的森林估计结果
    partial = zeros(Float64, n, thread_num)

    # 每个线程独立使用一套工作数组
    in_forests_workspace = [
        Vector{Bool}(undef, n) for _ in 1:thread_num
    ]

    nxt_workspace = [
        Vector{Int}(undef, n) for _ in 1:thread_num
    ]

    root_workspace = [
        Vector{Int}(undef, n) for _ in 1:thread_num
    ]

    root_count_workspace = [
        zeros(Int, n) for _ in 1:thread_num
    ]

    closed_count_workspace = [
        zeros(Int, n) for _ in 1:thread_num
    ]

    root_is_neighbor_workspace = [
        Vector{Bool}(undef, n) for _ in 1:thread_num
    ]

    return (
        d = d,
        nbr = nbr,
        inv_diag = inv_diag,
        offdiag_weight = offdiag_weight,
        At = At,
        thread_num = thread_num,
        partial = partial,
        in_forests_workspace = in_forests_workspace,
        nxt_workspace = nxt_workspace,
        root_workspace = root_workspace,
        root_count_workspace = root_count_workspace,
        closed_count_workspace = closed_count_workspace,
        root_is_neighbor_workspace = root_is_neighbor_workspace,
    )
end


# ============================================================
# 2. 支持预计算工作空间的并行森林估计器
# ============================================================

"""
    forestColumnSums_parallel(G, sample_num, prep; seed=1)

利用 `prep` 中的预计算信息生成 `sample_num` 个森林样本，
返回森林矩阵列和的无偏估计。

该函数适合在 pilot 阶段被重复调用。
"""
function forestColumnSums_parallel(
    G::Graph,
    sample_num::Int,
    prep;
    seed::Int = 1,
)

    sample_num > 0 ||
        throw(ArgumentError("sample_num must be positive"))

    seed >= 0 ||
        throw(ArgumentError("seed must be nonnegative"))

    n = G.n

    d = prep.d
    nbr = prep.nbr
    diag_weight = prep.inv_diag
    offdiag_weight = prep.offdiag_weight

    thread_num = prep.thread_num

    thread_num == nthreads() ||
        throw(ArgumentError(
            "The number of Julia threads has changed since prepare_pwfc was called."
        ))

    partial = prep.partial

    # 清空上一次调用留下的累计结果
    fill!(partial, 0.0)

    @threads :static for tid in 1:thread_num

        # 对基础 seed 和线程编号进行混合
        thread_seed =
            UInt64(seed) ⊻
            (UInt64(tid) * UInt64(0x9e3779b97f4a7c15))

        rng = Xoshiro(thread_seed)

        in_forests =
            prep.in_forests_workspace[tid]

        nxt =
            prep.nxt_workspace[tid]

        root =
            prep.root_workspace[tid]

        root_count =
            prep.root_count_workspace[tid]

        closed_count =
            prep.closed_count_workspace[tid]

        root_is_neighbor =
            prep.root_is_neighbor_workspace[tid]

        local_colsum = @view partial[:, tid]

        local_samples =
            sample_num ÷ thread_num +
            (tid <= sample_num % thread_num ? 1 : 0)

        for _ in 1:local_samples

            fill!(in_forests, false)

            # =================================================
            # Step 1: 构造一棵随机根森林
            # =================================================
            @inbounds for src in 1:n

                u = src

                while !in_forests[u]

                    # 以概率 1 / (1 + d[u]) 停止
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

            # =================================================
            # Step 2: 按根节点统计节点数
            # =================================================
            fill!(root_count, 0)

            @inbounds for j in 1:n
                root_count[root[j]] += 1
            end

            # =================================================
            # Step 3: 聚合闭邻域中的根节点计数
            # =================================================
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

            # =================================================
            # Step 4: 更新所有列和估计
            # =================================================
            @inbounds for i in 1:n

                if root_is_neighbor[i]
                    local_colsum[i] += diag_weight[i]
                end

                offdiag_count = closed_count[i]

                if root[i] == i || root_is_neighbor[i]
                    offdiag_count -= 1
                end

                local_colsum[i] +=
                    offdiag_count * offdiag_weight[i]
            end
        end
    end

    # w_ii 的确定性常数项
    colsum = copy(diag_weight)

    inv_sample_num = 1.0 / sample_num

    # 汇总各线程结果
    @inbounds for tid in 1:thread_num
        local_colsum = @view partial[:, tid]

        @. colsum +=
            inv_sample_num * local_colsum
    end

    return colsum
end


"""
保持原有调用形式。单次调用时自动创建工作空间。
Pilot 阶段不建议使用该包装函数，因为会重复分配工作数组。
"""
function forestColumnSums_parallel(
    G::Graph,
    sample_num::Int;
    seed::Int = 1,
)
    prep = prepare_pwfc(G)

    return forestColumnSums_parallel(
        G,
        sample_num,
        prep;
        seed = seed,
    )
end


# ============================================================
# 3. 从森林估计器生成完整迭代轨迹
# ============================================================

"""
    iterative_trajectory!(trajectory, z0, prep)

以森林估计 `z0` 为初值，生成

    z^(0), z^(1), ..., z^(K)

其中 `trajectory` 的大小必须为

    n × (K + 1).

第 j 列对应第 j-1 步迭代估计器。
"""
function iterative_trajectory!(
    trajectory::Matrix{Float64},
    z0::AbstractVector,
    prep,
)

    n, column_num = size(trajectory)
    max_depth = column_num - 1

    length(z0) == n ||
        throw(DimensionMismatch(
            "The length of z0 does not match trajectory."
        ))

    length(prep.inv_diag) == n ||
        throw(DimensionMismatch(
            "The graph size does not match trajectory."
        ))

    inv_diag = prep.inv_diag
    At = prep.At

    copyto!(@view(trajectory[:, 1]), z0)

    for step in 1:max_depth

        previous = @view trajectory[:, step]
        current = @view trajectory[:, step + 1]

        mul!(current, At, previous)

        @. current =
            (1.0 + current) * inv_diag
    end

    return trajectory
end


"""
    iterative_trajectory(z0, prep, max_depth)

分配并返回完整迭代轨迹。
"""
function iterative_trajectory(
    z0::AbstractVector,
    prep,
    max_depth::Int,
)

    max_depth >= 0 ||
        throw(ArgumentError(
            "max_depth must be nonnegative"
        ))

    trajectory = Matrix{Float64}(
        undef,
        length(z0),
        max_depth + 1,
    )

    iterative_trajectory!(
        trajectory,
        z0,
        prep,
    )

    return trajectory
end

function fixed_combination_from_trajectory(
    z0::AbstractVector,
    prep,
    depths::AbstractVector{Int},
    alpha::AbstractVector{Float64},
)
    length(depths) == length(alpha) ||
        throw(DimensionMismatch(
            "depths and alpha must have the same length."
        ))

    any(x -> x < 0, depths) &&
        throw(ArgumentError(
            "depths must be nonnegative."
        ))

    n = length(z0)
    max_depth = maximum(depths)
    estimate = zeros(Float64, n)
    z = Vector{Float64}(z0)
    next_z = similar(z)

    @inbounds for index in eachindex(depths)
        if depths[index] == 0
            @. estimate += alpha[index] * z
        end
    end

    for step in 1:max_depth
        mul!(next_z, prep.At, z)
        @. next_z = (1.0 + next_z) * prep.inv_diag
        z, next_z = next_z, z

        @inbounds for index in eachindex(depths)
            if depths[index] == step
                @. estimate += alpha[index] * z
            end
        end
    end

    return estimate
end

function tail_depths(K::Int; count::Int = 6, start_fraction::Float64 = 0.5)
    K >= 0 ||
        throw(ArgumentError(
            "K must be nonnegative"
        ))

    count > 0 ||
        throw(ArgumentError(
            "count must be positive"
        ))

    0.0 <= start_fraction <= 1.0 ||
        throw(ArgumentError(
            "start_fraction must be in [0, 1]"
        ))

    start_depth = min(K, max(0, floor(Int, start_fraction * K)))

    if count == 1 || start_depth == K
        return [K]
    end

    return unique(round.(Int, range(
        start_depth,
        K;
        length = count,
    )))
end

function fixed_geometric_weights(
    depths;
    lambda::Float64 = 0.85,
)
    depth_vector = Float64.(collect(depths))
    p = length(depth_vector)

    p >= 2 ||
        throw(ArgumentError(
            "At least two depths are required."
        ))

    0.0 < lambda < 1.0 ||
        throw(ArgumentError(
            "lambda must be in (0, 1)."
        ))

    constraint =
        vcat(
            ones(Float64, 1, p),
            reshape(lambda .^ depth_vector, 1, p),
        )

    target = [1.0, 0.0]

    alpha =
        transpose(constraint) *
        ((constraint * transpose(constraint)) \ target)

    alpha[end] += 1.0 - sum(alpha)

    return alpha
end


# ============================================================
# 4. 求解方差最优仿射组合系数
# ============================================================

"""
    solve_variance_optimal_weights(Gamma; ridge=1e-6)

求解约束优化问题

    min α' Gamma α
    s.t. 1'α = 1.

加入相对岭正则化

    Gamma_reg = Gamma + λI,

其中

    λ = ridge × mean(diag(Gamma)).

返回：

- `alpha`：方差最优仿射组合系数；
- `Gamma_reg`：正则化后的协方差矩阵；
- `ridge_absolute`：实际加入的岭参数；
- `predicted_variance`：未正则化协方差下的预测组合方差。
"""
function solve_variance_optimal_weights(
    Gamma::AbstractMatrix;
    ridge::Float64 = 1e-6,
)

    ridge >= 0.0 ||
        throw(ArgumentError(
            "ridge must be nonnegative"
        ))

    p, q = size(Gamma)

    p == q ||
        throw(DimensionMismatch(
            "Gamma must be square."
        ))

    # 消除浮点计算产生的轻微非对称性
    Gamma_sym =
        Matrix(Symmetric(
            (Gamma + transpose(Gamma)) / 2.0
        ))

    diagonal_scale =
        sum(abs, diag(Gamma_sym)) / p

    # 所有估计器都几乎没有随机波动
    if diagonal_scale <= eps(Float64)

        alpha = zeros(Float64, p)

        # 退化情况下直接使用最后一个估计器
        alpha[end] = 1.0

        return (
            alpha = alpha,
            Gamma_reg = Gamma_sym,
            ridge_absolute = 0.0,
            predicted_variance = 0.0,
            degenerate = true,
        )
    end

    ridge_absolute =
        ridge * diagonal_scale

    Gamma_reg = copy(Gamma_sym)

    @inbounds for i in 1:p
        Gamma_reg[i, i] += ridge_absolute
    end

    one_vector = ones(Float64, p)

    unnormalized = try
        Symmetric(Gamma_reg) \ one_vector
    catch
        # 极端病态情况下使用伪逆
        pinv(Gamma_reg) * one_vector
    end

    denominator = sum(unnormalized)

    if !isfinite(denominator) ||
       abs(denominator) <=
       sqrt(eps(Float64)) * max(norm(unnormalized, 1), 1.0)

        alpha = zeros(Float64, p)
        alpha[end] = 1.0

        return (
            alpha = alpha,
            Gamma_reg = Gamma_reg,
            ridge_absolute = ridge_absolute,
            predicted_variance = Gamma_sym[end, end],
            degenerate = true,
        )
    end

    alpha = unnormalized / denominator

    # 修正浮点误差，确保系数和严格接近 1
    alpha[end] += 1.0 - sum(alpha)

    predicted_variance =
        dot(alpha, Gamma_sym * alpha)

    return (
        alpha = alpha,
        Gamma_reg = Gamma_reg,
        ridge_absolute = ridge_absolute,
        predicted_variance = predicted_variance,
        degenerate = false,
    )
end


# ============================================================
# 5. Pilot 阶段：估计联合协方差矩阵和最优权重
# ============================================================

"""
    estimate_pwfc_combination_weights(
        G,
        prep,
        depths;
        pilot_reps=10,
        pilot_sample_num=1,
        ridge=1e-6,
        seed=1,
    )

使用 `pilot_reps` 条独立 pilot 轨迹估计指定迭代深度之间的
联合协方差矩阵。

`depths` 指定参与组合的迭代深度。例如：

    depths = 0:5

表示组合

    z^(0), z^(1), ..., z^(5).

    depths = 1:5

表示只组合

    z^(1), ..., z^(5).

联合协方差矩阵定义为

    Gamma[i,j] =
        sum_v Cov(z_v^(depths[i]), z_v^(depths[j])).

因此求得的权重最小化整个向量的总分量方差。
"""
function estimate_pwfc_combination_weights(
    G::Graph,
    prep,
    depths;
    pilot_reps::Int = 10,
    pilot_sample_num::Int = 1,
    ridge::Float64 = 1e-6,
    seed::Int = 1,
)

    pilot_reps >= 2 ||
        throw(ArgumentError(
            "pilot_reps must be at least 2"
        ))

    pilot_sample_num > 0 ||
        throw(ArgumentError(
            "pilot_sample_num must be positive"
        ))

    depth_vector = Int.(collect(depths))

    isempty(depth_vector) &&
        throw(ArgumentError(
            "depths must not be empty"
        ))

    any(x -> x < 0, depth_vector) &&
        throw(ArgumentError(
            "All iteration depths must be nonnegative"
        ))

    length(unique(depth_vector)) ==
    length(depth_vector) ||
        throw(ArgumentError(
            "depths must not contain duplicates"
        ))

    n = G.n
    estimator_num = length(depth_vector)
    max_depth = maximum(depth_vector)

    # 第 k 个深度对应 trajectory 的第 k+1 列
    column_indices = depth_vector .+ 1

    # Pilot 轨迹均值
    mean_estimates =
        zeros(Float64, n, estimator_num)

    # Welford 更新使用的差值矩阵
    delta =
        zeros(Float64, n, estimator_num)

    # 联合协方差的未归一化累计量
    covariance_accumulator =
        zeros(Float64, estimator_num, estimator_num)

    temporary_gram =
        zeros(Float64, estimator_num, estimator_num)

    # 重复使用轨迹矩阵，避免每轮重新分配
    trajectory =
        Matrix{Float64}(
            undef,
            n,
            max_depth + 1,
        )

    for repetition in 1:pilot_reps

        # 不同 pilot 重复使用不同随机种子
        repetition_seed =
            seed +
            (repetition - 1) * 1_000_003

        z0 = forestColumnSums_parallel(
            G,
            pilot_sample_num,
            prep;
            seed = repetition_seed,
        )

        iterative_trajectory!(
            trajectory,
            z0,
            prep,
        )

        selected_estimates =
            @view trajectory[:, column_indices]

        # Welford 向量化更新：
        # delta = X_r - mean_{r-1}
        @. delta =
            selected_estimates -
            mean_estimates

        # mean_r = mean_{r-1} + delta / r
        inv_repetition = 1.0 / repetition

        @. mean_estimates +=
            inv_repetition * delta

        if repetition > 1

            # delta' * delta 是 estimator_num × estimator_num
            mul!(
                temporary_gram,
                transpose(delta),
                delta,
            )

            # Welford 恒等式：
            #
            # M2_r =
            # M2_{r-1} +
            # ((r-1)/r) delta' delta
            factor =
                (repetition - 1.0) / repetition

            @. covariance_accumulator +=
                factor * temporary_gram
        end
    end

    Gamma =
        covariance_accumulator /
        (pilot_reps - 1)

    weight_result =
        solve_variance_optimal_weights(
            Gamma;
            ridge = ridge,
        )

    alpha = weight_result.alpha

    individual_variances = diag(Gamma)

    last_variance =
        individual_variances[end]

    variance_ratio_to_last =
        last_variance > 0.0 ?
        weight_result.predicted_variance / last_variance :
        NaN

    return (
        alpha = alpha,
        depths = depth_vector,
        Gamma = Gamma,
        Gamma_reg = weight_result.Gamma_reg,
        ridge_absolute = weight_result.ridge_absolute,
        predicted_variance =
            weight_result.predicted_variance,
        individual_variances =
            individual_variances,
        variance_ratio_to_last =
            variance_ratio_to_last,
        degenerate = weight_result.degenerate,
    )
end

function estimate_pwfc_mse_combination_weights(
    G::Graph,
    prep,
    depths;
    pilot_reps::Int = 10,
    pilot_sample_num::Int = 1,
    ridge::Float64 = 1e-3,
    seed::Int = 1,
    reference_extra_iters::Int = 100,
    alpha_bounds = (-5.0, 5.0),
)

    pilot_reps >= 1 ||
        throw(ArgumentError(
            "pilot_reps must be positive"
        ))

    pilot_sample_num > 0 ||
        throw(ArgumentError(
            "pilot_sample_num must be positive"
        ))

    reference_extra_iters >= 0 ||
        throw(ArgumentError(
            "reference_extra_iters must be nonnegative"
        ))

    depth_vector = Int.(collect(depths))

    isempty(depth_vector) &&
        throw(ArgumentError(
            "depths must not be empty"
        ))

    any(x -> x < 0, depth_vector) &&
        throw(ArgumentError(
            "All iteration depths must be nonnegative"
        ))

    length(unique(depth_vector)) ==
    length(depth_vector) ||
        throw(ArgumentError(
            "depths must not contain duplicates"
        ))

    n = G.n
    estimator_num = length(depth_vector)
    max_depth = maximum(depth_vector)
    reference_depth = max_depth + reference_extra_iters
    column_indices = depth_vector .+ 1

    quadratic = zeros(Float64, estimator_num, estimator_num)
    temporary_gram = zeros(Float64, estimator_num, estimator_num)

    trajectory =
        Matrix{Float64}(
            undef,
            n,
            reference_depth + 1,
        )

    for repetition in 1:pilot_reps
        repetition_seed =
            seed +
            (repetition - 1) * 1_000_003

        z0 = forestColumnSums_parallel(
            G,
            pilot_sample_num,
            prep;
            seed = repetition_seed,
        )

        iterative_trajectory!(
            trajectory,
            z0,
            prep,
        )

        selected_estimates =
            @view trajectory[:, column_indices]

        reference_estimate =
            @view trajectory[:, reference_depth + 1]

        @inbounds for column in 1:estimator_num
            selected_column = @view selected_estimates[:, column]
            @. selected_column -= reference_estimate
        end

        mul!(
            temporary_gram,
            transpose(selected_estimates),
            selected_estimates,
        )

        @. quadratic += temporary_gram
    end

    inv_reps = 1.0 / pilot_reps
    @. quadratic *= inv_reps

    weight_result =
        solve_affine_quadratic_weights(
            quadratic,
            zeros(Float64, estimator_num);
            ridge = ridge,
            alpha_bounds = alpha_bounds,
        )

    alpha = weight_result.alpha

    return (
        alpha = alpha,
        depths = depth_vector,
        quadratic = quadratic,
        linear = zeros(Float64, estimator_num),
        ridge_absolute = weight_result.ridge_absolute,
        predicted_objective = weight_result.objective_value,
        reference_depth = reference_depth,
        alpha_bounds = alpha_bounds,
    )
end

function prepare_pwfc_iteration(G::Graph)
    d = degree_vector(G)
    A = adjacency_matrix_directed(G)

    return (
        inv_diag = 1.0 ./ (1.0 .+ d),
        At = copy(transpose(A)),
    )
end


# ============================================================
# 6. Main 阶段：生成主轨迹并进行最优组合
# ============================================================

"""
    PW_FC_optimal_combination(
        G,
        K;
        depths=nothing,
        sample_num=10,
        pilot_reps=10,
        pilot_sample_num=1,
        ridge=1e-6,
        pilot_seed=1,
        main_seed=1_000_000_007,
        return_trajectory=false,
    )

基于森林采样和迭代轨迹构造多估计器方差最优组合。

默认：

    depths = 0:K

即组合 K+1 个估计器：

    z^(0), z^(1), ..., z^(K).

返回值为 NamedTuple，其中：

- `estimate`：最终组合估计；
- `alpha`：组合系数；
- `depths`：对应迭代深度；
- `Gamma`：pilot 阶段估计的联合协方差矩阵；
- `variance_ratio_to_last`：预测组合方差与最后一个估计器方差之比；
- `trajectory`：可选的主迭代轨迹。
"""
function PW_FC_optimal_combination(
    G::Graph,
    K::Int;
    depths = nothing,
    sample_num::Int = 10,
    pilot_reps::Int = 10,
    pilot_sample_num::Int = 1,
    ridge::Float64 = 1e-3,
    pilot_seed::Int = 1,
    main_seed::Int = 1_000_000_007,
    return_trajectory::Bool = false,
    objective::Symbol = :mse,
    reference_extra_iters::Int = 100,
    alpha_bounds = (-5.0, 5.0),
    tail_count::Int = 6,
    tail_start_fraction::Float64 = 0.5,
    fixed_alpha = nothing,
)

    K >= 0 ||
        throw(ArgumentError(
            "K must be nonnegative"
        ))

    sample_num > 0 ||
        throw(ArgumentError(
            "sample_num must be positive"
        ))

    pilot_seed != main_seed ||
        throw(ArgumentError(
            "pilot_seed and main_seed must be different"
        ))

    objective in (:mse, :variance) ||
        throw(ArgumentError(
            "objective must be :mse or :variance"
        ))

    depth_vector =
        depths === nothing ?
        tail_depths(
            K;
            count = tail_count,
            start_fraction = tail_start_fraction,
        ) :
        Int.(collect(depths))

    isempty(depth_vector) &&
        throw(ArgumentError(
            "depths must not be empty"
        ))

    maximum(depth_vector) <= K ||
        throw(ArgumentError(
            "All depths must be no larger than K"
        ))

    # 一次性预计算图数据及采样工作空间
    # ========================================================
    # Pilot stage
    # ========================================================
    if fixed_alpha === nothing
        prep = prepare_pwfc(G)

        pilot_result =
            objective == :variance ?
            estimate_pwfc_combination_weights(
                G,
                prep,
                depth_vector;
                pilot_reps = pilot_reps,
                pilot_sample_num = pilot_sample_num,
                ridge = ridge,
                seed = pilot_seed,
            ) :
            estimate_pwfc_mse_combination_weights(
                G,
                prep,
                depth_vector;
                pilot_reps = pilot_reps,
                pilot_sample_num = pilot_sample_num,
                ridge = ridge,
                seed = pilot_seed,
                reference_extra_iters = reference_extra_iters,
                alpha_bounds = alpha_bounds,
            )

        alpha = pilot_result.alpha
        alpha_fixed = false
    else
        alpha = Float64.(collect(fixed_alpha))

        length(alpha) == length(depth_vector) ||
            throw(DimensionMismatch(
                "fixed_alpha must have the same length as depths."
            ))

        alpha[end] += 1.0 - sum(alpha)

        pilot_result = nothing
        alpha_fixed = true
        prep = prepare_pwfc_iteration(G)
    end

    # ========================================================
    # Main stage
    # ========================================================
    if alpha_fixed
        Random.seed!(main_seed)
        z0 = forestColumnSums(G, sample_num)
    else
        z0 = forestColumnSums_parallel(
            G,
            sample_num,
            prep;
            seed = main_seed,
        )
    end

    max_depth = maximum(depth_vector)

    if alpha_fixed && !return_trajectory
        estimate =
            fixed_combination_from_trajectory(
                z0,
                prep,
                depth_vector,
                alpha,
            )

        selected_trajectory = nothing
    else
        trajectory = iterative_trajectory(
            z0,
            prep,
            max_depth,
        )

        column_indices = depth_vector .+ 1

        selected_trajectory =
            @view trajectory[:, column_indices]

        estimate =
            selected_trajectory * alpha
    end

    return (
        estimate = estimate,
        alpha = alpha,
        depths = depth_vector,
        objective = objective,
        alpha_fixed = alpha_fixed,
        Gamma =
            !alpha_fixed && objective == :variance ?
            pilot_result.Gamma :
            nothing,
        Gamma_reg =
            !alpha_fixed && objective == :variance ?
            pilot_result.Gamma_reg :
            nothing,
        individual_variances =
            !alpha_fixed && objective == :variance ?
            pilot_result.individual_variances :
            nothing,
        predicted_variance =
            !alpha_fixed && objective == :variance ?
            pilot_result.predicted_variance :
            nothing,
        variance_ratio_to_last =
            !alpha_fixed && objective == :variance ?
            pilot_result.variance_ratio_to_last :
            nothing,
        predicted_objective =
            !alpha_fixed && objective == :mse ?
            pilot_result.predicted_objective :
            nothing,
        reference_depth =
            !alpha_fixed && objective == :mse ?
            pilot_result.reference_depth :
            nothing,
        ridge_absolute =
            alpha_fixed ? 0.0 : pilot_result.ridge_absolute,
        degenerate =
            !alpha_fixed && objective == :variance ?
            pilot_result.degenerate :
            false,
        trajectory =
            return_trajectory && selected_trajectory !== nothing ?
            copy(selected_trajectory) :
            nothing,
    )
end

function solve_affine_quadratic_weights(
    quadratic::AbstractMatrix,
    linear::AbstractVector;
    ridge::Float64 = 1e-6,
    alpha_bounds = nothing,
)
    ridge >= 0.0 ||
        throw(ArgumentError(
            "ridge must be nonnegative"
        ))

    p, q = size(quadratic)

    p == q ||
        throw(DimensionMismatch(
            "quadratic must be square."
        ))

    length(linear) == p ||
        throw(DimensionMismatch(
            "linear term length does not match quadratic."
        ))

    Q_sym =
        Matrix(Symmetric(
            (quadratic + transpose(quadratic)) / 2.0
        ))

    diagonal_scale =
        sum(abs, diag(Q_sym)) / p

    ridge_absolute =
        diagonal_scale > eps(Float64) ?
        ridge * diagonal_scale :
        ridge

    Q_reg = copy(Q_sym)

    @inbounds for i in 1:p
        Q_reg[i, i] += ridge_absolute
    end

    one_vector = ones(Float64, p)
    kkt = zeros(Float64, p + 1, p + 1)
    rhs = zeros(Float64, p + 1)

    kkt[1:p, 1:p] .= Q_reg
    kkt[1:p, p + 1] .= one_vector
    kkt[p + 1, 1:p] .= one_vector
    rhs[1:p] .= linear
    rhs[p + 1] = 1.0

    solution = try
        kkt \ rhs
    catch
        pinv(kkt) * rhs
    end

    alpha = solution[1:p]

    if alpha_bounds !== nothing
        lower, upper = alpha_bounds

        lower <= upper ||
            throw(ArgumentError(
                "alpha_bounds must satisfy lower <= upper"
            ))

        alpha .= clamp.(alpha, lower, upper)
        alpha_sum = sum(alpha)

        if abs(alpha_sum) <=
           sqrt(eps(Float64)) * max(norm(alpha, 1), 1.0)
            fill!(alpha, 0.0)
            alpha[end] = 1.0
        else
            alpha ./= alpha_sum
        end
    else
        alpha[end] += 1.0 - sum(alpha)
    end

    objective_value =
        dot(alpha, Q_sym * alpha) -
        2.0 * dot(linear, alpha)

    return (
        alpha = alpha,
        quadratic_reg = Q_reg,
        ridge_absolute = ridge_absolute,
        objective_value = objective_value,
    )
end
