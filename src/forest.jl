

include("graph.jl")

using Random
using Base.Threads
# using LinearAlgebra, SparseArrays
# using StatsBase, Laplacians





function forestZ(G::Graph, sample_num::Int)
    n = G.n
    d=degree_vector(G)
    in_forests = Vector{Bool}(undef,n)
    next = Vector{Int}(undef,n)
    root = Vector{Int}(undef,n)

    nbr,weight=neighbor_weight(G)

    ans = zeros(Float64, n)
    for _ in 1:sample_num
        fill!(in_forests, false)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                   in_forests[u] = true
                   root[u] = u
                   ans[u]+=1/sample_num
                    break
               end
                next[u] = rand(nbr[u])
                u = next[u]
            end
            r = root[u]
            u = src
            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                ans[r]+=1/sample_num
                u = next[u]
            end
        end
    end
    return ans
end

function forestZ_parallel(G::Graph, sample_num::Int; seed::Int=1)
    n = G.n
    d = degree_vector(G)
    nbr,_ = neighbor_weight(G)
 
    thread_num = nthreads()
    partial = zeros(Float64, n, thread_num)

    @threads :static for tid in 1:thread_num
        rng = Xoshiro(seed + tid * 10_000_019)
        in_forests = Vector{Bool}(undef, n)
        nxt = Vector{Int}(undef, n)
        root = Vector{Int}(undef, n)
        local_ans = @view partial[:, tid]

        local_samples = sample_num ÷ thread_num + (tid <= sample_num % thread_num ?
          1 : 0)
        
        for _ in 1:local_samples
            fill!(in_forests,false)

            for src in 1:n
                u = src
                while in_forests[u] == false
                    if rand(rng) * (d[u] + 1) < 1
                        in_forests[u] = true
                        root[u] = u
                        local_ans[u] += 1.0
                        break
                    end
                    nxt[u] = rand(rng, nbr[u])
                    u = nxt[u]
                end
                r = root[u]
                u = src
                while in_forests[u] == false
                    in_forests[u] = true
                    root[u] = r
                    local_ans[r] += 1.0
                    u = nxt[u]
                end
            end
        end
    end
    ans = vec(sum(partial, dims=2))
    ans ./= sample_num
    return ans
end

function forestZ_iter1_from_estimate(G::Graph, rho_hat::AbstractVector)
    length(rho_hat) == G.n ||
        throw(DimensionMismatch("rho_hat length must match G.n"))

    d = degree_vector(G)
    rho_tilde = ones(Float64, G.n)

    for edge_idx in 1:G.m
        u = G.u[edge_idx]
        v = G.v[edge_idx]
        rho_tilde[v] += rho_hat[u]
    end

    rho_tilde ./= (1.0 .+ d)
    return rho_tilde
end

function forestZ_iter1(G::Graph, sample_num::Int)
    rho_hat = forestZ(G, sample_num)
    return forestZ_iter1_from_estimate(G, rho_hat)
end

function forestZ_iter1_in_forest(G::Graph, sample_num::Int)
    n = G.n
    d = degree_vector(G)
    nbr, _ = neighbor_weight(G)

    inv_diag = 1.0 ./ (1.0 .+ d)
    ans = copy(inv_diag)
    sample_weight = 1.0 / sample_num

    in_forests = Vector{Bool}(undef, n)
    nxt = Vector{Int}(undef, n)
    root = Vector{Int}(undef, n)
    root_count = zeros(Int, n)
    active_roots = Int[]

    for _ in 1:sample_num
        fill!(in_forests, false)
        for r in active_roots
            root_count[r] = 0
        end
        empty!(active_roots)

        @inbounds for src in 1:n
            u = src
            while !in_forests[u]
                if rand(Float64) * (d[u] + 1.0) < 1.0
                    in_forests[u] = true
                    root[u] = u
                    root_count[u] = 1
                    push!(active_roots, u)
                    break
                end
                nxt[u] = rand(nbr[u])
                u = nxt[u]
            end

            r = root[u]
            u = src
            while !in_forests[u]
                in_forests[u] = true
                root[u] = r
                root_count[r] += 1
                u = nxt[u]
            end
        end

        @inbounds for r in active_roots
            count = root_count[r]
            for i in nbr[r]
                ans[i] += count * inv_diag[i] * sample_weight
            end
        end
    end

    return ans
end

function forestZ_iter1_in_forest_parallel(
    G::Graph,
    sample_num::Int;
    seed::Int = 1,
)
    n = G.n
    d = degree_vector(G)
    nbr, _ = neighbor_weight(G)
    inv_diag = 1.0 ./ (1.0 .+ d)

    thread_num = nthreads()
    partial = zeros(Float64, n, thread_num)

    @threads :static for tid in 1:thread_num
        rng = Xoshiro(seed + tid * 10_000_019)
        local_ans = @view partial[:, tid]
        in_forests = Vector{Bool}(undef, n)
        nxt = Vector{Int}(undef, n)
        root = Vector{Int}(undef, n)
        root_count = zeros(Int, n)
        active_roots = Int[]

        local_samples = div(sample_num, thread_num) +
            (tid <= sample_num % thread_num ? 1 : 0)

        for _ in 1:local_samples
            fill!(in_forests, false)
            for r in active_roots
                root_count[r] = 0
            end
            empty!(active_roots)

            @inbounds for src in 1:n
                u = src
                while !in_forests[u]
                    if rand(rng) * (d[u] + 1.0) < 1.0
                        in_forests[u] = true
                        root[u] = u
                        root_count[u] = 1
                        push!(active_roots, u)
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
                    root_count[r] += 1
                    u = nxt[u]
                end
            end

            @inbounds for r in active_roots
                count = root_count[r]
                for i in nbr[r]
                    local_ans[i] += count * inv_diag[i]
                end
            end
        end
    end

    ans = copy(inv_diag)
    ans .+= vec(sum(partial, dims = 2)) ./ sample_num
    return ans
end

function forestZ1(G::Graph, sample_num::Int)
    n = G.n
    d=degree_vector(G)
    in_forests = Vector{Bool}(undef,n)
    next = Vector{Int}(undef,n)
    root = Vector{Int}(undef,n)

    nbr,weight=neighbor_weight(G)
    ans = zeros(Float64, n)
    for _ in 1:sample_num
        fill!(in_forests, false)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    ans[u]+=1/sample_num
                    break
                end
                next[u] = rand(nbr[u])
                u = next[u]
            end
            r = root[u]
            u = src
            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                ans[r]+=1/sample_num
                u = next[u]
            end
        end
    end
    return ans
end

function forestPlus(G::Graph, sample_num::Int)
    n = G.n
    A=adjacency_matrix_directed(G)
    d=degree_vector(G)
    in_forests = Vector{Bool}(undef,n)
    next = Vector{Int}(undef,n)
    root = Vector{Int}(undef,n)

    nbr,weight=neighbor_weight(G)
    ans = zeros(Float64, n)
    for u in 1:n
        ans[u]=1/(1+d[u])
    end
    for _ in 1:sample_num
        fill!(in_forests, false)
        fill!(next, -1)
        fill!(root, -1)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    break
                end
                next[u] = rand(nbr[u])
                u = next[u]
            end
            r = root[u]
            u = src
            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                if A[r,u]!=0
                    ans[u]+=1/(1+d[u])/sample_num
                end
                u = next[u]
            end
        end
    end
    return ans
end

function wij_N(G::Graph, sample_num::Int)
    n = G.n
    m=G.m
    A=adjacency_matrix(G)
    d=degree_vector(G)
    in_forests = Vector{Bool}(undef,n)
    next=fill(-1,n)
    root = Vector{Int}(undef,n)

    nbr,weight=neighbor_weight(G)
    ans = zeros(Float64, m)

    for _ in 1:sample_num
        fill!(in_forests, false)
        fill!(next, -1)
        fill!(root, -1)
        for src in 1:n
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    break
                end
                next[u] = StatsBase.sample(nbr[u], Weights(weight[u]))
                u = next[u]
            end
            r = root[u]
            u = src
            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                u = next[u]
            end
        end

        for i in 1:m
            u=G.u[i]
            v=G.v[i]
            k=root[u]
            if k==v
                ans[i]+=1/(2+d[v])/sample_num
            elseif A[k,v]!=0
                ans[i]+=1/(2+d[v])/sample_num
            end
        end
    end
    return ans
end

function forestColumnSum(G::Graph, i::Int, sample_num::Int)

    n = G.n

    A = adjacency_matrix_directed(G)
    d = degree_vector(G)

    in_forests = Vector{Bool}(undef, n)
    nxt = Vector{Int}(undef, n)
    root = Vector{Int}(undef, n)

    nbr, weight = neighbor_weight(G)

    # 第 i 列和估计值
    colsum = 0.0

    # w_ii 中的常数项：1 / (1 + d_i)
    colsum += 1 / (1 + d[i])

    for _ in 1:sample_num

        fill!(in_forests, false)
        fill!(nxt, -1)
        fill!(root, -1)

        # Step 1: 用扩展 Wilson 算法生成一棵随机根森林
        for src in 1:n

            u = src

            while in_forests[u] == false

                # 以概率 1 / (1 + d[u]) 停止，并令 u 成为根
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    break
                end

                # 否则随机走向邻居
                nxt[u] = rand(nbr[u])
                u = nxt[u]
            end

            r = root[u]

            u = src

            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                u = nxt[u]
            end
        end

        # Step 2: 森林构建完成后，根据 root 判断第 i 列贡献

        # 先处理 w_ii：
        # 判断节点 i 的根是否是 i 的邻居
        r = root[i]

        if A[r, i] != 0
            colsum += 1 / (1 + d[i]) / sample_num
        end

        # 再处理 w_ji, j != i：
        # 判断节点 j 的根是否是 i 或 i 的邻居
        for j in 1:n
            if j == i
                continue
            end

            r = root[j]

            if r == i || A[r, i] != 0
                colsum += 1 / (2 + d[i]) / sample_num
            end
        end
    end

    return colsum
end

function forestColumnSums(G::Graph, sample_num::Int)

    n = G.n

    d = degree_vector(G)

    in_forests = Vector{Bool}(undef, n)
    nxt = Vector{Int}(undef, n)
    root = Vector{Int}(undef, n)

    nbr, weight = neighbor_weight(G)

    # colsum[i] 估计第 i 列和：sum_j w_{ji}
    colsum = zeros(Float64, n)
    diag_weight = 1.0 ./ (1.0 .+ d)
    offdiag_weight = 1.0 ./ (2.0 .+ d)
    sample_weight = 1.0 / sample_num

    # w_ii 的常数项：1 / (1 + d_i)
    for i in 1:n
        colsum[i] = diag_weight[i]
    end

    # 每个样本中用于聚合 root 信息
    root_count = zeros(Int, n)
    closed_count = zeros(Int, n)
    root_is_neighbor = Vector{Bool}(undef, n)

    for _ in 1:sample_num

        fill!(in_forests, false)

        # =====================================================
        # Step 1: 构建一棵随机根森林
        # =====================================================
        for src in 1:n

            u = src

            while in_forests[u] == false

                # 以概率 1 / (1 + d[u]) 停止
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    break
                end

                # 否则随机走向一个邻居
                nxt[u] = rand(nbr[u])
                u = nxt[u]
            end

            r = root[u]

            u = src

            while in_forests[u] == false
                in_forests[u] = true
                root[u] = r
                u = nxt[u]
            end
        end

        # Step 2: 按 root 聚合所有节点 j 的贡献
        # root_weight[r] = sum_{j: root[j]=r} 1/(2+d[j])
        fill!(root_count, 0)

        for j in 1:n
            r = root[j]
            root_count[r] += 1
        end

        # =====================================================
        # Step 3: 计算每个 i 的 closed neighborhood 聚合量
        # closed_weight[i] =
        # root_weight[i] + sum_{r in N(i)} root_weight[r]
        # =====================================================
        copyto!(closed_count, root_count)
        fill!(root_is_neighbor, false)

        for edge_idx in 1:G.m
            u = G.u[edge_idx]
            v = G.v[edge_idx]
            closed_count[v] += root_count[u]

            if root[v] == u
                root_is_neighbor[v] = true
            end
        end

        # =====================================================
        # Step 4: 更新所有列和
        # =====================================================
        for i in 1:n

            r = root[i]

            # 判断 root(i) 是否是 i 的邻居
            # 对应 w_ii 中的概率项
            if root_is_neighbor[i]
                colsum[i] += diag_weight[i] * sample_weight
            end

            # closed_weight[i] 包含所有满足
            # root(j) in {i} union N(i)
            # 的 j 的贡献。
            #
            # 但其中包含了 j=i 自己，需要从 off-diagonal 部分中扣掉。
            root_is_closed = (r == i || root_is_neighbor[i])

            offdiag_count = closed_count[i]

            if root_is_closed
                offdiag_count -= 1
            end

            colsum[i] += offdiag_count * offdiag_weight[i] * sample_weight
        end
    end

    return colsum
end

function forestColumnSums_parallel(G::Graph, sample_num::Int; seed::Int=1)
    n = G.n
    d = degree_vector(G)
    nbr,_ = neighbor_weight(G)

    diag_weight = 1.0 ./ (1.0 .+ d)
    offdiag_weight = 1.0 ./ (2.0 .+ d)

    thread_num = nthreads()
    partial = zeros(Float64, n, thread_num)
 
    @threads :static for tid in 1:thread_num
        rng = Xoshiro(seed + tid * 10_000_019)
        in_forests = Vector{Bool}(undef, n)
        nxt = Vector{Int}(undef, n)
        root = Vector{Int}(undef, n)
        root_count = zeros(Int, n)
        closed_count = zeros(Int, n)
        root_is_neighbor = Vector{Bool}(undef, n)
        local_colsum = @view partial[:, tid]

        local_samples = sample_num ÷ thread_num + (tid <= sample_num % thread_num ?
          1 : 0)
          for _ in 1:local_samples
            fill!(in_forests, false)

            for src in 1:n
                u = src
                while in_forests[u] == false
                    if rand(rng) * (d[u] + 1) < 1
                        in_forests[u] = true
                        root[u] = u
                        break
                    end
                    nxt[u] = rand(rng, nbr[u])
                    u = nxt[u]
                end

                r = root[u]
                u = src

                while in_forests[u] == false
                    in_forests[u] = true
                    root[u] = r
                    u = nxt[u]
                end
            end

            fill!(root_count, 0)
            for j in 1:n
                root_count[root[j]] += 1
            end

            copyto!(closed_count, root_count)
            fill!(root_is_neighbor, false)

            for edge_idx in 1:G.m
                u = G.u[edge_idx]
                v = G.v[edge_idx]
                closed_count[v] += root_count[u]

                if root[v] == u
                    root_is_neighbor[v] = true
                end
            end

            for i in 1:n
                if root_is_neighbor[i]
                    local_colsum[i] += diag_weight[i]
                end
                offdiag_count = closed_count[i]
                if root[i] == i || root_is_neighbor[i]
                    offdiag_count -= 1
                end

                local_colsum[i] += offdiag_count * offdiag_weight[i]
            end
        end
    end
    colsum=copy(diag_weight)
    colsum .+=vec(sum(partial, dims=2))./sample_num
    return colsum
end     

