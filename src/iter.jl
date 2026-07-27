using SparseArrays
using LinearAlgebra
using IterativeSolvers

function exact(G::Graph)
    e=ones(G.n)
    d=degree_vector(G)
    n = G.n
    A=adjacency_matrix_directed(G)
    #=M = sparse(I, n, n) + spdiagm(0 => d) - A
    @assert size(A, 1) == n "size(A,1) = $(size(A,1)), but G.n = $n"
    @assert size(A, 2) == n "size(A,2) = $(size(A,2)), but G.n = $n"
    @assert length(d) == n "length(d) = $(length(d)), but G.n = $n"
    z=M'\e=#
    z=e'*inv(Matrix(I+spdiagm(0=>d)-A))
    return z
end

function Iter_exact_residual(G::Graph; tol=1e-12, maxiter=2000000)
    n = G.n
    A = adjacency_matrix_directed(G)
    d = degree_vector(G)

    @assert size(A, 1) == n "size(A,1) = $(size(A,1)), but G.n = $n"
    @assert size(A, 2) == n "size(A,2) = $(size(A,2)), but G.n = $n"
    @assert length(d) == n "length(d) = $(length(d)), but G.n = $n"

    e = ones(Float64, n)
    At = transpose(A)
    denom = 1.0 .+ d

    # 初始向量
    z = zeros(Float64, n)

    for iter in 1:maxiter
        # 原始固定点迭代：
        # z^{(t+1)} = (I+D)^{-1}(1 + A^T z^{(t)})
        z_new = (e .+ At * z) ./ denom

        # 残差：
        # r = 1 - (I + D - A^T) z_new
        #   = 1 + A^T z_new - (I + D) z_new
        r = e .+ At * z_new .- denom .* z_new

        # 相对残差停止准则
        if norm(r) <= tol * norm(e)
            z = z_new
            break
        end

        z = z_new
    end

    return z
end

function Iter_fast(G::Graph; tol=1e-12, maxiter=20000)
    n = G.n
    A=adjacency_matrix_directed(G)
    d=degree_vector(G)
    
    @assert size(A, 1) == n "size(A,1) = $(size(A,1)), but G.n = $n"
    @assert size(A, 2) == n "size(A,2) = $(size(A,2)), but G.n = $n"
    @assert length(d) == n "length(d) = $(length(d)), but G.n = $n"

    inv_diag = 1.0 ./ (1.0 .+ d)
    At = transpose(A)
    z = zeros(Float64, n)
    next_z = similar(z)

    for _ in 1:maxiter
        mul!(next_z, At, z)
        @. next_z = (1.0 + next_z) * inv_diag

        if norm(next_z - z) <= tol * max(norm(next_z), eps(Float64))
            z, next_z = next_z, z
            break
        end

        z, next_z = next_z, z
    end

    return z
end

function solver_reference_colsum_directed(G::Graph; tol=1e-10, maxiter=10_000, restart=100)
    n = G.n
    A = adjacency_matrix_directed(G)
    d = degree_vector(G)

    @assert size(A, 1) == n
    @assert size(A, 2) == n
    @assert length(d) == n

    M = spdiagm(0 => 1.0 .+ d) - A
    b = ones(Float64, n)

    x, hist = gmres(
        transpose(M), b;
        restart = restart,
        maxiter = maxiter,
        reltol = tol,
        abstol = tol,
        log = true
    )

    r = b - transpose(M) * x

    return x, norm(r, 2), norm(r, Inf), hist
end