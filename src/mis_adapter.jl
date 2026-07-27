using LinearAlgebra
using SparseArrays

struct MISGraph
    n::Int
    m::Int
    s::Vector{Float64}
    f::Vector{Float64}
    d::Vector{Float64}
    A::SparseMatrixCSC{Float64, Int}
end

function MISGraph(G::Graph, s::AbstractVector)
    length(s) == G.n ||
        throw(DimensionMismatch("s must have length G.n"))

    d = degree_vector(G)
    f = 1.0 ./ (1.0 .+ d)
    A = SparseMatrixCSC{Float64, Int}(adjacency_matrix_directed(G))

    return MISGraph(G.n, G.m, Vector{Float64}(s), f, d, A)
end

function globalInfApprox(g::MISGraph, epsilon::Float64; maxiter::Int = 20_000)
    epsilon > 0.0 ||
        throw(ArgumentError("epsilon must be positive"))

    scale = zeros(Float64, g.n)
    for i in 1:g.n
        if g.d[i] > 0.0
            scale[i] = (1.0 - g.f[i]) / g.d[i]
        end
    end

    y = zeros(Float64, g.n)
    next_y = similar(y)
    e = ones(Float64, g.n)

    timed = @timed begin
        for _ in 1:maxiter
            mul!(next_y, transpose(g.A), scale .* y)
            next_y .+= e

            if norm(next_y - y) <= epsilon * max(norm(next_y), eps(Float64))
                y, next_y = next_y, y
                break
            end

            y, next_y = next_y, y
        end

        g.f .* y
    end

    return timed.value, timed.time
end

function globalInfApprox(G::Graph, epsilon::Float64; maxiter::Int = 20_000)
    g = MISGraph(G, zeros(Float64, G.n))
    return globalInfApprox(g, epsilon; maxiter = maxiter)
end

function maxInfluenceSeletor(g::MISGraph, k::Int, epsilon::Float64 = 1e-3)
    k > 0 ||
        throw(ArgumentError("k must be positive"))

    take = min(k, g.n)
    rho, elapsed = globalInfApprox(g, epsilon)
    growth = rho .* (1.0 .- g.s)
    selected = partialsortperm(growth, 1:take, rev = true)

    return selected, elapsed
end

function maxInfluenceSeletor(
    G::Graph,
    k::Int,
    s::AbstractVector;
    epsilon::Float64 = 1e-3,
)
    g = MISGraph(G, s)
    return maxInfluenceSeletor(g, k, epsilon)
end
