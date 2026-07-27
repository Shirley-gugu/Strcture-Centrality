include("src/forest.jl")
include("src/FoIter.jl")
include("src/centrality.jl")

function max_from_centrality(w::AbstractVector, k::Int, s::AbstractVector)
    return partialsortperm(w .* (1 .- s), 1:k, rev=true)
end

function topk_precision(v_method, v_reference, k::Int)
    k > 0 || throw(ArgumentError("k must be positive"))
    reference_set = Set(v_reference)
    hit = count(node -> node in reference_set, v_method)
    return hit / k
end

function dcg_at_k(ranking, relevance::AbstractVector, k::Int)
    k > 0 || throw(ArgumentError("k must be positive"))
    take = min(k, length(ranking))
    dcg = 0.0
    for rank in 1:take
        node = ranking[rank]
        dcg += relevance[node] / log2(rank + 1)
    end
    return dcg
end

function ndcg_at_k(v_method, relevance::AbstractVector, k::Int)
    k > 0 || throw(ArgumentError("k must be positive"))
    ideal = partialsortperm(relevance, 1:k, rev=true)
    ideal_dcg = dcg_at_k(ideal, relevance, k)
    if ideal_dcg == 0.0
        return 0.0
    end
    return dcg_at_k(v_method, relevance, k) / ideal_dcg
end

function max_Iter(g::Graph,tol,maxiter,k, s)
    w = Iter_exact_residual(g, tol=1e-12, maxiter=maxiter)

    return max_from_centrality(w, k, s)
end

function max_Forest(g::Graph,k, l,s)
    w = forestZ(g, l)

    return max_from_centrality(w, k, s)
end

function max_Forest_columnsums(g::Graph,k, l,s; seed::Int=2)
    w = forestColumnSums_parallel(g, l; seed=seed)

    return max_from_centrality(w, k, s)
end

function max_Pwfc(g::Graph,k, l,s)
    w = PW_FC(g, l)

    return max_from_centrality(w, k, s)
end

function max_PD_structural(g::Graph, k, tau, s; undirected::Bool=false, seed::Int=3)
    w = PD_structural_centrality(
        g,
        tau,
        collect(1:g.n);
        undirected = undirected,
        seed = seed,
    )

    return max_from_centrality(w, k, s)
end

function max_Degree(g::Graph, k, s; mode::Symbol = :out)
    w = degree_centrality(g; mode = mode)
    return max_from_centrality(w, k, s)
end

function max_PageRank(
    g::Graph,
    k,
    s;
    directed::Bool = true,
    damping::Float64 = 0.85,
    tol::Float64 = 1e-8,
    maxiter::Int = 100,
)
    w = pagerank_centrality(
        g;
        directed = directed,
        damping = damping,
        tol = tol,
        maxiter = maxiter,
    )
    return max_from_centrality(w, k, s)
end

function max_Closeness(
    g::Graph,
    k,
    s;
    directed::Bool = true,
    harmonic::Bool = true,
    sample_num::Union{Nothing, Int} = nothing,
    seed::Int = 1,
)
    w = closeness_centrality(
        g;
        directed = directed,
        harmonic = harmonic,
        sample_num = sample_num,
        seed = seed,
    )
    return max_from_centrality(w, k, s)
end

function max_Betweenness(
    g::Graph,
    k,
    s;
    directed::Bool = true,
    sample_num::Union{Nothing, Int} = nothing,
    seed::Int = 1,
)
    w = betweenness_centrality(
        g;
        directed = directed,
        sample_num = sample_num,
        seed = seed,
    )
    return max_from_centrality(w, k, s)
end
