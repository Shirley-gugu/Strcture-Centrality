using Random

function centrality_neighbors(G::Graph; directed::Bool = true, reverse::Bool = false)
    nbr = [Int[] for _ in 1:G.n]
    for edge_idx in 1:G.m
        u = G.u[edge_idx]
        v = G.v[edge_idx]
        u == v && continue

        if reverse
            push!(nbr[v], u)
        else
            push!(nbr[u], v)
        end
        if !directed
            if reverse
                push!(nbr[u], v)
            else
                push!(nbr[v], u)
            end
        end
    end
    return nbr
end

function degree_centrality(
    G::Graph;
    mode::Symbol = :out,
    normalized::Bool = true,
)
    mode in (:out, :in, :total) ||
        throw(ArgumentError("mode must be :out, :in, or :total"))

    score = zeros(Float64, G.n)
    for edge_idx in 1:G.m
        u = G.u[edge_idx]
        v = G.v[edge_idx]
        u == v && continue

        if mode == :out || mode == :total
            score[u] += 1.0
        end
        if mode == :in || mode == :total
            score[v] += 1.0
        end
    end

    if normalized && G.n > 1
        score ./= (G.n - 1)
    end
    return score
end

function pagerank_centrality(
    G::Graph;
    directed::Bool = true,
    damping::Float64 = 0.85,
    tol::Float64 = 1e-8,
    maxiter::Int = 100,
)
    0.0 < damping < 1.0 ||
        throw(ArgumentError("damping must be in (0, 1)"))
    tol > 0.0 || throw(ArgumentError("tol must be positive"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive"))

    nbr = centrality_neighbors(G; directed = directed)
    n = G.n
    pr = fill(1.0 / n, n)
    next_pr = similar(pr)
    teleport = (1.0 - damping) / n

    for _ in 1:maxiter
        dangling_mass = 0.0
        fill!(next_pr, teleport)

        for u in 1:n
            outdeg = length(nbr[u])
            if outdeg == 0
                dangling_mass += pr[u]
            else
                share = damping * pr[u] / outdeg
                for v in nbr[u]
                    next_pr[v] += share
                end
            end
        end

        if dangling_mass > 0.0
            next_pr .+= damping * dangling_mass / n
        end

        if sum(abs, next_pr .- pr) <= tol
            pr, next_pr = next_pr, pr
            break
        end

        pr, next_pr = next_pr, pr
    end

    total = sum(pr)
    if total > 0.0
        pr ./= total
    end
    return pr
end

function closeness_centrality(
    G::Graph;
    directed::Bool = true,
    harmonic::Bool = true,
    normalized::Bool = true,
    sample_num::Union{Nothing, Int} = nothing,
    seed::Int = 1,
)
    if sample_num !== nothing && sample_num > 0 && sample_num < G.n
        harmonic ||
            throw(ArgumentError("sampled closeness is implemented for harmonic=true"))

        rng = Xoshiro(seed)
        targets = randperm(rng, G.n)[1:sample_num]
        reverse_nbr = centrality_neighbors(G; directed = directed, reverse = true)
        score = zeros(Float64, G.n)
        dist = fill(-1, G.n)
        queue = Vector{Int}(undef, G.n)

        for target in targets
            fill!(dist, -1)
            head = 1
            tail = 1
            queue[tail] = target
            dist[target] = 0

            while head <= tail
                u = queue[head]
                head += 1

                for v in reverse_nbr[u]
                    if dist[v] == -1
                        dist[v] = dist[u] + 1
                        tail += 1
                        queue[tail] = v
                    end
                end
            end

            for u in 1:G.n
                if u != target && dist[u] > 0
                    score[u] += 1.0 / dist[u]
                end
            end
        end

        if normalized
            score ./= sample_num
        elseif G.n > 1
            score .*= (G.n - 1) / sample_num
        end
        return score
    end

    nbr = centrality_neighbors(G; directed = directed)
    score = zeros(Float64, G.n)
    dist = fill(-1, G.n)
    queue = Vector{Int}(undef, G.n)

    for src in 1:G.n
        fill!(dist, -1)
        head = 1
        tail = 1
        queue[tail] = src
        dist[src] = 0

        while head <= tail
            u = queue[head]
            head += 1

            for v in nbr[u]
                if dist[v] == -1
                    dist[v] = dist[u] + 1
                    tail += 1
                    queue[tail] = v
                end
            end
        end

        if harmonic
            total = 0.0
            for v in 1:G.n
                if v != src && dist[v] > 0
                    total += 1.0 / dist[v]
                end
            end
            score[src] = normalized && G.n > 1 ? total / (G.n - 1) : total
        else
            reachable = 0
            dist_sum = 0.0
            for v in 1:G.n
                if v != src && dist[v] > 0
                    reachable += 1
                    dist_sum += dist[v]
                end
            end

            if dist_sum > 0.0
                score[src] = reachable / dist_sum
                if normalized && G.n > 1
                    score[src] *= reachable / (G.n - 1)
                end
            end
        end
    end

    return score
end

function betweenness_centrality(
    G::Graph;
    directed::Bool = true,
    normalized::Bool = true,
    sources = nothing,
    sample_num::Union{Nothing, Int} = nothing,
    seed::Int = 1,
)
    nbr = centrality_neighbors(G; directed = directed)
    cb = zeros(Float64, G.n)
    source_nodes =
        sources === nothing ? collect(1:G.n) : collect(sources)

    if sources === nothing && sample_num !== nothing &&
       sample_num > 0 && sample_num < G.n
        rng = Xoshiro(seed)
        source_nodes = randperm(rng, G.n)[1:sample_num]
    end

    stack = Vector{Int}(undef, G.n)
    queue = Vector{Int}(undef, G.n)
    pred = [Int[] for _ in 1:G.n]
    sigma = zeros(Float64, G.n)
    dist = fill(-1, G.n)
    delta = zeros(Float64, G.n)

    for src in source_nodes
        for v in 1:G.n
            empty!(pred[v])
        end
        fill!(sigma, 0.0)
        fill!(dist, -1)

        sigma[src] = 1.0
        dist[src] = 0

        stack_len = 0
        head = 1
        tail = 1
        queue[tail] = src

        while head <= tail
            v = queue[head]
            head += 1
            stack_len += 1
            stack[stack_len] = v

            for w in nbr[v]
                if dist[w] < 0
                    tail += 1
                    queue[tail] = w
                    dist[w] = dist[v] + 1
                end

                if dist[w] == dist[v] + 1
                    sigma[w] += sigma[v]
                    push!(pred[w], v)
                end
            end
        end

        fill!(delta, 0.0)
        for idx in stack_len:-1:1
            w = stack[idx]
            if sigma[w] > 0.0
                coeff = (1.0 + delta[w]) / sigma[w]
                for v in pred[w]
                    delta[v] += sigma[v] * coeff
                end
            end
            if w != src
                cb[w] += delta[w]
            end
        end
    end

    if length(source_nodes) < G.n
        cb .*= G.n / length(source_nodes)
    end

    if !directed
        cb ./= 2.0
    end

    if normalized && G.n > 2
        scale = directed ? 1.0 / ((G.n - 1) * (G.n - 2)) :
            2.0 / ((G.n - 1) * (G.n - 2))
        cb .*= scale
    end

    return cb
end
