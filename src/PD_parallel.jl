using Random
using SparseArrays
using Base.Threads

function pd_neighbor_list(G; undirected::Bool = true)
    if hasproperty(G, :nbr)
        return deepcopy(getproperty(G, :nbr))
    end

    nbr = [Int[] for _ in 1:G.n]
    for i in 1:G.m
        u = G.u[i]
        v = G.v[i]
        u == v && continue

        push!(nbr[u], v)
        if undirected
            push!(nbr[v], u)
        end
    end
    return nbr
end

function lapsp(G::Graph; undirected::Bool = true)
    nbr = pd_neighbor_list(G; undirected = undirected)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for u in 1:G.n
        for v in nbr[u]
            push!(rows, u)
            push!(cols, v)
            push!(vals, -1.0)
        end
    end

    for u in 1:G.n
        push!(rows, u)
        push!(cols, u)
        push!(vals, length(nbr[u]))
    end

    return sparse(rows, cols, vals, G.n, G.n)
end

function random_edgeset(L, G::Graph, k::Integer; rng = Random.default_rng())
    max_candidates = div(G.n * (G.n - 1), 2) - count_existing_undirected_edges(G)
    if k > max_candidates
        throw(ArgumentError("cannot sample $k non-existing edges; only $max_candidates candidates are available"))
    end

    edges = Set{Tuple{Int, Int}}()
    while length(edges) < k
        u = rand(rng, 1:G.n)
        v = rand(rng, 1:G.n)
        u == v && continue

        edge = u < v ? (u, v) : (v, u)
        if !in(edge, edges) && L[edge[1], edge[2]] == 0
            push!(edges, edge)
        end
    end
    return collect(edges)
end

function count_existing_undirected_edges(G::Graph)
    existing = Set{Tuple{Int, Int}}()
    for i in 1:G.m
        u = G.u[i]
        v = G.v[i]
        u == v && continue
        push!(existing, u < v ? (u, v) : (v, u))
    end
    return length(existing)
end

function PD_structural_centrality(
    G,
    tau::Integer,
    alist = 1:G.n;
    undirected::Bool = true,
    seed::Int = 1,
)
    tau > 0 || throw(ArgumentError("tau must be positive"))

    gnbr = pd_neighbor_list(G; undirected = undirected)
    d = [length(gnbr[j]) for j in 1:G.n]

    thread_num = nthreads()
    partial = zeros(Float64, G.n, thread_num)

    Threads.@threads :static for tid in 1:thread_num
        rng = Xoshiro(seed + tid * 10_000_019)
        in_forest = falses(G.n)
        nxt = fill(-1, G.n)
        root = zeros(Int, G.n)
        local_samples = div(tau, thread_num) + (tid <= tau % thread_num ? 1 : 0)
        local_z = @view partial[:, tid]

        for _ in 1:local_samples
            fill!(in_forest, false)
            fill!(nxt, -1)
            fill!(root, 0)

            for src in alist
                u = src
                while !in_forest[u]
                    r = rand(rng)
                    if r <= 1 / (1 + d[u])
                        in_forest[u] = true
                        root[u] = u
                    else
                        next_idx = floor(Int, r * (1 + d[u]))
                        nxt[u] = gnbr[u][next_idx]
                        u = nxt[u]
                    end
                end

                rootnow = root[u]
                local_z[rootnow] += 1.0

                u = src
                while root[u] == 0
                    in_forest[u] = true
                    root[u] = rootnow
                    u = nxt[u]
                end
            end
        end
    end

    z = vec(sum(partial, dims = 2))
    z ./= tau
    return z
end

function PD_parallel(G, tau, c, Ev, alist, kk; undirected::Bool = true)
    # G = deepcopy(G)
    S = Vector{Tuple{Int, Int}}()
    sev = length(Ev)
    kk <= sev || throw(ArgumentError("kk=$kk exceeds candidate edge count $sev"))
    cho = zeros(Bool, sev)
    c = deepcopy(c)  
    gnbr = pd_neighbor_list(G; undirected = undirected)
    d = [length(gnbr[j]) for j in 1:G.n]
    a_threads = [Int[] for _ in 1:nthreads()]
    b_threads = [Int[] for _ in 1:nthreads()]
    InForest_pool = [falses(G.n) for _ in 1:nthreads()]
    Next_pool = [fill(-1, G.n) for _ in 1:nthreads()]
    Rootindex_pool = [zeros(Int, G.n) for _ in 1:nthreads()]
    for aaa = 1:kk 
        for i in 1:nthreads()
            empty!(a_threads[i])
            empty!(b_threads[i])
            fill!(InForest_pool[i], false)
            fill!(Next_pool[i], -1)
            fill!(Rootindex_pool[i], 0)
        end

        Threads.@threads for p in 1:tau
            thread_id = threadid()

            InForest = InForest_pool[thread_id]
            Next = Next_pool[thread_id]
            Rootindex = Rootindex_pool[thread_id]

            fill!(InForest, false)
            fill!(Next, -1)
            fill!(Rootindex, 0)

            cntnow = 0; 
            for i in alist
                u = i
                # cntnow += 1;
                while !InForest[u]
                    seed = rand()
                    if seed <= 1 / (1 + d[u])
                        InForest[u] = true
                        Rootindex[u] = u
                    else
                        k = floor(Int, seed * (1 + d[u]))
                        Next[u] = gnbr[u][k]
                        u = Next[u]
                        cntnow += 1;
                    end
                end

                rootnow = Rootindex[u]
                push!(a_threads[thread_id], i)
                push!(b_threads[thread_id], rootnow)

                u = i
                while Rootindex[u] == 0
                    InForest[u] = true
                    Rootindex[u] = rootnow
                    u = Next[u]
                end
            end

        end

        a = vcat(a_threads...)
        b = vcat(b_threads...)
        W = sparse(a,b,1.0,G.n,G.n)
        W = W./tau
        q = W * c

        dz = zeros(sev)
        
        for j = 1 : sev
            if cho[j]
                continue
            end
            (u, v) = Ev[j]
            dz[j] = (q[u] - q[v])^2 / (1.0 + W[u, u] + W[v, v] - 2*W[u, v])
        end
        xx = argmax(dz)
        cho[xx] = true
        (su, sv) = Ev[xx]
        # W = updateW(W, su, sv)
        d[su] += 1
        push!(gnbr[su], sv)
        if undirected
            d[sv] += 1
            push!(gnbr[sv], su)
        end
        push!(S, Ev[xx])
    end
    return S
end
