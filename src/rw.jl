function randomwalkZ(graph::Graph,sample_num::Int)
    n = graph.n
    d = degree_vector(graph)
    nbr,_ = neighbor_weight(graph)
    
    ans=zeros(Float64,n)
    for ll in 1:sample_num
        #@show ll
        for src in 1:n
            u = src
            while true
                if rand() * (d[u] + 1) < 1
                    ans[u]+=1/sample_num
                    # push!(row, src)
                    # push!(col, u)
                    # push!(val, 1 / sample_num)
                    break
                end
                u = rand(nbr[u])
            end
        end
    end
    return ans
end

function randomwalkZ_parallel(graph::Graph, sample_num::Int; seed::Int=1)
    n = graph.n
    d = degree_vector(graph)
    nbr,_ = neighbor_weight(graph)
    stop_prob = @. 1.0 / (d + 1.0)

    thread_num = nthreads()
    partial = zeros(Float64, n, thread_num)
    rngs = [Xoshiro(seed + tid * 10_000_019) for tid in 1:thread_num]

    @threads :static for src in 1:n
        tid = threadid()
        rng = rngs[tid]
        local_ans = @view partial[:, tid]

        for _ in 1:sample_num
            u = src
            while true
                if rand(rng) < stop_prob[u]
                    local_ans[u] += 1.0
                    break
                end
                nu = nbr[u]
                u = nu[rand(rng, 1:length(nu))]
            end
        end
    end

    ans = vec(sum(partial, dims=2))
    ans ./= sample_num
    return ans
end

function randomwalkZ_fast_unweighted(
      graph::Graph,
      sample_num::Int,
      s::Vector{Float64},
      nbr;
      seed::Int = 1,
  )
      return randomwalkZ_parallel(graph, sample_num; seed=seed)
end