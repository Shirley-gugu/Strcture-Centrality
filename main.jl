include("src/graph.jl")
include("src/forest.jl")
include("src/rw.jl")
include("src/FoIter.jl")
include("src/variance.jl")
include("src/iter.jl")
include("src/varopt.jl")
include("src/covpoly.jl")
include("opt.jl")
include("distribution.jl")
include("max.jl")
include("src/mis_adapter.jl")
include("src/PD_parallel.jl")

function main(i)
    #G=read_data("Wiki-Vote.txt")
    G=read_data(i)
    println("dataset=", i)
    target_error = 1e-2

    
    function generate_opinion(dist::String, n)

    if dist == "uni"
        s, _ = generate_samples(Uniform(), n)

    elseif dist == "exp"
        s, _ = generate_samples(ScaledExponential(), n)

    elseif dist == "pow"
        s, _ = generate_samples(ScaledPowerLaw(), n)

    else
        error("Unknown distribution")
    end

    return s
    end

    d = degree_vector(G)

    s = generate_opinion("uni", G.n)

    A = adjacency_matrix_directed(G)


    # z=ones(G.n)'*inv(Matrix(I+D-A))
    #@time z=exact(G)'
    #@time z,res2_ref, resinf_ref, hist_ref=solver_reference_colsum_directed(G::Graph; tol=1e-10, maxiter=10_000, restart=100)
    

#方差计算
#=    reference = Iter_exact_residual(
      G;
      tol = 1e-12,
      maxiter = 20_000,
  )

  results = compare_sampling_variance(
      G;
      sample_num = 1000,
      pw_iterations = 100,
      repetitions = 30,
      seed = 2026,
      reference = reference,
  )=#


    #中心性计算
    #=@time z1=randomwalkZ(G, 10000)
    #@time z1=randomwalkZ_parallel(G, 10000)
    #@time z1=Iter_exact_residual(G, tol=1e-1, maxiter=5000)
    #@time z3=forestColumnSum(G, 7, 1000)
    #@time z3=forestColumnSums(G, 1000)
    #@time z2 = forestZ_parallel(G, 1000)
    #@time z4=PW(G, z1, 100)
    #@time z5=PW(G, z2, 100)
    #@time z7=PW(G, z4, 10)
    # z2=inv(I+D)*ones(n)+inv(I+D)*A'*z1
    @time z=Iter_exact_residual(G, tol=1e-12, maxiter=20000)
    @time z1=randomwalkZ(G, 10000)
    #@time z1=randomwalkZ_parallel(G, 10000)
    @time z2=forestZ(G, 1000)
    @time z3 = forestColumnSums_parallel(G, 1000)
    @time z4, balanced_info = PW_FC_balanced(G; sample_num = 10, residual_tol = target_error, marginal_tol = 1e-3,
    miniter = 20, maxiter = 300, check_every = 5)
    @time z6, pw_fc_iters, pw_fc_relres = PW_FC_tol(G; sample_num = 10, tol = target_error, maxiter = 10_000,)=#
    

#并行版版中心性计算
    @time z=Iter_exact_residual(G, tol=1e-12, maxiter=20000)
    #@time z1=randomwalkZ(G, 10000)
    @time z1=randomwalkZ_parallel(G, 10000)
    @time z2=forestZ(G, 10000)
    #@time z1=forestZ_parallel(G, 1000)
    #@time z7 = forestColumnSums_parallel(G, 1000)
    #@time z3=forestZ_iter1_in_forest(G, 1000)
    @time z3=forestZ_iter1_in_forest_parallel(G, 1000)
    @time z4=PW_FC(G, 100; sample_num = 10)
    @time z5,_=globalInfApprox(G, 1e-6)
    @time z6=PD_structural_centrality(G, 1000, collect(1:G.n); undirected=false, seed=3)

    @show error1=norm(z-z1)/norm(z)
    #@show error7=norm(z-z7)/norm(z)
    @show error2=norm(z-z2)/norm(z)
    @show error3=norm(z-z3)/norm(z)
    @show error4=norm(z-z4)/norm(z)
    @show error5=norm(z-z5)/norm(z)
    @show error6=norm(z-z6)/norm(z)

    function compute_overall_opinion(name, z, v, k, s)
        original_overall_opinion = dot(z, s)
        gain = sum(z[v] .* (1.0 .- s[v]))
        overall_opinion = original_overall_opinion + gain
        println("$name: $k: ", overall_opinion)
    end

    function print_topk_precision(name, v_method, v_reference, k)
        precision = topk_precision(v_method, v_reference, k)
        hits = round(Int, precision * k)
        println("$name precision@$k: ", precision, " (", hits, "/", k, ")")
    end

    function print_topk_ndcg(name, v_method, relevance, k)
        ndcg = ndcg_at_k(v_method, relevance, k)
        println("$name NDCG@$k: ", ndcg)
    end


# K,T网格搜索法
#=    result = compare_error_time(
      G;
      sample_nums = [1, 2, 5, 10, 20, 50, 100],
      pw_iterations = [0, 1, 2, 5, 10, 20, 50, 100, 200],
      global_tolerances = [
          1e-1, 5e-2, 1e-2, 5e-3,
          1e-3, 1e-4, 1e-5, 1e-6,
      ],
      repetitions = 20,
      seed = 2026,
      pd_undirected = false,
      csv_path = "results/error_time.csv",
  )
=#

    #=baseline_sample_num = G.n <= 20_000 ? nothing : min(G.n, 256)
    @time degree_score = degree_centrality(G; mode = :out)
    @time pagerank_score = pagerank_centrality(
        G;
        directed = true,
        damping = 0.85,
        tol = 1e-8,
        maxiter = 100,
    )
    @time closeness_score = closeness_centrality(
        G;
        directed = true,
        harmonic = true,
        sample_num = baseline_sample_num,
        seed = 4,
    )
    @time betweenness_score = betweenness_centrality(
        G;
        directed = true,
        sample_num = baseline_sample_num,
        seed = 5,
    )=#


    #max和v
    #for k in 2 .^ (1:10)
    #=true_gain = z .* (1.0 .- s)
    for k in (1, 100, 200, 300, 400, 500,600,700,800,900,1000)
        v = max_Iter(G, 1e-12, 20000, k, s)
        v1 = max_from_centrality(z1, k, s)
        v2 = max_from_centrality(z2, k, s)
        v3 = max_from_centrality(z3, k, s)
        v4 = max_from_centrality(z4, k, s)
        #v_deg = partialsortperm(degree_score .* (1.0 .- s), 1:k, rev = true)
        #v_pr = partialsortperm(pagerank_score .* (1.0 .- s), 1:k, rev = true)
        #v_close = partialsortperm(closeness_score .* (1.0 .- s), 1:k, rev = true)
        #v_bet = partialsortperm(betweenness_score .* (1.0 .- s), 1:k, rev = true)
        v6 = max_from_centrality(z6, k, s)=#



        #precision和NDCG
        #=print_topk_precision("Forest", v1, v, k)
        print_topk_precision("ForestColumnSums", v2, v, k)
        print_topk_precision("PW_FC", v3, v, k)
        print_topk_precision("globalInfApprox", v4, v, k)
        print_topk_precision("PD_structural", v6, v, k)

        print_topk_ndcg("Forest", v1, true_gain, k)
        print_topk_ndcg("ForestColumnSums", v2, true_gain, k)
        print_topk_ndcg("PW_FC", v3, true_gain, k)
        print_topk_ndcg("globalInfApprox", v4, true_gain, k)
        print_topk_ndcg("PD_structural", v6, true_gain, k)=#

        #=compute_overall_opinion("Iter", z, v4, k, s)
        compute_overall_opinion("Forest", z, v1, k, s)
        compute_overall_opinion("ForestColumnSums", z, v2, k, s)
        compute_overall_opinion("PW_FC", z, v3, k, s)
        compute_overall_opinion("MIS", z, v5, k, s)
        #compute_overall_opinion("Degree", z, v_deg, k, s)
        #compute_overall_opinion("PageRank", z, v_pr, k, s)
        #compute_overall_opinion("Closeness", z, v_close, k, s)
        #compute_overall_opinion("Betweenness", z, v_bet, k, s)
        compute_overall_opinion("PD_structural", z, v_pd, k, s)
    end=#


    # z4, balanced_info = PW_FC_balanced(
    #     G;
    #     sample_num = 10,
    #     residual_tol = target_error,
    #     marginal_tol = 1e-3,
    #     miniter = 20,
    #     maxiter = 300,
    #     check_every = 5,
    # )
    
    # pw_fc_time = @elapsed z6, pw_fc_iters, pw_fc_relres = PW_FC_tol(
    #     G;
    #     sample_num = 10,
    #     tol = target_error,
    #     maxiter = 10_000,
    # )

    #=max

    function compute_overall_opinion(name, z, v, k, s)
        original_overall_opinion = dot(z, s)
        gain = sum(z[v] .* (1.0 .- s[v]))
        overall_opinion = original_overall_opinion + gain
        #println(original_overall_opinion)
        println("$name: $k: ", overall_opinion)
    end

    function compute_overall_opinion_edges(
        name,
        G,
        edges,
        k,
        s;
        directed::Bool = true,
        centrality_samples::Int = 1000,
        seed::Int = 3,
    )
        add_edges = Tuple{Int, Int}[]
        for (u, v) in edges
            push!(add_edges, (Int(u), Int(v)))
            if !directed
                push!(add_edges, (Int(v), Int(u)))
            end
        end

        newG = add_directed_edges(G, add_edges)
        new_z = PD_structural_centrality(
            newG,
            centrality_samples,
            collect(1:newG.n);
            undirected = !directed,
            seed = seed,
        )
        println("$name: $k: ", dot(new_z, s))
    end

    Ev = random_directed_candidates(G, 10000; seed=1)

    alist = collect(1:G.n)
    
for k in (1, 10, 20, 30, 40, 50)
    v1=max_Forest(G, k, 4000, s)
    v2=max_Forest_columnsums(G, k, 4000, s; seed=2)
    v3=max_Pwfc(G, k, 4000, s)
    v4=max_Iter(G, 1e-12, 20000, k, s)
    v5, mis_time = maxInfluenceSeletor(G, k, s; epsilon = 1e-3)
    e6=PD_parallel(G, 1000, s, Ev, alist, k; undirected=false)
    v6=max_PD_structural(G, k, 1000, s; undirected=false, seed=3)
    #println("max_Forest=", v1)
    #println("max_Forest_columnsums=", v2)
    #println("max_Pwfc=", v3)

    
    compute_overall_opinion("Forest", z1, v1, k, s)
    compute_overall_opinion("ForestColumnSums", z2, v2, k, s)
    compute_overall_opinion("PW_FC", z3, v3, k, s)
    compute_overall_opinion("Iter", z, v4, k, s)
    compute_overall_opinion("MIS", z4, v5, k, s)
    compute_overall_opinion("PD_structural", z6, v6, k, s)
    #compute_overall_opinion_edges("PD_parallel_edges", G, e6, k, s; directed=true)

end=#

    #println("MIS time=", mis_time)

    #= Selected nodes returned by MIS, RWB, or Forest
    selected_nodes = v1

    # Reset the internal opinions of selected nodes to 1
    s_intervened = copy(s)
    s_intervened[selected_nodes] .= 1.0

    # Compute the new equilibrium opinion
    M = spdiagm(0 => ones(G.n) .+ d) - A
    z_intervened = M \ s_intervened

    # Aggregate equilibrium opinion after intervention
    overall_opinion = sum(z_intervened)
    println(overall_opinion, overall_opinion-sum(z2))=#

    # @show z
    # @show z1
    # @show z2
    # @show z3
    # @show z4

    #@show error0=norm(z-z0)/norm(z)
    #@show error1=norm(z-z1)/norm(z)
    #初始意见s

    #优化
    #=opinion_k = min(1000, G.n)
    method_results = [
        (name = :reference_iter, rho = z, time = ref_time),
        (name = :forestZ, rho = z2, time = forest_time),
        (name = :forestColumnSums_parallel, rho = z3, time = forest_col_time),
        (name = :PW_FC_balanced, rho = z4, time = balanced_time),
        (name = :PW_FC_tol, rho = z6, time = pw_fc_time),
    ]

    s = initial_opinion_vector(
        G.n;
        distribution = :normal,
        seed = 20260718,
    )

    opinion_results = opinion_minimization_comparison_from_rhos(
        z,
        s,
        method_results,
        opinion_k,
    )

    println("opinion minimization: distribution=normal, k=", opinion_k)
    println("method,time,error,decrease_ratio,final_value")
    for row in opinion_results
        println(
            row.name,
            ",",
            row.time,
            ",",
            row.rho_error,
            ",",
            row.decrease_ratio,
            ",",
            row.final_value,
        )
    end
    =#


    #@time z1, iters, relres = PW_FC_tol(G; sample_num=10, tol=1e-8)
    #@show iters relres

    #新方法
    #=@time z1=PW_FC(G, 100)
    @time result = PW_FC_VOE(
      G,
      50;
      sample_num = 10,
      seed = 20260714,
      fixed_alpha = -1.5,
  )
rho_hat = result.estimate
#@show result.alpha
#@show result.diagnostics
    combo_depths = [100, 105, 110, 115, 120]
    combo_alpha = fixed_geometric_weights(
        combo_depths;
        lambda = 0.85,
    )

    @time result1 = PW_FC_optimal_combination(
        G,
        maximum(combo_depths);
        depths = combo_depths,
        sample_num = 10,
        main_seed = 20260715,
        fixed_alpha = combo_alpha,
    )

@show error1 = norm(z - z1) / norm(z)
@show error2 = norm(z - rho_hat) / norm(z)
@show error3 = norm(z - result1.estimate) / norm(z)
@show result1.depths
@show result1.alpha=#

#网格搜索T,K
#=result = grid_search_pw_fc(
      G;
      sample_nums = [1, 2, 5, 10, 20, 50, 100],
      iteration_nums = [0, 1, 2, 5, 10, 20, 50, 100, 200],
      repetitions = 20,
      seed = 2026,
  )=#

end

main("soc-LiveJournal1.txt")
#=for i in ["Wiki-Vote.txt", "out.dblp-cite","Twitterlists","web-Stanford.txt","web-Google.txt","soc-pokec-relationships.txt","soc-LiveJournal1"]
    main(i)
end=#
#i="Wiki-Vote.txt"
#i="out.dblp-cite"


#=
#G=read_data("Wiki-Vote.txt")
G=read_data("com.txt")
@show G.m,G.n

n=G.n
A=adjacency_matrix_directed(G)
D=diagm(degree_vector(G))

# z=ones(G.n)'*inv(Matrix(I+D-A))
@time z=exact(G)'


@time z1=randomwalkZ(G, 1000)
@time z2=forestZ(G, 1000)
#@time z3=forestColumnSum(G, 7, 1000)
@time z3=forestColumnSums(G, 1000)
@time z4=PW(G, z1, 100)
@time z5=PW(G, z2, 100)
@time z6=PW(G, z3, 100)
#@time z7=PW(G, z4, 10)
# z2=inv(I+D)*ones(n)+inv(I+D)*A'*z1

# @show z
# @show z1
# @show z2
# @show z3
# @show z4

@show error1=norm(z-z1)/norm(z)
@show error2=norm(z-z2)/norm(z)
@show error3=norm(z-z3)/norm(z)
@show error4=norm(z-z4)/norm(z)
@show error5=norm(z-z5)/norm(z)
@show error6=norm(z-z6)/norm(z)
# k=10
# zz=zeros(n,k)
# for i in 1:k
#     global zz[:,i]=inv(I+D)*ones(n)+inv(I+D)*A'*z1
#     @show norm(z-zz[:,i])/norm(z)
#     global z1=zz[:,i]
# end
=#




