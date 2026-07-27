include("graph.jl")


using LinearAlgebra
using Random
using Statistics
using Laplacians
using SparseArrays
using DataFrames
using IterativeSolvers
using SparseArrays
using Plots
using Base.Threads
 
using CSV

 

function sqrtsample(n::Int, p::Float64)
    result = Int[]
    i = 0
    while i <= n
        ii = floor(Int, log(rand()) / log(1 - p))
        i +=ii;
        if (ii != 0) && (i<=n)
            push!(result, i)
        end
    end
    return result
end
 

 
 

function rf(G,tau) #algorithm SCF
    d = [length(G.nbr[j]) for j = 1:G.n];
    InForest = [false for j = 1:G.n];
    Next = [-1 for j = 1:G.n];
    Rootindex = zeros(Int, G.n);         
    # Q = zeros(G.n,G.n)
    a = Int[]
    b = Int[]
    for p = 1:tau
        InForest .= false;
        Next .= -1;
        Rootindex .= 0;
        # pp = 1/sqrt(G.n)
        for i = 1:G.n
            u = i;
            while !InForest[u];
                seed = rand();
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true;
                    Next[u] = -1;
                    push!(a,u)
                    push!(b,u)
                    # Q[u,u]+=1;
                    Rootindex[u] = u;
                else
                    k = floor(Int, seed * (1 + d[u]));
                    Next[u] = G.nbr[u][k];
                    u = Next[u];
                end
            end
            rootnow = Rootindex[u];
            # Q[i,rootnow] += 1;
            u = i
            while !InForest[u]
                InForest[u] = true;
                Rootindex[u] = rootnow; 
                push!(a,u)
                push!(b,rootnow)
                # Q[u,rootnow]+=1;            
                u = Next[u];
            end
        end
    end
    Q = sparse(a,b,1.0,G.n,G.n)
    return Q./tau
end



function rfz(G,tau,s) #algorithm SCF
    d = [length(G.nbr[j]) for j = 1:G.n];
    InForest = [false for j = 1:G.n];
    Next = [-1 for j = 1:G.n];
    Rootindex = zeros(Int, G.n);         
    z = zeros(G.n)
    for p = 1:tau
        InForest .= false;
        Next .= -1;
        Rootindex .= 0;
        for i = 1:G.n
            u = i;
            while !InForest[u];
                seed = rand();
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true;
                    Next[u] = -1;
                    z[u] += s[u]
                    Rootindex[u] = u;
                else
                    k = floor(Int, seed * (1 + d[u]));
                    Next[u] = G.nbr[u][k];
                    u = Next[u];
                end
            end
            rootnow = Rootindex[u];
            u = i
            while !InForest[u]
                InForest[u] = true;
                Rootindex[u] = rootnow; 
                z[u] += s[rootnow]
                u = Next[u];
            end
        end
    end
    return z./tau
end
function top_k_elements(arr::Vector, k::Int)
    indices = partialsortperm(arr, 1:k, rev=true) 
    values = arr[indices]
    return values, indices
end

function min_sumz(L,s)
    LL = Matrix(L+I);
    @time rho = gmres(LL',ones(G.n))
    return rho.*s
end

function min_sumzset(rhos,setk)
    alist = Float64[]
    len = length(setk)
    for i =1:len
        push!(alist,sum(rhos)-sum(rhos[setk[1:i]]))
    end
    return alist
end

 
function lazyz(G,l,r,s,d,pp)
    a = Int[];
    b = Float64[];
    alist =   sqrtsample(G.n, pp)
    for i in alist
        c = lazy(G,i,l,r,s,d)
        push!(a,i)
        push!(b,c)
    end
    return sparsevec(a,b,G.n)
end




function rerror(x,y)
    return norm(x-y)/norm(y)
end


 



function sqrtzzpre_tau_parallel(G, tau, c,alist)
    d = [length(G.nbr[j]) for j in 1:G.n]             

    a_threads = [Int[] for _ in 1:nthreads()]        
    b_threads = [Float64[] for _ in 1:nthreads()]    
    cnt = zeros(Int, nthreads())                   
    InForest_pool = [falses(G.n) for _ in 1:nthreads()]
    Next_pool = [fill(-1, G.n) for _ in 1:nthreads()]
    Rootindex_pool = [zeros(Int, G.n) for _ in 1:nthreads()]

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
                    Next[u] = G.nbr[u][k]
                    u = Next[u]
                    cntnow += 1;
                end
            end

            rootnow = Rootindex[u]
            push!(a_threads[thread_id], rootnow)
            push!(b_threads[thread_id], c[u])

            u = i
            while Rootindex[u] == 0
                InForest[u] = true
                Rootindex[u] = rootnow
                u = Next[u]
            end
        end
        #记录一下inforest里面有多少是true

        cnt[thread_id] += sum(InForest.==true)
    end

    # 合并各线程的结果
    a = vcat(a_threads...)
    b = vcat(b_threads...)
    println("Total nodes processed: ", sum(cnt)/length(alist)/tau)
    return sparsevec(a, b ./ tau, G.n)  
end





function sqrtrhopre(G,tau,pp,c)  
    d = [length(G.nbr[j]) for j = 1:G.n];
    InForest = [false for j = 1:G.n];
    Next = [-1 for j = 1:G.n];
    Rootindex = zeros(Int, G.n);         
    a = Int[]
    b = Float64[]
    alist = sqrtsample(G.n, pp)
    cnt = 0 
    for p = 1:tau
        InForest .= false;
        Next .= -1;
        Rootindex .= 0;
        for i in alist
            u = i;
            while !InForest[u];
                seed = rand();
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true;
                    Next[u] = -1;
                    Rootindex[u] = u;
                else
                    cnt += 1
                    k = floor(Int, seed * (1 + d[u]));
                    Next[u] = G.nbr[u][k];
                    u = Next[u];
                end
            end
            rootnow = Rootindex[u];
            push!(a,rootnow);
            push!(b,c[u])
            u = i
            while !InForest[u]
                InForest[u] = true;
                Rootindex[u] = rootnow;             
                u = Next[u];
            end
        end
    end
    cnt = cnt/tau;
    # println(cnt)
    b = b/tau/length(alist)*G.n;
    Q = sparsevec(a,b,G.n)
    return Q
end

function rfrho(G,tau,c) #algorithm SCF
    d = [length(G.nbr[j]) for j = 1:G.n];
    InForest = [false for j = 1:G.n];
    Next = [-1 for j = 1:G.n];
    Rootindex = zeros(Int, G.n);         
    z = zeros(G.n)
    for p = 1:tau
        InForest .= false;
        Next .= -1;
        Rootindex .= 0;
        for i = 1:G.n
            u = i;
            while !InForest[u];
                seed = rand();
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true;
                    Next[u] = -1;
                    z[u] += c[u];
                    Rootindex[u] = u;
                else
                    k = floor(Int, seed * (1 + d[u]));
                    Next[u] = G.nbr[u][k];
                    u = Next[u];
                end
            end
            rootnow = Rootindex[u];
            u = i
            while !InForest[u]
                InForest[u] = true;
                Rootindex[u] = rootnow; 
                z[rootnow] += c[u];
                u = Next[u];
            end
        end
    end
    return z./tau
end


function rfrho_parallel(G, tau, c)  # Parallel version of SCF algorithm without atomic operations
    d = [length(G.nbr[j]) for j = 1:G.n]
    num_threads = nthreads()
    z_thread = [zeros(G.n) for _ in 1:num_threads]  # Each thread has its own accumulator

    @threads for p = 1:tau
        thread_id = threadid()
        z_local = z_thread[thread_id]  # Get the local accumulator for the current thread

        InForest = falses(G.n)
        Next = fill(-1, G.n)
        Rootindex = zeros(Int, G.n)

        for i = 1:G.n
            u = i
            while !InForest[u]
                seed = rand()
                if seed <= 1 / (1 + d[u])
                    InForest[u] = true
                    Next[u] = -1
                    z_local[u] += c[u]
                    Rootindex[u] = u
                else
                    k = floor(Int, seed * (1 + d[u]));
                    Next[u] = G.nbr[u][k]
                    u = Next[u]
                end
            end

            rootnow = Rootindex[u]
            u = i
            while !InForest[u]
                InForest[u] = true
                Rootindex[u] = rootnow
                z_local[rootnow] += c[u]
                u = Next[u]
            end
        end
    end

    # Combine results from all threads
    z_total = zeros(G.n)
    for z_local in z_thread
        z_total .+= z_local
    end

    return z_total ./ tau
end
 
