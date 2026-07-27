include("graph.jl")

using LinearAlgebra, SparseArrays
using StatsBase,Statistics
using DataFrames,CSV

function exact(G::Graph)
    A=adjacency_matrix(G)
    D=degree_matrix(G)
    Q=inv(Matrix(I+D-A))
    qn=zeros(G.n)
    for i=1:G.n
        qn[i]=Q[i,i]
    end
    return qn
end

function wii(G::Graph, sample_num::Int)
    G2=gremban_expansion(G)
    G=G2
    n=G.n
    num=Int(n/2)
    d=degree_vector(G)
    in_forests = fill(false,n)
    next = fill(-1,n)
    root = fill(-1,n)
    nbr,weight=neighbor_weight(G)
    ans=zeros(Float64,Int(n/2))
    # @show n2
    for _ in 1:sample_num
        fill!(in_forests, false)
        for src in 1:num
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    if u<=num
                        ans[u] += 1/sample_num
                    end
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
                if r==u+num
                    ans[u]-=1/sample_num
                end
                u = next[u]
            end
        end
    end
    return ans
end

function wii_reduce(G::Graph, sample_num::Int)
    G2=gremban_expansion(G)
    G=G2
    n=G.n
    num=Int(n/2)
    d=degree_vector(G)
    in_forests = fill(false,n)
    next = fill(-1,n)
    root = fill(-1,n)
    nbr,weight=neighbor_weight(G)
    A=adjacency_matrix(G)
    ans=zeros(Float64,Int(n/2))
    for i in 1:num
        ans[i]=1/(1+d[i])
    end
    for _ in 1:sample_num
        fill!(in_forests, false)
        for src in 1:num
            u = src
            while in_forests[u] == false
                if rand(Float64) * (d[u] + 1) < 1
                    in_forests[u] = true
                    root[u] = u
                    # if u<=num
                    #     ans[u] += 1/sample_num
                    # end
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
                if u<=num
                    ru=A[r,u]
                    ru2=A[r,u+num]
                    if ru>0
                        ans[u] += 1/ (1 +d[u])/sample_num
                    end
                    if ru2>0
                        ans[u] -= 1/ (1 +d[u])/sample_num
                    end
                    u = next[u]
                end
            end
        end
    end
    return ans
end

# filelist=["test.txt","tribes.txt","bitcoinalpha.txt","bitcoinotc.txt"]
filelist=["wikipedialinks.txt"]
for file in filelist
    G=read_data(file)
    @show G.n,G.m

    @timev q=@timed exact(G)

    time=zeros(Float64,7)
    error=zeros(Float64,7)
    time[1]=q.time
    q= q.value

    for k in 1:3
        l=500*k
        @timev q1 = @timed wii(G,l)
        @timev q2 = @timed wii_reduce(G,l)

        time[2*k]= q1.time
        time[2*k+1]= q2.time

        error[2*k]= mean(abs.(q1.value - q)./q) 
        error[2*k+1]= mean(abs.(q2.value - q)./q)
    end
    df=DataFrame(time=time, error=error)
    @show df
    file ="syb"* file[1:end-4] * ".csv"
    CSV.write(file, df)
end