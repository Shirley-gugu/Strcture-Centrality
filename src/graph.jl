using LinearAlgebra
using SparseArrays
using Serialization

struct Graph
    n :: Int 
    m :: Int 
    u :: Vector{Int}
    v :: Vector{Int}
    w :: Vector{Float32}
end

function graph_data_path(filename::AbstractString)
    return isfile(filename) ? String(filename) : joinpath("data", filename)
end

function read_data(
    filename::AbstractString;
    remap_ids::Bool = true,
    id_offset::Int = 0,
    estimated_edges::Int = 0,
    progress_every::Int = 0,
)
    estimated_edges >= 0 ||
        throw(ArgumentError("estimated_edges must be nonnegative"))
    progress_every >= 0 ||
        throw(ArgumentError("progress_every must be nonnegative"))

    path = graph_data_path(filename)
    u = Int[]
    v = Int[]
    w = Float32[]
    if estimated_edges > 0
        sizehint!(u, estimated_edges)
        sizehint!(v, estimated_edges)
        sizehint!(w, estimated_edges)
    end

    mapping = remap_ids ? Dict{Int, Int}() : nothing
    maximum_id = 0
    edge_count = 0

    open(path, "r") do io
        for line in eachline(io)
            isempty(line) && continue
            first_character = line[firstindex(line)]
            (first_character == '#' || first_character == '%') && continue
            fields = split(line; limit = 3)
            length(fields) >= 2 || continue
            source_id = parse(Int, fields[1])
            destination_id = parse(Int, fields[2])

            if remap_ids
                source = get(mapping, source_id, 0)
                if source == 0
                    source = length(mapping) + 1
                    mapping[source_id] = source
                end
                destination = get(mapping, destination_id, 0)
                if destination == 0
                    destination = length(mapping) + 1
                    mapping[destination_id] = destination
                end
            else
                source = source_id + id_offset
                destination = destination_id + id_offset
                source > 0 && destination > 0 || throw(ArgumentError(
                    "id_offset must convert all node IDs to positive Julia indices",
                ))
                maximum_id = max(maximum_id, source, destination)
            end

            push!(u, source)
            push!(v, destination)
            push!(w, 1.0f0)
            edge_count += 1
            if progress_every > 0 && edge_count % progress_every == 0
                println("loaded ", edge_count, " edges from ", path)
            end
        end
    end

    node_count = remap_ids ? length(mapping) : maximum_id
    return Graph(node_count, edge_count, u, v, w)
end

function read_data_cached(
    filename::AbstractString;
    cache_path = nothing,
    remap_ids::Bool = true,
    id_offset::Int = 0,
    estimated_edges::Int = 0,
    progress_every::Int = 0,
)
    source_path = graph_data_path(filename)
    mode = remap_ids ? "remapped" : "direct"
    resolved_cache_path = cache_path === nothing ?
        string(source_path, ".", mode, ".jls") : String(cache_path)
    source_size = filesize(source_path)
    source_mtime = mtime(source_path)

    if isfile(resolved_cache_path)
        cached = open(deserialize, resolved_cache_path)
        if cached.source_size == source_size &&
           cached.source_mtime == source_mtime &&
           cached.remap_ids == remap_ids &&
           cached.id_offset == id_offset
            return cached.graph
        end
    end

    graph = read_data(
        source_path;
        remap_ids = remap_ids,
        id_offset = id_offset,
        estimated_edges = estimated_edges,
        progress_every = progress_every,
    )
    open(resolved_cache_path, "w") do io
        serialize(io, (
            source_size = source_size,
            source_mtime = source_mtime,
            remap_ids = remap_ids,
            id_offset = id_offset,
            graph = graph,
        ))
    end
    return graph
end

# 无向图 度矩阵 
function degree_matrix(G)
	u=zeros(G.n);
	d=zeros(G.n);
	for i=1:G.n
		u[i]=i
	end
	for i=1:G.m
		if G.w[i]>0
			d[G.u[i]]+=G.w[i];
			d[G.v[i]]+=G.w[i];
		elseif G.w[i]<0
			d[G.u[i]]-=G.w[i];
			d[G.v[i]]-=G.w[i];
		end
	end
	return sparse(u,u,d)
end

function degree_vector(G)
	d=zeros(G.n);
	for i=1:G.m
		if G.w[i]>0
			d[G.u[i]]+=G.w[i];
			# d[G.v[i]]+=G.w[i];
		elseif G.w[i]<0
			d[G.u[i]]-=G.w[i];
			# d[G.v[i]]-=G.w[i];
		end
	end
	return d
end

# 无向图 邻接矩阵
# function adjacency_matrix(G)
# 	u=zeros(2*G.m);
# 	v=zeros(2*G.m);
# 	w=zeros(2*G.m);
# 	for i=1:G.m
# 		u[i]=G.u[i];
# 		v[i]=G.v[i];
# 		w[i]=G.w[i];
# 		u[G.m+i]=G.v[i];
# 		v[G.m+i]=G.u[i];
# 		w[G.m+i]=G.w[i];
# 	end
# 	return sparse(u,v,w)
# end

function adjacency_matrix_directed(G)
	u=zeros(G.m);
	v=zeros(G.m);
	w=zeros(G.m);
	for i=1:G.m
		u[i]=G.u[i];
		v[i]=G.v[i];
		w[i]=G.w[i];
		# u[G.m+i]=G.v[i];
		# v[G.m+i]=G.u[i];
		# w[G.m+i]=G.w[i];
	end
	return sparse(u,v,w,G.n,G.n)
end

#拉普拉斯矩阵
function laplacian_matrix(G :: Graph)
	d=zeros(G.n);
	for i=1:G.m
		if G.w[i]>0
			d[G.u[i]]+=G.w[i];
			d[G.v[i]]+=G.w[i];
		elseif G.w[i]<0
			d[G.u[i]]-=G.w[i];
			d[G.v[i]]-=G.w[i];
		end
	end
	uu=zeros(2*G.m+G.n);
	vv=zeros(2*G.m+G.n);
	ww=zeros(2*G.m+G.n);
	a=zeros(G.n);
	for i=1:G.n
		a[i]=i;
	end
	uu[1:G.m]=G.u;
	uu[G.m+1:2*G.m]=G.v;
	uu[2*G.m+1:2*G.m+G.n]=a;
	vv[1:G.m]=G.v;
	vv[G.m+1:2*G.m]=G.u;
	vv[2*G.m+1:2*G.m+G.n]=a;
	ww[1:G.m].=-G.w;
	ww[G.m+1:2*G.m].=-G.w;
	ww[2*G.m+1:2*G.m+G.n]=d;
    return sparse(uu,vv,ww)
end



function gremban_expansion(G::Graph)
	n = G.n
	m = G.m
	u = zeros(Int, 2*m)
	v = zeros(Int, 2*m)
	w = zeros(Float64, 2*m)
	for i in 1:G.m
		if G.w[i] > 0
			u[i] = G.u[i]
			v[i] = G.v[i]
			w[i] = G.w[i]
			u[i+m] = G.u[i]+n
			v[i+m] = G.v[i]+n
			w[i+m] = G.w[i]
		elseif G.w[i] < 0
			u[i] = G.u[i]
			v[i] = G.v[i]+n
			w[i] = -G.w[i]
			u[i+m] = G.u[i]+n
			v[i+m] = G.v[i]
			w[i+m] = -G.w[i]
		end
	end
	return Graph(2*n, 2*m, u, v, w)
end


function neighbor_weight(G::Graph)
	n = G.n
	m = G.m
	nbr = Array{Array{Int, 1}}(undef, n)
	weight = Array{Array{Float64, 1}}(undef, n)
	for i in 1:n
		nbr[i] = Int[]
		weight[i] = Float64[]
	end
	for i in 1:m
		push!(nbr[G.u[i]], G.v[i])
		# push!(nbr[G.v[i]], G.u[i])
		push!(weight[G.u[i]], abs(G.w[i]))
		# push!(weight[G.v[i]], abs(G.w[i]))
	end
	return nbr, weight
end

function transition_matrix(G::Graph)
	n=G.n
	m=G.m
	d=zeros(G.n);
	for i=1:G.m
		d[G.u[i]]+=abs(G.w[i]);
		d[G.v[i]]+=abs(G.w[i]);	
	end
	u=zeros(2*G.m);
	v=zeros(2*G.m);
	w=zeros(2*G.m);
	absw=zeros(2*G.m);
	for i=1:G.m
		u[i]=G.u[i];
		v[i]=G.v[i];
		w[i]=G.w[i]/d[G.u[i]];
		absw[i]=abs(w[i]);
		u[G.m+i]=G.v[i];
		v[G.m+i]=G.u[i];
		w[G.m+i]=G.w[i]/d[G.v[i]];
		absw[G.m+i]=abs(w[G.m+i]);
	end
	return sparse(u,v,absw),sparse(u,v,w)
end


