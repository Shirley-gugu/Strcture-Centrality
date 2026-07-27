include("Forest.jl")

function PW(G::Graph, z1, k)
    n = G.n
    A = adjacency_matrix_directed(G)
    d = degree_vector(G)
    inv_diag = 1.0 ./ (1.0 .+ d)
    At = transpose(A)
    z = Vector{Float64}(z1)
    next_z = similar(z)

    for _ in 1:k
        mul!(next_z, At, z)
        @. next_z = (1.0 + next_z) * inv_diag
        z, next_z = next_z, z
    end

    return z
end

function PW_FC(
    G::Graph,
    k::Int;
    sample_num::Int = 10,
    seed::Int = 1,
)
    k >= 0 || throw(ArgumentError("k must be nonnegative"))
    sample_num > 0 || throw(ArgumentError("sample_num must be positive"))

    n = G.n
    A = adjacency_matrix_directed(G)
    d = degree_vector(G)
    inv_diag = 1.0 ./ (1.0 .+ d)
    At = transpose(A)
    z1 = forestZ_parallel(G, sample_num; seed = seed)
    #z1 = forestColumnSums(G, 10)
    z = Vector{Float64}(z1)
    next_z = similar(z)

    for _ in 1:k
        mul!(next_z, At, z)
        @. next_z = (1.0 + next_z) * inv_diag
        z, next_z = next_z, z
    end

    return z
end

function PW_FC_tol(G::Graph; sample_num=10, tol=1e-8, maxiter=10_000)
      A = adjacency_matrix_directed(G)
      d = degree_vector(G)
      inv_diag = 1.0 ./ (1.0 .+ d)
      At = transpose(A)
      denom = 1.0 .+ d

      z = Vector{Float64}(forestColumnSums(G, sample_num))
      next_z = similar(z)
      e = ones(Float64, G.n)

      for iter in 1:maxiter
          mul!(next_z, At, z)
          @. next_z = (1.0 + next_z) * inv_diag

          r = e .+ At * next_z .- denom .* next_z
          relres = norm(r) / norm(e)

          if relres <= tol
              return next_z, iter, relres
          end

          z, next_z = next_z, z
      end

      r = e .+ At * z .- denom .* z
      return z, maxiter, norm(r) / norm(e)
end

function PW_FC_balanced(
    G::Graph;
    sample_num::Int = 10,
    residual_tol::Float64 = 1e-4,
    marginal_tol::Float64 = 1e-3,
    miniter::Int = 20,
    maxiter::Int = 300,
    check_every::Int = 5,
)
      sample_num > 0 ||
          throw(ArgumentError("sample_num must be positive"))

      residual_tol > 0.0 ||
          throw(ArgumentError("residual_tol must be positive"))

      marginal_tol > 0.0 ||
          throw(ArgumentError("marginal_tol must be positive"))

      miniter >= 0 ||
          throw(ArgumentError("miniter must be nonnegative"))

      maxiter >= miniter ||
          throw(ArgumentError("maxiter must be at least miniter"))

      check_every > 0 ||
          throw(ArgumentError("check_every must be positive"))

      A = adjacency_matrix_directed(G)
      d = degree_vector(G)
      inv_diag = 1.0 ./ (1.0 .+ d)
      At = transpose(A)
      denom = 1.0 .+ d

      z = Vector{Float64}(forestColumnSums(G, sample_num))
      next_z = similar(z)
      e = ones(Float64, G.n)

      last_step_change = Inf
      relres = Inf
      stop_reason = :maxiter

      for iter in 1:maxiter
          mul!(next_z, At, z)
          @. next_z = (1.0 + next_z) * inv_diag

          step_change =
              norm(next_z - z) /
              max(norm(next_z), eps(Float64))

          should_check =
              iter >= miniter &&
              (iter % check_every == 0 || iter == maxiter)

          if should_check
              r = e .+ At * next_z .- denom .* next_z
              relres = norm(r) / norm(e)

              if relres <= residual_tol
                  stop_reason = :residual_tol
                  z, next_z = next_z, z
                  return z, (
                      iterations = iter,
                      relres = relres,
                      step_change = step_change,
                      stop_reason = stop_reason,
                  )
              end

              marginal_gain =
                  isfinite(last_step_change) ?
                  (last_step_change - step_change) /
                  max(last_step_change, eps(Float64)) :
                  Inf

              if marginal_gain >= 0.0 &&
                 marginal_gain <= marginal_tol
                  stop_reason = :marginal_tol
                  z, next_z = next_z, z
                  return z, (
                      iterations = iter,
                      relres = relres,
                      step_change = step_change,
                      marginal_gain = marginal_gain,
                      stop_reason = stop_reason,
                  )
              end

              last_step_change = step_change
          end

          z, next_z = next_z, z
      end

      r = e .+ At * z .- denom .* z
      relres = norm(r) / norm(e)

      return z, (
          iterations = maxiter,
          relres = relres,
          step_change = last_step_change,
          stop_reason = stop_reason,
      )
end
