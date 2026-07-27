using Distributions
using Random

# Using high-performance RNG and encapsulating random seed settings
const RNG = Xoshiro(round(Int, time()))

# struct Uniform <: Distribution{Univariate,Continuous} end
# struct Exponential <: Distribution{Univariate,Continuous} end
# struct Normal <: Distribution{Univariate,Continuous} end
struct PowerLaw <: Distribution{Univariate,Continuous} end
# struct Beta <: Distribution{Univariate,Continuous} end
# struct Poisson <: Distribution{Univariate,Continuous} end
struct TruncatedUniform <: Distribution{Univariate,Continuous} end
struct ScaledExponential <: Distribution{Univariate,Continuous} end
struct ScaledPowerLaw <: Distribution{Univariate,Continuous} end
"""
    generate_samples(dist::Distribution, n::Int; kwargs...)

Generate n random samples from a specified distribution, return normalized samples and distribution description strings.
"""
function generate_samples end

# Specific distribution implementation
function generate_samples(::Uniform, n::Int)
    rand(RNG, Float64, n), "Uniform(0,1)"
end

function generate_samples(::Exponential, n::Int; λ=1.0)
    x =  (-1 / λ) .* log.(1 .- rand(n))
    x ./= maximum(x)
    x, "Exponential(λ=$λ)"
end

function generate_samples(::Normal, n::Int; μ=0.0, σ=1.0)
    samples = rand(RNG, Normal(μ, σ), n)
    normalize_minmax(samples), "Normal(μ=$μ, σ=$σ)"
end

function generate_samples(::PowerLaw, n::Int; α=2.5, xmin=1.0)
    x = @. xmin * (1 - rand(RNG, Float64)) ^ (-1/(α - 1))
    (x .- 1) ./ (maximum(x) - 1), "PowerLaw(α=$α, xmin=$xmin)"
end

function generate_samples(::Beta, n::Int; α=0.5, β=0.5)
    rand(RNG, Beta(α, β), n), "Beta(α=$α, β=$β)"
end

function generate_samples(::Poisson, n::Int; λ=3.0)
    rand(RNG, Poisson(λ), n), "Poisson(λ=$λ)"
end

# auxiliary function
@inline function normalize_minmax(x::AbstractVector{T}) where T<:Real
    min_val, max_val = extrema(x)
    (x .- min_val) ./ (max_val - min_val)
end


# Truncate uniform distribution
function generate_samples(::TruncatedUniform, n::Int; m=0.01)
    rand(RNG, Uniform(m, 1.0), n), "TruncatedUniform(m=$m)"
end

# Scaling version exponential distribution
function generate_samples(::ScaledExponential, n::Int; m=0.01, λ=1.0)
    x, _ = generate_samples(Exponential(), n; λ)
    m .+ (1 - m) .* x, "ScaledExponential(m=$m, λ=$λ)"
end

# Scaling version power-law distribution
function generate_samples(::ScaledPowerLaw, n::Int; m=0.01, α=2.5, xmin=1.0)
    x, _ = generate_samples(PowerLaw(), n; α, xmin)
    m .+ (1-m) * (x .- 1) ./ (maximum(x) - 1), "ScaledPowerLaw(m=$m, α=$α, xmin=$xmin)"
end