"""
GeneralizedGamma(a,b,k)

The *Generalized gamma distribution* with shape parameters `a`, scale `b`, and power `k` has probability density
function

```math
f(x; a, b) = \\frac{(k/b^a) x^{a-1} e^{-(x/b)^k}{\\Gamma(a/k)},
\\quad x > 0
```

```julia
GeneralizedGamma(a, b, k)      # GeneralizedGamma distribution with shape a, scale b, power k

params(d)        # Get the parameters, i.e. (a, b, k)
shape(d)         # Get the shape parameter, i.e. a
scale(b)         # Get the scale parameter, i.e. b
```

External links

* [GeneralizedGamma distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_gamma_distribution)

"""

struct GeneralizedGamma{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    k::T
end

#### Parameters

params(d::GeneralizedGamma) = (d.a, d.b, d.k)

shape(d::GeneralizedGamma) = d.a
scale(d::GeneralizedGamma) = d.b
power(d::GeneralizedGamma) = 1 / d.k

partype(d::GeneralizedGamma{T}) where {T} = T

#### Evaluation

function gamma_cdf(a, x)
    return gamma_inc(a,x,0)[1]
end

function gamma_ccdf(a, x)
    return gamma_inc(a,x,0)[2]
end

function cdf(d::GeneralizedGamma, x)
    shp, scl, pwr = params(d)
    return gamma_cdf(shp/pwr, (x/scl)^pwr)
end

function ccdf(d::GeneralizedGamma, x)
    shp, scl, pwr = params(d)
    return gamma_ccdf(shp/pwr, (x/scl)^pwr)
end

ccdf2(d::GeneralizedGamma, x) = ccdf(d::GeneralizedGamma, x)

function pdf(d::GeneralizedGamma, x)
    shp, scl, pwr = params(d)
    return (pwr/(scl^shp))*x^(shp-1)*exp(-(x/scl)^pwr)/gamma(shp/pwr)
end

function logpdf(d::GeneralizedGamma, x)
    shp, scl, pwr = params(d)
    return log(pwr)-shp*log(scl) + (shp-1)*log(x) -(x/scl)^pwr -loggamma(shp/pwr)
end

function eqcdf(d::GeneralizedGamma, x)
    shp, scl, pwr = params(d)
    out = zero(x)
    if x >= zero(x)
      out = (one(x) + x*gamma(shp/pwr)*gamma_ccdf(shp/pwr, (x/scl)^pwr)/(scl*gamma((shp+1)/pwr)) - gamma_ccdf((shp+1)/pwr, (x/scl)^pwr))
    end
    return out
end

# p = -log.(rand(3))
# plot(x -> eqcdf(GeneralizedGamma(p[1],p[2],p[3]),x), 0, 20, legend=false)

function quantile(d::GeneralizedGamma, p)
    shp, scl, pwr = params(d)
    r = gamma_inc_inv(shp/pwr, p, 1-p)
    return scl*(r^inv(pwr))
end

function rand(rng::AbstractRNG, d::GeneralizedGamma)
    p = rand(rng)
    return quantile(d, p)
end


#### Statistics

function mean(d::GeneralizedGamma)
    shp, scl, pwr = params(d)
    return scl*gamma((shp+1)/pwr))/gamma(shp/pwr)
end

#mean(d::Gamma) = d.α * d.θ

#var(d::Gamma) = d.α * d.θ^2

#skewness(d::Gamma) = 2 / sqrt(d.α)

#kurtosis(d::Gamma) = 6 / d.α

# function mode(d::Gamma)
#     (α, θ) = params(d)
#     α >= 1 ? θ * (α - 1) : error("Gamma has no mode when shape < 1")
# end

# function entropy(d::Gamma)
#     (α, θ) = params(d)
#     α + loggamma(α) + (1 - α) * digamma(α) + log(θ)
# end

# mgf(d::Gamma, t::Real) = (1 - t * d.θ)^(-d.α)

# cf(d::Gamma, t::Real) = (1 - im * t * d.θ)^(-d.α)

# function kldivergence(p::Gamma, q::Gamma)
#     # We use the parametrization with the scale θ
#     αp, θp = params(p)
#     αq, θq = params(q)
#     θp_over_θq = θp / θq
#     return (αp - αq) * digamma(αp) - loggamma(αp) + loggamma(αq) -
#         αq * log(θp_over_θq) + αp * (θp_over_θq - 1)
# end

#### Evaluation & Sampling

# @_delegate_statsfuns Gamma gamma α θ

# gradlogpdf(d::Gamma{T}, x::Real) where {T<:Real} =
#     insupport(Gamma, x) ? (d.α - 1) / x - 1 / d.θ : zero(T)

# function rand(rng::AbstractRNG, d::Gamma)
#     if shape(d) < 1.0
#         # TODO: shape(d) = 0.5 : use scaled chisq
#         return rand(rng, GammaIPSampler(d))
#     elseif shape(d) == 1.0
#         return rand(rng, Exponential(d.θ))
#     else
#         return rand(rng, GammaGDSampler(d))
#     end
# end

# function sampler(d::Gamma)
#     if shape(d) < 1.0
#         # TODO: shape(d) = 0.5 : use scaled chisq
#         return GammaIPSampler(d)
#     elseif shape(d) == 1.0
#         return sampler(Exponential(d.θ))
#     else
#         return GammaGDSampler(d)
#     end
# end

#### Fit model

# struct GammaStats <: SufficientStats
#     sx::Float64      # (weighted) sum of x
#     slogx::Float64   # (weighted) sum of log(x)
#     tw::Float64      # total sample weight

#     GammaStats(sx::Real, slogx::Real, tw::Real) = new(sx, slogx, tw)
# end

# function suffstats(::Type{<:Gamma}, x::AbstractArray{T}) where T<:Real
#     sx = zero(T)
#     slogx = zero(T)
#     for xi = x
#         sx += xi
#         slogx += log(xi)
#     end
#     GammaStats(sx, slogx, length(x))
# end

# function suffstats(::Type{<:Gamma}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
#     n = length(x)
#     if length(w) != n
#         throw(DimensionMismatch("Inconsistent argument dimensions."))
#     end

#     sx = zero(T)
#     slogx = zero(T)
#     tw = zero(T)
#     for i = 1:n
#         @inbounds xi = x[i]
#         @inbounds wi = w[i]
#         sx += wi * xi
#         slogx += wi * log(xi)
#         tw += wi
#     end
#     GammaStats(sx, slogx, tw)
# end

# function gamma_mle_update(logmx::Float64, mlogx::Float64, a::Float64)
#     ia = 1 / a
#     z = ia + (mlogx - logmx + log(a) - digamma(a)) / (abs2(a) * (ia - trigamma(a)))
#     1 / z
# end

# function fit_mle(::Type{<:Gamma}, ss::GammaStats;
#     alpha0::Float64=NaN, maxiter::Int=1000, tol::Float64=1e-16)

#     mx = ss.sx / ss.tw
#     logmx = log(mx)
#     mlogx = ss.slogx / ss.tw

#     a::Float64 = isnan(alpha0) ? (logmx - mlogx)/2 : alpha0
#     converged = false

#     t = 0
#     while !converged && t < maxiter
#         t += 1
#         a_old = a
#         a = gamma_mle_update(logmx, mlogx, a)
#         converged = abs(a - a_old) <= tol
#     end

#     Gamma(a, mx / a)
# end