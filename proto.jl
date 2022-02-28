using Distributions
using SpecialFunctions
using Random
using Plots, StatsPlots
using ForwardDiff
import LinearAlgebra: dot
import Distributions: ccdf, cdf, logpdf, pdf,
       mean, rand, params, shape, scale

# abstract type survdist end
# abstract type GeneralizedGamma <: survdist end
# abstract type Gamma <: survdist end
# abstract type Weibull <: survdist end
# abstract type Exponential <: survdist end

struct GeneralizedGamma <: ContinuousUnivariateDistribution
    a
    b
    k
end

#### Parameters
params(d::GeneralizedGamma) = (d.a, d.b, d.k)
shape(d::GeneralizedGamma) = d.a
scale(d::GeneralizedGamma) = d.b
power(d::GeneralizedGamma) = d.k

# Moment
function mean(d::GeneralizedGamma)
    shp, scl, pwr = params(d)
    return scl*gamma((shp+1)/pwr)/gamma(shp/pwr)
end

function logmean(d::GeneralizedGamma)
    shp, scl, pwr = params(d)
    return log(scl) + loggamma((shp+1)/pwr)-loggamma(shp/pwr)
end

# Evaluation
function gamma_cdf(a::Real, x::Real)
    return gamma_inc(a,x,0)[1]
end

function gamma_ccdf(a::Real, x::Real)
    return gamma_inc(a,x,0)[2]
end

function gamma_cdf(a::ForwardDiff.Dual{T}, x::Real) where {T} #where {T} の意味わかっていない
    y = gamma_inc(ForwardDiff.value(a),x,0)[1]
    y_a = (log(x) - digamma(a))*y-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x)
    return ForwardDiff.Dual{T}(y,y_a)
end

function gamma_cdf(a::Real, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_inc(a,ForwardDiff.value(x),0)[1]
    y_x = exp(-x + (a-1)*log(x) - loggamma(a))
    return ForwardDiff.Dual{T}(y,y_x)
end

function gamma_ccdf(a::ForwardDiff.Dual{T}, x::Real) where {T}
    y = gamma_inc(ForwardDiff.value(a),x,0)[2]
    y_a = -((log(x) - digamma(a))*(1-y)-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x))
    return ForwardDiff.Dual{T}(y,y_a)
end

function gamma_ccdf(a::Real, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_inc(a,ForwardDiff.value(x),0)[2]
    y_x = -exp(-x + (a-1)*log(x) - loggamma(a))
    return ForwardDiff.Dual{T}(y,y_x)
end

function gamma_cdf(a::ForwardDiff.Dual{T}, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_cdf(ForwardDiff.value(a),ForwardDiff.value(x))
    y_a = (log(x) - digamma(a))*y-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x)
    y_x = exp(-x + (a-1)*log(x) - loggamma(a))
    pa = ForwardDiff.partials(a).values
    px = ForwardDiff.partials(x).values
    ga = y_a*pa[1] + y_x*px[1]
    gx = y_a*pa[2] + y_x*px[2]
    return ForwardDiff.Dual{T}(y,ga,gx)
end

function gamma_ccdf(a::ForwardDiff.Dual{T}, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_ccdf(ForwardDiff.value(a),ForwardDiff.value(x))
    y_a = -((log(x) - digamma(a))*(1-y)-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x))
    y_x = -exp(-x + (a-1)*log(x) - loggamma(a))
    pa = ForwardDiff.partials(a).values
    px = ForwardDiff.partials(x).values
    ga = y_a*pa[1] + y_x*px[1]
    gx = y_a*pa[2] + y_x*px[2]
    return ForwardDiff.Dual{T}(y,ga,gx)
end

function quantile(d::GeneralizedGamma, p::Real)
    shp, scl, pwr = params(d)
    r = gamma_inc_inv(shp/pwr, p, 1-p)
    return scl*(r^inv(pwr))
end

function rand(rng::AbstractRNG, d::GeneralizedGamma)
    p = rand(rng)
    return quantile(d, p)
end

#d = GeneralizedGamma(2.0, 2.0, 2.0)
