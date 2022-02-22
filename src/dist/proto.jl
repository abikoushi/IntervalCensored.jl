using Distributions
using SpecialFunctions
using Random
using Plots
import Statistics: quantile
import Base: rand

struct GeneralizedGamma{T<:Real} <: ContinuousUnivariateDistribution
    a::T
    b::T
    k::T
end

function gamma_cdf(a, x)
    return gamma_inc(a,x,0)[1]
end

function gamma_ccdf(a, x)
    return gamma_inc(a,x,0)[2]
end

params(d::GeneralizedGamma) = (d.a, d.b, d.k)

function cdf(d::GeneralizedGamma, x)
    shp, scl, pwr = params(d)
    return gamma_cdf(shp/pwr, (x/scl)^pwr)
end

function ccdf(d::GeneralizedGamma, x)
    shp, scl, pwr = params(d)
    return gamma_ccdf(shp/pwr, (x/scl)^pwr)
end

function mean(d::GeneralizedGamma)
    shp, scl, pwr = params(d)
    return scl*gamma((shp+1)/pwr))/gamma(shp/pwr)
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

function Mstep(d::GeneralizedGamma, y)
    a, b, k = params(d)
    n = length(y)
    sumlogy = sum(log, y)
    powklogy = (y.^k)' * (log.(y) .- log(b))
    powklogy2 = (y.^k)' * (log.(y) .- log(b)).^2

    fx = [-log(b) + (b-1)*sumlogy - n*digamma(a/k)/k, inv(k) + b^(-k) * powklogy + n*a*digamma(a/k)/k^2]
    dfx = [- n*trigamma(a/k)/k^2   (n/k^2)*digamma(a/k)+(n/k^3)*trigamma(a/k)
        (n/k^2)*digamma(a/k)+(n/k^3)*trigamma(a/k)  -n/(k^2) + n*log(b)*b^(-k) + powklogy2 + n*(2*a/k^3)*digamma(a/k)+ n*(2*a^2/k^4)*trigamma(a/k)]
        
    Delta = dfx \ fx
    a += Delta[1]
    k += Delta[2]
    b = (mean(x -> k*x^k, y)^inv(a))/a
    return GeneralizedGamma(a,b,k)
end


y = rand(MersenneTwister(1234), GeneralizedGamma(2.0,2.0,0.8), 1000)
histogram(y)
d = Mstep(GeneralizedGamma(2.0,2.0,0.8), y)
d = Mstep(d,y)