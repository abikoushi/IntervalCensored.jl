using Distributions
using SpecialFunctions
using Random
using Plots, StatsPlots
using ForwardDiff
import Distributions: ccdf, cdf, logpdf, pdf, mean, rand

abstract type GeneralizedGammaFamily <: ContinuousUnivariateDistribution end
abstract type Gamma <: GeneralizedGammaFamily end
abstract type Weibull <: GeneralizedGammaFamily end
abstract type Exponential <: GeneralizedGammaFamily end

struct GeneralizedGamma <: GeneralizedGammaFamily
    a
    b
    k
end

#### Parameters
params(d::GeneralizedGamma) = (d.a, d.b, d.k)
logparams(d::GeneralizedGamma) = (log(d.a), log(d.b), log(d.k))
shape(d::GeneralizedGamma) = d.a
scale(d::GeneralizedGamma) = d.b
power(d::GeneralizedGamma) = d.k

# moment
function mean(d::GeneralizedGamma)
    shp, scl, pwr = params(d)
    return scl*gamma((shp+1)/pwr)/gamma(shp/pwr)
end

function logmean(d::GeneralizedGamma)
    shp, scl, pwr = params(d)
    return log(scl) + loggamma((shp+1)/pwr)-loggamma(shp/pwr)
end

# Evaluation
function gamma_cdf(a::Number, x::Number)
    return gamma_inc(a,x,0)[1]
end

function gamma_ccdf(a::Number, x::Number)
    return gamma_inc(a,x,0)[2]
end

#where {T} の意味わかっていない
function gamma_cdf(a::ForwardDiff.Dual{T}, x::Number) where {T}
    y = gamma_inc(ForwardDiff.value(a),x,0)[1]
    y_a = (log(x) - digamma(a))*y-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x)
    return ForwardDiff.Dual{T}(y,y_a)
end

function gamma_cdf(a::Number, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_inc(a,ForwardDiff.value(x),0)[1]
    y_x = exp(-x + (a-1)*log(x) - loggamma(a))
    return ForwardDiff.Dual{T}(y,y_x)
end

function gamma_ccdf(a::ForwardDiff.Dual{T}, x::Number) where {T}
    y = gamma_inc(ForwardDiff.value(a),x,0)[2]
    y_a = -((log(x) - digamma(a))*(1-y)-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x))
    return ForwardDiff.Dual{T}(y,y_a)
end

function gamma_ccdf(a::Number, x::ForwardDiff.Dual{T}) where {T}
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

function cdf(d::Gamma, x)
    if x < zero(x)
        return zero(x)
    else
        shp, scl = params(d)
        return gamma_cdf(shp, x/scl)
    end
end

function ccdf(d::Gamma, x)
    if x < zero(x)
        return one(x)
    else
        shp, scl = params(d)
        return gamma_ccdf(shp, x/scl)
    end
end

function cdf(d::GeneralizedGamma, x)
    if x < zero(x)
        return zero(x)
    else
        shp, scl, pwr = params(d)
        return gamma_cdf(shp/pwr, (x/scl)^pwr) 
    end
end

function ccdf(d::GeneralizedGamma, x)
    if x < zero(x)
        return one(x)
    else
        shp, scl, pwr = params(d)
        return gamma_ccdf(shp/pwr, (x/scl)^pwr)
    end
end

function pdf(d::GeneralizedGamma, x)
    if x < zero(x)
        return zero(x)
    else
        shp, scl, pwr = params(d)
        return (pwr/(scl^shp))*x^(shp-1)*exp(-(x/scl)^pwr)/gamma(shp/pwr)
    end
end

function logpdf(d::GeneralizedGamma, x)
    if x < zero(x)
        return zero(x)
    else
        shp, scl, pwr = params(d)
        return log(pwr)-shp*log(scl) + (shp-1)*log(x) - (x/scl)^pwr -loggamma(shp/pwr)
    end
end

function eqcdf(d::GeneralizedGamma, x)
    if x < zero(x)
        return zero(x)
    else
        shp, scl, pwr = params(d)
        out = (one(x) + x*gamma(shp/pwr)*gamma_ccdf(shp/pwr, (x/scl)^pwr)/(scl*gamma((shp+1)/pwr)) - gamma_ccdf((shp+1)/pwr, (x/scl)^pwr))
        return out
    end
end

function quantile(d::GeneralizedGamma, p)
    shp, scl, pwr = params(d)
    r = gamma_inc_inv(shp/pwr, p, 1-p)
    return scl*(r^inv(pwr))
end

function Mstep(d::GeneralizedGamma, y)
    shp, scl, pwr = params(d)
    v = [pwr, shp]
    n = length(y)
    #log(pwr)-shp*log(scl) + (shp-1)*log(x) -(x/scl)^pwr -loggamma(shp/pwr)
    g1 = n/pwr - (log.(y/scl)' * (y/scl).^pwr) + n*digamma(shp/pwr) * (shp/pwr^2)
    g2 = -n*log(scl) + sum(log, y) - n*digamma(shp/pwr)/pwr
    h11 = -n/pwr^2 - (log.(y/scl).^2)' * (y/scl).^pwr - n*trigamma(shp/pwr) * (shp^2/pwr^4) - 2*n*digamma(shp/pwr) * (shp/pwr^3)
    h12 = n*(trigamma(shp/pwr)*(shp/pwr^3) + digamma(shp/pwr)/(pwr^2))
    h22 = -n*trigamma(shp/pwr)/(pwr^2)
    g = [g1, g2]
    H = [[h11, h12] [h12, h22]]
    v -= H \ g
    b = (mean(x -> x^v[1], y)^inv(v[1]))
    return GeneralizedGamma(v[2], b, v[1]), mean(logpdf.(GeneralizedGamma(v[2], b, v[1]),y))
end

function rand(rng::AbstractRNG, d::GeneralizedGamma)
    p = rand(rng)
    return quantile(d, p)
end

function Mstep(d::GeneralizedGamma, y)
    shp, scl, pwr = params(d)
    v = [log(pwr), log(shp)]
    n = length(y)
    f(x,pwr,shp)=log(pwr)-shp*log(scl) + (shp-1)*log(x) - (x/scl)^pwr -loggamma(shp/pwr)
    g = ForwardDiff.gradient(x -> sum(y->f(y,exp(x[1]),exp(x[2])), y), v)
    H = ForwardDiff.hessian(x -> sum(y->f(y,exp(x[1]),exp(x[2])), y), v)
    v -= H \ g
    b = mean(x -> x^exp(v[1]), y)^exp(-v[1])
    return GeneralizedGamma(exp(v[2]), b, exp(v[1]))
end

y = rand(MersenneTwister(444), GeneralizedGamma(2.0,2.0,2.0), 1000)
density(y, legend=false)
plot!(x -> pdf(GeneralizedGamma(2.0,2.0,2.0),x))
plot!(x -> pdf(d,x))
mean(x->logpdf(GeneralizedGamma(2.0,2.0,2.0),x),y)
d = Mstep(GeneralizedGamma(2.0,2.0,2.0), y)
mean(x->logpdf(d,x),y)
d = Mstep(d, y)
