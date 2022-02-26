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

function cdf(d::Gamma, x::Real)
    if x < zero(x)
        return zero(x)
    else
        shp, scl = params(d)
        return gamma_cdf(shp, x/scl)
    end
end

function ccdf(d::Gamma, x::Real)
    if x < zero(x)
        return one(x)
    else
        shp, scl = params(d)
        return gamma_ccdf(shp, x/scl)
    end
end

function cdf(d::GeneralizedGamma, x::Real)
    if x < zero(x)
        return zero(x)
    else
        shp, scl, pwr = params(d)
        return gamma_cdf(shp/pwr, (x/scl)^pwr) 
    end
end

function ccdf(d::GeneralizedGamma, x::Real)
    if x < zero(x)
        return one(x)
    else
        shp, scl, pwr = params(d)
        return gamma_ccdf(shp/pwr, (x/scl)^pwr)
    end
end

function pdf(d::GeneralizedGamma, x::Real)
    if x < zero(x)
        return zero(x)
    else
        shp, scl, pwr = params(d)
        return (pwr/(scl^shp))*x^(shp-1)*exp(-(x/scl)^pwr)/gamma(shp/pwr)
    end
end

function logpdf(d::GeneralizedGamma, x::Real)
    if x < zero(x)
        return zero(x)
    else
        shp, scl, pwr = params(d)
        return log(pwr)-shp*log(scl) + (shp-1)*log(x) - (x/scl)^pwr -loggamma(shp/pwr)
    end
end

function eqcdf(d::GeneralizedGamma, x::Real)
    if x < zero(x)
        return zero(x)
    else
        shp, scl, pwr = params(d)
        out = (one(x) + x*gamma(shp/pwr)*gamma_ccdf(shp/pwr, (x/scl)^pwr)/(scl*gamma((shp+1)/pwr)) - gamma_ccdf((shp+1)/pwr, (x/scl)^pwr))
        return out
    end
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
function Mstep(d::GeneralizedGamma, y)
    shp, scl, pwr = params(d)
    v = [log(shp), log(pwr)]
    g = ForwardDiff.gradient(v -> sum(y->logpdf(GeneralizedGamma(exp(v[1]),scl,exp(v[2])),y), y), v)
    H = ForwardDiff.hessian(x -> sum(y->f(y,exp(x[1]),exp(x[2])), y), v)
    v -= H \ g
    b = mean(x -> x^exp(v[2]), y)^exp(-v[2])
    return GeneralizedGamma(exp(v[1]), b, exp(v[2]))
end

function Mstep(d::Gamma, y)
    shp, scl = params(d)
    rho = log(shp)
    meanlogy = mean(log,y)
    #f(x,shp) = -shp*log(scl) + (shp-1)*log(x) - (x/scl) -loggamma(shp)
    g = shp*log(scl) - shp*meanlogy + digamma(shp)*shp
    H = shp*log(scl) - shp*meanlogy + trigamma(shp)*shp^2 + digamma(shp)*shp
    rho -= g/H
    b = mean(y)/exp(rho)
    return Gamma(exp(rho), b)
end

function Mstep(d::Weibull, y)
    shp, scl = params(d)
    rho = log(shp)
    logy = sum(log, y)
    n = length(y)
    A = ((y/scl).^shp)'*(log.(y/scl))
    B = ((y/scl).^shp)'*(log.(y/scl).^2)
    #f(x,shp) = log(shp)-shp*log(scl) + (shp-1)*log(x) - (x/scl)^shp
    g = n*(shp*log(scl)-1) - shp*logy + shp*A
    H = n*(shp*log(scl)-1) - shp*logy + shp*A + (shp^2)*B
    rho -= g/H
    b = mean(x -> x^exp(rho), y)^exp(-rho)
    return Weibull(exp(rho), b)
end

y = rand(MersenneTwister(444), GeneralizedGamma(2.0,2.0,2.0), 100)
density(y, legend=false)
plot!(x -> pdf(GeneralizedGamma(2.0,2.0,2.0),x))
mean(x->logpdf(GeneralizedGamma(2.0,2.0,2.0),x),y)
d = Mstep(GeneralizedGamma(1.0,1.0,1.0), y)
mean(x->logpdf(d,x),y)
d = Mstep(d, y)

y = rand(MersenneTwister(111), Gamma(2.0,2.0), 100)
d = Mstep(Gamma(1.0,1.0), y)


function mynewton(d,y)
    p = zeros(10,2)
    for i in 1:10
    d = Mstep(d, y)
    p[i,:] .= log.(params(d))
    end
    return p
end

y = rand(MersenneTwister(12345), Weibull(2.0,2.0), 100)
theta = mynewton(Weibull(3,1),y)
maximum(theta[:,1])
minimum(theta[:,1])
maximum(theta[:,2])
minimum(theta[:,2])
av = 0.8:0.01:1.5
bv =  0.65:0.01:0.81
Xv = repeat(reshape(av, 1, :), length(bv), 1)
Yv = repeat(bv, 1, length(av))
Zv = map((a,b) -> exp(mean(y -> logpdf(Weibull(exp(a),exp(b)),y), y)), Xv, Yv)
contour(av, bv, Zv, linewidth=1, colorbar=false)
scatter!(theta[:,1],theta[:,2],legend=false)
#plot(Xv,Yv,Zv,st=:surface,camera=(70,40))
