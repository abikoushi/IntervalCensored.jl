function Mstep(d::LogNormal, y)
    mu = mean(log, y)
    sigma = sqrt(mean(x -> (log(x)-mu)^2, y))
    return LogNormal(mu, sigma)
end

function Mstep(d::GeneralizedGamma, y)
    shp, scl, pwr = params(d)
    v = [log(shp), log(pwr)]
    g = ForwardDiff.gradient(v -> sum(y-> -logpdf(GeneralizedGamma(exp(v[1]),scl,exp(v[2])),y)), v)
    H = ForwardDiff.hessian(x -> sum(y-> -logpdf(GeneralizedGamma(exp(v[1]),scl,exp(v[2])),y)), v)
    v -= (H \ g)
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
    rho -= (g/H)
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
    rho -= (g/H)
    b = mean(x -> x^exp(rho), y)^exp(-rho)
    return Weibull(exp(rho), b)
end

# function Mstep(d::Gamma, y)
#     mlog = log(mean(y))
#     logm = mean(log, y)
#     a = shape(d)
#     ia = inv(a)
#     a = inv(ia + (-mlog + logm + log(a) - digamma(a)) / ((a^2) * (ia - trigamma(a))))
#     b = mean(y) / a
#     return Gamma(a,b)
# end

# function Mstep(d::Weibull, y)
#     a, b = params(d)
#     n = length(y)
#     powa = (y.^a)'
#     logy = log.(y)

#     fa = n*(inv(a) - inv(b)) + sum(logy)- n*log(b) - powa*(logy-log(b))/(b^a)
#     dfa = -n*inv(a^2) - (powa*(logy-log(b)).^2)/(b^a)

#     Delta = fx / dfx
#     a += Delta
#     b = mean(x -> x^a, ytilde)^inv(a)
#     return Weibull(a,b)
# end

function Mstep(d::Exponential, y)
    b = mean(y)
    return Exponential(b)
end

####

function Estep(rng::AbstractRNG, dist::ContinuousUnivariateDistribution, x::IC)
    ytilde = rand(rng, truncated(dist, x.S-x.ER, x.S-x.EL))
    return ytilde
end

function Estep(rng::AbstractRNG, dist::ContinuousUnivariateDistribution, x::ICRT)
    ys = rand(rng, truncated(dist, x.S-x.ER, x.S-x.EL))
    u = rand(rng)
    E = x.EL + (x.ER-x.EL)*u
    q = cdf(dist, x.TR - E)
    yb = []
    if zero(q) < q < one(q)
        B = rand(rng, Geometric(q))
        if B > 0
            push!(yb,[rand(rng, truncated(dist,Tmax-E,Inf)) for i in UnitRange(1,B)])
        end
    end
    return [ys ; yb]
end

function Estep(rng::AbstractRNG, dist::ContinuousUnivariateDistribution, x::DIC)
    u = rand(rng)
    S = x.SL + (x.SR-x.SL)*u
    ytilde = rand(rng, truncated(dist, S-x.ER, S-x.EL))
    return ytilde
end

function Estep(rng::AbstractRNG, dist::ContinuousUnivariateDistribution, x::DICRT)
    u = rand(rng)
    S = x.SL + (x.SR-x.SL)*u
    rand!(rng, u)
    E = x.EL + (x.ER-x.EL)*u
    q = cdf(dist, x.TR - E)
    yb = []
    if zero(q) < q < one(q)
        B = rand(rng, Geometric(q))
        if B > 0
            push!(yb,[rand(rng, truncated(dist,Tmax-E,Inf)) for i in UnitRange(1,B)])
        end
    end
    ytilde = rand(rng, truncated(dist, S-x.ER, S-x.EL))
    return ytilde
end

####

function MCEM(rng::AbstractRNG, dist::ContinuousUnivariateDistribution, x::Vector{Any}, iter::Int)
    @assert support(dist) == RealInterval{AbstractFloat}(0.0, Inf)
    lp = zeros(iter)
    pars = params(dist)
    for it in UnitRange(1,iter)
    ytilde = reduce(vcat, [Estep(rng, dist, x[i]) for i in eachindex(x)])
    dist = Mstep(dist, ytilde)
    lp[it] = sum(x-> -logpdf(dist, x), ytilde)
    end
    return dist,lp 
end
