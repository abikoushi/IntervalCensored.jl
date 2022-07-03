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
    v -= lr*(H \ g)
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
    rho -= lr*(g/H)
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
    rho -= lr*(g/H)
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

#######

function Estep(rng::AbstractRNG, dist::UnivariateDistribution, x::IC)
    ytilde = rand(rng, truncated(dist, x.S-x.ER, x.S-x.EL))
    return ytilde
end

function Estep(rng::AbstractRNG, dist::UnivariateDistribution, x::ICRT)
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

function Estep(rng::AbstractRNG, dist::UnivariateDistribution, x::DIC)
    u = rand(rng)
    S = x.SL + (x.SR[i]-x.SL)*u
    ytilde = rand(rng, truncated(dist, S-x.ER, S-x.EL))
    return ytilde
end

function Estep(rng::AbstractRNG, dist::UnivariateDistribution, x::DICRT)
    u = rand(rng)
    S = x.SL + (x.SR[i]-x.SL)*u
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


function MCEM(rng::AbstractRNG, dist::UnivariateDistribution, x, iter)
    lp = zeros(iter)
    pars = params(dist)
    for it in UnitRange(1,iter)
    ytilde = reduce(vcat, [Estep(rng, dist, x[i]) for i in eachindex(x)])
    dist = Mstep(dist, ytilde)
    lp[it] = sum(x-> -logpdf(dist, x), ytilde)
    end
    return dist,lp 
end

####

#interval censored
function MCEMic(rng, dist, iter, EL, ER, S, lr=1)
    lp = zeros(iter)
    pars = params(dist)
    for it in 1:iter
    ytilde = [rand(rng, truncated(dist,S[i]-ER[i],S[i]-EL[i])) for i in eachindex(S)]
    dist = Mstep(dist, ytilde, lr)
    lp[it] = sum(x-> -logpdf(dist,x), ytilde)
    end
    return dist,lp
end

#interval censored with right truncated
function Estep_icrt(rng, dist, EL, ER, S, Tmax)
    N = length(S)
    ys = zeros(N)
    yb = []
    for i in 1:N
        ys[i] = rand(rng, truncated(dist,S[i]-ER[i],S[i]-EL[i]))
        u = rand(rng)
        E = EL[i] + (ER[i]-EL[i])*u
        q = cdf(dist, Tmax - E)
        if 0.0 < q < 1.0
            B = rand(rng, Geometric(q))
            if B > 0
                ysub = [rand(rng, truncated(dist,Tmax-E,Inf)) for i in 1:B]
                yb = [yb;ysub]
            end
        end
    end
    return [ys ; yb] 
end

function MCEMicrt(rng, dist, iter, EL, ER, S, Tmax, lr=1)
    lp = zeros(iter)
    pars = params(dist)
    for it in 1:iter
    ytilde = Estep_icrt(rng, dist, EL, ER, S, Tmax)
    dist = Mstep(dist, ytilde, lr)
    lp[it] = sum(x-> -logpdf(dist,x), ytilde)
    end
    return dist,lp
end

######
#doubly interval censored
# function Estep_dic(rng, dist, EL, ER, SL, SR)
#     N = length(EL)
#     ys = zeros(N)
#     for i in 1:N
#         u = rand(rng)
#         E = EL[i] + (ER[i]-EL[i])*u
#         ys[i] = rand(rng, truncated(dist, SL[i]-E, SR[i]-E))
#     end
#     return ys
# end

function Estep_dic(rng, dist, EL, ER, SL, SR)
    N = length(EL)
    ys = zeros(N)
    for i in 1:N
        u = rand(rng)
        S = SL[i] + (SR[i]-SL[i])*u
        ys[i] = rand(rng, truncated(dist, S-ER[i], S-EL[i]))
    end
    return ys
end


function MCEMdic(rng, dist, iter, EL, ER, SL, SR, lr=1)
    lp = zeros(iter)
    pars = params(dist)
    for it in 1:iter
    ytilde = Estep_dic(rng, dist, EL, ER, SL, SR)
    dist = Mstep(dist, ytilde, lr)
    lp[it] = sum(x-> -logpdf(dist, x), ytilde)
    end
    return dist,lp 
end
