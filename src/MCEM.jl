function Mstep(d::LogNormal, ytilde)
    mu = mean(log, ytilde)
    sigma = sqrt(mean(x -> abs2(log(x)-mu), ytilde))
    return LogNormal(mu, sigma)
end

function Mstep(d::Gamma, ytilde)
    mlog = log(mean(ytilde))
    logm = mean(log, ytilde)
    a = shape(d)
    ia = inv(a)
    a = inv(ia + (-mlog + logm + log(a) - digamma(a)) / ((a^2) * (ia - trigamma(a))))
    b = mean(ytilde) / a
    return Gamma(a,b)
end

function Mstep(d::Weibull, ytilde)
    a = shape(d)
    logy = mean(log, ytilde, dims=2)
    powa = mean(x->x^a, ytilde, dims=2)
    mlog = mean(logy)
    spow = sum(powa)
    dot_powlog = dot(powa, logy)

    fx = dot_powlog/spow - mlog - inv(a)
    ∂fx = (-dot_powlog^2 + spow * dot(logy.^2, powa)) / (spow^2) + inv(a^2)

    Δa = fx / ∂fx
    a -= Δa
    b = mean(x -> x^a, ytilde)^inv(a)
    return Weibull(a,b)
end

function Mstep(d::Exponential, ytilde)
    b = mean(ytilde)
    return Exponential(b)
end

function MCEMic(rng, dist, iter, EL, ER, S, np=1)
    lp = zeros(iter)
    pars = params(dist)
    for it in 1:iter
    ytilde = [rand(rng, truncated(dist,S[i]-ER[i],S[i]-EL[i])) for i in eachindex(S), j in 1:np]
    dist = Mstep(dist, ytilde)
    lp[it] = mean(x-> -logpdf(dist,x), ytilde)
    end
    return dist,lp
end

function Estep_icrt(rng, dist, EL, ER, S, Tmax, np)
    N = length(S)
    ys = zeros(N, np)
    yb = zeros(0, np)
    for i in 1:N
        for j in 1:np
            ys[i,j] = rand(rng, truncated(dist,S[i]-ER[i],S[i]-EL[i]))
        end
        E = rand(rng, Uniform(EL[i], ER[i]))
        q = cdf(dist, Tmax - E)
        if 0.0 < q < 1.0
            B = rand(rng, Geometric(q))
        end
        yb = [yb; rand(rng, truncated(dist,Tmax-E,Inf)) j in 1:np]
    end
    return [ys ; yb] 
end

function MCEMicrt(rng, dist, iter, EL, ER, S, Tmax, np=1)
    lp = zeros(iter)
    pars = params(dist)
    for it in 1:iter
    ytilde = Estep_icrt(rng, dist, EL, ER, S, Tmax, np)
    dist = Mstep(dist, ytilde)
    lp[it] = mean(x-> -logpdf(dist,x), ytilde)
    end
    return dist,lp
end