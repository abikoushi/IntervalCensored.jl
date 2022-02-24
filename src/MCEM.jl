function Mstep(d::LogNormal, y)
    mu = mean(log, y)
    sigma = sqrt(mean(x -> (log(x)-mu)^2, y))
    return LogNormal(mu, sigma)
end

function Mstep(d::Gamma, y)
    mlog = log(mean(y))
    logm = mean(log, y)
    a = shape(d)
    ia = inv(a)
    a = inv(ia + (-mlog + logm + log(a) - digamma(a)) / ((a^2) * (ia - trigamma(a))))
    b = mean(ytilde) / a
    return Gamma(a,b)
end

function Mstep(d::Weibull, y)
    a, b = params(d)
    n = length(y)
    powa = (y.^a)'

    fa = n*(inv(a) - inv(b)) + sum(logy) - n*log(b) - powa*(log(y)-log(b))/(b^a)
    dfa = -n*inv(a^2) - (powa*(logy-log(b)).^2)/(b^a)

    Delta = fx / dfx
    a += Delta
    b = mean(x -> x^a, ytilde)^inv(a)
    return Weibull(a,b)
end

function Mstep(d::Exponential, y)
    b = mean(y)
    return Exponential(b)
end

#######
#interval censored
function MCEMic(rng, dist, iter, EL, ER, S)
    lp = zeros(iter)
    pars = params(dist)
    for it in 1:iter
    ytilde = [rand(rng, truncated(dist,S[i]-ER[i],S[i]-EL[i])) for i in eachindex(S)]
    dist = Mstep(dist, ytilde)
    lp[it] = mean(x-> -logpdf(dist,x), ytilde)
    end
    return dist,lp
end

########
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

######
#doubly interval censored
function Estep_dic(rng, dist, EL, ER, SL, SR)
    N = length(EL)
    ys = zeros(N)
    for i in 1:N
        u = rand(rng)
        E = EL[i] + (ER[i]-EL[i])*u
        ys[i] = rand(rng, truncated(dist, SL[i]-E, SR[i]-E))
    end
    return ys
end

function MCEMdic(rng, dist, iter, EL, ER, SL, SR, np=1)
    lp = zeros(iter)
    pars = params(dist)
    for it in 1:iter
    ytilde = Estep_dic(rng, dist, EL, ER, SL, SR, np)
    dist = Mstep(dist, ytilde)
    lp[it] = mean(x-> -logpdf(dist, x), ytilde)
    end
end
