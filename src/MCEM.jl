function Mstep(d::LogNormal, ytilde)
    mu = mean(log, ytilde)
    sigma = sqrt(mean(x -> abs2(log(x)-mu), ytilde))
    return LogNormal(mu, sigma)
end

function MCEMfic(rng, dist, iter, EL, ER, S, np=1)
    lp = zeros(iter)
    pars = params(dist)
    for it in 1:iter
    ytilde = [rand(rng, truncated(dist,S[i]-ER[i],S[i]-EL[i])) for i in eachindex(S), j in 1:np]
    dist = Mstep(dist, ytilde)
    lp[it] = mean(x->logpdf(dist,x),ytilde)
    end
    return dist,lp
end
