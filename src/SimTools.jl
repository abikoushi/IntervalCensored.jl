#####
#inteval censored
#####
function makeIC(rng::AbstractRNG, dt::UnivariateDistribution, N::Int)
    at = 0.0
    X = []
    for i in 1:N
        ue = rand(rng)
        at -= log(rand(rng))
        tau = -log(rand(rng))
        y = rand(rng, dt)
        S = at + y
        ud = rand(rng)
        EL = max(0.0, at - tau * (1.0 - ue))
        ER = min(S, at + tau * ue)
        push!(X, IC(EL,ER,S))
    end
    return X
end

#inteval censored (infinite)
function makeIC(rng::AbstractRNG, dt::UnivariateDistribution, N::Int, p)
    at = 0.0
    X = []
    for i in 1:N
        ue = rand(rng)
        at -= log(rand(rng))
        tau = -log(rand(rng))
        y = rand(rng,dt)
        S = at + y
        ud = rand(rng)
        EL = -Inf
        if ud <= p
            EL = max(0.0, at - tau * (1.0 - ue))
        end
        ER = min(S, at + tau * ue)
        push!(X, IC(EL,ER,S))
    end
    return X
end

#inteval censored with right truncated
function makeICRT(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, Tmax)
    X = []
    for i in 1:N
        ue = rand(rng)
        y = rand(rng,dt)
        at = rand(rng) * Tmax
        tau = -log(rand(rng))
        y = rand(rng,dt)
        S = at + y
        EL = max(0.0, at - tau * (1.0 - ue))
        ER = min(S, at + tau * ue)
        if S <= Tmax
            push!(X, ICRT(EL, ER, S, Tmax))
        end
    end
    return X
end

#inteval censored (infinite) with right truncated
function makeICRT(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, Tmax, p)
    X = []
    for i in 1:N
        ue = rand(rng)
        y = rand(rng,dt)
        at = rand(rng) * Tmax
        S = at + y
        tau = -log(rand(rng))
        ud = rand(rng)
        EL = - Inf
        if ud <= p
            EL = max(0.0, at - tau * (1.0 - ue))
        end
        ER = min(S, at + tau * ue)
        if S <= Tmax
            push!(X, ICRT(EL, ER, S, Tmax))
        end
    end
    return X
end

#####
#doubly inteval censored
#####
function makeDIC(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int)
    X=[]
    at = 0.0
    for i in 1:N
        ue = rand(rng)
        y = rand(rng, dt)
        at -= log(rand(rng))
        S = y + at
        ue = rand(rng)
        us = rand(rng)
        tau_e = -log(rand(rng))
        tau_s = -log(rand(rng))
        EL = max(0.0, at - tau_e * (1.0 - ue))
        SR = S + tau_s * us
        SL = max(EL, S - tau_s * (1.0 - us))
        ER = min(SR, at + tau_e * ue)
        push!(X, DIC(EL, ER, SL, SR))
    end
    return X
end

#infinite
function makeDIC(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, p)
    X=[]
    at = 0.0
    for i in 1:N
        at -= log(rand(rng))
        y = rand(rng,dt)
        S = y + at
        tau_e = -log(rand(rng))
        tau_s = -log(rand(rng))
        ue = rand(rng)
        ud = rand(rng)
        EL = -Inf
        if ud <= p
            EL = max(0.0, at - tau_e * (1.0 - ue))
        end
        us = rand(rng)
        SR = S + tau_s * us
        SL = max(EL, S - tau_s * (1.0 - us))
        ER = min(SR, at + tau_e * ue)
        push!(X, DIC(EL, ER, SL, SR))
    end
    return X
end

#right truncated (finite)
function makeDICRT(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, Tmax)
    X=[]
    for i in 1:N
        ue = rand(rng)
        y = rand(rng,dt)
        at = rand(rng) * Tmax
        S = y + at
        tau_e = -log(rand(rng))
        tau_s = -log(rand(rng))
        ue = rand(rng)
        ud = rand(rng)
        EL = -Inf
        if ud <= p
            EL = max(0.0, at - tau_e * (1.0 - ue))
        end
        us = rand(rng)
        SR = S + tau_s * us
        SL = max(EL, S - tau_s * (1.0 - us))
        ER = min(SR, at + tau_e * ue)
        if SR <= Tmax
            push!(X, DICRT(EL, ER, SL, SR, Tmax))
        end
    end
    return X
end

#doubly inteval censored (infinite) with right truncated
function makeDICRT(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, Tmax, p)
    X=[]
    for i in 1:N
        ue = rand(rng)
        y = rand(rng,dt)
        at = rand(rng) * Tmax
        S = y + at
        tau_e = -log(rand(rng))
        tau_s = -log(rand(rng))
        ue = rand(rng)
        ud = rand(rng)
        EL = -Inf
        if ud <= p
            EL = max(0.0, at - tau_e * (1.0 - ue))
        end
        us = rand(rng)
        SR = S + tau_s * us
        SL = max(EL, S - tau_s * (1.0 - us))
        ER = min(SR, at + tau_e * ue)
        if SR <= Tmax
            push!(X, DICRT(EL, ER, SL, SR, Tmax))
        end
    end
    return X
end
