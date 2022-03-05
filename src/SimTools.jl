#####
#inteval censored
#####
function make_ic(rng::AbstractRNG, dt::UnivariateDistribution, N::Int)
    ER = zeros(N)
    EL = zeros(N)
    S = zeros(N)
    at = 0.0
    for i in 1:N
        ue = rand(rng)
        at -= log(rand(rng))
        tau = -log(rand(rng))
        y = rand(rng, dt)
        S[i] = at + y
        ud = rand(rng)
        EL[i] = max(0.0, at - tau * (1.0-ue))
        ER[i] = min(S[i], at + tau * ue)
    end
    return EL, ER, S
end

#inteval censored (infinite)
function make_ic(rng::AbstractRNG, dt::UnivariateDistribution, N::Int, p)
    ER = zeros(N)
    EL = zeros(N)
    S = zeros(N)
    at = 0.0
    for i in 1:N
        ue = rand(rng)
        at -= log(rand(rng))
        tau = -log(rand(rng))
        y = rand(rng,dt)
        S[i] = at + y
        ud = rand(rng)
        if ud <= p
            EL[i] = max(0.0, at - tau * (1.0-ue))
        else 
            EL[i] = -Inf
        end
        ER[i] = min(S[i], at + tau * ue)
    end
    return EL, ER, S
end

#inteval censored with right truncated
function make_icrt(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, Tmax)
    L = zeros(N)
    R = zeros(N)
    S = zeros(N)
    d = trues(N)
    for i in 1:N
        ue = rand(rng)
        y = rand(rng,dt)
        at = rand(rng) * Tmax
        tau = -log(rand(rng))
        y = rand(rng,dt)
        S[i] = at + y
        L[i] = max(0.0, at - tau * (1.0-ue))
        R[i] = min(S[i], at + tau * ue)
        d[i] = (S[i] <= Tmax)
        d[i] = (S[i] <= Tmax)
    end
    return L[d], R[d], S[d]
end

#inteval censored (infinite) with right truncated
function make_icrt(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, Tmax, p)
    L = zeros(N)
    R = zeros(N)
    S = zeros(N)
    d = trues(N)
    for i in 1:N
        ue = rand(rng)
        y = rand(rng,dt)
        at = rand(rng) * Tmax
        tau = -log(rand(rng))
        ud = rand(rng)
        if ud <= p
            L[i] = max(0.0, at - tau * (1.0-ue))
        else 
            L[i] = -Inf
        end
        R[i] = min(S[i], at + tau * ue)
        d[i] = (S[i] <= Tmax)
    end
    return L[d], R[d], S[d]
end

#####
#doubly inteval censored
#####
function make_dic(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int)
    EL = zeros(N)
    ER = zeros(N)
    SL = zeros(N)
    SR = zeros(N)
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
        EL[i] = max(0.0, at - tau_e * (1.0 - ue))
        SR[i] = S + tau_s * us
        SL[i] = max(EL[i], S - tau_s * (1.0 - us))
        ER[i] = min(SR[i], at + tau_e * ue)
    end
    return EL, ER, SL, SR
end

#infinite
function make_dic(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, p)
    EL = zeros(N)
    ER = zeros(N)
    SL = zeros(N)
    SR = zeros(N)
    at = 0.0
    for i in 1:N
        at -= log(rand(rng))
        y = rand(rng,dt)
        S = y + at
        tau_e = -log(rand(rng))
        tau_s = -log(rand(rng))
        ue = rand(rng)
        ud = rand(rng)
        if ud <= p
            EL[i] = max(0.0, at - tau_e * (1.0 - ue))
        else
            EL[i] = -Inf
        end
        us = rand(rng)
        SR[i] = S + tau_s * us
        SL[i] = max(EL[i], S - tau_s * (1.0 - us))
        ER[i] = min(SR[i], at + tau_e * ue)
    end
    return EL, ER, SL, SR
end

#right truncated
function make_dicrt(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, Tmax)
    EL = zeros(N)
    ER = zeros(N)
    SL = zeros(N)
    SR = zeros(N)
    d = trues(N)
    for i in 1:N
        ue = rand(rng)
        y = rand(rng,dt)
        at = rand(rng) * Tmax
        S = y + at
        tau_e = -log(rand(rng))
        tau_s = -log(rand(rng))
        ue = rand(rng)
        ud = rand(rng)
        if ud <= p
            EL[i] = max(0.0, at - tau_e * (1d - ue))
        else
            EL[i] = -Inf
        end
        us = rand(rng)
        SR[i] = S + tau_s * us
        SL[i] = max(EL[i], S - tau_s * (1 - us))
        ER[i] = min(SR[i], at + tau_e * ue)
        d[i] = SR <= Tmax
    end
    return EL[d], ER[d], SL[d], SR[d]
end

#doubly inteval censored (infinite) with right truncated
function make_dicrt(rng::AbstractRNG, dt::ContinuousUnivariateDistribution, N::Int, Tmax, p)
    EL = zeros(N)
    ER = zeros(N)
    SL = zeros(N)
    SR = zeros(N)
    d = trues(N)
    for i in 1:N
        ue = rand(rng)
        y = rand(rng,dt)
        at = rand(rng) * Tmax
        S = y + at
        tau_e = -log(rand(rng))
        tau_s = -log(rand(rng))
        ue = rand(rng)
        ud = rand(rng)
        if ud <= p
            EL[i] = max(0.0, at - tau_e * (1.0 - ue))
        else
            EL[i] = -Inf
        end
        us = rand(rng)
        SR[i] = S + tau_s * us
        SL[i] = max(EL[i], S - tau_s * (1.0 - us))
        ER[i] = min(SR[i], at + tau_e * ue)
        d[i] = SR <= Tmax
    end
    return EL[d], ER[d], SL[d], SR[d]
end
