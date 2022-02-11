
function make_ic(rng::AbstractRNG, y::Vector, at::Vector, tau::Vector)
    n = length(y)
    E_R = zeros(n)
    E_L = zeros(n)
    S = zeros(n)
    for i in 1:n
        ue = rand(rng)
        E_R[i] = at[i] + tau[i] * ue
        E_L[i] = max(0.0, at[i] - tau[i] * (1.0 - ue))
        S[i] = y[i] + at[i]
    end
    return E_L, E_R, S
end

function make_ic(rng::AbstractRNG, y::Number, at::Number, tau::Number)
    ue = rand(rng)
    E_R = at + tau * ue
    E_L = max(0.0, at - tau * (1.0 - ue))
    S = y + at
    return E_L, E_R, S
end

function make_icrt(rng::AbstractRNG, dist::UnivariateDistribution, Tmax,  N)
    L = zeros(N)
    R = zeros(N)
    S = zeros(N)
    for i in 1:N
        y = rand(rng,dist)
        at = rand(rng, Uniform(0.0, Tmax))
        tau = rand(rng, Exponential(1.0))
        L[i], R[i], S[i] = make_ic(rng,y,at,tau)
    end
    d = S .<= Tmax
    return L[d], R[d], S[d]
end


function make_dic(rng::AbstractRNG, y, at, tau_e, tau_s)
    S = y + at
    ue = rand(rng)
    E_R = at + tau_e * ue
    E_L = max(0.0, at - tau_e * (1.0 - ue))
    us = rand(rng)
    S_R = S + tau_s * us
    S_L = max(0.0, S - tau_s * (1.0 - us))
    S = y + at
    return E_L, E_R, S_L, S_R
end

function make_dicrt(rng::AbstractRNG, dist::UnivariateDistribution, Tmax,  N)
    EL = zeros(N)
    ER = zeros(N)
    SL = zeros(N)
    SR = zeros(N)
    for i in 1:N
        y = rand(rng, dist)
        at = rand(rng, Uniform(0.0, Tmax))
        tau_e = rand(rng, Exponential(1.0))
        tau_s = rand(rng, Exponential(1.0))
        EL[i], ER[i], SL[i], SR[i] = make_dic(rng, y, at, tau_e, tau_s)
    end
    d = SR .<= Tmax
    return EL[d], ER[d], SL[d], SR[d]
end
