function d_up(p,j1,j2,n,m)
    dj = zeros(m)
    for i in 1:n
        if j1[i]>0 && j2[i]>0
            q = p[j1[i]:j2[i]]
            q ./= sum(q)
            dj[j1[i]:j2[i]] += q
        end
    end
    return dj
end

function setbreaks(trow, o=0)
    tj = sort(unique(max.(o, trow)))
    return tj
end

function p_up!(p,n,m,j1,j2,dj)
    bj = zeros(m)
    for i in 1:n
        if j1[i]>0 && j2[i]>0
        q = sum(p[j1[i]:j2[i]])
        if q > 0.0
        b = (1-q)/q
        ind = setdiff(1:m,j1[i]:j2[i])
        r = vec(p[ind])
        r ./= sum(r)
        bj[ind] += b*r
    end
    end
    end
    num = dj + bj
    copy!(p, num ./ sum(num))
end

function acount(L, R, breaks, n, m)
    afirst = zeros(Int,n)
    alast = zeros(Int,n)
    for i in 1:n
        alpha = L[i] .<= breaks[1:(m-1)] .&& breaks[2:m] .<= R[i]
        if any(alpha)
            afirst[i] = findfirst(alpha)
            alast[i] =  findlast(alpha)
        end
    end
    return afirst, alast
end

function acount(x, breaks, m)
    L = x.EL
    R = x.ER
    S = ES(x)
    if isfinite(L)
        alpha = breaks[2:m] .<= (S - R)
        afirst = findfirst(alpha)
        alast =  findlast(alpha)
        return afirst, alast
    else
        alpha = (S - L) .<= breaks[1:(m-1)] .&& breaks[2:m] .<= (S - R)
        afirst = findfirst(alpha)
        alast =  findlast(alpha)
        return afirst, alast
    end
end

function bcount(U, breaks, n, m)
    bfirst = zeros(Int,n)
    blast = zeros(Int,n)
    for i in 1:n
        beta = breaks .<= U[i]
        if any(beta)
            bfirst[i] = findfirst(beta)
            blast[i] =  findlast(beta)
        end
    end
    return bfirst, blast
end

function ES(x::DIC, midp)
    S=midp*(x.SL+x.SR)
    return S
end

function ES(x::DICRT, midp)
    S=midp*(x.SL+x.SR)
    return S
end

function ES(x::IC, midp)
    return x.S
end

function ES(x::ICRT, midp)
    return x.S
end

function truncpoint(x::ICRT)
    return x.TR - x.S
end

function truncpoint(x::DICRT)
    return x.TR - x.SR
end

function truncpoint(x::IC)
    return Inf
end

function truncpoint(x::DIC)
    return Inf
end

function eccdfEM(y, midp = 0.5, iter = 100, tol=1e-4)
    n = length(y)
    S0 = zeros(n)
    L = Vector{Union{Float64, Missing}}(undef, n)
    R = zeros(n)
    TP = zeros(n)
    for i in 1:n
        L[i] = y[i].EL
        R[i] = y[i].ER
        S0[i] = ES(y[i], midp)
        TP[i] = truncpoint(y[i])
    end
    tj = setbreaks([S0-L;S0-R])
    m = length(tj)
    aind = acount(S0-R, S0-L,tj,n,m)
    p = inv(m)*ones(m)
    bind = bcount(TP,tj,n,m)
    dj = d_up(p,aind[1],aind[2],n,m)
    p_up!(p, n, m, bind[1], bind[2],dj)
    con = false
    count = 0
    p2 = copy(p)
    for it in 1:iter
        count += 1
        dj = d_up(p, aind[1], aind[2], n, m)
        p_up!(p, n, m, bind[1], bind[2], dj)
        con = all(abs.(p2-p) .< tol)
        copy!(p, p2)
    if con || any(isnan.(p))
        break
    end
    end
    return tj, 1 .- cumsum(p), con, count
end

########
#following functions are deprecated
#

function SurvICm(Y, iter=100, tol=1e-6)
    n = size(Y, 1)
    tj = setbreaks([Y.S[i]-Y.R[i] for i in eachindex(Y)])
    m = length(tj)
    aind1 = zeros(Int, n)
    aind2 = zeros(Int, n)
    for i in eachindex(R)
        aind[1], aind[2] = acount(Y[i], tj, m)
    end
    p = inv(m)*ones(m)
    q = cumsum(p)
    Delta = diff(tj)
    den = sum(Delta)
    mu = (m-1) - sum(q[i]*Delta[i] for i in 1:(m-1))
    dj = d_up(p, aind1, aind2, n, m)
    copy!(p, dj*mu ./ (den*sum(dj)))
    con = false
    count = 0
    p2 = copy(p)
    for it in 1:iter
        count += 1
        dj = d_up(p, aind1, aind2, n, m)
        copy!(p2, dj ./ sum(dj))
        con = all(abs.(p2-p) .< tol)
        copy!(p, p2)
    if con || any(isnan.(p))
        break
    end
    end
    return tj, 1 .- cumsum(p), con, count
end

function SurvIC(L, R, S, iter=100, tol=1e-8)
    tj = setbreaks([S-L;S-R])
    n = length(L)
    m = length(tj)
    aind = acount(S-R, S-L,tj,n,m)
    p = inv(m)*ones(m)
    dj = d_up(p, aind[1], aind[2], n, m)
    copy!(p, dj ./ sum(dj))
    con = false
    count = 0
    p2 = copy(p) #こういうとこcopy!()とか使ったほうがいいですか？
    for it in 1:iter
        count += 1
        dj = d_up(p,aind[1],aind[2],n,m)
        copy!(p2, dj ./ sum(dj))
        con = all(abs.(p2-p) .< tol)
        copy!(p, p2)
    if con || any(isnan.(p))
        break
    end
    end
    return tj, 1.0 .- cumsum(p), con, count
end


function SurvICRT(L, R, S, Tmax, iter=100, tol=1e-10)
    tj = setbreaks([S-L;S-R])
    n = length(L)
    m = length(tj)
    aind = acount(S-R, S-L,tj,n,m)
    p = inv(m)*ones(m)
    bind = bcount(Tmax.-S,tj,n,m)
    dj = d_up(p,aind[1],aind[2],n,m)
    p_up!(p, n, m, bind[1], bind[2], dj)
    con = false
    count = 0
    p2 = p #こういうとこcopy!()とか使ったほうがいいですか？
    for it in 1:iter
        count += 1
        dj = d_up(p,aind[1],aind[2],n,m)
        p_up!(p, n, m, bind[1], bind[2], dj)
        con = all(abs.(p2-p) .< tol)
        p = p2
    if con || any(isnan.(p))
        break
    end
    end
    return tj, 1.0 .- cumsum(p), con, count
end

function SurvDIC(EL, ER, SL, SR, midp = 0.5, iter = 100, tol = 1e-8)
    S = midp * (SL+SR)
    tj = setbreaks([S-EL;S-ER])
    n = length(EL)
    m = length(tj)
    aind = acount(S-ER, S-EL,tj,n,m)
    p = inv(m)*ones(m)
    dj = d_up(p,aind[1],aind[2],n,m)
    copy!(p, dj ./ sum(dj))
    con = false
    count = 0
    p2 = p #こういうとこcopy!()とか使ったほうがいいですか？
    for it in 1:iter
        count += 1
        dj = d_up(p,aind[1],aind[2],n,m)
        copy!(p2, dj ./ sum(dj))
        con = all(abs.(p2-p) .< tol)
        p = p2
    if con || any(isnan.(p))
        break
    end
    end
    return tj, 1.0 .- cumsum(p), con, count
end

function SurvDICRT(EL, ER, SL, SR, Tmax, midp = 0.5, iter=100, tol=1e-8)
    S =  midp *(SL+SR)
    tj = setbreaks([S-EL;S-ER])
    n = length(EL)
    m = length(tj)
    aind = acount(S-ER, S-EL,tj,n,m)
    p = inv(m)*ones(m)
    bind = bcount(Tmax.-SR,tj,n,m)
    dj = d_up(p,aind[1],aind[2],n,m)
    p_up!(p, n, m, bind[1], bind[2],dj)
    con = false
    count = 0
    p2 = p #こういうとこcopy!()とか使ったほうがいいですか？
    for it in 1:iter
        count += 1
        dj = d_up(p,aind[1],aind[2],n,m)
        p_up!(p, n, m, bind[1], bind[2], dj)
        con = all(abs.(p2-p) .< tol)
        p = p2
    if con || any(isnan.(p))
        break
    end
    end
    return tj, 1.0 .- cumsum(p), con, count
end
