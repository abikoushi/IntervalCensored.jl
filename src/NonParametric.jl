function d_up(p,j1,j2,n,m)
    dj = zeros(m)
    for i in 1:n
        q = p[j1[i]:j2[i]]
        q ./= sum(q)
        dj[j1[i]:j2[i]] += q
    end
    return dj
end

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

function setbreaks(trow, O=0.0)
    tj = sort(unique(max.(O, trow)))
    return tj
end

function p_up!(p,n,m,j1,j2,dj)
    bj = zeros(m)
    for i in 1:n
        if j1[i]>0 && j2[i]>0
        q = sum(p[j1[i]:j2[i]])
        if q > 0.0
        b = (1.0-q)/q
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

function acount(L,R,breaks,n,m)
    afirst = zeros(Int,n)
    alast = zeros(Int,n)
    for i in 1:n
        alpha = L[i] .<= breaks[1:(m-1)] .&& breaks[2:m] .<= R[i]
        afirst[i] = findfirst(alpha)
        alast[i] =  findlast(alpha)
    end
    return afirst, alast
end


function bcount(U,breaks,n,m)
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

function SurvICRT(L, R, S, Tmax, iter=100, tol=1e-10)
    tj = setbreaks([S-L;S-R])
    n = length(L)
    m = length(tj)
    aind = acount(S-R, S-L,tj,n,m)
    p = (1.0/m)*ones(m)
    bind = bcount(Tmax.-S,tj,n,m)
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

function SurvDICRT(EL, ER, SL, SR, Tmax, iter=100, tol=1e-8)
    S = 0.5*(SL+SR)
    tj = setbreaks([S-EL;S-ER])
    n = length(EL)
    m = length(tj)
    aind = acount(S-ER, S-EL,tj,n,m)
    p = (1.0/m)*ones(m)
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
