function getE(x::NC)
    return x.E, x.E
end

function getE(x::ICS)
    return x.E, x.E
end

function getE(x::ICE)
    return x.EL, x.ER
end

function getE(x::DIC)
    return x.EL, x.ER
end

function getS(x::NC)
    return x.S, x.S
end

function getE(x::ICS)
    return x.SL, x.SR
end

function getS(x::ICE)
    return x.S, x.S
end

function getS(x::DIC)
    return x.SL, x.SR
end

function lp(A, h, lam)
    m = size(A,1)
    lp = 0.
    for e  in 1:(m-1)
      for s in (e+1):m
          lp += xlogy(A[e,s],h[s-e])
          lp += xlogy(A[e,s],lam[e])
      end
    end
  return lp
end

function lp(A, B, h, lam)
    lp = 0.
    m = size(A,1)
    for e in 1:m
      for s in (e+1):m
          lp += xlogy(A[e,s]+B[e,s],h[s-e])
          lp += xlogy(A[e,s]+B[e,s],lam[e])
      end
    end
    return lp
end

function Aup2(le_rank,re_rank,s_rank,h,lam)
    m = size(lam,1)
    A = UpperTriangular(zeros(m,m))
    a = h[(s_rank - re_rank+1):(s_rank - le_rank+1)] .* lam[le_rank:re_rank]
    A[le_rank:re_rank, s_rank] += a/sum(a)
    return A
end

function Aup(le_rank,re_rank,s_rank,h,lam)
    m = size(lam,1)
    A = UpperTriangular(zeros(m,m))
    for i in eachindex(s_rank)
      A += Aup2(le_rank[i],re_rank[i],s_rank[i],h,lam)
    end
    return A
  end

#double
function Aup3(le_rank,re_rank,ls_rank,rs_rank,h,lam)
    m = size(lam,1)
    A = UpperTriangular(zeros(m,m))
    a = zeros(m,m)
    for i in (le_rank):(re_rank)
      for j in (ls_rank):(rs_rank)
        if j>i
          a[i,j] += h[j-i] * lam[i]
        end
      end
    end
    a /= sum(a)
    return a
end

function Aup(le_rank,re_rank,ls_rank,rs_rank,h,lam)
    m = size(lam,1)
    A = UpperTriangular(zeros(m,m))
    for i in eachindex(rs_rank)
      A += Aup3(le_rank[i],re_rank[i],ls_rank[i],rs_rank[i],h,lam)
    end
    return A
end

#double
function Aup3(le_rank,re_rank,ls_rank,rs_rank,h)
    m = size(lam,1)
    A = UpperTriangular(zeros(m,m))
    a = zeros(m,m)
    for i in (le_rank):(re_rank)
      for j in (ls_rank):(rs_rank)
        if j>i
          a[i,j] += h[j-i]
        end
      end
    end
    a /= sum(a)
    return a
end

function Aup(le_rank,re_rank,ls_rank,rs_rank,h)
    m = size(lam,1)
    A = UpperTriangular(zeros(m,m))
    for i in eachindex(rs_rank)
      A += Aup3(le_rank[i],re_rank[i],ls_rank[i],rs_rank[i],h)
    end
    return A
end

function paramup(A)
    m = size(A,1)
    h = zeros(m)
    for e in 1:(m-1)
      for s in (e+1):m
        h[s-e] += A[e,s]
      end
    end
   h /= sum(h)
   return h
end

function paramup(A)
    m = size(A,1)
    h = zeros(m)
    lam = zeros(m)
    for e in 1:(m-1)
      lam[e] += sum(A[e,:])
      for s in (e+1):m
        h[s-e] += A[e,s]
      end
    end
    h /= sum(h)
    lam /= sum(lam)
    return h, lam
end


function bup(le_rank,re_rank,h,lam)
    m = size(lam,1)
    B = UpperTriangular(zeros(m,m))
    rho = zeros(2)
    #convolution
    for i in le_rank:re_rank
      rho[1] += lam[i]*sum(h[1:(m-i)])
      rho[2] += lam[i]*sum(h[(m-i+1):end])
    end
    rho /= sum(rho)
    #expectation
    for i in le_rank:re_rank
      for j in (i+1):m
        B[i,j] += h[j-i+1]*rho[2]/rho[1]
      end
    end
    return B
  end
  
  function Bup(le_rank,re_rank,h,lam)
    m = size(h,1)
    b = UpperTriangular(zeros(m,m))
    for i in eachindex(re_rank)
      b += bup(le_rank[i],re_rank[i],h,lam)
    end
    return b
  end

  function paramup(A,B)
    m = size(A,1)
    h = zeros(m)
    lam = zeros(m)
    for e in 1:(m-1)
      lam[e] += sum(A[e,:]) + sum(B[e,:])
      for s in (e+1):m
        h[s-e] += A[e,s]
        h[s-e] += B[e,s]
      end
    end
    h /= sum(h)
    lam /= sum(lam)
    return h, lam
  end

  function jointecdfEM(y,Tmax,iter)
    n = length(y)
    LE = zeros(n)
    RE = zeros(n)
    LS = zeros(n)
    RS = zeros(n)
    for i in 1:n
        LE[i], RE[i] = getE(y[i])
        LS[i], RS[i] = getS(y[i])
    end
    ti = sort(unique([LE;RE;LS;RS;Tmax]))
    le_rank = indexin(LE, ti)
    re_rank = indexin(RE, ti)
    ls_rank = indexin(LS, ti)
    rs_rank = indexin(RS, ti)
    m = length(ti)
    h = ones(m)
    h = h/sum(h)
    lam = ones(m)
    lam = lam/sum(lam)
    A = UpperTriangular(zeros(m,m))
    B = UpperTriangular(zeros(m,m))
    logprob = zeros(iter)
    for i in 1:iter
      A = Aup(le_rank, re_rank, ls_rank, rs_rank, h, lam)
      B = Bup(le_rank, re_rank, h, lam)
      h, lam = paramup(A,B)
      logprob[i] = lp(A, B, h, lam)
    end
    return ti, h, lam, A, B, logprob
  end


  function jointecdfEM(y,iter)
    n = length(y)
    LE = zeros(n)
    RE = zeros(n)
    LS = zeros(n)
    RS = zeros(n)
    for i in 1:n
        LE[i], RE[i] = getE(y[i])
        LS[i], RS[i] = getS(y[i])
    end
    ti = sort(unique([LE;RE;LS;RS]))
    le_rank = indexin(LE, ti)
    re_rank = indexin(RE, ti)
    ls_rank = indexin(LS, ti)
    rs_rank = indexin(RS, ti)
    m = length(ti)
    h = ones(m)
    h = h/sum(h)
    lam = ones(m)
    lam = lam/sum(lam)
    A = UpperTriangular(zeros(m,m))
    logprob = zeros(iter)
    for i in 1:iter
      A = Aup(le_rank, re_rank, ls_rank, rs_rank, h, lam)
      h, lam = paramup(A)
      logprob[i] = lp(A, h, lam)
    end
    return ti, h, lam, A, logprob
  end

  
function ecdfEM(y, iter, tol)
    n = length(y)
    LE = zeros(n)
    RE = zeros(n)
    LS = zeros(n)
    RS = zeros(n)
    for i in 1:n
        LE[i], RE[i] = getE(y[i])
        LS[i], RS[i] = getS(y[i])
    end
    ti = sort(unique([LE;RE;LS;RS]))
    le_rank = indexin(LE, ti)
    re_rank = indexin(RE, ti)
    ls_rank = indexin(LS, ti)
    rs_rank = indexin(RS, ti)
    m = length(ti)
    h = ones(m)
    h = h/sum(h)
    lam = ones(m)
    lam = lam/sum(lam)
    A = UpperTriangular(zeros(m,m))
    logprob = zeros(iter)
    for i in 1:iter
      A = Aup(le_rank, re_rank, ls_rank, rs_rank, h)
      h = paramup(A)
      logprob[i] = lp(A, h)
    end
    return ti, h, A, logprob
end

  
 colmarginal(x) = cumsum(vec(sum(x,dims=2)))
 h2ccdf(x) = reverse(cumsum(reverse(x)))
