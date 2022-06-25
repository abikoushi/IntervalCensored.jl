function gamma_cdf(a::Number, x::Number)
    return gamma_inc(a,x,0)[1]
end

function gamma_ccdf(a::Number, x::Number)
    return gamma_inc(a,x,0)[2]
end

#where {T} の意味わかっていない
function gamma_cdf(a::ForwardDiff.Dual{T}, x::Number) where {T}
    y = gamma_inc(ForwardDiff.value(a),x,0)[1]
    y_a = (log(x) - digamma(a))*y-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x)
    return ForwardDiff.Dual{T}(y,y_a)
end

function gamma_cdf(a::Number, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_inc(a,ForwardDiff.value(x),0)[1]
    y_x = exp(-x + (a-1)*log(x) - loggamma(a))
    return ForwardDiff.Dual{T}(y,y_x)
end

function gamma_ccdf(a::ForwardDiff.Dual{T}, x::Number) where {T}
    y = gamma_inc(ForwardDiff.value(a),x,0)[2]
    y_a = -((log(x) - digamma(a))*(1-y)-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x))
    return ForwardDiff.Dual{T}(y,y_a)
end

function gamma_ccdf(a::Number, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_inc(a,ForwardDiff.value(x),0)[2]
    y_x = -exp(-x + (a-1)*log(x) - loggamma(a))
    return ForwardDiff.Dual{T}(y,y_x)
end

function gamma_cdf(a::ForwardDiff.Dual{T}, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_cdf(ForwardDiff.value(a),ForwardDiff.value(x))
    y_a = (log(x) - digamma(a))*y-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x)
    y_x = exp(-x + (a-1)*log(x) - loggamma(a))
    pa = ForwardDiff.partials(a).values
    px = ForwardDiff.partials(x).values
    ga = y_a*pa[1] + y_x*px[1]
    gx = y_a*pa[2] + y_x*px[2]
    return ForwardDiff.Dual{T}(y,ga,gx)
end

function gamma_ccdf(a::ForwardDiff.Dual{T}, x::ForwardDiff.Dual{T}) where {T}
    y = gamma_ccdf(ForwardDiff.value(a),ForwardDiff.value(x))
    y_a = -((log(x) - digamma(a))*(1-y)-exp(a*log(x)-log(a)-loggamma(a+1))*pFq(SA[a,a], SA[a+1,a+1], -x))
    y_x = -exp(-x + (a-1)*log(x) - loggamma(a))
    pa = ForwardDiff.partials(a).values
    px = ForwardDiff.partials(x).values
    ga = y_a*pa[1] + y_x*px[1]
    gx = y_a*pa[2] + y_x*px[2]
    return ForwardDiff.Dual{T}(y,ga,gx)
end

function cdf(d::Gamma, x)
    shp, scl = params(d)
    return gamma_cdf(shp, x/scl)
end

function ccdf(d::Gamma, x)
    shp, scl = params(d)
    return gamma_ccdf(shp, x/scl)
end

#integral erfc((log(x) - a)/b) dx = x erfc((log(x) - a)/b) - e^(a + b^2/4) erf((2 a + b^2 - 2 log(x))/(2 b)) + 定数
function eqcdf(d::LogNormal, x)
    mu, sigma = params(d)
    return 0.5*(1+x*erfc((log(x) - mu)/(sigma*sqrt(2)))*exp(-(mu + (sigma^2)/2)) - erf((mu + sigma^2 - log(x))/(sigma*sqrt(2))))
end

function eqcdf(d::Weibull, x)
    out = zero(x)
    if x >= zero(x)
    shp, scl = params(d)
    out = gamma_cdf(inv(shp),(max(x, 0) / scl) ^ shp)
    end
    return out
end

function eqcdf(d::Gamma, x)
    out = zero(x)
    if x >= zero(x)
        shp, scl = params(d)
        out = gamma_cdf(shp+1, x/scl)+(x/scl)*gamma_ccdf(shp, x/scl)/shp
    end
    return out
end

function eqcdf(d::Exponential, x)
    out = zero(x)
    if x >= zero(x)
        b = rate(d)
        out = b * exp(-b * x)
    end
    return out
end

function  logmean(d::LogNormal)
    mu, sigma = params(d)
    return mu + (sigma^2)/2
end

function  logmean(d::Weibull)
    shp, scl = params(d)
    return log(scl) + loggamma(1+inv(shp))
end

function  logmean(d::Gamma)
    shp, scl= params(d)
    return log(shp) + log(scl)
end

function  logmean(d::Exponential)
    b = scale(d)
    return log(b)
end

function logeqpdf(d::UnivariateDistribution,x)
    log(ccdf(d,x))-logmean(d)
end

function eqpdf(d::UnivariateDistribution,x)
    ccdf(d,x)/mean(d)
end
