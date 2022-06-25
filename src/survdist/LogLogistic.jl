"""
    LogLogistic(a,b)

The *log-logistic distribution* with shape `a` and scale `b` has probability density function

```math
f(x; a, b)=\\frac{(a / b )(x/b)^{a-1}}{1+(x/b)^a)^2}}
```

```julia
LogLogistic()       # LogLogistic distribution with zero location and unit scale, i.e. LogLogistic(0, 1)
LogLogistic(a)      # LogLogistic distribution with location a and unit scale, i.e. LogLogistic(a, 1)
LogLogistic(a, b)   # LogLogistic distribution with location a and scale b

params(d)       # Get the parameters, i.e. (a, b)
shape(d)     # Get the location parameter, i.e. a
scale(d)        # Get the scale parameter, i.e. b
```

External links

* [Log-Logistic distribution on Wikipedia](https://en.wikipedia.org/wiki/Log-logistic_distribution)

"""
struct LogLogistic(Ta, Tb) <: ContinuousUnivariateDistribution
    a::Ta
    b::Tb
end

#### Parameters

shape(d::LogLogistic) = d.a
scale(d::LogLogistic) = d.b

params(d::LogLogistic) = (d.a, d.b)
#@inline partype(d::LogLogistic{T}) where {T<:Real} = T


#### Statistics

function mean(d::LogLogistic)
    a, b = params(d)
    if a>1
        return (b*pi/a)/sin(pi/a)
    else
        return missing
    end
end
median(d::Logistic) = d.b
#mode(d::Logistic) = d.b
#	{\displaystyle \alpha \left({\frac {\beta -1}{\beta +1}}\right)^{1/\beta }}\alpha\left(\frac{\beta-1}{\beta+1}\right)^{1/\beta}

# std(d::Logistic) = π * d.θ / sqrt3
# var(d::Logistic) = (π * d.θ)^2 / 3
# skewness(d::Logistic{T}) where {T<:Real} = zero(T)
# kurtosis(d::Logistic{T}) where {T<:Real} = T(6)/5

# entropy(d::Logistic) = log(d.θ) + 2


#### Evaluation

# zval(d::Logistic, x::Real) = (x - d.μ) / d.θ
# xval(d::Logistic, z::Real) = d.μ + z * d.θ

function pdf(d::LogLogistic, x)
    a, b = params(d)
    return ((a/b)*(x/b )^(a-1))/((1+(x/b)^a))^2
end

function logpdf(d::LogLogistic, x)
    a, b = params(d)
    return log(a)-log(b)+(a-1)*log(x/b) - 2*log1p((x/b)^a)
end

function cdf(d::LogLogistic, x)
    a, b = params(d)
    return (x^a) / (b^a +x^a)
end

function ccdf(d::LogLogistic, x)
    a, b = params(d)
    return one(x) - (x^a) / (b^a +x^a)
end


# logcdf(d::Logistic, x::Real) = -log1pexp(-zval(d, x))
# logccdf(d::Logistic, x::Real) = -log1pexp(zval(d, x))
function quantile(d::LogLogistic,p)
    a, b = params(d)
    return b * ((p)/(1-p))^inv(a)
end

function rand(rng::AbstractRNG, d::LogLogistic)
    p = rand(rng)
    return quantile(d, p)
end

# quantile(d::Logistic, p::Real) = xval(d, logit(p))
# cquantile(d::Logistic, p::Real) = xval(d, -logit(p))
# invlogcdf(d::Logistic, lp::Real) = xval(d, -logexpm1(-lp))
# invlogccdf(d::Logistic, lp::Real) = xval(d, logexpm1(-lp))

# function gradlogpdf(d::Logistic, x::Real)
#     e = exp(-zval(d, x))
#     ((2e) / (1 + e) - 1) / d.θ
# end

# mgf(d::Logistic, t::Real) = exp(t * d.μ) / sinc(d.θ * t)

# function cf(d::Logistic, t::Real)
#     a = (π * t) * d.θ
#     a == zero(a) ? complex(one(a)) : cis(t * d.μ) * (a / sinh(a))
# end
