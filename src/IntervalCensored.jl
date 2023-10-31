module IntervalCensored

using Distributions
using Random
using SpecialFunctions
using HypergeometricFunctions
using ForwardDiff
using StaticArrays
using StatsFuns
using LogExpFunctions
import LinearAlgebra: dot, UpperTriangular
import Distributions: ccdf, cdf, logpdf, pdf, quantile, mean, rand, params, shape, scale
import Distributions: @distr_support

abstract type IntervalData end

struct NC{TE, TS} <: IntervalData
    E::TE
    S::TS
end

struct ICE{TEL, TER, TS} <: IntervalData
    EL::TEL
    ER::TER
    S::TS
end

struct ICS{TE, TSL, TSR} <: IntervalData
    E::TE
    SL::TSL
    SR::TSR
end

struct DIC{TEL, TER, TSL, TSR} <: IntervalData
    EL::TEL
    ER::TER
    SL::TSL
    SR::TSR
end

include("NonParametric.jl")
include("SimTools.jl")
include("calclp.jl")
include("./survdist/survdist.jl")
include("./survdist/GeneralizedGamma.jl")
include("./survdist/LogLogistic.jl")
include("MCEM.jl")

export evaluatelp, IntervalData, NC, ICE, ICS, DIC
export makeIC, makeDIC
export jointecdfEM, ecdfEM,  colmarginal,  h2ccdf
export eqcdf, cdf, ccdf, pdf, logpdf, mean, shape, scale, params, rand
export MCEM
export GeneralizedGamma, LogLogistic

end
