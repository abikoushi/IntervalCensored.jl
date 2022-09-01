module IntervalCensored

using Distributions
using Random
using SpecialFunctions
using HypergeometricFunctions
using ForwardDiff
using StaticArrays
using StatsFuns
import LinearAlgebra: dot
import Distributions: ccdf, cdf, logpdf, pdf, quantile, mean, rand, params, shape, scale
import Distributions: @distr_support

abstract type IntervalData end

struct IC{TEL, TER, TS} <: IntervalData
    EL::Union{TEL, Missing}
    ER::TER
    S::TS
end

struct ICRT{TEL, TER, TS, TTR} <: IntervalData
    EL::Union{TEL, Missing}
    ER::TER
    S::TS
    TR::TTR
end

struct ICT{TEL, TER, TS, TTL, TTR} <: IntervalData
    EL::Union{TEL, Missing}
    ER::TER
    S::TS
    TL::TTL
    TR::TTR
end

struct DIC{TEL, TER, TSL, TSR} <: IntervalData
    EL::Union{TEL, Missing}
    ER::TER
    SL::TSL
    SR::TSR
end

struct DICRT{TEL, TER, TSL, TSR, TTR} <: IntervalData
    EL::Union{TEL, Missing}
    ER::TER
    SL::TSL
    SR::TSR
    TR::TTR
end

struct DICT{TEL, TER, TSL, TSR, TTL, TTR} <: IntervalData
    EL::Union{TEL, Missing}
    ER::TER
    SL::TSL
    SR::TSR
    TL::TTL
    TR::TTR
end

function setDICdata(EL, ER, SL, SR)
    out = Vector{IntervalData}(undef, length(EL))
    for i in eachindex(EL)
        if SL[i] == SR[i]
            out[i] = IC(EL[i], ER[i], SL[i])
        else
            out[i] = DIC(EL[i], ER[i], SL[i], SR[i])
        end
    end
    return out
end


include("NonParametric.jl")
include("SimTools.jl")
include("calclp.jl")
include("./survdist/survdist.jl")
include("./survdist/GeneralizedGamma.jl")
include("./survdist/LogLogistic.jl")
include("MCEM.jl")

export calclp, IntervalData, IC, ICRT, DIC, DICRT
export makeIC, makeICRT, makeDIC, makeDICRT
export eccdfEM
export eqcdf, cdf, ccdf, pdf, logpdf, mean, shape, scale, params, rand
export MCEM
export GeneralizedGamma, LogLogistic

end
