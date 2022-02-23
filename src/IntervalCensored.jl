module IntervalCensored

using Distributions
using Random
using SpecialFunctions
using HypergeometricFunctions
using ForwardDiff
using StaticArrays
using StatsFuns
import LinearAlgebra: dot
import Base: rand

include("survdist.jl")
include("NonParametric.jl")
include("SimTools.jl")
include("calclp.jl")
include("MCEM.jl")
# include("./dist/GeneralizedGamma.jl")
# include("./dist/LogLogistic.jl")

export calclp_ic, calclp_icrt, calclp_dic, 
 make_ic, make_icrt, make_dic, make_dicrt,
 SurvIC, SurvICRT, SurvDIC, SurvDICRT,
 eqcdf,
 MCEMic, MCEMicrt, MCEMdic
 #GeneralizedGamma, LogLogistic

end
