## The package 'IntervalCensored' is Available on https://github.com/abikoushi/IntervalCensored.jl
using Distributions
using Random
using Plots
using StatsPlots
using QuadGK
using DataFrames
#using RCall
#using Optim
#using ForwardDiff
using StatsFuns
#using SpecialFunctions
#using LinearAlgebra
#using ProgressMeter
#]add "./projects/IntervalCensored"
#]add "https://github.com/abikoushi/IntervalCensored.jl"
using IntervalCensored

function calclp_dic2(d, EL, ER, SL, SR)
    n = length(EL)
    ll = zero(EL[1])
    logmu = log(mean(d))
    for i in eachindex(EL)
        if isfinite(EL[i])
            if ER[i] < SL[i]
                ll += logmu+log(eqcdf(d,SR[i]-ER[i])-eqcdf(d,SL[i]-ER[i])-(eqcdf(d,SR[i]-EL[i])-eqcdf(d,SL[i]-EL[i]))) -
                 log(ER[i] - EL[i]) - log(SR[i] - SL[i])
            elseif ER[i] < SR[i]
                ll += logmu+log(eqcdf(d,SR[i]-ER[i])-(eqcdf(d,SR[i]-EL[i])-eqcdf(d,SL[i]-EL[i]))) - 
                 log(ER[i] - EL[i]) - log(SR[i] - SL[i])
            elseif SR[i] <= ER[i]
                ll += logmu+log(eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i])) - 
                  log(ER[i] - EL[i]) - log(SR[i] - SL[i])
            end
        else
            if ER[i] < SL[i]
                ll += log(eqcdf(d,SR[i]-ER[i])-eqcdf(d,SL[i]-ER[i])) - log(SR[i] - SL[i])
            elseif ER[i] < SR[i]
                ll += log(eqcdf(d,SR[i]-ER[i])) - log(SR[i] - SL[i])
            end
        end
    end
    return -ll
end

function sim_dic(td, md, N, iter, seed)
    rng = MersenneTwister()
    aic1 = zeros(iter)
    aic2 = zeros(iter)
    ge = zeros(iter)
    K = length(params(md))
    theta = zeros(iter,K)
    for i in 1:iter
        dat = make_dic(rng, td, N)
        fit = MCEMdic(rng, md, 10, dat[1], dat[2], dat[3], dat[4])
        ge[i] = quadgk(x -> -logpdf(fit[1],x)*pdf(td,x), 0, Inf)[1]
        aic1[i] = (calclp_dic(fit[1], dat[1], dat[2], dat[3], dat[4]) + K)/N
        aic2[i] = fit[2][end]+K/N
        theta[i,:] .= params(fit[1])
    end
    return aic1, aic2, ge, theta
end

plot(x->eqcdf(Weibull(1.5,7), x),0,9, legend=false)

rng = MersenneTwister()
EL, ER, SL, SR = make_dic(rng, Weibull(1.5,7), 100)
k = findfirst(SR .<= ER)
eqcdf(Weibull(1.5,7),SR[k]-EL[k])
eqcdf(Weibull(1.5,7),SL[k]-EL[k])
@time simout_dic = sim_dic(Weibull(1.5,7), Weibull(2, 7), 100, 100, 1234)
ms = [mean(simout_dic[1]-simout_dic[3]), mean(simout_dic[2]-simout_dic[3])]
ss = [std(simout_dic[1]-simout_dic[3]), std(simout_dic[2]-simout_dic[3])]
df = stack(DataFrame(AIC1=simout_dic[1]-simout_dic[3], AIC2=simout_dic[2]-simout_dic[3]))
@df df violin(:variable, :value, fill="white", legend=false, tick_direction=:out, xtickfontsize=12, trim=false)
scatter!(["AIC1","AIC2"], ms[1:2], yerror = ss[1:2], color="black", ms=6)
Plots.abline!(0, 0, ls=:dash, color="black")


function sim_ic(td, md, N, iter, seed)
    rng = MersenneTwister(seed)
    aic1 = zeros(iter)
    aic2 = zeros(iter)
    ge = zeros(iter)
    K = length(params(md))
    theta = zeros(iter,K)
    for i in 1:iter
        dat = make_ic(rng, td, N)
        fit = MCEMic(rng, md, 100, dat[1], dat[2], dat[3])
        ge[i] = quadgk(x -> -logpdf(fit[1],x)*pdf(td,x), 0, Inf)[1]
        aic1[i] = (calclp_ic(fit[1], dat[1], dat[2], dat[3]) + K)/N
        aic2[i] = fit[2][end]+K/N
        theta[i,:] .= params(fit[1])
    end
    return aic1, aic2, ge, theta
end

function sim_ic(td, md, N, p, iter, seed)
    rng = MersenneTwister(seed)
    aic1 = zeros(iter)
    aic2 = zeros(iter)
    ge = zeros(iter)
    K = length(params(md))
    theta = zeros(iter,K)
    for i in 1:iter
        dat = make_ic(rng, td, N, p)
        fit = MCEMic(rng, md, 100, dat[1], dat[2], dat[3])
        ge[i] = quadgk(x -> -logpdf(fit[1],x)*pdf(td, x), 0, Inf)[1]
        aic1[i] = (calclp_ic(fit[1], dat[1], dat[2], dat[3]) + K)/N
        aic2[i] = fit[2][end]+K/N
        theta[i,:] .= params(fit[1])
    end
    return aic1, aic2, ge, theta
end

@time simout_ic = sim_ic(Weibull(1.5, 3), Weibull(1.0, 3.0), 100, 500, 2222)
mean(simout_ic[4], dims=1)
ms = [mean(simout_ic[1]-simout_ic[3]), mean(simout_ic[2]-simout_ic[3])]
ss = [std(simout_ic[1]-simout_ic[3]), std(simout_ic[2]-simout_ic[3])]
df = stack(DataFrame(AIC1=simout_ic[1]-simout_ic[3], AIC2=simout_ic[2]-simout_ic[3]))

@df df violin(:variable, :value, fill="white", legend=false, tick_direction=:out, xtickfontsize=12, trim=false)
scatter!(["AIC1","AIC2"], ms[1:2], yerror = ss[1:2], color="black", ms=6)
Plots.abline!(0, 0, ls=:dash, color="black")

@time simout_icm = sim_ic(Gamma(1.5, 2.0), Gamma(1.0, 2.0), 100, 0.8, 500, 1111)
ms = [mean(simout_icm[1]-simout_icm[3]), mean(simout_icm[2]-simout_icm[3])]
ss = [std(simout_icm[1]-simout_icm[3]), std(simout_icm[2]-simout_icm[3])]
df2 = stack(DataFrame(AIC1=simout_icm[1]-simout_icm[3], AIC2=simout_icm[2]-simout_icm[3]))

@df df2 violin(:variable, :value, fill="white", legend=false, tick_direction=:out, xtickfontsize=12, trim=false)
scatter!(["AIC1","AIC2"], ms[1:2], yerror = ss[1:2], color="black", ms=6)
Plots.abline!(0, 0, ls=:dash, color="black")

########
function sim_icrt(td, md, N, tmax, iter, seed)
    rng = MersenneTwister(seed)
    aic1 = zeros(iter)
    aic2 = zeros(iter)
    ge = zeros(iter)
    K = length(params(md))
    theta = zeros(iter,K)
    for i in 1:iter
        dat = make_icrt(rng, td, tmax, N)
        fit = MCEMicrt(rng, md, 100, dat[1], dat[2], dat[3], tmax, 1)
        ge[i] = quadgk(x -> -logpdf(td,x)*pdf(td,x),0,Inf)[1]
        n = length(dat[1])
        aic1[i] = (calclp_icrt(fit[1] ,dat[1], dat[2], dat[3], tmax) + K)/n 
        aic2[i] = fit[2][end]+K/n
        theta[i,:] .= params(fit[1])
    end
    return aic1, aic2, ge, theta
end
Tmax = 30
@time simout_icrt = sim_icrt(Weibull(1.5, 4), Weibull(1.5, 4), 100, Tmax, 500, 101)
rng = MersenneTwister()
dat = make_icrt(rng, Weibull(1.5, 4), Tmax, 100)
all(dat[2]-dat[1].>0)
plot(x -> (eqcdf(Gamma(1.5, 4),x-2)-eqcdf(Gamma(1.5, 4),x)),0,9,legend=false)
plot(x -> (ccdf(Gamma(1.5, 4),x-2)-ccdf(Gamma(1.5, 4),x-1)),0,20,legend=false)

f(t) = quadgk(x -> (ccdf(Gamma(1.5, 4),x-3)-cdf(Gamma(1.5, 4),x-1))/2,3,t)[1]
g(t) = mean(Gamma(1.5,4))*(-eqcdf(Gamma(1.5, 4),max(0,t-3))+eqcdf(Gamma(1.5, 4),max(t-1,0)))/2
f(5), g(5)
min(t-1,1)

