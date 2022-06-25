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
#using StatsFuns
#using SpecialFunctions
#using LinearAlgebra
#using ProgressMeter
#]add "./projects/IntervalCensored"
#]add "https://github.com/abikoushi/IntervalCensored.jl"
using IntervalCensored

function calclp_dic_s(rng, d, EL, ER, SL, SR)
    ll = zero(EL[1])
    for i in eachindex(EL)
        if isfinite(EL[i])
            u = rand(rng)
            E = EL[i] + (ER[i]-EL[i])*u
            ys = rand(rng, truncated(d, SL[i]-E, SR[i]-E))
            ll += logpdf(d, ys)
        end
    end
    return -ll
end

function calclp_dic2(d, EL, ER, SL, SR)
    ll = zero(EL[1])
    mu = mean(d)
    for i in eachindex(EL)
        ll -= log(ER[i] - EL[i]) + log(SR[i] - SL[i])
        if ER[i] < SL[i]
            ll += log(mu)
            ll += log(eqcdf(d,SR[i]-ER[i]) - eqcdf(d,SL[i]-ER[i]) - (eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i])))
        elseif SL[i] < ER[i] < SR[i]
            ll += log((ER[i]-SL[i]) + mu*(eqcdf(d,SR[i]-ER[i]) - (eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i]))))
        elseif SR[i] <= ER[i]
            ll += log( (SR[i] - SL[i]) - mu*(eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i])) )
        end
    end
    return -ll
end

function simentropy(td, n, iter)
    rng = MersenneTwister()
    lp1 = zeros(iter)
    lp2 = zeros(iter)
    for i in 1:iter
        dat = make_dic(rng, td, n)
        lp1[i] = calclp_dic_s(rng, td, dat[1], dat[2], dat[3], dat[4])/n
        lp2[i] = calclp_dic2(td, dat[1], dat[2], dat[3], dat[4])/n
    end
    return lp1, lp2
end

lp1, lp2 = simentropy(Weibull(2.0,7.0),50,10000)

ms = [mean(lp1), mean(lp2)]
ss = [std(lp1), std(lp2)]
df = stack(DataFrame(Hs=lp1, Hm=lp2))
@df df violin(:variable, :value, fill="white", legend=false, tick_direction=:out, xtickfontsize=12, trim=false)
scatter!(["Hs","Hm"], ms[1:2], yerror = ss[1:2], color="black", ms=6)
Plots.abline!(0, entropy(Weibull(2.0,7.0)), ls=:dash, color="black")
mean(lp1)-entropy(Weibull(1.5,7))
mean(lp2)-entropy(Weibull(1.5,7))
histogram(lp2,legend=false, color = "lightgrey")
vline!([entropy(Weibull(1.5,7))], linewidth=3, color = "royalblue")
vline!([mean(lp2)], linewidth=3, color = "tomato")

d = Weibull(2.0, 2.0)
EL, ER, SL, SR = 1.0, 1.5, 2.0, 3.0
quadgk(s -> ccdf(d,s-ER)-ccdf(d,s-EL), SL, SR)
mean(d)*(eqcdf(d,SR-ER) - eqcdf(d,SL-ER) - (eqcdf(d,SR-EL) - eqcdf(d,SL-EL)))

EL, SL, ER, SR = 0.25, 1.7, 2.0, 3.0
p1, _ = quadgk(s -> ccdf(d,s-ER)-ccdf(d,s-EL), SL, SR)
p1
mean(d)*(eqcdf(d,SR-ER) - (eqcdf(d,SR-EL) - eqcdf(d,SL-EL))) + ER-SL

EL, SL, SR, ER = 0.25, 1.7, 2.0, 2.0
p1, _ = quadgk(s -> ccdf(d,s-ER)-ccdf(d,s-EL), SL, SR)
p1
(SR - SL)  - mean(d)*(eqcdf(d,SR-EL) - eqcdf(d,SL-EL)) 

function calclp_dic3(d, EL, ER, SL, SR)
    ll = zero(EL[1])
    for i in eachindex(EL)
        ll -= log(ER[i] - EL[i]) + log(SR[i] - SL[i])
        prob, _ = quadgk(s -> ccdf(d,s-ER[i])-ccdf(d,s-EL[i]), SL[i], SR[i])
        ll += log(prob)
    end
    return ll
end

function calclp_ic_s(rng, d, EL, ER, S)
    ll = zero(EL[1])
    for i in eachindex(EL)
        if isfinite(EL[i])
            u = rand(rng)
            E = EL[i] + (ER[i]-EL[i])*u
            ll += logpdf(d,S[i]-E)
        end
    end
    return -ll
end




function simentropy_ic(td, n, iter)
    rng = MersenneTwister()
    lp1 = zeros(iter)
    lp2 = zeros(iter)
    for i in 1:iter
        dat = make_ic(rng, td, n)
        lp1[i] = calclp_ic_s(rng, td, dat[1], dat[2], dat[3])/n
        lp2[i] = calclp_ic(td, dat[1], dat[2], dat[3])/n
    end
    return lp1, lp2
end

dist0 = Weibull(1.5,5)
n = 100
rng = MersenneTwister()
dat = make_dic(rng, dist0, n)
calclp_dic3(dist0, dat[1], dat[2], dat[3], dat[4])/n
calclp_dic2(dist0, dat[1], dat[2], dat[3], dat[4])/n


lp1, lp2 = simentropy(dist0, 1000, 10000)

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
        ge[i] = N*quadgk(x -> -logpdf(fit[1],x)*pdf(td,x), 0, Inf)[1]
        aic1[i] = (calclp_dic2(fit[1], dat[1], dat[2], dat[3], dat[4]) + K)
        aic2[i] = N*fit[2][end]+K
        theta[i,:] .= params(fit[1])
    end
    return aic1, aic2, ge, theta
end

@time simout_dic = sim_dic(Weibull(1.5,7), Weibull(2,7), 100, 1000, 1234)
ms = [mean(simout_dic[1]-simout_dic[3]), mean(simout_dic[2]-simout_dic[3])]
ss = [std(simout_dic[1]-simout_dic[3]), std(simout_dic[2]-simout_dic[3])]
df = stack(DataFrame(AICm=simout_dic[1]-simout_dic[3], AICs=simout_dic[2]-simout_dic[3]))
@df df violin(:variable, :value, fill="white", legend=false, tick_direction=:out, xtickfontsize=12, trim=false)
scatter!(["AICm","AICs"], ms[1:2], yerror = ss[1:2], color="black", ms=6)
Plots.abline!(0, 0, ls=:dash, color="black")
mean(simout_dic[4], dims=1)
histogram(simout_dic[1]-simout_dic[3],legend=false, color = "lightgrey", tick_direction=:out)
vline!([0], linewidth=3, color = "royalblue")
vline!([mean(simout_dic[1]-simout_dic[3])], linewidth=3, color = "tomato")
#png("./Desktop/miss.png")

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
        ge[i] = N*quadgk(x -> -logpdf(fit[1],x)*pdf(td, x), 0, Inf)[1]
        aic1[i] = (calclp_ic(fit[1], dat[1], dat[2], dat[3]) + K)
        aic2[i] = fit[2][end]+K
        theta[i,:] .= params(fit[1])
    end
    return aic1, aic2, ge, theta
end

@time simout_ic = sim_ic(Weibull(1.5, 3), Weibull(2, 3), 100, 500, 123)
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

