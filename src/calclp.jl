function calclp_dic(d, EL, ER, SL, SR)
    logmu = logmean(d)
    ll = zero(EL[1])
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

# function calclp_dicrt(d, EL, ER, SL, SR, Tmax)
#     n = length(EL)
#     ll = 0
#     logmu = logmean(d)
#     for i in 1:n
#         if ER[i]<SL[i]
#             ll += logmu+logsubexp(log(eqcdf(d,SR[i]-ER[i])-eqcdf(d,SL[i]-ER[i])),log(eqcdf(d,SR[i]-EL[i])-eqcdf(d,SL[i]-EL[i])))
#         else 
#             ll += logmu+logsubexp(log(eqcdf(d,SR[i]-ER[i])-(SL[i]-ER[i])),log(eqcdf(d,SR[i]-EL[i])-eqcdf(d,SL[i]-EL[i])))
#         end
#         ll += log(cdf2(d, Tmax-SR[i]))
#     end
#     return -ll
# end

#######
#interval censored
function calclp_ic(d, EL, ER, S)
    n = length(S)
    ll = 0.0
    mu = mean(d)
    for i in 1:n
        if isfinite(EL[i])
            ll += log(ccdf(d, S[i]-ER[i]) - ccdf(d, S[i]-EL[i])) - log(ER[i] - EL[i])
        else
            ll += log(ccdf(d, S[i]-ER[i])) - logmean(d)
        end
    end
    return -ll
end

function calclp_icrt(d, EL, ER, S, Tmax)
    n = length(S)
    ll = 0.0
    for i in 1:n
        if isfinite(EL[i])
            ll += log(ccdf(d,S[i]-ER[i]) - ccdf(d,S[i]-EL[i])) - (logmean(d)+log(eqcdf(d,Tmax-ER[i])-eqcdf(d,Tmax-EL[i])))
        else
            ll += log(ccdf(d,S[i]-ER[i])) - log(eqcdf(d,Tmax-ER[i]))
        end
    end
    return -ll
end
