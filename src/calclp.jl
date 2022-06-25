function calclp_dic(d, EL, ER, SL, SR)
    ll = zero(EL[1])
    mu = mean(d)
    for i in eachindex(EL)
        if isfinite(EL[i])
            ll -= log(ER[i] - EL[i]) + log(SR[i] - SL[i])
            if ER[i] < SL[i]
                ll += log(mu)
                ll += log(eqcdf(d,SR[i]-ER[i]) - eqcdf(d,SL[i]-ER[i]) - (eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i])))
            elseif ER[i] < SR[i]
                ll += log((ER[i]-SL[i]) + mu*(eqcdf(d,SR[i]-ER[i]) - (eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i]))))
            elseif SR[i] <= ER[i]
                ll += log( (SR[i] - SL[i]) - mu*(eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i])) )
            end
        else #EL is missing
            if ER[i] < SL[i]
                ll += log(eqcdf(d,SR[i]-ER[i])-eqcdf(d,SL[i]-ER[i]))  - log(SR[i] - SL[i])
            elseif ER[i] < SR[i]
                ll += log(eqcdf(d,SR[i]-ER[i]))  - log(SR[i] - SL[i])
            end
        end
    end
    return -ll
end

function calclp_dicrt(d, EL, ER, SL, SR, Tmax)
    ll = zero(EL[1])
    mu = mean(d)
    for i in eachindex(EL)
        if isfinite(EL[i])
            ll -= log(ER[i] - EL[i]) + log(SR[i] - SL[i])
            if ER[i] < SL[i]
                ll += log(mu)
                ll += log(eqcdf(d,SR[i]-ER[i]) - eqcdf(d,SL[i]-ER[i]) - (eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i])))
            elseif ER[i] < SR[i]
                ll += log((ER[i]-SL[i]) + mu*(eqcdf(d,SR[i]-ER[i]) - (eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i]))))
            elseif SR[i] <= ER[i]
                ll += log( (SR[i] - SL[i]) - mu*(eqcdf(d,SR[i]-EL[i]) - eqcdf(d,SL[i]-EL[i])) )
            end
        else #EL is missing
            if ER[i] < SL[i]
                ll += log(eqcdf(d,SR[i]-ER[i])-eqcdf(d,SL[i]-ER[i]))  - log(SR[i] - SL[i])
            elseif ER[i] < SR[i]
                ll += log(eqcdf(d,SR[i]-ER[i]))  - log(SR[i] - SL[i])
            end
        end
        ll += logcdf(d, Tmax-SR[i])
    end
    return -ll
end

#######
#interval censored
function calclp_ic(d, EL, ER, S)
    ll = zero(EL[1])
    mu = mean(d)
    for i in eachindex(EL)
        if isfinite(EL[i])
            ll += logdiffcdf(d, S[i]-EL[i], S[i]-ER[i]) - log(ER[i] - EL[i])
        else
            ll += logccdf(d, S[i]-ER[i]) - logmean(d)
        end
    end
    return -ll
end

function calclp_icrt(d, EL, ER, S, Tmax)
    ll = zero(EL[1])
    for i in eachindex(EL)
        if isfinite(EL[i])
            ll += logdiffcdf(d, S[i]-EL[i], S[i]-ER[i]) - (logmean(d)+log(eqcdf(d,Tmax-ER[i])-eqcdf(d,Tmax-EL[i])))
        else
            ll += logccdf(d,S[i]-ER[i]) - log(eqcdf(d,Tmax-ER[i]))
        end
    end
    return -ll
end
