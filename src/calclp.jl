#doubly interval censored
function evaluatelp(d::UnivariateDistribution, x::DIC)
    ll = zero(x.EL)
    mu = mean(d)
    if isfinite(x.EL)
        ll -= (log(x.ER - x.EL) + log(x.SR - x.SL))
        if x.ER < x.SL
            ll += log(mu)
            ll += log(eqcdf(d, x.SR-x.ER) - eqcdf(d, x.SL-x.ER) - (eqcdf(d, x.SR-x.EL) - eqcdf(d, x.SL-x.EL)))
        elseif x.SL < x.ER < x.SR
            ll += log((x.ER - x.SL) + mu*(eqcdf(d, x.SR-x.ER) - (eqcdf(d, x.SR-x.EL) - eqcdf(d, x.SL-x.EL))))
        elseif x.SR <= x.ER
            ll += log( (x.SR - x.SL) - mu*(eqcdf(d, x.SR-x.EL) - eqcdf(d, x.SL-x.EL)) )
        end
    else #EL is missing
        if x.ER < x.SL
            ll += log(eqcdf(d, x.SR-x.ER)-eqcdf(d, x.SL-x.ER))  - log(x.SR - x.SL)
        elseif x.ER <= x.SR
            ll += log(eqcdf(d, x.SR-x.ER))  - log(x.SR - x.SL)
        end
    end
    return ll
end

# function evaluatelp(d::UnivariateDistribution, x::DICRT)
#     ll = zero(x.EL)
#     mu = mean(d)
#     if isfinite(x.EL)
#         if x.ER < x.SL
#             ll += log(eqcdf(d, x.SR-x.ER) - eqcdf(d, x.SL-x.ER) - (eqcdf(d, x.SR-x.EL) - eqcdf(d, x.SL-x.EL))) - (log(eqcdf(d,x.TR-x.ER)-eqcdf(d,x.TR-x.EL)))
#         elseif x.SL < x.ER < x.SR
#             ll += log((x.ER-x.SL) + mu*(eqcdf(d, x.SR-x.ER) - (eqcdf(d, x.SR-x.EL) - eqcdf(d, x.SL-x.EL)))) - (log(mu) + log(eqcdf(d,x.TR-x.ER)-eqcdf(d,x.TR-x.EL)))
#         elseif x.SR <= x.ER
#             ll += log( (x.SR - x.SL) - mu*(eqcdf(d, x.SR-x.EL) - eqcdf(d, x.SL-x.EL)) ) - (log(mu) + log(eqcdf(d,x.TR-x.ER)-eqcdf(d,x.TR-x.EL)))
#         end
#     else #EL is missing
#         if x.ER < x.SL
#             ll += log(eqcdf(d, x.SR-x.ER)-eqcdf(d, x.SL-x.ER)) - (logmu+log(eqcdf(d,x.TR-x.ER))) #- log(x.SR - x.SL)
#         elseif x.ER <= x.SR
#             ll += log(eqcdf(d, x.SR-x.ER)) - (logmu+log(eqcdf(d,x.TR-x.ER))) 
#         end
#     end
#     return ll
# end

#interval censored
function evaluatelp(d::UnivariateDistribution, x::ICE)
    ll = zero(x.EL)
    logmu = logmean(d)
    if isfinite(x.EL)
        ll += log(cdf(d, x.S-x.EL) - cdf(d, x.S-x.ER)) - log(x.ER - x.EL)
    else
        ll += log(ccdf(d, x.S-x.ER)) - logmu
    end
    return ll
end

# function evaluatelp(d::UnivariateDistribution, x::ICRT)
#     ll = zero(x.EL)
#     logmu = logmean(d)
#     if isfinite(x.EL)
#         ll += log(cdf(d, x.S-x.EL) - cdf(d, x.S-x.ER)) - (logmu+log(eqcdf(d,x.TR-x.ER)-eqcdf(d,x.TR-x.EL))) #- log(x.ER - x.EL)
#     else
#         ll += log(ccdf(d, x.S-x.ER)) - logmu - log(eqcdf(d,x.TR-x.ER))
#     end
#     return ll
# end

function evaluatelp(d::UnivariateDistribution, x::NC)
    return logpdf(d, x.S-x.E)
end
