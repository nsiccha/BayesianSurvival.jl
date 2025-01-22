module GadflyExt
using BayesianSurvival, Gadfly, DataFrames

"Plots the observed groupwise survival functions."
BayesianSurvival.plot_observed_survival(df; by) = begin 
    cdf = transform(groupby(sort(df, :t), by)) do sdf
        (;survival=1 .- cumsum(sdf.event) ./ size(sdf, 1))
    end
    Gadfly.plot(cdf, x=:t, y=:survival, color=by, Geom.line)
end

end