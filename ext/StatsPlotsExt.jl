module StatsPlotsExt
using BayesianSurvival, StatsPlots
import StatsPlots: Plots, plot, plot!, histogram, boxplot, quantile
import BayesianSurvival: SurvivalCurve, survival_curve, survival

Plots.plot!(p::Plots.Plot, s::SurvivalCurve; kwargs...) = Plots.plot!(p, s.t, survival(s); kwargs...)
Plots.plot(s::SurvivalCurve; kwargs...) = Plots.plot!(Plots.plot(), s; kwargs...)

maps(f) = (args...)->map(f, args...)

"Plots a summary of a fit."
BayesianSurvival.plot_summary(lpdf, result; df, kwargs...) = begin 
    gqs = mapreduce(
        lpdf.gq, maps(hcat),eachcol(result.posterior_matrix)
    )
    n_covariates = 2
    n_timepoints, n_draws = size(gqs.log_hazard_timewise)
    (;t_pred) = gqs
    beta_histograms = if hasproperty(gqs, :beta)
        [Symbol("beta.", i)=>gqs.beta[i,:] for i in 1:n_covariates] 
    else
        []
    end 
    histograms = (;
        log_hazard_intercept=view(gqs.log_hazard_intercept, 1, :),
        beta_histograms...
    )
    beta_boxplots = if hasproperty(gqs, :beta_timewise)
        beta = reshape(gqs.beta_timewise, (n_covariates, n_timepoints, n_draws))
        [Symbol("beta_timewise.", i)=>beta[i,:,:] for i in 1:n_covariates] 
    else
        []
    end
    boxplots = (;
        gqs.log_hazard_timewise,
        beta_boxplots...
    )
    
    sorted_t = sort(df.t)
    survival_curves_preds = survival_curve.(eachcol(t_pred))
    survival_preds = survival.(survival_curves_preds, sorted_t')
    survival_quantiles = mapreduce(hcat, eachcol(survival_preds)) do col
        quantile(col, [.025, .5, .975])
    end
    
    Plots.plot(
        [
            Plots.histogram(col, title=key, c=2, label="")
            for (key, col) in pairs(histograms)
        ]..., 
        [
            Plots.boxplot(draws', title=key, c=2, label="")
            for (key, draws) in pairs(boxplots)
        ]...,
        Plots.plot!(
            Plots.plot!(
                Plots.plot(
                    survival_curve(sorted_t), label="Observed", ylim=[0,1], marker=:circle
                ),
                sorted_t, survival_quantiles[1, :], fillrange=survival_quantiles[3, :], fillalpha=.25, c=2, alpha=0,
                label="95% CrI"
            ),
            sorted_t, survival_quantiles[2, :], c=2, label="Median"
        );
        size=(800, 1600), layout=(:, 1), kwargs...
    )
end

end