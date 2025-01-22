module DynamicHMCExt
using BayesianSurvival, DynamicHMC, StanBlocks

BayesianSurvival.fit_survival_model(rng, lpdf; n_draws=1000, kwargs...) = begin 
    mcmc_with_warmup(rng, StanBlocks.with_gradient(lpdf), n_draws; kwargs...)
end

end