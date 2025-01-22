cd(dirname(Base.active_project()))
using BayesianSurvival, StanBlocks, DynamicObjects, Distributions, Random, DataFrames, StatsModels, Mooncake, DynamicHMC, LogDensityProblems, Chairmarks, Plots, StatsPlots
plotlyjs()

@dynamicstruct struct Simulation
    N
    censor_time
    rate_form
    rate_coefs
    seed = 0
    likelihood = true
    cache_path = joinpath("cache", "$N")
    @cached df = sim_data_exp_correlated(;N, censor_time, rate_form, rate_coefs)
    design_matrix = hcat(df.age .- mean(df.age), df.male)
    lpdf1 = survival_model(pem_survival_model; df, design_matrix, likelihood)
    lpdf2 = survival_model(pem_survival_model_randomwalk; df, design_matrix, likelihood)
    lpdf3 = survival_model(pem_survival_model_timevarying; df, design_matrix, likelihood)
    @cached result1 = fit_survival_model(Xoshiro(seed), lpdf1)
    @cached result2 = fit_survival_model(Xoshiro(seed), lpdf2)
    @cached result3 = fit_survival_model(Xoshiro(seed), lpdf3)
    lr1 = (lpdf1, result1)
    lr2 = (lpdf2, result2)
    lr3 = (lpdf3, result3)
end 
