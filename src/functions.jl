abstract type AbstractExponentialModel end
struct ConstantExponentialModel{H} <: AbstractExponentialModel
    hazard::H
end
Base.rand(rng::AbstractRNG, m::ConstantExponentialModel, args...) = rand(rng, Exponential(1/m.hazard))

# start snippet sim_data_exp_correlated
sim_data_exp_correlated(rng=Random.default_rng(); N, censor_time, rate_form, rate_coefs) = begin 
    idx = 1:N
    age = rand(rng, Poisson(55), N)
    male = rand(rng, Bernoulli(.5), N)
    rate = @. exp(rate_coefs[1] + male * rate_coefs[2])
    true_t = rand.(rng, ConstantExponentialModel.(rate))
    t = min.(true_t, censor_time)
    survived = true_t .> censor_time
    DataFrame((;age, male, rate, true_t, t, survived, idx))
end
# end snippet sim_data_exp_correlated


"Plots a summary of a fit. Requires loading StatsPlots"
function plot_summary end

function prep_data_long_surv end

"Instantiates a survival model"
survival_model(model; df, design_matrix, likelihood=true) = model(;df.survived, df.t, design_matrix, likelihood)
"Fit a survival model instantiated via `survival_model`. Requires loading DynamicHMC."
function fit_survival_model end 


struct SurvivalCurve{T}
    t::T
end
survival_curve(t) = SurvivalCurve(sort(t))
survival(s::SurvivalCurve) = 1 .- cumsum(s.t .< s.t[end]) ./ length(s.t)
survival(s::SurvivalCurve, t) = 1 - (searchsortedfirst(s.t, t)-1) / length(s.t)