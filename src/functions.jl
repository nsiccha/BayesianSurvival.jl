abstract type AbstractExponentialModel end
struct ConstantExponentialModel{H} <: AbstractExponentialModel
    hazard::H
end
Base.rand(rng::AbstractRNG, m::ConstantExponentialModel, args...) = rand(rng, Exponential(1/m.hazard))

sim_data_exp_correlated(rng=Random.default_rng(); N, censor_time, rate_form, rate_coefs) = begin 
    idx = 1:N
    age = rand(rng, Poisson(55), N)
    male = rand(rng, Bernoulli(.5), N)
    rate = @. exp(rate_coefs[1] + male * rate_coefs[2])
    true_t = rand.(rng, ConstantExponentialModel.(rate))
    t = min.(true_t, censor_time)
    event = true_t .<= censor_time
    DataFrame((;age, male, rate, true_t, t, event, idx))
end


"Requires loading Gadfly, implementation in `ext/GadflyExt.jl`."
function plot_observed_survival end

function prep_data_long_surv end
function fit_stan_survival_model end