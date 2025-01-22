import StanBlocks: @parameters
begin


prepare_survival(;
    t,
    design_matrix,
) = begin 

    n_persons, n_covariates = size(design_matrix)
    t1 = sort(unique(t))
    n_timepoints = length(t1)
    end_idxs = searchsortedfirst.(Ref(t1), t)
    t0 = vcat(0., t1[1:end-1])
    dt = t1 .- t0
    log_dts = log.(dt)
    (;n_persons, n_covariates, t1, n_timepoints, end_idxs, t0, dt, log_dts)
end
survival_lpdf(survived, log_hazards, log_dts) = begin
    log_survival = -exp(StanBlocks.logsumexp(StanBlocks.@broadcasted(log_dts + log_hazards)))
    if survived
        log_survival
    else
        log_hazards[length(log_hazards)] + log_survival
    end
end

# start snippet pem_survival_model
function pem_survival_model(;
    survived,
    t,
    design_matrix,
    likelihood=true
)
    (;
        n_persons, n_covariates, t1, n_timepoints, end_idxs, t0, dt, log_dts
    ) = prepare_survival(;t, design_matrix)
    StanBlocks.@stan begin 
        @parameters begin 
            log_hazard_intercept::real
            beta::vector[n_covariates]
            log_hazard_timewise_scale::real(lower=0)
            log_hazard_timewise::vector[n_timepoints]
        end
        log_hazard_personwise = design_matrix*beta
        StanBlocks.@model @views begin 
            log_hazard_intercept ~ normal(0, 1)
            beta ~ cauchy(0, 2)
            log_hazard_timewise_scale ~ normal(0, 1)
            log_hazard_timewise ~ normal(0, log_hazard_timewise_scale)
            log_lik = Base.broadcast(1:n_persons) do person 
                idxs = 1:end_idxs[person]
                survival_lpdf(
                    survived[person], 
                    StanBlocks.@broadcasted(log_hazard_intercept + log_hazard_personwise[person] + log_hazard_timewise[idxs]),
                    log_dts[idxs]
                )
            end
            likelihood && (target += sum(log_lik))
        end
        StanBlocks.@generated_quantities begin
            log_lik = collect(log_lik)
            t_pred = map(1:n_persons) do person 
                for timepoint in 1:n_timepoints
                    log_hazard = log_hazard_intercept + log_hazard_personwise[person] + log_hazard_timewise[timepoint]
                    rv = rand(Exponential(exp(-log_hazard)))
                    rv <= dt[timepoint] && return t0[timepoint] + rv
                end
                t1[end]
            end
        end
    end
end
# end snippet pem_survival_model

random_walk_lpdf(x::AbstractMatrix, scale) = @views StanBlocks.normal_lpdf(x[2:end, :], x[1:end-1, :], scale)
random_walk_lpdf(x::AbstractVector, scale) = @views StanBlocks.normal_lpdf(x[2:end], x[1:end-1], scale)

# start snippet pem_survival_model_randomwalk
function pem_survival_model_randomwalk(;
    survived,
    t,
    design_matrix,
    likelihood=true
)
    (;
        n_persons, n_covariates, t1, n_timepoints, end_idxs, t0, dt, log_dts
    ) = prepare_survival(;t, design_matrix)
    rw_sqrt_scale = @. sqrt(.5*(dt[1:end-1] + dt[2:end]))
    StanBlocks.@stan begin 
        @parameters begin 
            log_hazard_intercept::real
            beta::vector[n_covariates]
            log_hazard_timewise_scale::real(lower=0)
            log_hazard_timewise::vector[n_timepoints]
        end
        log_hazard_personwise = design_matrix*beta
        StanBlocks.@model @views begin 
            log_hazard_intercept ~ normal(0, 1)
            beta ~ cauchy(0, 2)
            log_hazard_timewise_scale ~ normal(0, 1)
            log_hazard_timewise[1] ~ normal(0, 1)
            log_hazard_timewise ~ random_walk(
                StanBlocks.@broadcasted(log_hazard_timewise_scale * rw_sqrt_scale)
            )
            log_lik = Base.broadcast(1:n_persons) do person 
                idxs = 1:end_idxs[person]
                survival_lpdf(
                    survived[person], 
                    StanBlocks.@broadcasted(log_hazard_intercept + log_hazard_personwise[person] + log_hazard_timewise[idxs]),
                    log_dts[idxs]
                )
            end
            likelihood && (target += sum(log_lik))
        end
        StanBlocks.@generated_quantities begin
            log_lik = collect(log_lik)
            t_pred = map(1:n_persons) do person 
                for timepoint in 1:n_timepoints
                    log_hazard = log_hazard_intercept + log_hazard_personwise[person] + log_hazard_timewise[timepoint]
                    rv = rand(Exponential(exp(-log_hazard)))
                    rv <= dt[timepoint] && return t0[timepoint] + rv
                end
                t1[end]
            end
        end
    end
end
# end snippet pem_survival_model_randomwalk

# start snippet pem_survival_model_timevarying
function pem_survival_model_timevarying(;
    survived,
    t,
    design_matrix,
    likelihood=true
)
    (;
        n_persons, n_covariates, t1, n_timepoints, end_idxs, t0, dt, log_dts
    ) = prepare_survival(;t, design_matrix)
    rw_sqrt_scale = @. sqrt(.5*(dt[1:end-1] + dt[2:end]))
    StanBlocks.@stan begin 
        @parameters begin 
            log_hazard_intercept::real
            beta_timewise_scale::real(lower=0)
            beta_timewise::matrix[n_covariates, n_timepoints]
            log_hazard_timewise_scale::real(lower=0)
            log_hazard_timewise::vector[n_timepoints]
        end
        log_hazard_personwise = design_matrix*beta_timewise
        StanBlocks.@model @views begin 
            log_hazard_intercept ~ normal(0, 1)
            beta_timewise_scale ~ cauchy(0, 1)
            beta_timewise[:, 1] ~ cauchy(0, 1) 
            beta_timewise' ~ random_walk(
                StanBlocks.@broadcasted(beta_timewise_scale * rw_sqrt_scale)
            )
            log_hazard_timewise_scale ~ normal(0, 1)
            log_hazard_timewise[1] ~ normal(0, 1)
            log_hazard_timewise ~ random_walk(
                StanBlocks.@broadcasted(log_hazard_timewise_scale * rw_sqrt_scale)
            )
            log_lik = Base.broadcast(1:n_persons) do person 
                idxs = 1:end_idxs[person]
                survival_lpdf(
                    survived[person], 
                    StanBlocks.@broadcasted(log_hazard_intercept + log_hazard_personwise[person, idxs] + log_hazard_timewise[idxs]),
                    log_dts[idxs]
                )
            end
            likelihood && (target += sum(log_lik))
        end
        StanBlocks.@generated_quantities begin
            log_lik = collect(log_lik)
            t_pred = map(1:n_persons) do person 
                for timepoint in 1:n_timepoints
                    log_hazard = log_hazard_intercept + log_hazard_personwise[person, timepoint] + log_hazard_timewise[timepoint]
                    rv = rand(Exponential(exp(-log_hazard)))
                    rv <= dt[timepoint] && return t0[timepoint] + rv
                end
                t1[end]
            end
        end
    end
end
# end snippet pem_survival_model_timevarying
end