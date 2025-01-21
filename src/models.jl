import StanBlocks: @parameters
begin


function pem_survival_model(;
    survived,
    t,
    design_matrix,
)
    n_persons, n_covariates = size(design_matrix)
    t1 = sort(unique(t))
    n_timepoints = length(t1)
    end_idxs = searchsortedfirst.(Ref(t1), t)
    t0 = vcat(0., t1[1:end-1])
    dt = t1 .- t0
    log_dts = log.(dt)
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
            for person in 1:n_persons
                idxs = 1:end_idxs[person]
                log_hazards = StanBlocks.@broadcasted(log_hazard_intercept + log_hazard_personwise[person] + log_hazard_timewise[idxs])
                log_survival = -exp(StanBlocks.logsumexp(StanBlocks.@broadcasted(log_dts[idxs] + log_hazards)))
                target += log_survival
                survived[person] && continue
                target += log_hazards[end_idxs[person]]
            end
        end
        StanBlocks.@generated_quantities begin
            log_lik = map(1:n_persons) do person
                idxs = 1:end_idxs[person]
                log_hazards = StanBlocks.@broadcasted(log_hazard_intercept + log_hazard_personwise[person] + log_hazard_timewise[idxs])
                log_survival = -exp(StanBlocks.logsumexp(StanBlocks.@broadcasted(log_dts[idxs] + log_hazards)))
                if survived[person]
                    log_survival
                else
                    log_hazards[end_idxs[person]] + log_survival
                end
            end
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

end