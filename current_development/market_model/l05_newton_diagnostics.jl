module FullBookNewtonDiagnostics

import CSV
import DataFrames
import LinearAlgebra
const PM = parentmodule(@__MODULE__).PooledMarket
const LA = LinearAlgebra

"""
Deterministic trace of ONE full-book filter at a recorded supported coordinate.
Inject a derivative observer into the UNCHANGED production joint_mode. Return
exactly the original derivatives; evaluate f(x) only for logging. No line-search,
mode rule, budget, population or numerical fallback is changed. Failed filters
are reported as failed, never fits. This diagnostic is not a sampler retry.
"""
function trace_fullbook_mode(generated,theta,out)
    rows = NamedTuple[]
    calls = Ref(0)
    ids = generated.panel.obs_match[1:2:end]
    function observed_solver(f,a; derivative,iterations=100)
        calls[] += 1
        id = ids[calls[]]
        derivative_calls = Ref(0)
        previous = Ref{Union{Nothing,Vector{Float64}}}(nothing)
        function observed_derivative(f,x)
            g,h = derivative(f,x)
            derivative_calls[] += 1
            e = LA.eigen(LA.Symmetric(-h))
            B = e.vectors*LA.Diagonal(max.(e.values,1e-6))*e.vectors'
            step = B\g
            movement = previous[] === nothing ? NaN : maximum(abs,x-previous[])
            previous[] = copy(x)
            push!(rows,(; match_id=id,derivative_call=derivative_calls[],
                mode_1=x[1],mode_2=x[2],gradient_1=g[1],gradient_2=g[2],
                gradient_norm=LA.norm(g),decrement=LA.dot(g,step)/2,
                newton_step_inf=maximum(abs,step),actual_movement_inf=movement,
                raw_precision_min=minimum(e.values),value=f(x)))
            return g,h
        end
        return PM.joint_mode(f,a; derivative=observed_derivative,iterations)
    end
    PM.reset_newton_accounting!()
    failure = nothing
    try
        PM.fullbook_filter(PM.FullBookRung(:C1),generated.panel,theta;
            markets=generated.markets,mode_solver=observed_solver)
    catch e
        failure = sprint(showerror,e)
    end
    mkpath(out)
    CSV.write(joinpath(out,"newton_mode_trace.csv"),DataFrames.DataFrame(rows))
    summary = DataFrames.DataFrame(status=[failure === nothing ? "completed filter" : "failed filter"],
        error=[something(failure,"")],fixture=[ids[calls[]]],
        derivative_calls=[last(rows).derivative_call],
        theta_q=[theta[1]],theta_s=[theta[2]],theta_u=[theta[3]],theta_n=[theta[4]],
        final_gradient_norm=[last(rows).gradient_norm],final_decrement=[last(rows).decrement],
        final_newton_step_inf=[last(rows).newton_step_inf],
        final_actual_movement_inf=[last(rows).actual_movement_inf])
    CSV.write(joinpath(out,"newton_mode_trace_summary.csv"),summary)
    PM.write_newton_accounting(out; run="initial_target_diagnostic")
    return summary
end

end # module
