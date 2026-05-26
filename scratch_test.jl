using Pkg; Pkg.activate(".")
using JLD2
using BayesianFootball

# Temporary JLD2 upgrade hook
function Base.convert(::Type{BayesianFootball.Training.AbstractExecutionStrategy}, rs::JLD2.ReconstructedStatic)
    if typeof(rs).parameters[1] == Symbol("BayesianFootball.Training.Independent")
        parallel = hasproperty(rs, :parallel) ? rs.parallel : false
        max_splits = hasproperty(rs, :max_concurrent_splits) ? rs.max_concurrent_splits : 1
        return BayesianFootball.Training.Independent(
            parallel = parallel,
            max_concurrent_splits = max_splits,
            max_concurrent_tasks = Threads.nthreads()
        )
    end
    throw(MethodError(convert, (BayesianFootball.Training.AbstractExecutionStrategy, rs)))
end

try
    saved_files = BayesianFootball.Experiments.list_experiments("./data/copula_ab_test/", data_dir="")
    expr_results = BayesianFootball.Experiments.load_experiment(saved_files, 4)
    println("SUCCESS")
catch e
    showerror(stdout, e, catch_backtrace())
end
