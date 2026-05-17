# exp/ablation_study/test_analysis.jl

using BayesianFootball
using DataFrames
using JSON3
using Statistics
using Printf

function run_test_analysis(; exp_dir="exp/ablation_study")
    println("============================================================")
    println(" 📊 Analyzing Ablation Study Results")
    println("============================================================")

    # 1. Get all experiment paths (Automatically sorted newest -> oldest)
    paths = BayesianFootball.Experiments.list_experiments(exp_dir)
    
    if isempty(paths)
        println("No experiments found.")
        return
    end

    df_rows = []
    seen_models = Set{String}()

    # 2. Iterate and Extract
    for path in paths
        meta_file = joinpath(path, "meta.json")
        
        if isfile(meta_file)
            meta = JSON3.read(read(meta_file, String))
            model_name = meta.name
            
            # Because paths are sorted newest -> oldest, the first time we 
            # see a model name, it is guaranteed to be the most recent run.
            if !(model_name in seen_models) && occursin("ablation", model_name)
                push!(seen_models, model_name)
                
                printstyled("Extracting: ", color=:light_black)
                println(model_name)
                
                # Extract Time from Meta
                time_str = get(meta, :time_taken, "N/A")
                
                # --- LOAD METRICS ---
                # Load the heavy JLD2 file to verify cross-validation succeeded
                # and extract actual predictive metrics (e.g., RQR, Log-Growth).
                # Adjust these field names based on your exact `TrainingResults` struct!
                res = BayesianFootball.Experiments.load_experiment(path)
                
                # Default placeholders
                rqr_val = missing
                log_growth_val = missing
                
                try
                    # Assuming your CV pipeline calculates summary metrics and stores
                    # them somewhere in the training_results struct. 
                    # Example: res.training_results.metrics[:rqr]
                    # Update this path to match exactly where your CV evaluator saves metrics:
                    
                    if hasproperty(res.training_results, :metrics)
                        metrics = res.training_results.metrics
                        rqr_val = round(get(metrics, :rqr, NaN), digits=4)
                        log_growth_val = round(get(metrics, :log_growth, NaN), digits=4)
                    end
                catch e
                    # Fails silently if metrics aren't pre-calculated in the struct
                end

                # Add to our DataFrame rows
                push!(df_rows, (
                    Model = model_name,
                    Time = time_str,
                    RQR = rqr_val,
                    LogGrowth = log_growth_val,
                    Path = basename(path)
                ))
            end
        end
    end

    # 3. Compile and Display
    df = DataFrame(df_rows)
    
    # Sort models sequentially (01 to 07)
    sort!(df, :Model)
    
    println("\n============================================================")
    println(" 📈 Forward-Step Ablation Summary")
    println("============================================================")
    
    # Print a beautiful markdown-style table to the REPL
    display(df)
    
    println("\n💡 Verify that RQR and LogGrowth are not 'missing'. If they are,")
    println("   you may need to run the `BayesianFootball.Metrics` evaluator")
    println("   on the folds inside `res.training_results` before saving.")
    println("============================================================")
    
    return df
end

# Execute
results_df = run_test_analysis()
