# exp/market_scot/april/dev_loaders/l05_compute_shift_ppd.jl
#
using Revise
using BayesianFootball
using DataFrames
using Dates

"""
    parse_duration(tags::Vector{String})

Helper to find a "time:..." tag and convert it to total seconds.
Supports formats like "43m 11s", "1h 5m", or just "10s".
"""
function parse_duration(tags::Vector{String})
    # Find the tag that starts with "time:"
    time_idx = findfirst(t -> startswith(t, "time:"), tags)
    isnothing(time_idx) && return 0.0
    
    time_str = replace(tags[time_idx], "time:" => "")
    
    seconds = 0.0
    # Match hours, minutes, and seconds using Regex
    m_h = match(r"(\d+)h", time_str)
    m_m = match(r"(\d+)m", time_str)
    m_s = match(r"(\d+)s", time_str)
    
    if !isnothing(m_h); seconds += parse(Float64, m_h.captures[1]) * 3600; end
    if !isnothing(m_m); seconds += parse(Float64, m_m.captures[1]) * 60;   end
    if !isnothing(m_s); seconds += parse(Float64, m_s.captures[1]);        end
    
    return seconds
end

function load_same_large_experiment_model(target_exp; dir="exp/ablation_study", data_dir="./data")
    # 1. List folders
    saved_folders = Experiments.list_experiments(dir; data_dir=data_dir)
    
    # 2. Load matching experiments
    matching_results = []
    
    for folder in saved_folders
        try
            res = Experiments.load_experiment(folder)
            # Check if the model type/structure matches
            if res.config.model == target_exp.config.model
                push!(matching_results, res)
            end
        catch e
            @warn "Could not load $folder: $e"
        end
    end

    if isempty(matching_results)
        error("No matching experiments found for model: $(target_exp.config.model)")
    end

    # 3. Find the one with the maximum time tag
    # argmax returns the element for which the function returns the highest value
    best_res = argmax(r -> parse_duration(r.config.tags), matching_results)
    
    # Optional: Print out what we found for confirmation
    best_time = parse_duration(best_res.config.tags)
    println("Selected Model: $(best_res.config.name) with duration $(best_time)s")
    
    return best_res
end



# ---- part 2 --- 
# compute the shift logic dic for the markets 

using DataFrames
using GLM
using Statistics

"""
Helper: Calculates the logit shift for a single market selection using GLM.
"""
function calculate_single_shift(ppd_df::DataFrame, market_df::DataFrame, target_selection::Symbol)
    println("--- Calibrating: :$target_selection ---")
    
    sub_df = filter(:selection => ==(target_selection), ppd_df)
    calib_df = innerjoin(market_df, sub_df, on = [:match_id, :market_name, :market_line, :selection])
    
    if isempty(calib_df)
        @warn "No matching market data found for :$target_selection. Returning 0.0 shift."
        return 0.0
    end

    calib_df.actual = Float64.(calib_df.is_winner)
    calib_df.mean_prob = [mean(dist) for dist in calib_df.distribution]
    
    eps = 1e-6
    clamped = clamp.(calib_df.mean_prob, eps, 1.0 - eps)
    calib_df.logit_prob = log.(clamped ./ (1.0 .- clamped))
    
    model = glm(@formula(actual ~ 1), calib_df, Binomial(), LogitLink(), offset=calib_df.logit_prob)
    
    C_shift = coef(model)[1]
    println(">> Calculated Logit Shift: ", round(C_shift, digits=4))
    
    return C_shift
end

"""
Helper: Applies a logit shift to an array of MCMC probabilities, returning a new array.
"""
function apply_logit_shift(dist_array::Vector{Float64}, C_shift::Float64)
    eps = 1e-6
    clamped = clamp.(dist_array, eps, 1.0 - eps)
    logits = log.(clamped ./ (1.0 .- clamped))
    shifted_logits = logits .+ C_shift
    return 1.0 ./ (1.0 .+ Base.exp.(.-shifted_logits))
end



"""
    compute_market_shifts(exp, ds, market_data_df, target_selections)

Loads the best matching model, generates OOS predictions, and computes 
the logit shift for a provided list of market selections.
Returns a Dict{Symbol, Float64}.
"""
function compute_market_shifts(exp_m1, ds, market_data_df::DataFrame, target_selections::Vector{Symbol})
    # 1. Load best model and extract Out-Of-Sample latents
    best_match = load_same_large_experiment_model(exp_m1)
    latents = BayesianFootball.Predictions.extract_oos_predictions(ds, best_match)
    
    # 2. Generate the PPD for calibration against known results
    calib_ppd = BayesianFootball.Predictions.model_inference(latents)
    
    # 3. Build the dictionary of shifts
    shifts = Dict{Symbol, Float64}()
    for sel in target_selections
        shifts[sel] = calculate_single_shift(calib_ppd.df, market_data_df, sel)
    end
    
    return shifts
end


"""
    apply_market_shifts(ppd, shift_dict)

Takes a PPD object and a dictionary of shifts. Returns a NEW PPD object 
where the distributions have been calibrated, leaving the original untouched.
"""
function apply_market_shifts(ppd::BayesianFootball.Predictions.PPD, shift_dict::Dict{Symbol, Float64})
    # 1. Copy the DataFrame so we don't mutate the original PPD
    new_df = copy(ppd.df)
    
    # 2. Map over the rows and apply the shift if the selection exists in our dictionary
    new_df.distribution = map(eachrow(new_df)) do row
        if haskey(shift_dict, row.selection)
            shift_val = shift_dict[row.selection]
            return apply_logit_shift(row.distribution, shift_val)
        else
            # If no shift is defined for this selection, return the original MCMC array
            return row.distribution
        end
    end
    
    println("Successfully applied calibration shifts to new PPD.")
    
    # 3. Return a brand new PPD struct
    return BayesianFootball.Predictions.PPD(new_df, ppd.model, ppd.config)
end




# --- dev 3 analyis compare 
