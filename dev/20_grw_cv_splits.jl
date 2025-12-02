using BayesianFootball
using DataFrames
using Statistics


using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 



# --- create a subset for one league for one season 
#  To improve the speed and sampling 
#  Add match week 

data_store = BayesianFootball.Data.load_default_datastore()
# ds = BayesianFootball.load_scottish_data("24/25", split_week=14)

filtered_matches = subset(data_store.matches,
                          :season => ByRow(isequal("24/25")),
                          :tournament_id => ByRow(isequal(54))
                          )

matches_df = BayesianFootball.Data.add_match_week_column(filtered_matches)


# create half way through the season
matches_df.split_col = max.(0, matches_df.match_week .- 20);

unique(matches_df.split_col)
combine(
  groupby(matches_df, :split_col), 
  nrow
)



odds_subset = semijoin(data_store.odds, matches_df, on = :match_id)
incidents_subset = semijoin(data_store.incidents, matches_df, on = :match_id)

# Create the DataStore
ds = BayesianFootball.Data.DataStore(
    matches_df,
    odds_subset,
    incidents_subset
)


model= BayesianFootball.Models.PreGame.GRWPoisson()

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)

splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :split_col, :sequential) #

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

# here we need to remove the first week since we need 2 week for the dynamic process
fs_modded = feature_sets[2:end]

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=3) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=1000, n_chains=2, n_warmup=500) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, fs_modded)


using JLD2

JLD2.save_object("grw_debug_results.jld2", results)

results = JLD2.load_object("grw_debug_results.jld2")

r = results[5][1]




split_col_sym = :split_col
all_split = sort(unique(ds.matches[!, split_col_sym]))
prediction_split_keys = all_split[3:end] 
grouped_matches = groupby(ds.matches, split_col_sym)

dfs_to_predict = [
    grouped_matches[(; split_col_sym => key)] 
    for key in prediction_split_keys
]

# this is not working
all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict, 
    vocabulary,
    results[1:10]
)
"""
julia> all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
           model,
           dfs_to_predict, 
           vocabulary,
           results[1:10]
       )
ERROR: TaskFailedException

    nested task error: DimensionMismatch: array could not be broadcast to match destination
    Stacktrace:
      [1] check_broadcast_shape
        @ ./broadcast.jl:559 [inlined]
      [2] check_broadcast_axes
        @ ./broadcast.jl:562 [inlined]
      [3] instantiate
        @ ./broadcast.jl:316 [inlined]
      [4] materialize!
        @ ./broadcast.jl:905 [inlined]
      [5] materialize!
        @ ./broadcast.jl:902 [inlined]
      [6] unwrap_ntuple(tuple_of_arrays::NTuple{12, AxisArrays.AxisMatrix{Float64, Matrix{…}, Tuple{…}}})
        @ BayesianFootball.Models.PreGame.Implementations ~/bet_project/BayesianFootball/src/models/pregame/models-src/grw-poisson.jl:156
      [7] extract_parameters(model::BayesianFootball.Models.PreGame.Implementations.GRWPoisson, df_to_predict::SubDataFrame{…}, vocabulary::Vocabulary, chains::MCMCChains.Chains{…})
        @ BayesianFootball.Models.PreGame.Implementations ~/bet_project/BayesianFootball/src/models/pregame/models-src/grw-poisson.jl:187
      [8] macro expansion
        @ ~/bet_project/BayesianFootball/src/models/pregame/pregame-module.jl:137 [inlined]
      [9] (::BayesianFootball.Models.PreGame.Implementations.var"#extract_parameters##0#extract_parameters##1"{…})(tid::Int64; onethread::Bool)
        @ BayesianFootball.Models.PreGame.Implementations ./threadingconstructs.jl:276
     [10] #extract_parameters##0
        @ ./threadingconstructs.jl:243 [inlined]
     [11] (::Base.Threads.var"#threading_run##0#threading_run##1"{…})()
        @ Base.Threads ./threadingconstructs.jl:177

...and 7 more exceptions.

Stacktrace:
 [1] threading_run(fun::BayesianFootball.Models.PreGame.Implementations.var"#extract_parameters##0#extract_parameters##1"{…}, static::Bool)
   @ Base.Threads ./threadingconstructs.jl:196
 [2] macro expansion
   @ ./threadingconstructs.jl:213 [inlined]
 [3] extract_parameters(model::BayesianFootball.Models.PreGame.Implementations.GRWPoisson, dfs_to_predict::Vector{…}, vocabulary::Vocabulary, results_vector::Vector{…})
   @ BayesianFootball.Models.PreGame.Implementations ~/bet_project/BayesianFootball/src/models/pregame/pregame-module.jl:129
 [4] top-level scope
   @ REPL[64]:1
Some type information was truncated. Use `show(err)` to see complete types.

"""

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

r = results[9][1]

mp = subset(ds.matches, :split_col => ByRow(isequal(10)))

row = mp[1, :]

m_e = BayesianFootball.Models.PreGame.extract_parameters(model, mp, vocabulary, r)

##### 
# de bug unwrap_ntuple 

params = get(r, [:home_adv, :σ_att, :σ_def, :z_att_init, :z_def_init, :z_att_steps, :z_def_steps]);

BayesianFootball.Models.PreGame.Implementations.unwrap_ntuple(params.z_att_init)
"""
julia> BayesianFootball.Models.PreGame.Implementations.unwrap_ntuple(params.z_att_init)
ERROR: DimensionMismatch: array could not be broadcast to match destination
Stacktrace:
 [1] check_broadcast_shape
   @ ./broadcast.jl:559 [inlined]
 [2] check_broadcast_axes
   @ ./broadcast.jl:562 [inlined]
 [3] instantiate
   @ ./broadcast.jl:316 [inlined]
 [4] materialize!
   @ ./broadcast.jl:905 [inlined]
 [5] materialize!
   @ ./broadcast.jl:902 [inlined]
 [6] unwrap_ntuple(tuple_of_arrays::NTuple{12, AxisArrays.AxisMatrix{Float64, Matrix{Float64}, Tuple{AxisArrays.Axis{…}, AxisArrays.Axis{…}}}})
   @ BayesianFootball.Models.PreGame.Implementations ~/bet_project/BayesianFootball/src/models/pregame/models-src/grw-poisson.jl:156
 [7] top-level scope
   @ REPL[91]:1
Some type information was truncated. Use `show(err)` to see complete types.

"""

tuple_of_arrays = params.z_att_init ;

n_features = length(tuple_of_arrays) # 12 

n_samples = length(tuple_of_arrays[1]) # 2_000


out = Matrix{Float64}(undef, n_features, n_samples)


for (i, arr) in enumerate(tuple_of_arrays)
    # We copy the data from the AxisArray 'arr' into the i-th row of 'out'
    # 'vec(arr)' creates a copy, so we just iterate/broadcast
    out[i, :] .= arr
end
"""
julia> tuple_of_arrays = params.z_att_init ;

julia> n_features = length(tuple_of_arrays) # 12 
12

julia> n_samples = length(tuple_of_arrays[1]) # 2_000
2000

julia> for (i, arr) in enumerate(tuple_of_arrays)
           # We copy the data from the AxisArray 'arr' into the i-th row of 'out'
           # 'vec(arr)' creates a copy, so we just iterate/broadcast
           out[i, :] .= arr
       end
ERROR: DimensionMismatch: array could not be broadcast to match destination
Stacktrace:
 [1] check_broadcast_shape
   @ ./broadcast.jl:559 [inlined]
 [2] check_broadcast_axes
   @ ./broadcast.jl:562 [inlined]
 [3] instantiate
   @ ./broadcast.jl:316 [inlined]
 [4] materialize!
   @ ./broadcast.jl:905 [inlined]
 [5] materialize!(dest::SubArray{…}, bc::Base.Broadcast.Broadcasted{…})
   @ Base.Broadcast ./broadcast.jl:902
 [6] top-level scope
   @ ./REPL[95]:4
Some type information was truncated. Use `show(err)` to see complete types.
"""
for (i, arr) in enumerate(tuple_of_arrays)
        # THE FIX:
        # parent(arr) -> Get raw Matrix (2000x1)
        # vec(...)    -> Create 1D View (2000)
        # .=          -> Broadcast copy into pre-allocated row
        out[i, :] .= vec(parent(arr))
end

out

typeof(tuple_of_arrays)
"""
julia> typeof(tuple_of_arrays)
NTuple{12, AxisArrays.AxisMatrix{Float64, Matrix{Float64}, Tuple{AxisArrays.Axis{:iter, StepRange{Int64, Int64}}, AxisArrays.Axis{:chain, UnitRange{Int64}}}
}}

"""

##################################################################################################

""""
old, pre Implementation into the code base

"""

##################################################################################################

using Revise
using BayesianFootball
using DataFrames
using Statistics


# --- HPC OPTIMIZATION START ---
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 



# --- create a subset for one league for one season 
#  To improve the speed and sampling 
#  Add match week 

data_store = BayesianFootball.Data.load_default_datastore()
# ds = BayesianFootball.load_scottish_data("24/25", split_week=14)

filtered_matches = subset(data_store.matches,
                          :season => ByRow(isequal("24/25")),
                          :tournament_id => ByRow(isequal(54))
                          )

matches_df = BayesianFootball.Data.add_match_week_column(filtered_matches)

using Dates 

time_step_summary = combine(
    groupby(matches_df, :match_week),
    nrow => :number_of_matches,
    :match_date => minimum => :start_date,
    # This stores the list of rounds as a vector [1, 2] instead of splitting rows
    :round => (x -> Ref(unique(x))) => :rounds_included 
)


"""
Here limit to one league to improve the sampling speed and reduce the compleity of the data 
"""
odds_subset = semijoin(data_store.odds, matches_df, on = :match_id)
incidents_subset = semijoin(data_store.incidents, matches_df, on = :match_id)

# Create the DataStore
ds = BayesianFootball.Data.DataStore(
    matches_df,
    odds_subset,
    incidents_subset
)




model= BayesianFootball.Models.PreGame.GRWPoisson()

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)

splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :match_week, :sequential) #

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

# here we need to remove the first week since we need 2 week for the dynamic process
fs_modded = feature_sets[2:end]

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=8) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=1, n_warmup=10) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)

results = BayesianFootball.Training.train(model, training_config, fs_modded)


results

###
all_split = sort(unique(ds.matches[!, :match_week]))

prediction_split_keys = all_split[3:end] 


grouped_matches = groupby(ds.matches, :match_week)


dfs_to_predict = [
    grouped_matches[(; :match_week => key)] 
    for key in prediction_split_keys
]

# this is not working
all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict, 
    vocabulary,
    results
)
""" yields the following error: 
julia> model= BayesianFootball.Models.PreGame.GRWPoisson()
BayesianFootball.Models.PreGame.Implementations.GRWPoisson()

julia> all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
           model,
           dfs_to_predict,
           vocabulary,
           results
       )
ERROR: MethodError: no method matching extract_parameters(::BayesianFootball.Models.PreGame.Implementations.GRWPoisson, ::Vector{…}, ::Vocabulary, ::Vector{…})
The function `extract_parameters` exists, but no method is defined for this combination of argument types.

Closest candidates are:
  extract_parameters(::BayesianFootball.Models.PreGame.Implementations.StaticPoisson, ::AbstractVector, ::Vocabulary, ::AbstractVector)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/models/pregame/pregame-module.jl:46
  extract_parameters(::BayesianFootball.Models.PreGame.Implementations.StaticDixonColes, ::AbstractVector, ::Vocabulary, ::AbstractVector)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/models/pregame/pregame-module.jl:77
  extract_parameters(::BayesianFootball.Models.PreGame.Implementations.GRWPoisson, ::AbstractDataFrame, ::Vocabulary, ::MCMCChains.Chains)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/models/pregame/models-src/grw-poisson.jl:145
  ...

Stacktrace:
 [1] top-level scope
   @ REPL[222]:1
Some type information was truncated. Use `show(err)` to see complete types.

julia>


"""



predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )

""" 
results[<iterations over the cv fold>] [ <1 - mcmc data, 2 - round number str] 

so results[1][1] is the mcmc data for round_2 - since we trimed it to remove round 1 as need > 1 rounds to train the grw. 

results[1] -> ( to predict) round 3 ( round we mean split_col / match_week 
- i dont think that is work yet the extraction function for this

"""


#####
# dev space 1 
#####
using MCMCChains

"""
Helper: Converts the NTuple of AxisArrays returned by `get(chain, :x)`
into a standard Matrix [Dimension, Samples]
"""
function unwrap_ntuple(tuple_of_arrays)
    # 1. Convert tuple elements to standard Vectors (strips Axis info)
    #    result: Vector of Vectors [[s1, s2...], [s1, s2...]]
    vectors = [vec(x) for x in tuple_of_arrays]
    
    # 2. Stack horizontally: Result is Matrix [Samples, Parameters]
    mat_samples_rows = reduce(hcat, vectors)
    
    # 3. Transpose: Result is Matrix [Parameters, Samples]
    #    Now row 1 is Team 1, Row 2 is Team 2, etc.
    return permutedims(mat_samples_rows, (2, 1))
end
params = get(chain_r2, [:home_adv, :σ_att, :σ_def, :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

params.home_adv
home_adv_vec = vec(params.home_adv)
σ_att_vec    = vec(params.σ_att)
σ_def_vec    = vec(params.σ_def)

Z_att_init = unwrap_ntuple(params.z_att_init)
Z_def_init = unwrap_ntuple(params.z_def_init)


Z_att_steps_raw = unwrap_ntuple(params.z_att_steps)
Z_def_steps_raw = unwrap_ntuple(params.z_def_steps)

# 3. Handle Dimensions
n_samples = length(σ_att_vec)
n_teams   = size(Z_att_init, 1) # Rows = Teams


# Calculate time steps
# Total flattened items / n_teams = n_steps
n_steps   = div(size(Z_att_steps_raw, 1), n_teams)

# 4. Reshape Steps to [Team, Time, Sample]
Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_steps, n_samples)
Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_steps, n_samples)


###
sum_z_steps_att = dropdims(sum(Z_att_steps, dims=2), dims=2)
sum_z_steps_def = dropdims(sum(Z_def_steps, dims=2), dims=2)

# 2. Reshape Sigmas for Broadcasting
# Currently [Samples] -> Make it [1, Samples] to broadcast across Teams
σ_att_row = reshape(σ_att_vec, 1, :) 
σ_def_row = reshape(σ_def_vec, 1, :)

# 3. Calculate Final Raw States (Vectorized)
# Math: Init*0.5 + (Total_Z_Steps * Sigma)
# Operations are now performed on the WHOLE matrix at once (BLAS optimized)

raw_att_T = (Z_att_init .* 0.5) .+ (sum_z_steps_att .* σ_att_row)
raw_def_T = (Z_def_init .* 0.5) .+ (sum_z_steps_def .* σ_def_row)

# 4. Center (Zero-Sum Constraint)
# Calculate mean across Teams (dims=1) and subtract
final_att_speed = raw_att_T .- mean(raw_att_T, dims=1)
final_def_speed = raw_def_T .- mean(raw_def_T, dims=1)

final_def .- final_att_speed



# 5. Reconstruct FINAL state (Time T)
final_att = zeros(n_teams, n_samples)
final_def = zeros(n_teams, n_samples)

for s in 1:n_samples
        σ_a = σ_att_vec[s]
        σ_d = σ_def_vec[s]

        # Extract paths for this sample 's'
        # Z_att_steps[:, :, s] is [Team, Time]
        steps_att = Z_att_steps[:, :, s] .* σ_a
        steps_def = Z_def_steps[:, :, s] .* σ_d

        # Combine Init (scaled 0.5) + Steps
        full_path_att = hcat(Z_att_init[:, s] .* 0.5, steps_att)
        full_path_def = hcat(Z_def_init[:, s] .* 0.5, steps_def)

        # Cumsum & take last column
        raw_att_end = sum(full_path_att, dims=2)
        raw_def_end = sum(full_path_def, dims=2)

        # Center
        final_att[:, s] = raw_att_end .- mean(raw_att_end)
        final_def[:, s] = raw_def_end .- mean(raw_def_end)
    end



att_h = final_att[h_id, :]
att_a = final_att[a_id, :]
def_h = final_def[h_id, :]
def_a = final_def[a_id, :]

λ_h = exp.(att_h .+ def_a .+ home_adv_vec)
λ_a = exp.(att_a .+ def_h)


""" compare slow and fast 

""" 

using Statistics, BenchmarkTools, LinearAlgebra

# --- 1. Fake Data Generation ---
n_teams = 12
n_samples = 100
n_steps = 10

# Random inputs similar to your Chains
σ_vec = rand(n_samples)                 # Vector
Z_init = randn(n_teams, n_samples)      # Matrix [Team, Sample]
Z_steps = randn(n_teams, n_steps, n_samples) # 3D [Team, Time, Sample]

# --- 2. The SLOW Way (Loop) ---
function run_slow(Z_init, Z_steps, σ_vec)
    n_t, n_s = size(Z_init)
    out = zeros(n_t, n_s)
    for s in 1:n_s
        σ = σ_vec[s]
        # Loop over steps
        steps = Z_steps[:, :, s] .* σ
        full_path = hcat(Z_init[:, s] .* 0.5, steps)
        raw = sum(full_path, dims=2)
        out[:, s] = raw .- mean(raw)
    end
    return out
end

# --- 3. The FAST Way (Vectorized) ---
function run_fast(Z_init, Z_steps, σ_vec)
    # Sum steps over time first
    sum_steps = dropdims(sum(Z_steps, dims=2), dims=2) 
    
    # Broadcast Sigma [1, Samples]
    σ_row = reshape(σ_vec, 1, :)
    
    # Vectorized Math
    raw = (Z_init .* 0.5) .+ (sum_steps .* σ_row)
    
    # Vectorized Centering
    return raw .- mean(raw, dims=1)
end

# --- 4. Validation ---
println("Running Validation...")
res_slow = run_slow(Z_init, Z_steps, σ_vec)
res_fast = run_fast(Z_init, Z_steps, σ_vec)

# Check difference
err = norm(res_slow - res_fast)
println("Total Difference between Slow and Fast: $err")

if err < 1e-12
    println("✅ SUCCESS: Vectorization Logic is Accurate.")
else
    println("❌ ERROR: Logic mismatch.")
end

# --- 5. Benchmark ---
println("\nBenchmarking Speed...")
println("Slow Loop:")
@btime run_slow($Z_init, $Z_steps, $σ_vec)
println("Fast Vectorized:")
@btime run_fast($Z_init, $Z_steps, $σ_vec)


"""
julia> @btime run_slow($Z_init, $Z_steps, $σ_vec)
  25.217 μs (1403 allocations: 392.27 KiB)
julia> println("Fast Vectorized:")

Fast Vectorized:
julia> @btime run_fast($Z_init, $Z_steps, $σ_vec)
  6.454 μs (19 allocations: 29.79 KiB)
"""


""" 
 Lets put it together - the extract functions 
"""

# --- Helper Function ---
"""
Helper: Converts the NTuple of AxisArrays returned by `get(chain, :x)`
into a standard Matrix [Dimension, Samples]
"""
function unwrap_ntuple(tuple_of_arrays)
    # 1. Convert tuple elements to standard Vectors (strips Axis info)
    vectors = [vec(x) for x in tuple_of_arrays]
    
    # 2. Stack horizontally: Result is Matrix [Samples, Parameters]
    mat_samples_rows = reduce(hcat, vectors)
    
    # 3. Transpose: Result is Matrix [Parameters, Samples]
    return permutedims(mat_samples_rows, (2, 1))
end

# --- Main Extraction Function ---
function extract_parameters_v1(model::BayesianFootball.Models.PreGame.GRWPoisson, df_to_predict::AbstractDataFrame, vocabulary::BayesianFootball.TypesInterfaces.Vocabulary, chains::Chains)
    
    # 1. Fetch Parameters
    # Using 'get' is safer than 'group' for reconstructing indices
    params = get(chains, [:home_adv, :σ_att, :σ_def, :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

    # 2. Unwrap Variables
    # Scalars (get returns a tuple of length 1 for scalars)
    home_adv_vec = vec(params.home_adv)
    σ_att_vec    = vec(params.σ_att)
    σ_def_vec    = vec(params.σ_def)

    # Vectors & Matrices
    # unwrap_ntuple returns [Features, Samples]
    Z_att_init = unwrap_ntuple(params.z_att_init)
    Z_def_init = unwrap_ntuple(params.z_def_init)
    
    Z_att_steps_raw = unwrap_ntuple(params.z_att_steps)
    Z_def_steps_raw = unwrap_ntuple(params.z_def_steps)

    # 3. Handle Dimensions
    n_samples = length(σ_att_vec)
    n_teams   = size(Z_att_init, 1)
    
    # Calculate time steps (Total flattened items / n_teams)
    n_steps   = div(size(Z_att_steps_raw, 1), n_teams)

    # Reshape Steps to [Team, Time, Sample]
    Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_steps, n_samples)
    Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_steps, n_samples)

    # --- VECTORIZED RECONSTRUCTION (The Fast Part) ---
    
    # A. Sum Z-steps over time (Collapse Dim 2)
    # Result: [Teams, Samples]
    sum_z_att = dropdims(sum(Z_att_steps, dims=2), dims=2)
    sum_z_def = dropdims(sum(Z_def_steps, dims=2), dims=2)

    # B. Reshape Sigmas to [1, Samples] for broadcasting
    σ_att_row = reshape(σ_att_vec, 1, :)
    σ_def_row = reshape(σ_def_vec, 1, :)

    # C. Calculate Final Strengths
    # Math: Init*0.5 + (Total_Z_Steps * Sigma)
    raw_att = (Z_att_init .* 0.5) .+ (sum_z_att .* σ_att_row)
    raw_def = (Z_def_init .* 0.5) .+ (sum_z_def .* σ_def_row)

    # D. Center columns (Zero-Sum Constraint)
    final_att = raw_att .- mean(raw_att, dims=1)
    final_def = raw_def .- mean(raw_def, dims=1)

    # --- PREDICTION LOOP ---
    
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    team_map = vocabulary.mappings[:team_map]

    for row in eachrow(df_to_predict)
        if !haskey(team_map, row.home_team) || !haskey(team_map, row.away_team)
            continue
        end

        h_id = team_map[row.home_team]
        a_id = team_map[row.away_team]

        # Slice columns from our pre-calculated matrices
        att_h = final_att[h_id, :]
        att_a = final_att[a_id, :]
        def_h = final_def[h_id, :]
        def_a = final_def[a_id, :]

        # Calculate Rates
        λ_h = exp.(att_h .+ def_a .+ home_adv_vec)
        λ_a = exp.(att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end


function extract_parameters_v2(model::BayesianFootball.Models.PreGame.GRWPoisson, df_to_predict::AbstractDataFrame, vocabulary::BayesianFootball.TypesInterfaces.Vocabulary, chains::Chains)
    
    # 1. Fetch Parameters
    # Using 'get' is safer than 'group' for reconstructing indices
    params = get(chains, [:home_adv, :σ_att, :σ_def, :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

    # 2. Unwrap Variables
    # Scalars (get returns a tuple of length 1 for scalars)
    home_adv_vec = vec(params.home_adv)
    σ_att_vec    = vec(params.σ_att)
    σ_def_vec    = vec(params.σ_def)

    # Vectors & Matrices
    # unwrap_ntuple returns [Features, Samples]
    Z_att_init = unwrap_ntuple(params.z_att_init)
    Z_def_init = unwrap_ntuple(params.z_def_init)
    
    Z_att_steps_raw = unwrap_ntuple(params.z_att_steps)
    Z_def_steps_raw = unwrap_ntuple(params.z_def_steps)

    # 3. Handle Dimensions
    n_samples = length(σ_att_vec)
    n_teams   = size(Z_att_init, 1)
    
    # Calculate time steps (Total flattened items / n_teams)
    n_steps   = div(size(Z_att_steps_raw, 1), n_teams)

    # Reshape Steps to [Team, Time, Sample]
    Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_steps, n_samples)
    Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_steps, n_samples)

    # --- VECTORIZED RECONSTRUCTION (The Fast Part) ---
    
    # A. Sum Z-steps over time (Collapse Dim 2)
    # Result: [Teams, Samples]
    sum_z_att = dropdims(sum(Z_att_steps, dims=2), dims=2)
    sum_z_def = dropdims(sum(Z_def_steps, dims=2), dims=2)

    # B. Reshape Sigmas to [1, Samples] for broadcasting
    σ_att_row = reshape(σ_att_vec, 1, :)
    σ_def_row = reshape(σ_def_vec, 1, :)

    # C. Calculate Final Strengths
    # Math: Init*0.5 + (Total_Z_Steps * Sigma)
    raw_att = (Z_att_init .* 0.5) .+ (sum_z_att .* σ_att_row)
    raw_def = (Z_def_init .* 0.5) .+ (sum_z_def .* σ_def_row)

    # D. Center columns (Zero-Sum Constraint)
    final_att = raw_att .- mean(raw_att, dims=1)
    final_def = raw_def .- mean(raw_def, dims=1)

    # --- PREDICTION LOOP ---
    # --- VECTORIZED PREDICTION ---
    team_map = vocabulary.mappings[:team_map]

    # 1. Pre-calculate ID vectors (Map String -> Int)
    # We filter valid rows first to avoid 'missing key' errors during mapping
    valid_rows = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), df_to_predict)
    
    if nrow(valid_rows) == 0
        return Dict{Int64, NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}}()
    end

    h_ids = [team_map[r.home_team] for r in eachrow(valid_rows)]
    a_ids = [team_map[r.away_team] for r in eachrow(valid_rows)]

    # 2. Batch Indexing
    # Resulting Shape: [N_Matches, N_Samples]
    # Julia allows indexing a matrix with a vector of rows: Matrix[vector_indices, :]
    att_h_mat = final_att[h_ids, :]
    def_a_mat = final_def[a_ids, :]
    att_a_mat = final_att[a_ids, :]
    def_h_mat = final_def[h_ids, :]

    # 3. Reshape Home Advantage for Broadcasting
    # Currently [N_Samples] -> make it [1, N_Samples] to broadcast down the matches
    home_adv_row = reshape(home_adv_vec, 1, :)

    # 4. Vectorized Math (BLAS Optimized)
    # This computes exp() for thousands of numbers in one compiled kernel call
    λ_h_mat = exp.(att_h_mat .+ def_a_mat .+ home_adv_row)
    λ_a_mat = exp.(att_a_mat .+ def_h_mat)

    # 5. Packaging Results
    # We still need a loop to stuff results into the Dictionary (Dicts are inherently sequential),
    # but the heavy math is already done.
    
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    
    # We iterate by index now, which is faster than iterating by row structs
    match_ids = valid_rows.match_id

    for i in 1:length(match_ids)
        # Slicing the pre-calculated row
        # Vectors in Julia are column-major, so this is contiguous memory access (Fast)
        lambdas = (
            λ_h = vec(λ_h_mat[i, :]), 
            λ_a = vec(λ_a_mat[i, :])
        )
        extraction_dict[match_ids[i]] = lambdas
    end

    return extraction_dict

end



function extract_parameters_v3(model::BayesianFootball.Models.PreGame.GRWPoisson, df_to_predict::AbstractDataFrame, vocabulary::BayesianFootball.TypesInterfaces.Vocabulary, chains::Chains)
    
    # 1. Fetch Parameters
    # Using 'get' is safer than 'group' for reconstructing indices
    params = get(chains, [:home_adv, :σ_att, :σ_def, :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

    # 2. Unwrap Variables
    # Scalars (get returns a tuple of length 1 for scalars)
    home_adv_vec = vec(params.home_adv)
    σ_att_vec    = vec(params.σ_att)
    σ_def_vec    = vec(params.σ_def)

    # Vectors & Matrices
    # unwrap_ntuple returns [Features, Samples]
    Z_att_init = unwrap_ntuple(params.z_att_init)
    Z_def_init = unwrap_ntuple(params.z_def_init)
    
    Z_att_steps_raw = unwrap_ntuple(params.z_att_steps)
    Z_def_steps_raw = unwrap_ntuple(params.z_def_steps)

    # 3. Handle Dimensions
    n_samples = length(σ_att_vec)
    n_teams   = size(Z_att_init, 1)
    
    # Calculate time steps (Total flattened items / n_teams)
    n_steps   = div(size(Z_att_steps_raw, 1), n_teams)

    # Reshape Steps to [Team, Time, Sample]
    Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_steps, n_samples)
    Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_steps, n_samples)

    # --- VECTORIZED RECONSTRUCTION (The Fast Part) ---
    
    # A. Sum Z-steps over time (Collapse Dim 2)
    # Result: [Teams, Samples]
    sum_z_att = dropdims(sum(Z_att_steps, dims=2), dims=2)
    sum_z_def = dropdims(sum(Z_def_steps, dims=2), dims=2)

    # B. Reshape Sigmas to [1, Samples] for broadcasting
    σ_att_row = reshape(σ_att_vec, 1, :)
    σ_def_row = reshape(σ_def_vec, 1, :)

    # C. Calculate Final Strengths
    # Math: Init*0.5 + (Total_Z_Steps * Sigma)
    raw_att = (Z_att_init .* 0.5) .+ (sum_z_att .* σ_att_row)
    raw_def = (Z_def_init .* 0.5) .+ (sum_z_def .* σ_def_row)

    # D. Center columns (Zero-Sum Constraint)
    final_att = raw_att .- mean(raw_att, dims=1)
    final_def = raw_def .- mean(raw_def, dims=1)

    # --- PREDICTION LOOP ---
    # --- VECTORIZED PREDICTION ---
    team_map = vocabulary.mappings[:team_map]

    # 1. Pre-calculate ID vectors (Map String -> Int)
    # We filter valid rows first to avoid 'missing key' errors during mapping
    valid_rows = filter(row -> haskey(team_map, row.home_team) && haskey(team_map, row.away_team), df_to_predict)
    
    if nrow(valid_rows) == 0
        return Dict{Int64, NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}}()
    end

    h_ids = [team_map[r.home_team] for r in eachrow(valid_rows)]
    a_ids = [team_map[r.away_team] for r in eachrow(valid_rows)]

    # 2. Batch Indexing
    # Resulting Shape: [N_Matches, N_Samples]
    # Julia allows indexing a matrix with a vector of rows: Matrix[vector_indices, :]
    att_h_mat = final_att[h_ids, :]
    def_a_mat = final_def[a_ids, :]
    att_a_mat = final_att[a_ids, :]
    def_h_mat = final_def[h_ids, :]

    # 3. Reshape Home Advantage for Broadcasting
    # Currently [N_Samples] -> make it [1, N_Samples] to broadcast down the matches
    home_adv_row = reshape(home_adv_vec, 1, :)

    # 4. Vectorized Math (BLAS Optimized)
    # This computes exp() for thousands of numbers in one compiled kernel call
    λ_h_mat = exp.(att_h_mat .+ def_a_mat .+ home_adv_row)
    λ_a_mat = exp.(att_a_mat .+ def_h_mat)

    # 5. Packaging Results
    # We still need a loop to stuff results into the Dictionary (Dicts are inherently sequential),
    # but the heavy math is already done.
    
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    extraction_dict = Dict{Int64, ExtractionValue}()
    
    # We iterate by index now, which is faster than iterating by row structs
    match_ids = valid_rows.match_id

    return DataFrame(
        match_id = match_ids,
        λ_h = [vec(r) for r in eachrow(λ_h_mat)],
        λ_a = [vec(r) for r in eachrow(λ_a_mat)]
    )

end


function extract_parameters_v4(model::BayesianFootball.Models.PreGame.GRWPoisson, df_to_predict::AbstractDataFrame, vocabulary::BayesianFootball.TypesInterfaces.Vocabulary, chains::Chains)
    
    # --- STEP 1: Fast Vectorized Reconstruction (Keep this!) ---
    params = get(chains, [:home_adv, :σ_att, :σ_def, :z_att_init, :z_def_init, :z_att_steps, :z_def_steps])

    home_adv_vec = vec(params.home_adv)
    σ_att_vec    = vec(params.σ_att)
    σ_def_vec    = vec(params.σ_def)

    Z_att_init = unwrap_ntuple(params.z_att_init)
    Z_def_init = unwrap_ntuple(params.z_def_init)
    Z_att_steps_raw = unwrap_ntuple(params.z_att_steps)
    Z_def_steps_raw = unwrap_ntuple(params.z_def_steps)

    n_samples = length(σ_att_vec)
    n_teams   = vocabulary.mappings[:n_teams]
    n_steps   = div(size(Z_att_steps_raw, 1), n_teams)

    Z_att_steps = reshape(Z_att_steps_raw, n_teams, n_steps, n_samples)
    Z_def_steps = reshape(Z_def_steps_raw, n_teams, n_steps, n_samples)

    # 1. Sum Z-steps (Vectorized)
    sum_z_att = dropdims(sum(Z_att_steps, dims=2), dims=2)
    sum_z_def = dropdims(sum(Z_def_steps, dims=2), dims=2)

    # 2. Reshape Sigmas
    σ_att_row = reshape(σ_att_vec, 1, :)
    σ_def_row = reshape(σ_def_vec, 1, :)

    # 3. Calculate Final Strengths (Vectorized)
    raw_att = (Z_att_init .* 0.5) .+ (sum_z_att .* σ_att_row)
    raw_def = (Z_def_init .* 0.5) .+ (sum_z_def .* σ_def_row)

    # 4. Center columns
    final_att = raw_att .- mean(raw_att, dims=1)
    final_def = raw_def .- mean(raw_def, dims=1)

    # --- STEP 2: Optimized Loop with Views ---
    
    ExtractionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    # Pre-allocate Dictionary size to avoid resizing overhead
    extraction_dict = Dict{Int64, ExtractionValue}()
    sizehint!(extraction_dict, nrow(df_to_predict))
    
    team_map = vocabulary.mappings[:team_map]

    for row in eachrow(df_to_predict)
        h_team = row.home_team
        a_team = row.away_team

        # Fast lookup with default to avoid try/catch overhead
        h_id = get(team_map, h_team, 0)
        a_id = get(team_map, a_team, 0)

        # --- THE OPTIMIZATION ---
        # @views ensures we DO NOT copy the columns from final_att.
        # We just point to them.
        att_h = @view final_att[h_id, :]
        def_a = @view final_def[a_id, :]
        att_a = @view final_att[a_id, :]
        def_h = @view final_def[h_id, :]

        # Broadcasting results into a new vector (Payload)
        # We do this directly, skipping intermediate 'matrix' creation
        λ_h = exp.(att_h .+ def_a .+ home_adv_vec)
        λ_a = exp.(att_a .+ def_h)

        extraction_dict[Int(row.match_id)] = (; λ_h, λ_a)
    end

    return extraction_dict
end






@btime extract_parameters_v1(model, mp, vocabulary, chain_r2)
extract_parameters_v2(model, mp, vocabulary, chain_r2)
@btime extract_parameters_v2(model, mp, vocabulary, chain_r2)


extract_parameters_v3(model, mp, vocabulary, chain_r2)
@btime extract_parameters_v3(model, mp, vocabulary, chain_r2)

@btime extract_parameters_v4(model, mp, vocabulary, chain_r2)


""" btime 
extract_parameters_v1: (byrow)
140.925 μs (1848 allocations: 412.88 KiB)  
141.305 μs (1848 allocations: 412.88 KiB)

extract_parameters_v2: ( vectorised byrow)
157.786 μs (1888 allocations: 425.20 KiB)
157.365 μs (1888 allocations: 425.20 KiB)


extract_parameters_v3:  ( v2 with returing df instead of dict) 
155.041 μs (1858 allocations: 415.55 KiB)
156.243 μs (1858 allocations: 415.55 KiB)

extract_parameters_v4: 
139.762 μs (1812 allocations: 393.56 KiB)
140.043 μs (1812 allocations: 393.56 KiB)



"""


######


function extract_parameters(
    model::BayesianFootball.Models.PreGame.GRWPoisson,
    dfs_to_predict::AbstractVector, 
    vocabulary::BayesianFootball.TypesInterfaces.Vocabulary,
    results_vector::AbstractVector
)

    # 1. Define the return type (Same as StaticPoisson)
    PredictionValue = NamedTuple{(:λ_h, :λ_a), Tuple{AbstractVector{Float64}, AbstractVector{Float64}}}

    # 2. Allocate output dictionary
    full_extraction_dict = Dict{Int64, PredictionValue}()

    # 3. Iterate in parallel
    # Note: This assumes results_vector[i] corresponds to the model trained *before* dfs_to_predict[i]
    for (result_tuple, df_for_this_split) in zip(results_vector, dfs_to_predict)
        
        # result_tuple is (Chains, "Label string")
        chains = result_tuple[1]

        # 4. Call the INNER function (defined in grw-poisson.jl)
        single_split_dict = extract_parameters_v4(model, df_for_this_split, vocabulary, chains)

        # 5. Merge
        merge!(full_extraction_dict, single_split_dict)
    end
    
    return full_extraction_dict
end

function extract_parameters_parallel(
    model::BayesianFootball.Models.PreGame.GRWPoisson,
    dfs_to_predict::AbstractVector, 
    vocabulary::BayesianFootball.TypesInterfaces.Vocabulary,
    results_vector::AbstractVector
)

    # 1. Define Types
    PredictionValue = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}
    DictType = Dict{Int64, PredictionValue}

    n_splits = length(dfs_to_predict)
    
    # 2. Allocate storage for threaded results
    partial_results = Vector{DictType}(undef, n_splits)

    # 3. Parallel Map
    Threads.@threads for i in 1:n_splits
        # Safety check: ensure we have a result for this data split
        if i <= length(results_vector)
            result_tuple = results_vector[i]
            df_curr = dfs_to_predict[i]
            chains = result_tuple[1]

            # Call the optimized inner function (v4)
            partial_results[i] = extract_parameters_v4(model, df_curr, vocabulary, chains)
        end
    end

    # 4. Sequential Reduce
    total_rows = sum(nrow, dfs_to_predict)
    full_extraction_dict = DictType()
    sizehint!(full_extraction_dict, total_rows)

    # FIXED LOOP: Iterate by integer index to safely check isassigned
    for i in 1:n_splits
        if isassigned(partial_results, i)
            merge!(full_extraction_dict, partial_results[i])
        end
    end

    return full_extraction_dict
end


using Base.Threads
println("Threads available: ", Threads.nthreads())



oos_results = extract_parameters(model, dfs_to_predict, vocabulary, results)

@btime extract_parameters(model, dfs_to_predict, vocabulary, results)

@btime extract_parameters(model, $dfs_to_predict, $vocabulary, $results)

oss_r =  extract_parameters_parallel(model, $dfs_to_predict, $vocabulary, $results)

@btime extract_parameters_parallel(model, $dfs_to_predict, $vocabulary, $results)

""" btime of the wrapper extract_parameters 
v1: 
20.662 ms (247945 allocations: 67.84 MiB)  
20.715 ms (247945 allocations: 67.84 MiB)   
20.702 ms (247945 allocations: 67.84 MiB)   

v1( optimized helper) 
20.184 ms (247311 allocations: 58.73 MiB)   
20.105 ms (247311 allocations: 58.73 MiB)  


v2: ( parallel) 
7.188 ms (247987 allocations: 67.82 MiB)    
7.166 ms (247987 allocations: 67.82 MiB) 

v3:( parrallel and optimized helper) 
6.540 ms (247337 allocations: 58.71 MiB)  
6.478 ms (247337 allocations: 58.71 MiB)
"""

"""
OPTIMIZED HELPER: unwraps NTuple directly into target shape
Avoids hcat and permutedims allocations.
"""
function unwrap_ntuple(tuple_of_arrays)
    # 1. Determine Dimensions
    # tuple_of_arrays is (AxisArray_1, AxisArray_2, ...)
    n_features = length(tuple_of_arrays)
    
    # Peek at the first element to get sample count (length of the array)
    n_samples = length(tuple_of_arrays[1])
    
    # 2. Pre-allocate the FINAL Matrix [Features, Samples]
    # We want Float64, assuming that's what comes out of Turing
    out = Matrix{Float64}(undef, n_features, n_samples)
    
    # 3. Fill directly (No temporary arrays)
    for (i, arr) in enumerate(tuple_of_arrays)
        # We copy the data from the AxisArray 'arr' into the i-th row of 'out'
        # 'vec(arr)' creates a copy, so we just iterate/broadcast
        out[i, :] .= arr
    end
    
    return out
end


