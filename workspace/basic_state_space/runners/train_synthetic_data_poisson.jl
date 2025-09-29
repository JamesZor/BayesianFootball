# workspace/basic_state_space/runners/train_synthetic_poisson.jl

using Turing
using Plots
using Statistics
using DataFrames

# Performance libraries
using ReverseDiff, Memoization
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# --- 1. SETUP AND INCLUDES ---

# Include our refactored modules
include("../models/ar1_poisson.jl")
using .AR1Poisson
include("../analysis/utils.jl")
using .SSMUtils
include("../analysis/plotting.jl")
using .SSMPlots

# --- 2. GENERATE SYNTHETIC DATA ---

println("Generating synthetic data...")
synth_data = generate_synthetic_data(n_teams=10, n_rounds=38)

# Convert the generated data into a DataFrame, which our model pipeline expects
# This mimics the structure of the real data
matches_df = DataFrame(
    global_round = vcat([fill(r, 5) for r in 1:synth_data.n_rounds]...), # 5 matches per round
    home_team_ids = synth_data.home_team_ids,
    away_team_ids = synth_data.away_team_ids
);

# --- 3. TRAIN THE MODEL ---

println("Training AR1 Poisson model on synthetic data...")

# Instantiate the model definition
model_def = AR1PoissonModel()

# The `build_turing_model` function expects features as a NamedTuple
features = (
    global_round = matches_df.global_round,
    home_team_ids = matches_df.home_team_ids,
    away_team_ids = matches_df.away_team_ids,
    n_teams = synth_data.n_teams
)

# Build the Turing model instance
turing_model = AR1Poisson.build_turing_model(
    model_def,
    features,
    synth_data.home_goals,
    synth_data.away_goals
)

# Sample from the model (using a small number of samples for a quick test)
chain = sample(turing_model, NUTS(0.65), 500, progress=true)

println("Training complete.")

# --- 4. ANALYZE AND VISUALIZE RESULTS ---

println("Extracting posterior samples and plotting results...")

# To visualize, we need to reconstruct the time series of alpha and beta
# This logic is identical to what's in our `extract_posterior_samples` function
# We are doing it manually here to show the process clearly.

n_samples = size(chain, 1) * size(chain, 3)
n_teams = synth_data.n_teams
n_rounds = synth_data.n_rounds

ρ_attack = vec(Array(chain[:ρ_attack]))
ρ_defense = vec(Array(chain[:ρ_defense]))
μ_log_σ_attack = vec(Array(chain[:μ_log_σ_attack]))
τ_log_σ_attack = vec(Array(chain[:τ_log_σ_attack]))
z_log_σ_attack = hcat(chain[:z_log_σ_attack]...)
σ_attack = exp.(μ_log_σ_attack .+ z_log_σ_attack' .* τ_log_σ_attack)
μ_log_σ_defense = vec(Array(chain[:μ_log_σ_defense]))
τ_log_σ_defense = vec(Array(chain[:τ_log_σ_defense]))
z_log_σ_defense = hcat(chain[:z_log_σ_defense]...)
σ_defense = exp.(μ_log_σ_defense .+ z_log_σ_defense' .* τ_log_σ_defense)

initial_α_z = hcat(chain[:initial_α_z]...)
initial_β_z = hcat(chain[:initial_β_z]...)
z_α_flat = hcat(chain[:z_α]...)
z_α_mat_reshaped = reshape(z_α_flat, n_teams, n_rounds, n_samples)
z_β_flat = hcat(chain[:z_β]...)
z_β_mat_reshaped = reshape(z_β_flat, n_teams, n_rounds, n_samples)

log_α_raw = Array{Float64, 3}(undef, n_samples, n_teams, n_rounds)
log_β_raw = Array{Float64, 3}(undef, n_samples, n_teams, n_rounds)

for s in 1:n_samples
    log_α_raw_t0 = initial_α_z[:, s] .* sqrt(0.5)
    log_β_raw_t0 = initial_β_z[:, s] .* sqrt(0.5)
    for t in 1:n_rounds
        if t == 1
            log_α_raw[s, :, 1] = log_α_raw_t0 .+ z_α_mat_reshaped[:, 1, s] .* σ_attack[s, :]
            log_β_raw[s, :, 1] = log_β_raw_t0 .+ z_β_mat_reshaped[:, 1, s] .* σ_defense[s, :]
        else
            log_α_raw[s, :, t] = ρ_attack[s] * log_α_raw[s, :, t-1] .+ z_α_mat_reshaped[:, t, s] .* σ_attack[s, :]
            log_β_raw[s, :, t] = ρ_defense[s] * log_β_raw[s, :, t-1] .+ z_β_mat_reshaped[:, t, s] .* σ_defense[s, :]
        end
    end
end

log_α_centered = similar(log_α_raw)
log_β_centered = similar(log_β_raw)
for s in 1:n_samples, t in 1:n_rounds
    log_α_centered[s, :, t] = log_α_raw[s, :, t] .- mean(log_α_raw[s, :, t])
    log_β_centered[s, :, t] = log_β_raw[s, :, t] .- mean(log_β_raw[s, :, t])
end

# Now use our plotting function to visualize the results for Team 1
# This team's attack strength was designed to improve over the season
team1_plot = plot_team_dashboard(1, synth_data, log_α_centered, log_β_centered)
display(team1_plot) # This will show the plot in the REPL or plotting pane

# Save the plot
savefig(team1_plot, "team1_synthetic_recovery.png")

println("\n✅ Script finished. Plot saved to team1_synthetic_recovery.png")
