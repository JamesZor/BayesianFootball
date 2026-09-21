# l01_market_inverse_loader.jl
# Loader for Market-Inverse State-Space & Dynamic GRW Volatility Models (TODO 023)

module MarketInverseDynamics

using DataFrames, Dates, LinearAlgebra, Statistics
using Turing, ReverseDiff, Distributions
import BayesianFootball: Data, Markets, Calibration, Features

export load_market_data, extract_market_rates, MarketInverseDataset

struct MarketInverseDataset
    matches::DataFrame
    rates::DataFrame
    teams::Vector{String}
    team_to_idx::Dict{String,Int}
    match_team_indices::Matrix{Int} # [N_matches x 2] (home, away)
    log_targets::Matrix{Float64}    # [N_matches x 2] (log_lambda_h, log_lambda_a)
    time_indices::Vector{Int}       # matchday bi-week index per match
end

function load_market_data()
    ds = Data.load_datastore_cached(Data.ScottishLower())
    return ds
end

function extract_market_rates(ds)
    odds = ds.odds
    # Invert closing odds to get Poisson intensities
    raw_fits = Calibration.invert_market_rates(odds)
    rates_df = Calibration.inversion_frame(raw_fits)
    accepted_df = filter(:accepted => identity, rates_df)
    return accepted_df
end

end # module
