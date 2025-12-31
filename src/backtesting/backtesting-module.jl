# src/backtesting/backtesting-module.jl 

module BackTesting

using DataFrames
using ..Data
using ..Experiments
using ..Predictions
using ..Signals
using ..Models

include("./types.jl")
include("./processor.jl")
include("./analysis.jl")


export run_backtest, BacktestLedger

end

