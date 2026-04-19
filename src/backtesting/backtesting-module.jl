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
# include("./analysis.jl")


# metrics
include("./metrics/types.jl")
include("./metrics/interfaces.jl")

include("./metrics/implentations/cumulative_wealth.jl")

include("./metrics/implentations/sharpe.jl")
include("./metrics/implentations/calmar.jl")
include("./metrics/implentations/burke.jl")
include("./metrics/implentations/sortino.jl")
include("./metrics/implentations/sterling.jl")

include("./metrics/processing.jl")



export generate_tearsheet, AbstractWealthMetric


export run_backtest, BacktestLedger

end

