# src/Calibration/calibration-module.jl

module Calibration

using DataFrames
using Base.Threads
using GLM
using StatsModels
using StatsFuns: logit, logistic
using Statistics
using Optim # If you moved your Kelly stuff here, otherwise ignore

# Assuming you need to reference your parent modules
using ..Data
using ..Predictions
using ..TypesInterfaces

# 1. Export the Structs
export CalibrationConfig, CalibrationResults
export BasicLogitShift

# 2. Export the Core Functions
export train_calibrators, apply_calibrators

# 3. Export the Analytics
export build_evaluation_df, summarize_metrics, compare_models

# 4. Include the files (Order matters! Types first)
include("./types.jl")
include("./data_l2_prep.jl")
include("./trainer.jl")
include("./basic_metrics.jl")

include("./shift_models/basic_logit.jl")
include("./shift_models/fitted_logit.jl")


end
