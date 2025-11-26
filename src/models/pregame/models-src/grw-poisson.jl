using DataFrames
using Turing
using LinearAlgebra



export GRWPoisson, build_turing_model, predict

struct GRWPoisson <: AbstractGRWPoissonModel end 


@model function gaussian_random_walk_poisson_model(n_teams, home_ids, away_ids, home_goals, away_goals) 




end
