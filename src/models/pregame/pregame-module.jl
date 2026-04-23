# src/models/pregame/PreGame.jl

module PreGame

# We import the Types to extend them, but we don't need Reexport
using ...TypesInterfaces

using Turing, Distributions, DataFrames
using ..MyDistributions 
using ...Features
using LinearAlgebra
using Statistics
using Dates


# 1. Load Common Helpers (The Orchestrator)
include("common.jl")
include("./grw_helpers.jl")
include("./hierarchical_helpers.jl")

# 2. Load Models (The Workers)
# Since these files are included, their 'extract_parameters' overloads 
# merge into the function defined in common.jl
include("./implementations/static_poisson.jl")
include("./implementations/static_hierarchical_poisson.jl")
include("./implementations/static_hierarchical_poisson_NCP.jl")
include("./implementations/bivariate_poisson_ncp.jl")

include("./implementations/static_mvpln.jl")
export StaticMVPLN

include("./implementations/dixon_coles.jl")
export DixonColesNCP

include("./implementations/static_mixture_copula.jl")
export StaticMixtureCopula

include("./implementations/static_dixoncoles.jl")

include("./implementations/grw_poisson.jl")
export GRWPoisson

include("./implementations/grw_dixon_coles.jl")
export GRWDixonColes

include("./implementations/static_double_neg_bin.jl")
export StaticDoubleNegBin
include("./implementations/grw_double_neg_bin.jl")
export GRWNegativeBinomial
include("./implementations/grw_double_neg_bin_phi.jl")
export GRWNegativeBinomialPhi


include("./implementations/grw_double_neg_bin_mu.jl")
export GRWNegativeBinomialMu

include("./implementations/grw_double_neg_bin_mu_phi.jl")
export GRWNegativeBinomialMuPhi

include("./implementations/grw_double_neg_bin_delta.jl")
export GRWNegativeBinomialDelta


include("./implementations/grw_double_neg_bin_full.jl")
export GRWNegativeBinomialFull



include("./implementations/grw_bivariate_poisson.jl")
export GRWBivariatePoisson

include("./implementations/multi_grw_neg_bin.jl")
export MSNegativeBinomial 

include("./implementations/multi_grw_neg_bin_delta.jl")
export MSNegativeBinomialDelta

include("./implementations/multi_grw_neg_bin_kappa.jl")
export MSNegativeBinomialKappa

include("./implementations/multi_grw_neg_bin_r_all.jl")
export MSNegativeBinomialRho

include("./implementations/multi_grw_neg_bin_gamma.jl")
export MSNegativeBinomialGamma


include("./implementations/multi_grw_neg_bin_dc.jl")
export MSNegativeBinomialDC



include("./implementations/sequential_funnel.jl")
export SequentialFunnelModel


# ==============================================================================
# ABLATION STUDY MODELS
# ==============================================================================
include("./implementations/ablation_study/nb_baseline.jl")
export AblationStudy_NB_baseLine

include("./implementations/ablation_study/nb_env.jl")
export AblationStudy_NB_env

include("./implementations/ablation_study/nb_home_hierarchy.jl")
export AblationStudy_NB_home_hierarchy

include("./implementations/ablation_study/nb_team_dispersion.jl")
export AblationStudy_NB_team_dispersion

include("./implementations/ablation_study/nb_mu_month.jl")
export AblationStudy_NB_baseline_month_mu

include("./implementations/ablation_study/nb_r_month.jl")
export AblationStudy_NB_baseline_month_r

include("./implementations/ablation_study/nb_baseline_ha_r.jl")
export AblationStudy_NB_baseline_HA_r

include("./implementations/ablation_study/nb_ha_r_home_hierarchy.jl")
export AblationStudy_NB_HA_r_home_hierarchy

include("./implementations/ablation_study/nb_bloated.jl")
export AblationStudy_NB_KitchenSink



# ---- ----
# 3. Export
export StaticPoisson, StaticDixonColes, StaticHierarchicalPoisson, StaticHierarchicalPoissonNCP, BivariatePoissonNCP
export build_turing_model, extract_parameters

end # module
