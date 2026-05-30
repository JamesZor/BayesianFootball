import re

file_path = "src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson_no_market.jl"
with open(file_path, "r") as f:
    code = f.read()

# 1. Rename struct
code = code.replace("DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel", "DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel")
code = code.replace("build_double_poisson_xg_market_player_engine", "build_double_poisson_xg_no_market_player_engine")

# 2. Remove market fields from struct
code = re.sub(r'\s*market_feature_config::M = Features\.DoublePoissonMarketFeature\(\)', '', code)
code = re.sub(r'\s*market_σ::Distribution = truncated\(Normal\(0\.1, 0\.2\), lower=0\.01\)', '', code)
code = re.sub(r'\s*market_weight::Float64 = 1\.0', '', code)

# Fix type parameters
code = code.replace("struct DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel{I<:AbstractInterceptionConfig, P<:OutfieldPlayerDynamicsConfig, D<:AbstractDispersionConfig, H<:AbstractHomeAdvantageConfig, K<:AbstractKappaConfig, R<:AbstractFeatureConfig, M<:AbstractMarketFeatureConfig} <: AbstractPreGameModel", "struct DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel{I<:AbstractInterceptionConfig, P<:OutfieldPlayerDynamicsConfig, D<:AbstractDispersionConfig, H<:AbstractHomeAdvantageConfig, K<:AbstractKappaConfig, R<:AbstractFeatureConfig} <: AbstractPreGameModel")

# 3. Remove market arguments from @model
code = re.sub(r'\s*# --- Market Data ---\s*market_log_λ_h::Vector\{Float64\},\s*market_log_λ_a::Vector\{Float64\},\s*idx_market::Vector\{Int\},', '', code)

# 4. Remove market parameter from @model
code = re.sub(r'\s*σ_market ~ config\.market_σ', '', code)

# 5. Remove market likelihood
market_lik = r'\s*# --- Pillar C: The Market \(Normal\) ---.*?end'
code = re.sub(market_lik, '', code, flags=re.DOTALL)

# 6. Remove market from required_features
code = re.sub(r'\s*model\.market_feature_config,', '', code)

# 7. Remove market from build_turing_model call
code = re.sub(r'\s*market_log_h, market_log_a, idx_market,', '', code)

# 8. Remove market data extraction in build_turing_model
code = re.sub(r'\s*idx_market = Int\[\]', '', code)
code = re.sub(r'\s*market_log_h = Vector\{Float64\}\(coalesce\.\(log\.\(data\[:flat_market_λ_home\]\), NaN\)\)', '', code)
code = re.sub(r'\s*market_log_a = Vector\{Float64\}\(coalesce\.\(log\.\(data\[:flat_market_λ_away\]\), NaN\)\)', '', code)
code = re.sub(r'\s*for i in 1:length\(match_weights\)\s*if !isnan\(market_log_h\[i\]\) && !isnan\(market_log_a\[i\]\)\s*push!\(idx_market, i\)\s*end\s*end', '', code)


with open(file_path, "w") as f:
    f.write(code)

print("Done stripping market features")
