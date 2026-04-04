using Revise
using BayesianFootball
using DataFrames
using Dates
using JSON3
using CSV
using Statistics
using SHA

# ==============================================================================
# PART 1: DEFINITIONS (Scraper & Types)
# ==============================================================================
# (You can move this entire section to src/matchday/scraper.jl later)

struct ScraperMatch
    id::String            
    event_name::String    
    start_time::Time      
    date::Date            
    league::String
end

struct ScraperRow
    match_id::String
    event_name::String
    market_name::String   
    selection::Symbol     
    back_price::Float64   
    lay_price::Float64    
    timestamp::DateTime
end

# --- CONFIGURATION ---
# UPDATE THESE PATHS to match your machine!
const PYTHON_EXE = "/home/james/.conda/envs/webscrape/bin/python"
const CLI_PATH = "/home/james/bet_project/whatstheodds/live_odds_cli.py"

function _run_cli(args::Vector{String})
    cmd = Cmd(vcat([PYTHON_EXE, CLI_PATH], args))
    try
        return read(setenv(cmd, dir=dirname(CLI_PATH)), String)
    catch e
        @error "Python CLI Failed: $e"
        return nothing
    end
end

function get_matches(leagues::Vector{String}; time_window_hours::Union{Float64, Nothing}=nothing)
    output = _run_cli(["list", "-f", leagues...])
    isnothing(output) && return ScraperMatch[]

    matches = ScraperMatch[]
    current_time = now()
    today_date = today()

    for line in split(output, '\n')
        m = match(r"^\s*-\s*(\d{2}:\d{2})\s*\|\s*(.+?)\s+v\s+(.+?)$", line)
        if !isnothing(m)
            t_str, home, away = m.captures
            match_time = Time(t_str, "HH:MM")
            
            if !isnothing(time_window_hours)
                match_dt = DateTime(today_date, match_time)
                diff = (match_dt - current_time) / Hour(1)
                if diff < -0.5 || diff > time_window_hours
                    continue 
                end
            end
            
            event_name = "$home v $away"
            id_str = bytes2hex(sha256(event_name * string(today_date)))[1:8]
            push!(matches, ScraperMatch(id_str, event_name, match_time, today_date, "Unknown"))
        end
    end
    return matches
end

function fetch_odds(matches::Vector{ScraperMatch})
    results = ScraperRow[]
    for m in matches
        json_str = _run_cli(["odds", m.event_name, "-d"])
        if !isnothing(json_str) && !isempty(json_str)
            try
                data = JSON3.read(json_str, Dict)
                _parse_data!(results, m, data)
            catch e
                @warn "Error processing data for $(m.event_name): $e"
            end
        end
    end
    return DataFrame(results)
end

function _parse_data!(out, m, data)
    get_p(o) = (
        Float64(get(get(o, "back", Dict()), "price", NaN)), 
        Float64(get(get(o, "lay", Dict()), "price", NaN))
    )

    if haskey(data, "ft")
        ft = data["ft"]
        if haskey(ft, "Match Odds")
            for (t, o) in ft["Match Odds"]
                sel = t == "The Draw" ? :draw : (startswith(m.event_name, t) ? :home : :away)
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "1X2", sel, b, l, now()))
            end
        end

        # Dynamically handle all Over/Under markets
        for market_key in ["0.5", "1.5", "2.5", "3.5", "4.5", "5.5"]
            full_key = "Over/Under $market_key Goals"
            if haskey(ft, full_key)
                for (k, o) in ft[full_key]
                    base_sym = replace(market_key, "." => "")
                    sel = startswith(k, "Over") ? Symbol("over_$base_sym") : Symbol("under_$base_sym")
                    b, l = get_p(o)
                    push!(out, ScraperRow(m.id, m.event_name, "OverUnder", sel, b, l, now()))
                end
            end
        end

        if haskey(ft, "Both teams to Score?")
            for (k, o) in ft["Both teams to Score?"]
                sel = k == "Yes" ? :btts_yes : :btts_no
                b, l = get_p(o)
                push!(out, ScraperRow(m.id, m.event_name, "BTTS", sel, b, l, now()))
            end
        end
    end
end

function load_mappings(mapping_dir::String=joinpath(dirname(CLI_PATH), "mappings"))
    master_map = Dict{String, String}()
    if !isdir(mapping_dir)
        @warn "Mapping directory not found: $mapping_dir"
        return master_map
    end
    
    for file in readdir(mapping_dir)
        if endswith(file, ".json")
            try
                path = joinpath(mapping_dir, file)
                data = JSON3.read(read(path, String), Dict)
                for (model_id, scraper_name) in data
                    master_map[scraper_name] = model_id
                end
            catch e
                @warn "Could not read mapping file $file: $e"
            end
        end
    end
    return master_map
end

# ==============================================================================
# PART 2: MATCH DAY EXECUTION SCRIPT
# ==============================================================================

println("=== 1. INITIALIZATION & DATA LOADING ===")

ds = BayesianFootball.Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)

println("Loading Model & Features...")
# UPDATE THIS PATH to your target model!
saved_folders = BayesianFootball.Experiments.list_experiments("exp/market_runs/april"; data_dir="./data")
m1 = BayesianFootball.Experiments.load_experiment(saved_folders[1])

feature_collection = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m1.config.splitter),
    m1.config.model, 
    m1.config.splitter
)

last_split_idx = length(m1.training_results)
chain1 = m1.training_results[last_split_idx][1]
feature_set = feature_collection[last_split_idx][1]

println("=== 2. FETCH LIVE MARKET DATA ===")

matches = get_matches(["scotland"], time_window_hours=24.0)
if isempty(matches)
    println("No matches found in the next 24 hours. Exiting.")
    exit()
end

df_odds = fetch_odds(matches)
df_odds.odds = df_odds.back_price;
df_odds.is_winner .= missing;
df_odds.date = Date.(df_odds.timestamp);

println("=== 3. ID ALIGNMENT & MODEL INPUT ===")

team_mappings = load_mappings()
# Manual overrides
team_mappings["Spartans"] = "the-spartans-fc"
team_mappings["East Kilbride"] = "east-kilbride"

# Extract the vocabulary the model actually knows
model_team_vocab = feature_set.data[:team_map]

model_rows = []
valid_string_ids = Set{String}()

for m in matches
    parts = split(m.event_name, " v ")
    if length(parts) == 2
        h_name, a_name = parts[1], parts[2]
        
        # 1. Check if the scraper names have a known mapping
        if haskey(team_mappings, h_name) && haskey(team_mappings, a_name)
            canonical_home = team_mappings[h_name]
            canonical_away = team_mappings[a_name]
            
            # 2. CRITICAL FIX: Check if the mapped teams actually exist in the model's trained vocabulary
            if haskey(model_team_vocab, canonical_home) && haskey(model_team_vocab, canonical_away)
                push!(model_rows, (
                    match_id = m.id, 
                    match_week = 999, 
                    match_date = today(),
                    home_team = canonical_home, 
                    away_team = canonical_away
                ))
                push!(valid_string_ids, m.id)
            else
                println("⚠️ Skipping $(m.event_name) - Teams are in a different league not covered by the model.")
            end
        else
            println("⚠️ Skipping $(m.event_name) - Could not map raw scraper names to canonical IDs.")
        end
    end
end
df_model = DataFrame(model_rows)

if isempty(df_model)
    println("Could not map any valid matches to model vocabulary. Exiting.")
    exit()
end

# -- The Alignment Fix: Safely Map Strings to Int64s --
df_odds_clean = filter(row -> row.match_id in valid_string_ids, df_odds)
unique_str_ids = unique(df_model.match_id)
str_to_int = Dict(id => i for (i, id) in enumerate(unique_str_ids))
int_to_str = Dict(i => id for (id, i) in str_to_int)

df_model.match_id = [str_to_int[x] for x in df_model.match_id]
df_odds_clean.match_id = [str_to_int[x] for x in df_odds_clean.match_id]

println("✅ Alignment complete. $(nrow(df_model)) matches ready for inference.")

println("=== 4. INFERENCE & MARKET MAKING ===")



raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m1.config.model, df_model, feature_set, chain1
)

ids = collect(keys(raw_preds))
cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
for k in keys(raw_preds[ids[1]])
    cols[k] = [raw_preds[i][k] for i in ids]
end
latents = BayesianFootball.Experiments.LatentStates(DataFrame(cols), m1.config.model)
ppd = BayesianFootball.Predictions.model_inference(latents)

# # -- Quantile Market Making logic --
# ppd.df.prob_q05 = [quantile(d, 0.05) for d in ppd.df.distribution]
# ppd.df.prob_mean = [mean(d) for d in ppd.df.distribution]
# ppd.df.prob_q95 = [quantile(d, 0.95) for d in ppd.df.distribution]
#
# ppd.df.model_odds_mean = 1.0 ./ ppd.df.prob_mean
# ppd.df.market_make_ask = 1.0 ./ ppd.df.prob_q05 
# ppd.df.market_make_bid = 1.0 ./ ppd.df.prob_q95 
#

# 1. Define your percentiles (Easy to adjust here!)
q_low  = 0.3  # 10th percentile (Conservative / Ask)
q_high = 0.70  # 90th percentile (Optimistic / Bid)

# 2. Calculate probabilities for these quantiles
ppd.df.prob_lower = [quantile(d, q_low) for d in ppd.df.distribution]
ppd.df.prob_mean  = [mean(d) for d in ppd.df.distribution]
ppd.df.prob_upper = [quantile(d, q_high) for d in ppd.df.distribution]

# 3. Translate Probabilities to Odds (Odds = 1 / Probability)
ppd.df.odds_ask  = 1.0 ./ ppd.df.prob_lower # Max price you'd offer (Bookie Ask)
ppd.df.odds_mean = 1.0 ./ ppd.df.prob_mean  # True "Fair" Market Odds
ppd.df.odds_bid  = 1.0 ./ ppd.df.prob_upper # Min price you'd accept (Bookie Bid)

println("=== 5. SIGNAL GENERATION & EXECUTION ===")

# Use the Bayesian Kelly from your package
my_signals = [BayesianFootball.Signals.BayesianKelly()]

results = BayesianFootball.Signals.process_signals(
    ppd, 
    df_odds_clean, 
    my_signals;
    odds_column=:odds
)

results.df.real_match_id = [int_to_str[x] for x in results.df.match_id]
match_names = unique(df_odds[:, [:match_id, :event_name]])
results_final = leftjoin(results.df, match_names, on=:real_match_id => :match_id)

# Filter for profitable selections from your backtest 
# profitable_selections = Set([:over_15, :over_25,:home, :draw])
# final_slip = filter(row -> row.stake > 0 && row.selection in profitable_selections, results_final)


# Filter for profitable selections from the AblationStudy_NB_KitchenSink backtest 
profitable_selections = Set([
    # The Core
    :over_15, :over_25, :over_35, :btts_yes,
    # The Tail Fade (High-line unders)
    :under_45, :under_55,
    # The High-Variance Edge
    :draw 
])

final_slip = filter(row -> row.stake > 0 && row.selection in profitable_selections, results_final)

if isempty(final_slip)
    println("\n📉 No value bets found.")
else
    println("\n💰 RECOMMENDED BETS:")
    display(select(final_slip, :event_name, :market_name, :selection, :odds, :stake))
end

#=
💰 RECOMMENDED BETS:
7×5 DataFrame
 Row │ event_name                 market_name  selection  odds     stake      
     │ String?                    String       Symbol     Float64  Float64    
─────┼────────────────────────────────────────────────────────────────────────
   1 │ Spartans v East Kilbride   OverUnder    over_35       2.66  1.37345e-7
   2 │ Hamilton v Kelty Hearts    OverUnder    over_15       1.37  0.0135818
   3 │ Hamilton v Kelty Hearts    OverUnder    over_25       2.1   0.0219846
   4 │ Hamilton v Kelty Hearts    OverUnder    over_35       3.9   0.0249542
   5 │ Stirling v Edinburgh City  OverUnder    over_15       1.27  0.0150526
   6 │ Stirling v Edinburgh City  OverUnder    over_25       1.8   0.0240011
   7 │ Stirling v Edinburgh City  OverUnder    over_35       3.0   0.0240855


julia> final_slip = filter(row -> row.stake > 0 && row.selection in profitable_selections, results_final)
21×18 DataFrame
 Row │ match_id  date        market_name  selection  is_winner  signal_name    signal_params  odds_type  odds     stake        real_match_id  event_name                 prob_lower  prob_mean  prob_upper  odds_ask  odds_mean  odds_bid 
     │ Int64     Date        String       Symbol     Missing    String         String         String     Float64  Float64      String         String?                    Float64?    Float64?   Float64?    Float64?  Float64?   Float64? 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │        5  2026-03-28  1X2          home         missing  BayesianKelly  none           odds          2.62  0.000190349  c1b6bec8       Spartans v East Kilbride     0.335662   0.385421    0.431219   2.97919    2.59456   2.31901
   2 │        6  2026-03-28  1X2          home         missing  BayesianKelly  none           odds          1.86  0.0751454    5ecff49e       Hamilton v Kelty Hearts      0.566693   0.606468    0.649274   1.76462    1.64889   1.54018
   3 │        6  2026-03-28  BTTS         btts_no      missing  BayesianKelly  none           odds          1.9   1.54425e-5   5ecff49e       Hamilton v Kelty Hearts      0.491736   0.527093    0.562375   2.03361    1.8972    1.77817
   4 │        6  2026-03-28  OverUnder    over_25      missing  BayesianKelly  none           odds          2.1   0.0219846    5ecff49e       Hamilton v Kelty Hearts      0.467961   0.514433    0.561945   2.13693    1.94389   1.77953
   5 │        7  2026-03-28  1X2          home         missing  BayesianKelly  none           odds          2.58  0.0589848    d0755677       Stirling v Edinburgh City    0.414902   0.459436    0.500472   2.41021    2.17658   1.99811
   6 │        7  2026-03-28  OverUnder    over_25      missing  BayesianKelly  none           odds          1.8   0.0240011    d0755677       Stirling v Edinburgh City    0.549084   0.591401    0.636203   1.82122    1.6909    1.57183
   7 │        2  2026-03-28  1X2          home         missing  BayesianKelly  none           odds          5.0   0.0553963    ed3eab6d       East Fife v Stenhousemuir    0.236431   0.272168    0.305589   4.22956    3.6742    3.27237
   8 │        2  2026-03-28  BTTS         btts_no      missing  BayesianKelly  none           odds          1.96  0.0430957    ed3eab6d       East Fife v Stenhousemuir    0.524689   0.556547    0.58866    1.90589    1.79679   1.69877
   9 │        2  2026-03-28  OverUnder    under_25     missing  BayesianKelly  none           odds          1.87  0.0501079    ed3eab6d       East Fife v Stenhousemuir    0.548479   0.588219    0.62997    1.82323    1.70005   1.58738
  10 │        9  2026-03-28  1X2          home         missing  BayesianKelly  none           odds          2.02  0.000512628  047f9d9d       Alloa v Peterhead            0.456272   0.500356    0.541178   2.19168    1.99858   1.84782
  11 │        9  2026-03-28  BTTS         btts_no      missing  BayesianKelly  none           odds          2.42  0.0294948    047f9d9d       Alloa v Peterhead            0.420312   0.454685    0.489641   2.37918    2.19932   2.04231
  12 │        9  2026-03-28  OverUnder    under_25     missing  BayesianKelly  none           odds          2.24  0.000885726  047f9d9d       Alloa v Peterhead            0.410196   0.453606    0.499557   2.43786    2.20456   2.00177
  13 │        8  2026-03-28  1X2          away         missing  BayesianKelly  none           odds          2.8   0.00371619   abfd6570       Elgin City FC v Clyde        0.332597   0.373115    0.407843   3.00664    2.68014   2.45193
  14 │        8  2026-03-28  BTTS         btts_no      missing  BayesianKelly  none           odds          2.34  0.0194301    abfd6570       Elgin City FC v Clyde        0.42721    0.459966    0.494108   2.34077    2.17407   2.02385
  15 │        8  2026-03-28  OverUnder    under_25     missing  BayesianKelly  none           odds          2.16  0.00604544   abfd6570       Elgin City FC v Clyde        0.439334   0.481837    0.526533   2.27617    2.07539   1.89921
  16 │        3  2026-03-28  1X2          away         missing  BayesianKelly  none           odds          3.2   0.0143043    e4931cb8       Annan v Stranraer            0.306025   0.345954    0.381449   3.26771    2.89056   2.62158
  17 │        3  2026-03-28  BTTS         btts_no      missing  BayesianKelly  none           odds          2.4   0.0204374    e4931cb8       Annan v Stranraer            0.417164   0.450606    0.482418   2.39714    2.21923   2.07289
  18 │        3  2026-03-28  OverUnder    under_25     missing  BayesianKelly  none           odds          2.24  0.00747574   e4931cb8       Annan v Stranraer            0.426301   0.467978    0.51072    2.34576    2.13685   1.95802
  19 │        1  2026-03-28  1X2          away         missing  BayesianKelly  none           odds          3.3   0.00177483   382241b2       Dumbarton v Forfar           0.274484   0.314102    0.347048   3.64319    3.18368   2.88145
  20 │        1  2026-03-28  BTTS         btts_no      missing  BayesianKelly  none           odds          2.12  0.0213601    382241b2       Dumbarton v Forfar           0.472278   0.503944    0.537437   2.1174     1.98435   1.86068
  21 │        1  2026-03-28  OverUnder    under_25     missing  BayesianKelly  none           odds          1.91  0.00163055   382241b2       Dumbarton v Forfar           0.492463   0.532331    0.575482   2.03061    1.87853   1.73767

=#


# 4. Extract just the keys and the calculated metrics from the PPD
ppd_metrics = select(ppd.df, 
    :match_id, 
    :market_name, 
    :selection,
    :prob_lower, :prob_mean, :prob_upper, 
    :odds_ask, :odds_mean, :odds_bid
)

# 5. Join onto your final results
results_final = leftjoin(
    results_final, 
    ppd_metrics, 
    on = [:match_id, :market_name, :selection]
)

# 6. View the result side-by-side
display(select(results_final, 
    :event_name, :selection, :odds, :odds_mean, :odds_ask, :odds_bid, :stake
))

results_final


#=
julia> final_slip = filter(row -> row.stake > 0 && row.selection in profitable_selections, results_final)
4×18 DataFrame
 Row │ match_id  date        market_name  selection  is_winner  signal_name    signal_params  odds_type  odds     stake      real_match_id  event_name                 prob_lower  prob_mean  prob_upper  odds_ask  odds_mean  odds_bid 
     │ Int64     Date        String       Symbol     Missing    String         String         String     Float64  Float64    String         String?                    Float64?    Float64?   Float64?    Float64?  Float64?   Float64? 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │        6  2026-03-28  OverUnder    over_15      missing  BayesianKelly  none           odds          1.37  0.0135818  5ecff49e       Hamilton v Kelty Hearts      0.662409   0.747425    0.826509   1.50964    1.33793   1.20991
   2 │        6  2026-03-28  OverUnder    over_25      missing  BayesianKelly  none           odds          2.1   0.0219846  5ecff49e       Hamilton v Kelty Hearts      0.403827   0.514433    0.623631   2.47631    1.94389   1.60351
   3 │        7  2026-03-28  OverUnder    over_15      missing  BayesianKelly  none           odds          1.27  0.0150526  d0755677       Stirling v Edinburgh City    0.729958   0.802486    0.870982   1.36994    1.24613   1.14813
   4 │        7  2026-03-28  OverUnder    over_25      missing  BayesianKelly  none           odds          1.8   0.0240011  d0755677       Stirling v Edinburgh City    0.485213   0.591401    0.697238   2.06095    1.6909    1.43423
=#



#=
julia> tearsheet
27×18 DataFrame
 Row │ model_name                    model_parameters                   signal_name    signal_params  selection  opportunities  bets_placed  activity_pct  turnover  profit   roi_pct  win_rate_pct  SharpeRatio  CumulativeWealth  CalmarRatio  BurkeRatio   SterlingRatio  SortinoRatio 
     │ String                        String                             String         String         Symbol     Int64          Int64        Float64       Float64   Float64  Float64  Float64       Float64      Float64           Float64      Float64      Float64        Float64      
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           home                1214          376          31.0     21.85    -0.16    -0.73          32.2       -0.002            -0.16   -0.029       -0.001         -0.131        -0.003
   2 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           draw                1214          160          13.2      4.02     3.34    83.15          25.0        0.034             3.343   6.115        0.425          9.727         0.224
   3 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           away                1214          530          43.7     28.6     -4.09   -14.3           23.2       -0.037            -4.09   -0.619       -0.031         -2.088        -0.072
   4 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_yes            1208          183          15.1      7.17     0.78    10.82          53.0        0.021             0.776   0.771        0.038          1.848         0.039
   5 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           btts_no             1208          447          37.0     16.57    -1.88   -11.34          42.7       -0.045            -1.879  -0.646       -0.032         -2.238        -0.058
   6 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_05             1134           22           1.9      1.11     0.06     5.25         100.0        0.077             0.058   0.0          0.0            0.0         999.0
   7 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_05            1134          210          18.5      1.45    -0.76   -52.1            6.2       -0.072            -0.755  -1.0         -0.045         -1.0          -0.128
   8 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_15             1134          119          10.5      5.42     0.24     4.33          80.7        0.018             0.235   0.475        0.031          1.835         0.021
   9 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_15            1134          528          46.6     16.59    -3.78   -22.78          22.2       -0.067            -3.78   -0.857       -0.041         -0.857        -0.111
  10 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_25             1192          302          25.3     11.28     1.11     9.83          56.6        0.032             1.109   0.766        0.042          2.966         0.045
  11 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_25            1192          542          45.5     30.88    -0.35    -1.13          48.7       -0.005            -0.35   -0.132       -0.007         -0.394        -0.007
  12 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_35             1134          306          27.0      8.39     1.47    17.58          33.3        0.037             1.474   1.051        0.063          3.954         0.067
  13 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_35            1134          411          36.2     29.49    -1.28    -4.34          69.3       -0.025            -1.281  -0.521       -0.025         -0.521        -0.028
  14 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_45             1134          271          23.9      4.33    -0.44   -10.26          16.2       -0.015            -0.444  -0.437       -0.024         -0.998        -0.03
  15 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_45            1134          264          23.3     22.16     0.88     3.97          88.6        0.035             0.88    1.141        0.064          3.913         0.04
  16 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_55             1183          128          10.8      1.05    -0.21   -20.31          10.2       -0.013            -0.214  -0.497       -0.033         -0.497        -0.035
  17 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_55            1183          151          12.8     14.72     0.86     5.82          95.4        0.12              0.856  10.352        1.307         27.238         0.251
  18 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_65             1190           28           2.4      0.12    -0.06   -51.33           3.6       -0.027            -0.062  -0.617       -0.034         -0.617        -0.044
  19 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_65            1190           55           4.6      6.98     0.22     3.14         100.0        0.081             0.22    7.90858e15   6.93795e14     1.26537e16    2.22507e14
  20 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_75              981            9           0.9      0.01    -0.01  -100.0            0.0       -0.066            -0.015  -1.0         -0.043         -1.0          -0.065
  21 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_75             981           17           1.7      1.9      0.04     2.05         100.0        0.047             0.039   0.0          0.0            0.0         999.0
  22 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_85               76            0           0.0      0.0      0.0      0.0            0.0        0.0               0.0     0.0          0.0            0.0           0.0
  23 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_85              76            3           3.9      0.4      0.01     1.63         100.0        0.13              0.007   0.0          0.0            0.0         999.0
  24 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_95                8            0           0.0      0.0      0.0      0.0            0.0        0.0               0.0     0.0          0.0            0.0           0.0
  25 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_95               8            1          12.5      0.05     0.0      1.0          100.0        0.378             0.0     0.0          0.0            0.0         999.0
  26 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           over_105               1            0           0.0      0.0      0.0      0.0            0.0        0.0               0.0     0.0          0.0            0.0           0.0
  27 │ AblationStudy_NB_KitchenSink  μ=Normal(μ=0.2, σ=0.5), γ=Normal…  BayesianKelly  none           under_105              1            0           0.0      0.0      0.0      0.0            0.0        0.0               0.0     0.0          0.0            0.0           0.0


=#

using DataFramesMeta

# Group by match, and for each match, only keep the bet with the highest recommended stake
safeguarded_slip = combine(groupby(final_slip, :match_id)) do sdf
    # Find the index of the row with the maximum stake in this match's group
    best_row_idx = argmax(sdf.stake)
    return sdf[best_row_idx:best_row_idx, :]
end

display(safeguarded_slip[:, [:event_name, :selection, :odds, :prob_mean, :stake]])


using DataFramesMeta

# Group by match, and for each match, keep the profitable bet with the HIGHEST WIN PROBABILITY
steady_growth_slip = combine(groupby(final_slip, :match_id)) do sdf
    # Using prob_mean ensures we pick the safest positive-value bet
    safest_row_idx = argmax(sdf.prob_mean)
    return sdf[safest_row_idx:safest_row_idx, :]
end

# Let's view the new, lower-variance portfolio
display(steady_growth_slip[:, [:event_name, :selection, :odds, :prob_mean, :stake]])




println("=== 5. EXCHANGE VALUE TRADING (BACK & LAY) ===")

# 1. Join Model Posteriors with Live Exchange Prices
# We use innerjoin so we only evaluate lines that exist in both the model and the scraper
trading_df = innerjoin(
    ppd.df, 
    df_odds_clean, 
    on = [:match_id, :market_name, :selection]
)

# Bring in the real string match_ids just in case you need to debug later
trading_df.real_match_id = [int_to_str[x] for x in trading_df.match_id]

# (The event_name is already safely inside trading_df now!)

# 2. Calculate Expected Value (EV) for Both Sides
# EV Back = (True Prob * Back Payout) - 1
trading_df.ev_back = (trading_df.prob_mean .* trading_df.back_price) .- 1.0

# EV Lay = 1 - (True Prob * Lay Payout)
trading_df.ev_lay = 1.0 .- (trading_df.prob_mean .* trading_df.lay_price)

# 3. Apply the Adverse Selection Shield (The Percentile Filter)
# We ONLY Back if the exchange price is better than our 10th percentile worst-case scenario
trading_df.valid_back = trading_df.back_price .> trading_df.odds_ask

# We ONLY Lay if the exchange price is lower than our 90th percentile best-case scenario
trading_df.valid_lay = trading_df.lay_price .< trading_df.odds_bid

# 4. Filter and Sort the Opportunities
# The Strategy: Focus on the highly profitable, highly liquid core markets
profitable_selections = Set([:home, :draw, :away, :over_25, :under_25, :btts_yes, :btts_no])
trading_df = filter(row -> row.selection in profitable_selections, trading_df)

# Extract Actionable Backs
value_backs = filter(row -> row.ev_back > 0.02 && row.valid_back, trading_df)
sort!(value_backs, :ev_back, rev=true)

# Extract Actionable Lays
value_lays = filter(row -> row.ev_lay > 0.02 && row.valid_lay, trading_df)
sort!(value_lays, :ev_lay, rev=true)

# 5. Display the Execution Sheets
if isempty(value_backs) && isempty(value_lays)
    println("\n📉 No edge found against the exchange. The market is perfectly efficient today.")
else
    if !isempty(value_backs)
        println("\n🔥 VALUE BACKS (You act as the Punter):")
        println("   Rule: Exchange Back Price > Model Ask Bounds")
        display(select(value_backs, 
            :event_name, :market_name, :selection, 
            :prob_mean => (x -> round.(x, digits=3)) => :model_prob,
            :odds_ask => (x -> round.(x, digits=2)) => :req_odds,
            :back_price => :exchange_odds, 
            :ev_back => (x -> round.(x, digits=3)) => :EV
        ))
    end
    
    if !isempty(value_lays)
        println("\n🏦 VALUE LAYS (You act as the Bookmaker):")
        println("   Rule: Exchange Lay Price < Model Bid Bounds")
        display(select(value_lays, 
            :event_name, :market_name, :selection, 
            :prob_mean => (x -> round.(x, digits=3)) => :model_prob,
            :odds_bid => (x -> round.(x, digits=2)) => :req_odds,
            :lay_price => :exchange_odds, 
            :ev_lay => (x -> round.(x, digits=3)) => :EV
        ))
    end
end



# Set your Kelly Fraction (e.g., 0.25 for Quarter Kelly)
kelly_fraction = 0.25
bankroll = 1000.0 # Define your total trading bankroll

# Calculate Recommended Backer Stake (The amount the punter bets against you)
value_lays.backer_stake = value_lays.ev_lay .* bankroll .* kelly_fraction

# Calculate Your Actual Liability (The amount locked in your Betfair account)
# FIXED: Using lay_price instead of exchange_odds
value_lays.liability = value_lays.backer_stake .* (value_lays.lay_price .- 1.0)

# Display the final execution sheet with sizing
println("\n🏦 VALUE LAYS (You act as the Bookmaker):")
display(select(value_lays, 
    :event_name, :selection, 
    :lay_price => :exchange_odds, 
    :ev_lay => (x -> round.(x * 100, digits=1)) => :EV_pct,
    :backer_stake => (x -> round.(x, digits=2)) => :accept_stake_£,
    :liability => (x -> round.(x, digits=2)) => :liability_at_risk_£
))


# 5. Display the Execution Sheets
if isempty(value_backs) && isempty(value_lays)
    println("\n📉 No edge found against the exchange. The market is perfectly efficient today.")
else
    if !isempty(value_backs)
        println("\n🔥 VALUE BACKS (You act as the Punter):")
        println("   Rule: Exchange Back Price > Model Ask Bounds")
        display(select(value_backs, 
            :event_name, :market_name, :selection, 
            :prob_mean => (x -> round.(x, digits=3)) => :model_prob,
            :odds_ask => (x -> round.(x, digits=2)) => :req_odds,
            :back_price => :exchange_odds, 
            :ev_back => (x -> round.(x, digits=3)) => :EV
        ))
    end
    
    if !isempty(value_lays)
        println("\n🏦 VALUE LAYS (You act as the Bookmaker):")
        println("   Rule: Exchange Lay Price < Model Bid Bounds")
        display(select(value_lays, 
            :event_name, :market_name, :selection, 
            :prob_mean => (x -> round.(x, digits=3)) => :model_prob,
            :odds_bid => (x -> round.(x, digits=2)) => :req_odds,
            :lay_price => :exchange_odds, 
            :ev_lay => (x -> round.(x, digits=3)) => :EV
        ))
    end
end


#=
🔥 VALUE BACKS (You act as the Punter):
   Rule: Exchange Back Price > Model Ask Bounds
8×7 DataFrame
 Row │ event_name                 market_name  selection  model_prob  req_odds  exchange_odds  EV      
     │ String                     String       Symbol     Float64     Float64   Float64        Float64 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ East Fife v Stenhousemuir  1X2          home            0.272      4.23           5.0     0.361
   2 │ Stirling v Edinburgh City  1X2          home            0.459      2.41           2.58    0.185
   3 │ Hamilton v Kelty Hearts    1X2          home            0.606      1.76           1.86    0.128
   4 │ Alloa v Peterhead          BTTS         btts_no         0.455      2.38           2.42    0.1
   5 │ East Fife v Stenhousemuir  OverUnder    under_25        0.588      1.82           1.87    0.1
   6 │ East Fife v Stenhousemuir  BTTS         btts_no         0.557      1.91           1.96    0.091
   7 │ Annan v Stranraer          BTTS         btts_no         0.451      2.4            2.4     0.081
   8 │ Dumbarton v Forfar         BTTS         btts_no         0.504      2.12           2.12    0.068

🏦 VALUE LAYS (You act as the Bookmaker):
   Rule: Exchange Lay Price < Model Bid Bounds
13×7 DataFrame
 Row │ event_name                 market_name  selection  model_prob  req_odds  exchange_odds  EV      
     │ String                     String       Symbol     Float64     Float64   Float64        Float64 
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ East Fife v Stenhousemuir  1X2          away            0.456      2.02           1.81    0.174
   2 │ Hamilton v Kelty Hearts    1X2          draw            0.219      4.23           3.85    0.159
   3 │ Hamilton v Kelty Hearts    1X2          away            0.175      5.06           4.9     0.143
   4 │ Stirling v Edinburgh City  1X2          draw            0.228      4.18           3.85    0.121
   5 │ East Fife v Stenhousemuir  OverUnder    over_25         0.412      2.21           2.16    0.111
   6 │ Stirling v Edinburgh City  1X2          away            0.312      2.88           2.86    0.107
   7 │ East Fife v Stenhousemuir  BTTS         btts_yes        0.443      2.1            2.06    0.086
   8 │ Alloa v Peterhead          1X2          draw            0.234      4.05           3.95    0.078
   9 │ Alloa v Peterhead          BTTS         btts_yes        0.545      1.73           1.7     0.073
  10 │ Dumbarton v Forfar         BTTS         btts_yes        0.496      1.89           1.89    0.062
  11 │ Annan v Stranraer          BTTS         btts_yes        0.549      1.72           1.71    0.061
  12 │ Elgin City FC v Clyde      BTTS         btts_yes        0.54       1.75           1.74    0.06
  13 │ Elgin City FC v Clyde      1X2          draw            0.249      3.85           3.8     0.053

=#
