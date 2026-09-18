# ==============================================================================
# r04 — Out-of-sample proper scores and ground effects, hierarchical vs flat HA
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A predictive-fit comparison of the three hierarchical-HA candidates against their
# flat-HA twins and the Betfair closing line, plus the posterior of every club's
# home effect. It is not a betting study; `r05` re-prices the 2026-09-12 card.
#
# HYPOTHESES, written so they can fail visibly
#
#   H1  σ_γ is identified: its posterior moves away from the half-normal prior's
#       boundary mass (P(σ_γ < 0.02) well below the prior's 0.159).
#   H2  Turf grounds carry a larger home effect: P(mean γ_turf > mean γ_grass) ≥ 0.90
#       on the end-of-25/26 fold.
#   H3  Team HA improves proper scores: paired ΔLogLoss(candidate − flat twin) < 0
#       with a 95% fixture-clustered interval excluding zero, overall and on 1X2.
#   H4  The gain, if any, lives where the hypothesis says it should: ΔLogLoss on
#       fixtures with a turf home ground is more negative than on grass.
#
# COMPARABILITY CONTRACT
#
# * One panel: the 710 fixtures of 24/25 + 25/26 every arm holds a latent for;
#   extended runs are restricted BEFORE scoring.
# * One book: de-vigged Betfair TWA(−20, 0] close. `m12_hybrid_td_raw` must reproduce
#   its published LogLoss 0.64337 / ECE 0.0100 before any contrast is printed.
# * Every contrast is also reported without the fixtures whose home club the fold
#   never saw (T003: the hierarchical extraction prices them at γ = 0, the flat one
#   at γ_global). Those fixtures come from r02's `r02_unmapped_home_*.csv`.
# * Sampler budgets differ between candidates (4 × 500+1000, δ 0.80) and the TD
#   controls (4 × 800+800, δ 0.65). That changes Monte-Carlo noise, not the target.
#
# USAGE (mcmc-beast, from /root/BF_hier_ha, after r02)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/hierarchical_home_advantage/r04_evaluate.jl
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf
using Statistics

include(joinpath(@__DIR__, "l01_loader.jl"))
include(joinpath(@__DIR__, "l02_evaluation.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R04_CONFIG = HHAConfig()
const R04_BOOTSTRAP_B = 10_000
const R04_SEED = 20260912
const R04_OUT_DIR = joinpath(R04_CONFIG.save_root, "evaluation")
const R04_M12_TD_LOGLOSS = 0.64337
const R04_M12_TD_ECE = 0.0100
# Fold 20 closes 24/25, fold 40 closes 25/26 — the two most-conditioned posteriors.
const R04_GROUND_FOLDS = [20, 40]
const R04_SCOPES = ("all", "1X2", "OU2.5", "BTTS")

mkpath(R04_OUT_DIR)
println("\n" * "="^96)
println("  r04 PROPER SCORES — hierarchical vs flat home advantage, 24/25 + 25/26")
println("  bootstrap  : ", R04_BOOTSTRAP_B, " fixture-clustered paired resamples, seed ", R04_SEED)
println("="^96)

# %%
# ===================================================================
# 3. Data snapshot, the book and the arms
# ===================================================================
r04_ds = gph_load_data()
r04_odds = gph_betfair_closing_odds(r04_ds)
r04_families = gph_family_selections(r04_odds)
r04_splitter = gph_splitter(R04_CONFIG.target_seasons)

r04_arms = hha_arms(R04_CONFIG)
r04_raw_fits = Dict{String,Any}()
r04_fits = Dict{String,Any}()
for arm in r04_arms
    fit = gph_load_arm(arm)
    panel = gph_season_panel(r04_ds, fit, R04_CONFIG.target_seasons)
    length(panel) == R04_CONFIG.expected_oos || error(
        "$(arm.label) holds $(length(panel)) 24/25+25/26 fixtures; expected $(R04_CONFIG.expected_oos)")
    r04_raw_fits[arm.label] = fit
    r04_fits[arm.label] = gph_restrict(fit, panel)
    @printf("  %-38s %-9s %-15s %2d folds  %4d → %d fixtures  run %s\n",
            arm.label, arm.role, arm.dynamics, length(fit.folds), n_matches(fit.latents),
            length(panel), arm.run_id)
end
r04_panel = Set(r04_fits[first(HHA_MODEL_NAMES)].latents.match_ids)
all(Set(f.latents.match_ids) == r04_panel for f in values(r04_fits)) ||
    error("arms do not cover the identical 710-fixture panel")

# T003 fixtures: identical across candidates because the split and team maps are.
r04_unmapped = Set{Int}()
for name in HHA_MODEL_NAMES
    path = joinpath(R04_CONFIG.save_root, "r02_unmapped_home_$(name).csv")
    isfile(path) || error("missing $path — r02 writes it before sampling")
    frame = CSV.read(path, DataFrame)
    union!(r04_unmapped, Int.(frame.match_id))
end
println("  T003 unmapped-home fixtures (γ = 0 under hierarchical extraction): ", length(r04_unmapped))

# %%
# ===================================================================
# 4. Proper scores per arm and market family
# ===================================================================
r04_score_rows = NamedTuple[]
r04_obs = Dict{String,DataFrame}()
for arm in r04_arms
    ctx = gph_context(r04_fits[arm.label], r04_odds, r04_ds)
    append!(r04_score_rows, [(; row..., dynamics = arm.dynamics, role = arm.role)
                             for row in gph_scores(arm.label, ctx, r04_families)])
    r04_obs[arm.label] = hha_annotate!(gph_observation_frame(arm.label, ctx, r04_odds),
                                       r04_ds, r04_unmapped)
end
r04_scores = DataFrame(r04_score_rows)

r04_m12_td = only(filter(r -> r.model == "m12_hybrid_td_raw" && r.scope == "all", r04_scores))
@printf("\n  reproduction: m12_hybrid_td_raw LogLoss %.5f (published %.5f)  ECE %.4f (published %.4f)\n",
        r04_m12_td.logloss, R04_M12_TD_LOGLOSS, r04_m12_td.ece, R04_M12_TD_ECE)
abs(r04_m12_td.logloss - R04_M12_TD_LOGLOSS) < 5e-5 || error("m12 TD control does not reproduce its published LogLoss")
abs(r04_m12_td.ece - R04_M12_TD_ECE) < 5e-4 || error("m12 TD control does not reproduce its published ECE")

println("\n=== PROPER SCORES (scope = all) ===")
show(stdout, MIME"text/plain"(),
     sort(select(filter(:scope => ==("all"), r04_scores),
                 :model, :role, :dynamics, :n_obs, :logloss, :market_logloss, :brier, :rps, :ece, :market_ece),
          :logloss); allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 5. Paired, fixture-clustered bootstrap (H3, H4)
# ===================================================================
r04_cuts = [
    ("all fixtures", df -> trues(nrow(df))),
    ("excluding T003", df -> .!df.unmapped_home),
    ("turf home", df -> df.home_surface .== "turf"),
    ("grass home", df -> df.home_surface .== "grass"),
    # One T003 fixture (East Kilbride's first home game after promotion) is a turf home
    # ground priced at γ = 0 by the hierarchical arm; the surface contrast must be
    # readable without it.
    ("turf home excluding T003", df -> (df.home_surface .== "turf") .& .!df.unmapped_home),
    ("grass home excluding T003", df -> (df.home_surface .== "grass") .& .!df.unmapped_home),
]

r04_boot_rows = NamedTuple[]
for scope in R04_SCOPES
    fam = scope == "all" ? nothing : scope
    for name in HHA_MODEL_NAMES
        control = HHA_CONTROLS[name].label
        for (cut, pred) in r04_cuts
            b = hha_paired(r04_obs[name], r04_obs[control]; subset = pred,
                           B = R04_BOOTSTRAP_B, seed = R04_SEED, family = fam)
            push!(r04_boot_rows, (; left = name, right = control, scope, cut, b...))
        end
        for label in (name, control)
            b = hha_paired(r04_obs[label], :market; B = R04_BOOTSTRAP_B, seed = R04_SEED, family = fam)
            push!(r04_boot_rows, (; left = label, right = "betfair_close", scope, cut = "all fixtures", b...))
        end
    end
end
r04_boot = DataFrame(r04_boot_rows)
r04_boot.significant = (r04_boot.hi .< 0) .| (r04_boot.lo .> 0)

println("\n=== PAIRED ΔLogLoss (hierarchical − flat twin) ===")
show(stdout, MIME"text/plain"(),
     select(filter(r -> r.right != "betfair_close" && r.scope in ("all", "1X2"), r04_boot),
            :left, :scope, :cut, :n_obs, :n_fixtures, :delta, :lo, :hi, :p_negative, :significant);
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 6. Ground effects: σ_γ identification (H1) and turf vs grass (H2)
# ===================================================================
r04_club_frames = DataFrame[]
r04_contrast_rows = NamedTuple[]
for fold in R04_GROUND_FOLDS
    team_map = hha_fold_team_map(r04_ds, r04_splitter, fold)
    for name in HHA_MODEL_NAMES
        clubs, contrast = hha_turf_contrast(r04_raw_fits[name], fold, team_map)
        clubs.model = fill(name, nrow(clubs))
        clubs.fold = fill(fold, nrow(clubs))
        push!(r04_club_frames, clubs)
        push!(r04_contrast_rows, (; model = name, contrast...))
    end
end
r04_clubs = vcat(r04_club_frames...)
r04_contrast = DataFrame(r04_contrast_rows)

println("\n=== σ_γ AND TURF − GRASS CONTRAST ===")
show(stdout, MIME"text/plain"(), r04_contrast; allrows = true, allcols = true)
println("\n\n=== CLUB HOME EFFECTS, fold 40, m12_joint_hybrid_synergy_hier_ha ===")
show(stdout, MIME"text/plain"(),
     select(filter(r -> r.fold == 40 && r.model == "m12_joint_hybrid_synergy_hier_ha", r04_clubs),
            :team, :surface, :gamma_mean, :gamma_sd, :gamma_q05, :gamma_q95, :home_multiplier, :p_above_base);
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 7. Final report
# ===================================================================
CSV.write(joinpath(R04_OUT_DIR, "r04_proper_scores.csv"), r04_scores)
CSV.write(joinpath(R04_OUT_DIR, "r04_paired_bootstrap.csv"), r04_boot)
CSV.write(joinpath(R04_OUT_DIR, "r04_club_home_effects.csv"), r04_clubs)
CSV.write(joinpath(R04_OUT_DIR, "r04_turf_contrast.csv"), r04_contrast)

r04_fmt5 = v -> gph_num(v; digits = 5)
r04_fmt4 = v -> gph_num(v; digits = 4)
r04_fmt3 = v -> gph_num(v; digits = 3)
open(joinpath(R04_OUT_DIR, "r04_evaluation_report.md"), "w") do io
    println(io, "# r04 proper scores — Task 008 Phase 1\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            ". Panel: ", length(r04_panel), " fixtures. Book: de-vigged Betfair TWA(−20, 0] close. ",
            "Control reproduction: `m12_hybrid_td_raw` ",
            @sprintf("LogLoss %.5f / ECE %.4f", r04_m12_td.logloss, r04_m12_td.ece),
            " vs published ", R04_M12_TD_LOGLOSS, " / ", R04_M12_TD_ECE, ". ",
            "T003 unmapped-home fixtures: ", length(r04_unmapped), ".\n")
    for scope in R04_SCOPES
        sub = sort(filter(:scope => ==(scope), r04_scores), :logloss)
        nrow(sub) == 0 && continue
        println(io, "## Scope: ", scope, "\n")
        cols = scope in ("all", "1X2") ?
            [:model, :role, :dynamics, :n_obs, :logloss, :market_logloss, :brier, :market_brier, :rps, :ece, :market_ece] :
            [:model, :role, :dynamics, :n_obs, :logloss, :market_logloss, :brier, :market_brier, :ece, :market_ece]
        print(io, gph_markdown_table(select(sub, cols);
            formats = Dict(:logloss => r04_fmt5, :market_logloss => r04_fmt5,
                           :brier => r04_fmt5, :market_brier => r04_fmt5,
                           :rps => r04_fmt5, :ece => r04_fmt4, :market_ece => r04_fmt4)))
        println(io)
    end
    println(io, "## Paired ΔLogLoss, fixture-clustered bootstrap (B = ", R04_BOOTSTRAP_B, ")\n")
    println(io, "Negative Δ favours the left arm. 95% percentile interval.\n")
    print(io, gph_markdown_table(select(r04_boot, :scope, :cut, :left, :right, :n_obs, :n_fixtures,
                                        :delta, :lo, :hi, :p_negative, :significant);
        formats = Dict(:delta => v -> gph_signed(v; digits = 5),
                       :lo => v -> gph_signed(v; digits = 5),
                       :hi => v -> gph_signed(v; digits = 5),
                       :p_negative => r04_fmt3)))
    println(io, "\n## σ_γ and the turf − grass contrast\n")
    print(io, gph_markdown_table(r04_contrast;
        formats = Dict(:delta_mean => v -> gph_signed(v; digits = 4),
                       :delta_q05 => v -> gph_signed(v; digits = 4),
                       :delta_q95 => v -> gph_signed(v; digits = 4))))
    println(io, "\n## Club home effects\n")
    print(io, gph_markdown_table(select(r04_clubs, :model, :fold, :team, :surface, :gamma_mean,
                                        :gamma_sd, :gamma_q05, :gamma_q95, :home_multiplier, :p_above_base)))
end
println("\nR04_DONE report=", joinpath(R04_OUT_DIR, "r04_evaluation_report.md"))
