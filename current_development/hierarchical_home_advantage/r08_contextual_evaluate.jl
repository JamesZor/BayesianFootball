# ==============================================================================
# r08 — Proper scores and coefficient audit, contextual HA ladder vs flat control
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# The predictive-fit comparison of the three Phase 2 rungs against the persisted flat
# control `m05_joint_td_raw` (Exp 06) and the Betfair close, plus the posterior of every
# contextual coefficient. Phase 1's RE-only `m05_joint_production_wealth_hier_ha` is a
# reference arm: it separates "context helps" from "stadium RE helps". Not a betting study.
#
# HYPOTHESES (WORK_PACKAGE_PHASE_2_TURF_TIMING.md §3C), written so they can fail
#
#   H1  P(β_turf_asym > 0) ≥ 0.90 on the end-of-25/26 fold. Read beside the PRIOR's 0.84
#       and the posterior contraction — a prior-dominated 0.90 is not evidence.
#   H2  β_turf_pace > 0 (turf fixtures score more); scored on OU2.5 at turf home grounds.
#   H3  P(β_midweek > 0) ≥ 0.90, same caveat as H1.
#   H4  ΔLogLoss(rung − control) < 0 with the 95% fixture-clustered interval excluding 0.
#   Stratified by surface (turf home / grass home / grass visitor at turf) and timing
#   (midweek / not).
#
# COMPARABILITY CONTRACT (Phase 1's r04, unchanged)
#   * One 710-fixture panel; one book: de-vigged Betfair TWA(−20, 0] close.
#   * The control must reproduce Phase 1's r04 numbers before any contrast prints.
#   * Every contrast is also reported without T003 fixtures (r07_unmapped_home_*.csv).
#   * Sampler budgets differ (rungs 4 × 500+1000 δ 0.80; control 4 × 800+800 δ 0.65).
#
# USAGE (mcmc-beast, from /root/BF_hier_ha, after r07)
#
#   julia --project -t 16 current_development/hierarchical_home_advantage/r08_contextual_evaluate.jl
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

include(joinpath(@__DIR__, "l04_contextual_loader.jl"))
include(joinpath(@__DIR__, "l02_evaluation.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R08_CONFIG = CtxConfig()
const R08_B = 10_000
const R08_SEED = 20260918
const R08_OUT_DIR = joinpath(R08_CONFIG.save_root, "evaluation")
const R08_CONTROL = CTX_CONTROL.label                 # m05_joint_td_raw
# Phase 1 r04, scope "all": the control must reproduce these.
const R08_CONTROL_LOGLOSS = 0.6429863825954205
const R08_CONTROL_ECE = 0.014930352713004068
const R08_RE_ONLY = GPHArm("m05_joint_production_wealth_hier_ha", "scottish_lower_hierarchical_ha",
                           UUID("6117c711-a9f9-4539-a36c-40d6ddb6593c"), "TimeDecay(180)", "reference")
const R08_COEF_FOLDS = [20, 40]                       # close of 24/25 and of 25/26
const R08_SCOPES = ("all", "1X2", "OU2.5", "BTTS")

mkpath(R08_OUT_DIR)
println("\n" * "="^96)
println("  r08 PROPER SCORES — contextual HA ladder vs flat control, 24/25 + 25/26")
println("  bootstrap  : ", R08_B, " fixture-clustered paired resamples, seed ", R08_SEED)
println("="^96)

# %%
# ===================================================================
# 3. Data, book, arms
# ===================================================================
r08_ds = gph_load_data()
r08_odds = gph_betfair_closing_odds(r08_ds)
r08_families = gph_family_selections(r08_odds)

r08_db = PostgresStorage(R08_CONFIG.experiment)
r08_arms = GPHArm[]
for name in CTX_MODEL_NAMES
    run_id = gph_run_by_name(r08_db, name)
    run_id === nothing && error("no completed run named $name in $(R08_CONFIG.experiment) — run r07 first")
    push!(r08_arms, GPHArm(name, R08_CONFIG.experiment, run_id, "TimeDecay(180)", "candidate"))
end
push!(r08_arms, GPHArm(R08_CONTROL, CTX_CONTROL.experiment, CTX_CONTROL.run_id, "TimeDecay(180)", "control"))
push!(r08_arms, R08_RE_ONLY)

r08_raw = Dict{String,Any}()
r08_fits = Dict{String,Any}()
for arm in r08_arms
    fit = gph_load_arm(arm)
    panel = gph_season_panel(r08_ds, fit, R08_CONFIG.target_seasons)
    length(panel) == R08_CONFIG.expected_oos || error(
        "$(arm.label) holds $(length(panel)) fixtures; expected $(R08_CONFIG.expected_oos)")
    r08_raw[arm.label] = fit
    r08_fits[arm.label] = gph_restrict(fit, panel)
    @printf("  %-38s %-9s %2d folds  %4d fixtures  run %s\n",
            arm.label, arm.role, length(fit.folds), length(panel), arm.run_id)
end
r08_panel = Set(r08_fits[R08_CONTROL].latents.match_ids)
all(Set(f.latents.match_ids) == r08_panel for f in values(r08_fits)) ||
    error("arms do not cover the identical 710-fixture panel")

r08_design = CSV.read(joinpath(R08_CONFIG.save_root, "r07_oos_design.csv"), DataFrame)
r08_design_of = Dict(Int(r.match_id) => r for r in eachrow(r08_design))
r08_unmapped = Set{Int}()
for name in CTX_MODEL_NAMES
    path = joinpath(R08_CONFIG.save_root, "r07_unmapped_home_$(name).csv")
    isfile(path) && union!(r08_unmapped, Int.(CSV.read(path, DataFrame).match_id))
end
println("  T003 unmapped-home fixtures: ", length(r08_unmapped),
        " | OOS design rows: ", nrow(r08_design))

function r08_annotate!(obs::DataFrame)
    d = [r08_design_of[Int(m)] for m in obs.match_id]
    obs.turf_home = [r.turf_home == 1 for r in d]
    obs.turf_asym = [r.turf_asym == 1 for r in d]
    obs.midweek = [r.midweek == 1 for r in d]
    obs.unmapped_home = [Int(m) in r08_unmapped for m in obs.match_id]
    return obs
end

# %%
# ===================================================================
# 4. Proper scores
# ===================================================================
r08_score_rows = NamedTuple[]
r08_obs = Dict{String,DataFrame}()
for arm in r08_arms
    ctx = gph_context(r08_fits[arm.label], r08_odds, r08_ds)
    append!(r08_score_rows, [(; row..., role = arm.role)
                             for row in gph_scores(arm.label, ctx, r08_families)])
    r08_obs[arm.label] = r08_annotate!(gph_observation_frame(arm.label, ctx, r08_odds))
end
r08_scores = DataFrame(r08_score_rows)

r08_ctl = only(filter(r -> r.model == R08_CONTROL && r.scope == "all", r08_scores))
@printf("\n  reproduction: %s LogLoss %.5f (r04 %.5f)  ECE %.4f (r04 %.4f)\n",
        R08_CONTROL, r08_ctl.logloss, R08_CONTROL_LOGLOSS, r08_ctl.ece, R08_CONTROL_ECE)
abs(r08_ctl.logloss - R08_CONTROL_LOGLOSS) < 5e-5 || error("control does not reproduce r04's LogLoss")
abs(r08_ctl.ece - R08_CONTROL_ECE) < 5e-4 || error("control does not reproduce r04's ECE")

println("\n=== PROPER SCORES (scope = all) ===")
show(stdout, MIME"text/plain"(),
     sort(select(filter(:scope => ==("all"), r08_scores),
                 :model, :role, :n_obs, :logloss, :market_logloss, :brier, :rps, :ece, :market_ece),
          :logloss); allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 5. Paired bootstrap, stratified (H4)
# ===================================================================
r08_cuts = [
    ("all fixtures", df -> trues(nrow(df))),
    ("excluding T003", df -> .!df.unmapped_home),
    ("turf home", df -> df.turf_home),
    ("grass home", df -> .!df.turf_home),
    ("grass visitor at turf", df -> df.turf_asym),
    ("midweek", df -> df.midweek),
    ("not midweek", df -> .!df.midweek),
]

r08_boot_rows = NamedTuple[]
for scope in R08_SCOPES
    fam = scope == "all" ? nothing : scope
    for left in vcat(CTX_MODEL_NAMES, [R08_RE_ONLY.label])
        for (cut, pred) in r08_cuts
            sub_l = r08_obs[left][pred(r08_obs[left]), :]
            sub_r = r08_obs[R08_CONTROL][pred(r08_obs[R08_CONTROL]), :]
            nrow(sub_l) == 0 && continue
            b = gph_paired_bootstrap(sub_l, sub_r; B = R08_B, seed = R08_SEED, family = fam)
            push!(r08_boot_rows, (; left, right = R08_CONTROL, scope, cut, b...))
        end
    end
    for label in vcat(CTX_MODEL_NAMES, [R08_CONTROL])
        b = gph_paired_bootstrap(r08_obs[label], :market; B = R08_B, seed = R08_SEED, family = fam)
        push!(r08_boot_rows, (; left = label, right = "betfair_close", scope, cut = "all fixtures", b...))
    end
end
r08_boot = DataFrame(r08_boot_rows)
r08_boot.significant = (r08_boot.hi .< 0) .| (r08_boot.lo .> 0)

println("\n=== PAIRED ΔLogLoss (arm − ", R08_CONTROL, ") ===")
show(stdout, MIME"text/plain"(),
     select(filter(r -> r.right == R08_CONTROL && r.scope in ("all", "1X2", "OU2.5"), r08_boot),
            :left, :scope, :cut, :n_fixtures, :delta, :lo, :hi, :p_negative, :significant);
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 6. Coefficients (H1, H2, H3) and a descriptive turf-goals check
# ===================================================================
r08_coef_frames = DataFrame[]
for name in CTX_MODEL_NAMES
    fit = r08_raw[name]
    for fold in R08_COEF_FOLDS
        fold <= length(fit.folds) || continue
        c = ctx_coefficients(fit, fold)
        c.model = fill(name, nrow(c))
        push!(r08_coef_frames, c)
    end
end
r08_coefs = vcat(r08_coef_frames...)
r08_coefs.h_pass = [r.site in ("turf_asym.w", "midweek.w") ? r.p_positive >= 0.90 :
                    r.site == "turf_pace.w" ? r.q05 > 0 : missing for r in eachrow(r08_coefs)]
println("\n=== COEFFICIENTS (folds ", join(R08_COEF_FOLDS, ", "), ") ===")
show(stdout, MIME"text/plain"(),
     select(r08_coefs, :model, :fold, :site, :mean, :q05, :q95, :p_positive, :prior_p_positive,
            :contraction, :h_pass); allrows = true, allcols = true)
println()

# Raw goals in the panel, no model: does a turf home ground see more goals?
r08_goals = let m = r08_ds.matches[[Int(x) in r08_panel for x in r08_ds.matches.match_id], :]
    m.turf_home = [r08_design_of[Int(x)].turf_home == 1 for x in m.match_id]
    m.midweek = [r08_design_of[Int(x)].midweek == 1 for x in m.match_id]
    m.total = m.home_score .+ m.away_score
    m.diff = m.home_score .- m.away_score
    combine(groupby(m, [:turf_home, :midweek]), nrow => :n, :total => mean => :mean_total,
            :home_score => mean => :mean_home, :away_score => mean => :mean_away,
            :diff => mean => :mean_goal_diff)
end
println("\n=== RAW PANEL GOALS BY SURFACE × TIMING ===")
show(stdout, MIME"text/plain"(), r08_goals; allrows = true)
println()

# %%
# ===================================================================
# 7. Report
# ===================================================================
CSV.write(joinpath(R08_OUT_DIR, "r08_proper_scores.csv"), r08_scores)
CSV.write(joinpath(R08_OUT_DIR, "r08_paired_bootstrap.csv"), r08_boot)
CSV.write(joinpath(R08_OUT_DIR, "r08_coefficients.csv"), r08_coefs)
CSV.write(joinpath(R08_OUT_DIR, "r08_panel_goals.csv"), r08_goals)

f5 = v -> gph_num(v; digits = 5)
f4 = v -> gph_num(v; digits = 4)
f3 = v -> gph_num(v; digits = 3)
open(joinpath(R08_OUT_DIR, "r08_evaluation_report.md"), "w") do io
    println(io, "# r08 proper scores — Task 008 Phase 2\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ". Panel ", length(r08_panel),
            " fixtures; de-vigged Betfair TWA(−20, 0] close. Control `", R08_CONTROL, "` reproduced ",
            @sprintf("LogLoss %.5f / ECE %.4f", r08_ctl.logloss, r08_ctl.ece),
            ". T003 fixtures: ", length(r08_unmapped), ".\n")
    for scope in R08_SCOPES
        sub = sort(filter(:scope => ==(scope), r08_scores), :logloss)
        nrow(sub) == 0 && continue
        println(io, "## Scope: ", scope, "\n")
        print(io, gph_markdown_table(select(sub, :model, :role, :n_obs, :logloss, :market_logloss,
                                            :brier, :rps, :ece, :market_ece);
            formats = Dict(:logloss => f5, :market_logloss => f5, :brier => f5, :rps => f5,
                           :ece => f4, :market_ece => f4)))
        println(io)
    end
    println(io, "## Paired ΔLogLoss, fixture-clustered bootstrap (B = ", R08_B, ")\n")
    println(io, "Negative Δ favours the left arm. 95% percentile interval.\n")
    print(io, gph_markdown_table(select(r08_boot, :scope, :cut, :left, :right, :n_obs, :n_fixtures,
                                        :delta, :lo, :hi, :p_negative, :significant);
        formats = Dict(:delta => v -> gph_signed(v; digits = 5), :lo => v -> gph_signed(v; digits = 5),
                       :hi => v -> gph_signed(v; digits = 5), :p_negative => f3)))
    println(io, "\n## Coefficients vs prior\n")
    print(io, gph_markdown_table(select(r08_coefs, :model, :fold, :site, :mean, :sd, :q05, :q95,
                                        :p_positive, :prior_p_positive, :contraction, :h_pass);
        formats = Dict(:mean => f4, :sd => f4, :q05 => f4, :q95 => f4, :p_positive => f3,
                       :prior_p_positive => f3, :contraction => f3)))
    println(io, "\n## Raw panel goals by surface × timing\n")
    print(io, gph_markdown_table(r08_goals; formats = Dict(:mean_total => f3, :mean_home => f3,
                                                          :mean_away => f3, :mean_goal_diff => f3)))
end
println("\nR08_DONE report=", joinpath(R08_OUT_DIR, "r08_evaluation_report.md"))
