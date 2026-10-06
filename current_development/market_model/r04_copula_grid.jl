# Phase B3 only: do Gaussian R6 team steps reproduce their own tail diagnostics,
# and which single-global-parameter score grid reproduces the quoted book?
# NOT Phase C: the human keeps double Poisson for C, and decides on C2 later.
# All inversions use the same Phase A accepted population, lines, KL and gates.
# 10a theta is retrospective; 10b theta uses honest_train only. Both descriptive
# FFBS PPCs condition on the full observed panel; they are NOT honest forecasts.
# PPC refilters newly generated panels at every theta draw, avoiding a naive
# comparison of shrunk FFBS steps with Gaussian theory.
# Replaceable scientific CSVs: results/B3; binary chains ignored. No DB writes.
# Run ONLY in a fresh owned beast tmux REPL, --project -t 16, pinned Manifest/cache.
# Two full fresh runs at the same HEAD/seeds must reproduce CSV bytes.
# Anchored completion marker: R04_DONE.

# %%
# ===================================================================
# 1. Packages and local implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, ThreadPinning, CSV, DataFrames, SHA
ENV["GKSwstype"] = "100"
using Plots
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l04_copula_grid.jl"))
const B3 = CopulaGridMarket

# %%
# ===================================================================
# 2. Visible configuration, runtime and scope gates
# ===================================================================
const B3_CONFIG = B3.MM.scottish_lower_2425_2526()
const B3_OUT = joinpath(@__DIR__,"results","B3")
const B3_FIG = joinpath(@__DIR__,"results","figures")
const B3_PPC_DRAWS = 200
const B3_PPC_SEED = 3942
const B3_PROFILE_CANDIDATES = Dict(
    1=>[-0.15,-0.10,-0.05,0.0,0.03],
    2=>[0.0,0.05,0.10,0.20,0.40],
    3=>[-1.0,0.0,0.5,1.0,2.0,4.0])
mkpath(B3_OUT)
mkpath(B3_FIG)
b3_started = time()
b3_load = parse.(Float64,split(read("/proc/loadavg",String))[1:3])
println("load averages = ",b3_load)
if maximum(b3_load) > 20
    println("PHASEB3_BLOCKED")
    error("load average exceeds 20: $b3_load")
end
println("B3 SHA = ",strip(read(`git rev-parse HEAD`,String)),"; threads = ",Threads.nthreads())

# %%
# ===================================================================
# 3. Pinned books and panel: Phase A identities and Phase B outcome columns
# ===================================================================
b3_cache = joinpath(pwd(),".cache","datastore_ScottishLower.jls")
b3_hash = open(io->bytes2hex(SHA.sha256(io)),b3_cache)
b3_hash == "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4" || error("wrong pinned cache")
b3_ds = BayesianFootball.Data.load_datastore_cached(B3_CONFIG.segment; max_age_hours=10^6)
b3_bridge = B3.TB.phase_b_panel(b3_ds; config=B3_CONFIG)
b3_panel = b3_bridge.panel
b3_rates = filter(:accepted=>identity,b3_bridge.rates)
b3_book,_ = B3.MM.gated_close(b3_ds,B3_CONFIG)
b3_books = Dict(Int(first(g.match_id))=>DataFrame(g) for g in groupby(b3_book,:match_id))
println("Accepted books = ",nrow(b3_rates),"; structural fixtures = ",B3.MID.n_fixtures(b3_panel))
println("10b theta train/test = ",B3_CONFIG.honest_train," / ",B3_CONFIG.honest_test)

# %%
# ===================================================================
# 4. Global dependence profiles: all accepted books, multistart KL inversion
# ===================================================================
b3_grids = Dict(0=>B3.ScoreGrid(0))
b3_frames = Dict(0=>B3.invert_accepted(b3_grids[0],b3_rates,b3_books))
b3_profiles = DataFrame[]
for kind in 1:3
    g,frame,profile = B3.profile_grid(kind,b3_rates,b3_books,B3_OUT;
        candidates=B3_PROFILE_CANDIDATES[kind])
    b3_grids[kind] = g
    b3_frames[kind] = frame
    push!(b3_profiles,profile)
end
b3_profile = vcat(b3_profiles...)
CSV.write(joinpath(B3_OUT,"grid_profile.csv"),b3_profile)

# %%
# ===================================================================
# 5. Book fit, heldout lines, 1X2-only bias, paired outcome losses and rates
# ===================================================================
# The global parameter is estimated on full books. Heldout means heldout lines
# at fixed global parameter, not independent fixture CV; disclose this leakage.
b3_comparison = B3.compare_grids(b3_ds,b3_rates,b3_books,b3_grids,b3_frames,B3_OUT)
b3_best = b3_comparison.winner
println("Lowest heldout mean absolute selection error: G",b3_best)

# %%
# ===================================================================
# 6. R6 fits: same priors/budgets as B2, both protocols, convergence before use
# ===================================================================
b3_g0_fits = B3.refit_r6(b3_panel,B3_CONFIG,B3_OUT,"G0")
b3_r6_tables = [B3.r6_measures(b3_panel,B3_CONFIG,b3_g0_fits,"G0")]
if b3_best != 0
    b3_best_panel = B3.panel_with_rates(b3_panel,b3_frames[b3_best])
    b3_best_fits = B3.refit_r6(b3_best_panel,B3_CONFIG,B3_OUT,"G$b3_best")
    push!(b3_r6_tables,B3.r6_measures(b3_best_panel,B3_CONFIG,b3_best_fits,"G$b3_best"))
end
b3_best_bias = only(filter(:grid=>==("G$b3_best"),b3_comparison.bias).mean)
if abs(b3_best_bias) < 0.03
    # Config drives inclusion: retain A's other refusals; remove only thin-book exclusion.
    b3_allframe = copy(b3_bridge.frame)
    b3_byid = Dict(r.match_id=>r for r in eachrow(b3_frames[b3_best]))
    for r in eachrow(b3_allframe)
        haskey(b3_byid,r.match_id) || continue
        fit = b3_byid[r.match_id]
        r.accepted = true
        r.lambda_mkt_h = fit.lambda_h
        r.lambda_mkt_a = fit.lambda_a
        r.sse = fit.kl
        r.reason = ""
    end
    b3_extended = B3.MID.build_market_panel(b3_ds,b3_allframe; config=B3_CONFIG)
    b3_extended_fits = B3.refit_r6(b3_extended,B3_CONFIG,B3_OUT,"G$(b3_best)_including_thin")
    push!(b3_r6_tables,B3.r6_measures(b3_extended,B3_CONFIG,b3_extended_fits,"G$(b3_best)_including_thin"))
end
CSV.write(joinpath(B3_OUT,"r6_on_best_grid.csv"),vcat(b3_r6_tables...))

# %%
# ===================================================================
# 7. Gaussian dependence PPC: observed AND refiltered replicate FFBS paths
# ===================================================================
# Endpoints from first through last accepted match in each team-season, t>=2.
# Every theta draw regenerates all weeks/fixtures including summer; active mask
# is identical for observed and replicate statistics. Centred observable steps.
b3_tail,b3_largest = DataFrame[],DataFrame[]
for protocol in ("10a","10b")
    fit = b3_g0_fits[protocol]
    # Deterministic stratification across chains and retained samples, >=200 draws.
    thetas = vcat([fit.udraws[:,:,c] for c in axes(fit.udraws,3)]...)
    raw,quality,keys = B3.tail_ppc(b3_panel,thetas; draws=B3_PPC_DRAWS,
        seed=B3_PPC_SEED+(protocol == "10b" ? 1 : 0))
    raw.protocol .= protocol
    push!(b3_tail,raw)
    push!(b3_largest,B3.largest_steps(b3_panel,quality,keys,protocol))
end
b3_tail_raw = vcat(b3_tail...)
b3_tail_summary = B3.tail_summary(b3_tail_raw)
CSV.write(joinpath(B3_OUT,"tail_ppc.csv"),b3_tail_summary)
CSV.write(joinpath(B3_OUT,"tail_ppc_draws.csv"),b3_tail_raw)
CSV.write(joinpath(B3_OUT,"largest_quality_steps.csv"),vcat(b3_largest...))
if any(b3_tail_summary.extreme)
    error("Extreme Gaussian PPC: Student-t quality-step alternative must be implemented/fitted before B3 promotion")
else
    println("No PPC p-value <0.01 or >0.99; no alternative step model fitted.")
end

# %%
# ===================================================================
# 8. Figures and final provenance (wall times never enter scientific CSVs)
# ===================================================================
B3.b3_figures(Plots,b3_profile,b3_comparison,b3_tail_raw,B3_FIG)
open(joinpath(B3_OUT,"RUN_PROVENANCE.md"),"a") do io
    println(io,"- Full R04 SHA `",strip(read(`git rev-parse HEAD`,String)),"`; ",
        time()-b3_started," seconds; threads=",Threads.nthreads(),"; R04_DONE.")
end
println("R04_DONE")
