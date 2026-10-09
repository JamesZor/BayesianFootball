# Wave 2 Phase 4: adapted wave-1 r05_score.jl, same scorer/goal channels/bootstrap seeds.
# Saved UUIDs only; no sampling, reference refits or database writes.
# Reuses r05_block_check.jl and r05_goal_cluster.jl unchanged.
using BayesianFootball, CSV, DataFrames, Dates, Distributions, LinearAlgebra, Random
using Serialization, Statistics, ThreadPinning, UUIDs
pinthreads(:cores)
BLAS.set_num_threads(1)
const W1 = joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market")
isdefined(Main, :QSMarketArm) || include(joinpath(W1, "market_arm.jl"))
isdefined(Main, :QSScoringAdapter) || include(joinpath(W1, "scoring_adapter.jl"))
const S5C = Module(:QSScoreCandidates)
Base.include(S5C, joinpath(@__DIR__, "candidates.jl"))
const H5 = BayesianFootball.Harness
const S5_OUT = "/root/BF_runs/qs_experiment_w2_out/phase4"
mkpath(S5_OUT)
const S5_B = 999
const S5_BLOCK = 8
const S5_SEED = 20261009

# %% 1. Load immutable wave-2 and wave-1 references (never fit)
s5_ds = Main.QSMarketArm.checked_datastore()
s5_db = Training.PostgresStorage(S5C.EXPERIMENT)
s5_fits = Dict{String,Any}()
s5_refs = H5.RunRef[]
s5_ids = CSV.read(joinpath(@__DIR__, "results", "phase3", "RUNS.csv"), DataFrame)
for c in S5C.CANDIDATES
    id = only(filter(r -> r.label == c.name, s5_ids)).run_id
    fit = Training.load_fit(s5_db, UUID(string(id)))
    length(fit.folds) == 40 || error("incomplete $(c.name)")
    H5._structural_equal(fit.config.model, c.model) || error("model mismatch $(c.name)")
    H5._structural_equal(fit.config.sampler, c.sampler) || error("budget mismatch $(c.name)")
    s5_fits[c.name] = fit
    push!(s5_refs, H5.RunRef(c.name, S5C.EXPERIMENT, id, c.role))
end
s5_ctl = only(filter(r -> r.role === :control, s5_refs))
s5_w1db = Training.PostgresStorage("scottish_lower_quality_style_2426")
for (arm, id) in (("control_grw", "a036d22a-ff32-404c-b801-5f928d8a89f4"),
                  ("qs_weak_r", "21f2a9f9-b96f-4034-97de-767704a9d54a"))
    s5_fits[arm] = Training.load_fit(s5_w1db, UUID(id))
    push!(s5_refs, H5.RunRef(arm, "scottish_lower_quality_style_2426", id, :candidate))
end
s5_template = s5_fits["control_grw"]
s5_market = Serialization.deserialize("/root/BF_runs/qs_experiment_out/market_grid/market_latents.jls")
const S5_MARKET_ID = uuid5(UUID("6f1c3c2e-7a3b-4d55-9a8f-0b2d5c1e9a40"), "qs_market_c0_40fold")
s5_market_ref = H5.RunRef("market_c0", "scottish_lower_quality_style_2426", S5_MARKET_ID, :candidate)

# %% 2. Unchanged harness scorer; references checked against committed wave-1 scores
s5_pairs = Pair[r => s5_fits[r.label] for r in s5_refs]
push!(s5_pairs, s5_market_ref => QSScoringAdapter.wrap(s5_template, s5_market))
s5_tiers = H5.club_season_tiers(s5_ds)
s5_scores = H5.score_fits(s5_pairs; ds = s5_ds, tiers = s5_tiers, control = s5_ctl,
    target_seasons = ["24/25", "25/26"], expected_fixtures = 710, bootstrap_B = 10_000)
s5_scores.stage .= "grid"
s5_reference = CSV.read(joinpath(W1, "results", "harness_scores_vs_control_grw.csv"), DataFrame)
s5_parity = NamedTuple[]
for arm in ("control_grw", "qs_weak_r", "market_c0")
    independent(df) = filter(r -> r.model == arm && !startswith(r.metric, "delta_"), df)
    a, b = independent(s5_reference), independent(s5_scores)
    gap = QSScoringAdapter.max_difference(a, b)
    gap == 0 || error("saved reference scores differ $arm: $gap")
    push!(s5_parity, (; arm, rows=nrow(a), max_abs_diff=gap, exact=true))
end
CSV.write(joinpath(S5_OUT, "reference_score_parity.csv"), DataFrame(s5_parity))
# Publish the saved reference rows, not their newly paired-control deltas.
CSV.write(joinpath(S5_OUT, "wave1_reference_scores.csv"),
    filter(r -> r.model in ("control_grw", "qs_weak_r", "market_c0"), s5_reference))
CSV.write(joinpath(S5_OUT, "harness_scores_vs_grw_joint.csv"), s5_scores)
CSV.write(joinpath(S5_OUT, "RUNS.csv"), DataFrame(label = [r.label for r in vcat(s5_refs, [s5_market_ref])],
    experiment = [r.experiment for r in vcat(s5_refs, [s5_market_ref])],
    run_id = [string(r.run_id) for r in vcat(s5_refs, [s5_market_ref])]))

# %% 3. Per-observation frame for every arm (harness functions, same odds and context)
s5_odds = H5._betfair_closing_odds(s5_ds)
s5_lower = Set(Int(r.match_id) for r in eachrow(s5_ds.matches) if Int(r.tournament_id) in (56, 57))
s5_odds = s5_odds[in.(s5_odds.match_id, Ref(s5_lower)), :]
function s5_frame(label, fit)
    panel = H5._season_panel(s5_ds, fit, ["24/25", "25/26"])
    length(panel) == 710 || error("$label covers $(length(panel)) fixtures")
    ctx = H5._context(H5._restrict(fit, panel), s5_odds, s5_ds)
    return H5._observation_frame(label, ctx, s5_odds)
end
s5_frames = Dict(r.label => s5_frame(r.label, s5_fits[r.label]) for r in s5_refs)
s5_frames["market_c0"] = s5_frame("market_c0", QSScoringAdapter.wrap(s5_template, s5_market))
# Market close: p_model := p_market on the same observation rows.
s5_close = copy(s5_frames[s5_ctl.label])
s5_close.p_model .= s5_close.p_market
s5_close.ll_model .= s5_close.ll_market
s5_close.model .= "market_close"
s5_frames["market_close"] = s5_close

# %% 4. Calendar weeks within season, for 8-week moving blocks
s5_match = Dict(Int(r.match_id) => r for r in eachrow(s5_ds.matches))
function s5_week(id)
    d = Date(s5_match[id].match_date)
    return (String(s5_match[id].season), d - Day(dayofweek(d) - 1))
end

"""
999 noncircular moving blocks of whole weeks within league season (R07 scheme), 90% CI.
`sums[i]`/`counts[i]` are fixture i's summed difference and observation count, so every
replicate is an observation-weighted mean, exactly like the point estimate.
"""
function s5_block_bootstrap(fixture_ids, sums, counts = ones(length(sums));
                            B = S5_B, block = S5_BLOCK, seed = S5_SEED)
    weeks = Dict{Tuple{String,Date},Vector{Int}}()
    for (i, id) in enumerate(fixture_ids)
        push!(get!(weeks, s5_week(id), Int[]), i)
    end
    seasons = Dict{String,Vector{Vector{Int}}}()
    for key in sort!(collect(keys(weeks)))
        push!(get!(seasons, key[1], Vector{Int}[]), weeks[key])
    end
    rng = Random.Xoshiro(seed)
    stats = zeros(B)
    for b in 1:B
        total, n = 0.0, 0
        for season in sort!(collect(keys(seasons)))
            ws = seasons[season]
            L = min(block, length(ws))
            chosen = Int[]
            while length(chosen) < length(ws)
                s = rand(rng, 1:(length(ws) - L + 1))
                append!(chosen, s:(s + L - 1))
            end
            for w in chosen[1:length(ws)], i in ws[w]
                total += sums[i]
                n += counts[i]
            end
        end
        stats[b] = total / n
    end
    return (; mean = sum(sums) / sum(counts), lo = quantile(stats, 0.05),
              hi = quantile(stats, 0.95))
end

classify(lo, hi) = hi < 0 ? "better" : lo > 0 ? "worse" : "no detectable difference"

# %% 5. Paired market metrics, both bootstraps
s5_pairs_spec = vcat(
    [("primary_A", "qs_joint", "grw_joint")],
    [("primary_B", a, b) for a in ("grw_marketobs", "qs_marketobs") for b in ("market_c0", "grw_joint")],
    [("secondary", a, b) for a in ("grw_joint", "qs_joint") for b in ("control_grw", "qs_weak_r")],
    [("secondary", "qs_marketobs", "grw_marketobs")],
    [("secondary", a, "market_close") for a in ("grw_joint", "qs_joint", "grw_marketobs", "qs_marketobs")])
s5_ll(frame, family) = family == "all" ? frame : frame[frame.family .== family, :]
s5_rows = NamedTuple[]
for (tier, a, b) in s5_pairs_spec, family in ("1X2", "OU2.5", "BTTS", "all")
    fa, fb = s5_ll(s5_frames[a], family), s5_ll(s5_frames[b], family)
    j = innerjoin(select(fa, :match_id, :selection, :ll_model => :la),
                  select(fb, :match_id, :selection, :ll_model => :lb); on = [:match_id, :selection])
    nrow(j) == nrow(fa) == nrow(fb) || error("$a/$b $family observation sets differ")
    perfix = combine(groupby(transform(j, [:la, :lb] => ((x, y) -> x .- y) => :d), :match_id),
                     :d => sum => :d, nrow => :n)
    block = s5_block_bootstrap(perfix.match_id, perfix.d, perfix.n)
    # Harness fixture-clustered paired bootstrap (its own 95% interval, seed 20260911).
    clustered = H5._paired_bootstrap(fa, b == "market_close" ? :market : fb; B = 10_000)
    push!(s5_rows, (; tier, arm = a, reference = b, market = family, n_obs = nrow(j),
        n_fixtures = nrow(perfix), delta_logloss = mean(j.la .- j.lb),
        block8_lo90 = block.lo, block8_hi90 = block.hi,
        cluster_lo95 = clustered.lo, cluster_hi95 = clustered.hi,
        class_block8 = family in ("1X2", "OU2.5") ? classify(block.lo, block.hi) : ""))
end
s5_paired = DataFrame(s5_rows)
CSV.write(joinpath(S5_OUT, "paired_logloss.csv"), s5_paired)

# %% 6. Goal log score: joint double Poisson = total (Poisson Λ) + allocation (Binomial),
# integrated over posterior draws as a mixture (R07: log-mean-exp of per-draw densities).
logmeanexp(v) = (m = maximum(v); m + log(mean(exp.(v .- m))))
function s5_goal_scores(label, latents)
    rows = NamedTuple[]
    ids = BayesianFootball.Models.latent_match_ids(latents)
    for (i, id) in enumerate(ids)
        m = s5_match[Int(id)]
        (ismissing(m.home_score) || ismissing(m.away_score)) && error("no score for $id")
        String(m.season) in ("24/25", "25/26") || continue
        h, a = Int(m.home_score), Int(m.away_score)
        λh, λa = view(latents.λ_home, i, :), view(latents.λ_away, i, :)
        joint = logmeanexp(logpdf.(Poisson.(λh), h) .+ logpdf.(Poisson.(λa), a))
        total = logmeanexp(logpdf.(Poisson.(λh .+ λa), h + a))
        push!(rows, (; arm = label, match_id = Int(id), joint, total, allocation = joint - total))
    end
    return DataFrame(rows)
end
s5_goal = vcat([s5_goal_scores(r.label, s5_fits[r.label].latents) for r in s5_refs]...,
               s5_goal_scores("market_c0", s5_market))
CSV.write(joinpath(S5_OUT, "goal_logscore_fixtures.csv"), s5_goal)
s5_goal_rows = NamedTuple[]
for (tier, a, b) in s5_pairs_spec
    b == "market_close" && continue
    j = innerjoin(filter(:arm => ==(a), s5_goal), filter(:arm => ==(b), s5_goal);
                  on = :match_id, makeunique = true)
    nrow(j) == 710 || error("goal panel for $a/$b is $(nrow(j))")
    for channel in (:joint, :total, :allocation)
        # Loss orientation: negative log score difference (a − b); < 0 means a is better.
        d = -(j[!, channel] .- j[!, Symbol(channel, "_1")])
        bb = s5_block_bootstrap(j.match_id, d)
        fa = DataFrame(match_id = j.match_id, selection = fill(channel, nrow(j)),
                       ll_model = -j[!, channel])
        fb = DataFrame(match_id = j.match_id, selection = fill(channel, nrow(j)),
                       ll_model = -j[!, Symbol(channel, "_1")])
        clustered = H5._paired_bootstrap(fa, fb; B = 10_000)
        isapprox(clustered.delta, bb.mean; atol = 1e-12, rtol = 0) ||
            error("goal point estimates differ for $a/$b/$channel")
        # Goal scores have no market-close counterpart: the close is not a rate pair for the
        # 193 fixtures without an invertible book, so market_close pairs are skipped above.
        push!(s5_goal_rows, (; tier, arm = a, reference = b, channel = String(channel),
            n = nrow(j), delta_neg_logscore = bb.mean, block8_lo90 = bb.lo, block8_hi90 = bb.hi,
            class_block8 = classify(bb.lo, bb.hi),
            clustered_lo95 = clustered.lo, clustered_hi95 = clustered.hi))
    end
end
CSV.write(joinpath(S5_OUT, "paired_goal_logscore.csv"), DataFrame(s5_goal_rows))

# %% 7. Posterior r by fold for both QS arms, and implied step correlation ρ = (r²−1)/(r²+1)
# rₖ is a site only in folds with target steps (the first fold of each season has none).
s5_r_rows = NamedTuple[]
for arm in ("qs_joint", "qs_marketobs"), (k, fold) in enumerate(s5_fits[arm].folds)
    present = Set(string.(names(fold.chain)))
    for scale in ("r₀", "rₛ", "rₖ")
        "dyn.$scale" in present || continue
        x = vec(Array(fold.chain[Symbol("dyn.$scale")]))
        ρ = (x .^ 2 .- 1) ./ (x .^ 2 .+ 1)
        q = quantile(x, [0.05, 0.5, 0.95])
        push!(s5_r_rows, (; arm, fold = k, scale, r_q05 = q[1], r_median = q[2], r_q95 = q[3],
            rho_median = median(ρ)))
    end
end
CSV.write(joinpath(S5_OUT, "posterior_r_by_fold.csv"), DataFrame(s5_r_rows))
# %% 8. Measurement noise: full saved posterior, no C0 refit
s5_sigma = NamedTuple[]
for arm in ("grw_marketobs", "qs_marketobs"), fold in s5_fits[arm].folds
    sites = filter(x -> occursin("σ_obs", string(x)), names(fold.chain))
    site = only(sites)
    q = quantile(vec(Array(fold.chain[site])), [0.05, 0.5, 0.95])
    push!(s5_sigma, (; arm, fold=fold.fold, q05=q[1], median=q[2], q95=q[3]))
end
for k in 1:40
    fit = Serialization.deserialize(joinpath("/root/BF_runs/qs_experiment_out/market_grid",
        "fold_$(lpad(k, 2, '0'))", "fit.jls"))
    q = quantile(exp.(vec(fit.udraws[:, 1, :])), [0.05, 0.5, 0.95])
    push!(s5_sigma, (; arm="market_c0", fold=k, q05=q[1], median=q[2], q95=q[3]))
end
CSV.write(joinpath(S5_OUT, "posterior_sigma_by_fold.csv"), DataFrame(s5_sigma))
# Exact saved goal-score reference parity at fixture level
saved_goals = CSV.read("/root/BF_runs/qs_experiment_out/phase5/goal_logscore_fixtures.csv", DataFrame)
goal_parity = NamedTuple[]
for arm in ("control_grw", "qs_weak_r", "market_c0")
    a = sort(filter(:arm => ==(arm), saved_goals), :match_id)
    b = sort(filter(:arm => ==(arm), s5_goal), :match_id)
    isequal(a, b) || error("saved goal reference differs $arm")
    push!(goal_parity, (; arm, rows=nrow(a), exact=true))
end
CSV.write(joinpath(S5_OUT, "reference_goal_parity.csv"), DataFrame(goal_parity))
CSV.write(joinpath(S5_OUT, "wave1_reference_goal_fixtures.csv"),
    filter(:arm => x -> x in ("control_grw", "qs_weak_r", "market_c0"), saved_goals))
# Reuse wave-1 circular/noncircular and goal cluster paths without changing them.
include(joinpath(W1, "r05_block_check.jl"))
ENV["QSX_OUT"] = "/root/BF_runs/qs_experiment_w2_out/cluster_adapter"
mkpath(joinpath(ENV["QSX_OUT"], "phase5"))
for file in ("goal_logscore_fixtures.csv", "paired_goal_logscore.csv")
    cp(joinpath(S5_OUT, file), joinpath(ENV["QSX_OUT"], "phase5", file); force=true)
end
include(joinpath(W1, "r05_goal_cluster.jl"))
read(joinpath(S5_OUT, "paired_goal_logscore.csv")) ==
    read(joinpath(ENV["QSX_OUT"], "phase5", "paired_goal_logscore.csv")) || error("goal cluster byte parity")
println("PHASE4_SCORE_DONE")
