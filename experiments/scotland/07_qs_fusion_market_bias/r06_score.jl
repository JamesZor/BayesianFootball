# Wave 3 Phase 4: wave 2's r06_score.jl, r06_interval_audit.jl and r11_score_panel_audit.jl on
# the three fusion grids, scored against fusion_qs_bias (the control). Saved UUIDs only: no
# sampling, reference refits or database writes. Same harness scorer, goal channels, bootstrap
# seeds and blocks as wave 2. Every interval lives in one table: noncircular 8-week 90% (the
# classification), circular 8-week 90% (sensitivity, contradictions flagged) and the harness's
# fixture-clustered 95%.
using BayesianFootball, CSV, DataFrames, Dates, Distributions, LinearAlgebra, Printf, Random
using Statistics, ThreadPinning, UUIDs
pinthreads(:cores)
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" || error("score on beast only")
# candidates.jl loads wave 1's market module at top level as Main.QSMarketArm (provenance rule).
const P4C = Module(:Wave3ScoreCandidates)
Base.include(P4C, joinpath(@__DIR__, "candidates.jl"))
const W1 = joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market")
const W2 = joinpath(@__DIR__, "..", "06_qs_joint_and_market_observation")
isdefined(Main, :QSScoringAdapter) || include(joinpath(W1, "scoring_adapter.jl"))
const H = BayesianFootball.Harness
const P4_E = BayesianFootball.Evaluation
const P4_OUT = "/root/BF_runs/qs_experiment_w3_out/phase4"
mkpath(P4_OUT)
const P4_B, P4_BLOCK, P4_SEED = 999, 8, 20261009      # wave 2's block bootstrap
const P4_SEASONS = ["24/25", "25/26"]
const P4_W2 = "scottish_lower_qs_wave2_2426"
const P4_REFERENCES = ["qs_marketobs" => "81fdf817-f82d-4f66-9bfd-5f456c4fe29f",
                       "qs_joint" => "2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea"]

# %% 1. Saved wave-3 grids and wave-2 references (never fit)
p4_ds = Main.QSMarketArm.checked_datastore()
p4_runs = CSV.read(joinpath(@__DIR__, "results", "phase3", "RUNS.csv"), DataFrame)
p4_db = Training.PostgresStorage(P4C.EXPERIMENT)
p4_fits = Dict{String,Any}()
p4_refs = H.RunRef[]
for c in P4C.CANDIDATES
    id = string(only(filter(r -> r.label == c.name, p4_runs)).run_id)
    receipt = strip(read(joinpath(@__DIR__, "results", "phase3", "run_$(c.name).txt"), String))
    id == receipt || error("RUNS.csv and receipt differ for $(c.name)")
    fit = Training.load_fit(p4_db, UUID(id))
    length(fit.folds) == 40 || error("incomplete $(c.name)")
    H._structural_equal(fit.config.model, c.model) || error("model mismatch $(c.name)")
    H._structural_equal(fit.config.sampler, c.sampler) || error("budget mismatch $(c.name)")
    p4_fits[c.name] = fit
    push!(p4_refs, H.RunRef(c.name, P4C.EXPERIMENT, UUID(id), c.role))
end
p4_w2db = Training.PostgresStorage(P4_W2)
p4_w2runs = CSV.read(joinpath(W2, "results", "phase3", "RUNS.csv"), DataFrame)
for (arm, id) in P4_REFERENCES
    string(only(filter(r -> r.label == arm, p4_w2runs)).run_id) == id || error("reference id $arm")
    fit = Training.load_fit(p4_w2db, UUID(id))
    length(fit.folds) == 40 || error("incomplete reference $arm")
    p4_fits[arm] = fit
    push!(p4_refs, H.RunRef(arm, P4_W2, UUID(id), :candidate))
end
p4_ctl = only(filter(r -> r.role === :control, p4_refs))
p4_ctl.label == P4C.CONTROL || error("control is not $(P4C.CONTROL)")

# %% 2. Unchanged harness scorer; reference rows must equal wave 2's saved rows exactly
p4_scores = H.score_fits(Pair[r => p4_fits[r.label] for r in p4_refs]; ds = p4_ds,
    tiers = H.club_season_tiers(p4_ds), control = p4_ctl, target_seasons = P4_SEASONS,
    expected_fixtures = 710, bootstrap_B = 10_000)
p4_scores.stage .= "grid"
CSV.write(joinpath(P4_OUT, "harness_scores_vs_fusion_qs_bias.csv"), p4_scores)
p4_w2scores = CSV.read(joinpath(W2, "results", "phase4", "harness_scores_vs_grw_joint.csv"), DataFrame)
p4_independent(df, arm) = filter(r -> r.model == arm && !startswith(r.metric, "delta_"), df)
p4_parity = NamedTuple[]
for (arm, _) in P4_REFERENCES
    gap = QSScoringAdapter.max_difference(p4_independent(p4_w2scores, arm), p4_independent(p4_scores, arm))
    push!(p4_parity, (; check = "harness_independent_scores", arm,
        rows = nrow(p4_independent(p4_scores, arm)), max_abs_diff = gap))
end

# %% 3. Per-observation frames (harness functions, same odds and context as wave 2)
p4_odds = H._betfair_closing_odds(p4_ds)
p4_lower = Set(Int(r.match_id) for r in eachrow(p4_ds.matches) if Int(r.tournament_id) in (56, 57))
p4_odds = p4_odds[in.(p4_odds.match_id, Ref(p4_lower)), :]
p4_ctx = Dict{String,Any}()
function p4_frame(label, fit)
    panel = H._season_panel(p4_ds, fit, P4_SEASONS)
    length(panel) == 710 || error("$label covers $(length(panel)) fixtures")
    ctx = H._context(H._restrict(fit, panel), p4_odds, p4_ds)
    p4_ctx[label] = ctx
    frame = H._observation_frame(label, ctx, p4_odds)
    frame.selection = String.(frame.selection)
    frame.family = String.(frame.family)
    return frame
end
p4_frames = Dict(r.label => p4_frame(r.label, p4_fits[r.label]) for r in p4_refs)
# Market close: p_model := p_market on the control's observation rows (wave 2's construction).
p4_close = copy(p4_frames[P4C.CONTROL])
p4_close.p_model .= p4_close.p_market
p4_close.ll_model .= p4_close.ll_market
p4_close.model .= "market_close"
p4_frames["market_close"] = p4_close
const P4_ARMS = [P4C.CONTROL, "fusion_qs_nobias", "fusion_grw_bias", "qs_marketobs", "qs_joint",
                 "market_close"]

# Reference frames must equal wave 2's committed frames value for value.
p4_w2frames = CSV.read(joinpath(W2, "results", "phase4", "observation_frames.csv"), DataFrame)
for arm in ("qs_marketobs", "qs_joint", "market_close")
    k = [:family, :match_id, :selection]
    old = filter(:model => ==(arm), p4_w2frames)
    old.selection, old.family = String.(old.selection), String.(old.family)
    new = copy(p4_frames[arm])
    sort!(old, k); sort!(new, k)
    nrow(old) == nrow(new) && isequal(select(old, k), select(new, k)) || error("frame keys differ $arm")
    gap = maximum(maximum(abs.(old[!, c] .- new[!, c])) for c in
                  (:p_model, :p_market, :y, :ll_model, :ll_market))
    push!(p4_parity, (; check = "observation_frame", arm, rows = nrow(new), max_abs_diff = gap))
end
CSV.write(joinpath(P4_OUT, "observation_frames.csv"), vcat((p4_frames[a] for a in P4_ARMS)...))

# %% 4. Standard 3-way categorical LogLoss and RPS on the common 1X2 panel (wave 2's r11 law)
p4_common = filter(:family => ==("1X2"), p4_frames["market_close"])
p4_keys = sort([(Int(r.match_id), r.selection) for r in eachrow(p4_common)])
p4_ids = Set(first.(p4_keys))
length(p4_ids) == 595 && length(p4_keys) == 1785 || error("common 1X2 panel is not 595/1785")
function p4_categorical(arm)
    f = filter(:family => ==("1X2"), p4_frames[arm])
    sort([(Int(r.match_id), r.selection) for r in eachrow(f)]) == p4_keys || error("1X2 keys differ $arm")
    rows = NamedTuple[]
    for g in groupby(f, :match_id)
        ix = [only(findall(==(s), g.selection)) for s in ("home", "draw", "away")]
        p, y = g.p_model[ix], g.y[ix]
        sum(y) == 1 || error("noncategorical outcome $arm")
        push!(rows, (; arm, match_id = Int(first(g.match_id)),
            categorical_ll = -log(p[only(findall(==(1.0), y))] / sum(p)),
            rps = ((p[1] - y[1])^2 + (p[1] + p[2] - y[1] - y[2])^2) / 2,
            binary_ll_sum = sum(g.ll_model), prob_sum = sum(p)))
    end
    length(rows) == 595 || error("categorical panel $arm")
    return sort!(DataFrame(rows), :match_id)
end
p4_cat = vcat((p4_categorical(a) for a in P4_ARMS)...)
CSV.write(joinpath(P4_OUT, "categorical_1x2_fixtures.csv"), p4_cat)
p4_w2panel = CSV.read(joinpath(W2, "results", "score_panel_audit.csv"), DataFrame)
for arm in ("qs_marketobs", "qs_joint", "market_close")
    old = only(filter(:model => ==(arm), p4_w2panel))
    c = filter(:arm => ==(arm), p4_cat)
    gap = max(abs(mean(c.categorical_ll) - old.categorical_logloss), abs(mean(c.rps) - old.common_rps),
              abs(sum(c.binary_ll_sum) / 1785 - old.binary_logloss))
    push!(p4_parity, (; check = "common_panel_scores", arm, rows = nrow(c), max_abs_diff = gap))
end

# The harness's own RPS population per arm (Evaluation.ranked_probability_score's traversal);
# the close's is 1528 fixtures, so the headline uses the common panel instead.
function p4_rps_population(ctx; source)
    slots = Dict{Int,Vector{Float64}}()
    for i in 1:ctx.odds.n
        sel = ctx.odds.selection[i]
        k = sel === :home ? 1 : sel === :draw ? 2 : sel === :away ? 3 : 0
        k == 0 && continue
        p = source === :market ? (ctx.odds.has_fair[i] ? ctx.odds.prob_fair_close[i] : NaN) :
            something(P4_E.prob_mean(ctx.probs, ctx.odds.match_id[i], sel), NaN)
        isfinite(p) || continue
        get!(slots, ctx.odds.match_id[i], fill(NaN, 3))[k] = p
    end
    ids = [id for (id, p) in slots if all(isfinite, p) && P4_E.outcome_of(ctx.outcomes, id) !== nothing]
    actual, n = P4_E.ranked_probability_score(ctx; source)
    length(ids) == n || error("RPS traversal mismatch")
    return ids, actual
end
p4_panel = NamedTuple[]
for arm in P4_ARMS
    source = arm == "market_close" ? :market : :model
    ids, harness_rps = p4_rps_population(p4_ctx[arm == "market_close" ? P4C.CONTROL : arm]; source)
    c = filter(:arm => ==(arm), p4_cat)
    push!(p4_panel, (; model = arm, harness_rps_fixtures = length(ids), harness_rps,
        harness_extra_fixtures = length(setdiff(Set(ids), p4_ids)),
        harness_missing_common = length(setdiff(p4_ids, Set(ids))),
        common_fixtures = 595, common_selections = 1785,
        binary_logloss_1x2 = sum(c.binary_ll_sum) / 1785, categorical_logloss_1x2 = mean(c.categorical_ll),
        rps_1x2 = mean(c.rps), max_probability_sum_error = maximum(abs.(c.prob_sum .- 1))))
end
CSV.write(joinpath(P4_OUT, "score_panel_audit.csv"), DataFrame(p4_panel))

# %% 5. Goal log score: joint double Poisson = total (Poisson Λ) + allocation (Binomial), mixed
# over posterior draws (wave 2's law). All 710 fixtures; no market-close counterpart.
p4_match = Dict(Int(r.match_id) => r for r in eachrow(p4_ds.matches))
p4_logmeanexp(v) = (m = maximum(v); m + log(mean(exp.(v .- m))))
function p4_goal_scores(label, latents)
    rows = NamedTuple[]
    for (i, id) in enumerate(BayesianFootball.Models.latent_match_ids(latents))
        m = p4_match[Int(id)]
        (ismissing(m.home_score) || ismissing(m.away_score)) && error("no score for $id")
        String(m.season) in P4_SEASONS || continue
        h, a = Int(m.home_score), Int(m.away_score)
        λh, λa = view(latents.λ_home, i, :), view(latents.λ_away, i, :)
        joint = p4_logmeanexp(logpdf.(Poisson.(λh), h) .+ logpdf.(Poisson.(λa), a))
        total = p4_logmeanexp(logpdf.(Poisson.(λh .+ λa), h + a))
        push!(rows, (; arm = label, match_id = Int(id), joint, total, allocation = joint - total))
    end
    return DataFrame(rows)
end
p4_goal = vcat((p4_goal_scores(a, p4_fits[a].latents) for a in P4_ARMS[1:5])...)
CSV.write(joinpath(P4_OUT, "goal_logscore_fixtures.csv"), p4_goal)
p4_w2goal = CSV.read(joinpath(W2, "results", "phase4", "goal_logscore_fixtures.csv"), DataFrame)
for (arm, _) in P4_REFERENCES
    a = sort(filter(:arm => ==(arm), p4_w2goal), :match_id)
    b = sort(filter(:arm => ==(arm), p4_goal), :match_id)
    nrow(a) == nrow(b) == 710 && a.match_id == b.match_id || error("goal panel $arm")
    gap = maximum(maximum(abs.(a[!, c] .- b[!, c])) for c in (:joint, :total, :allocation))
    push!(p4_parity, (; check = "goal_logscore_fixtures", arm, rows = nrow(b), max_abs_diff = gap))
end
p4_parity_df = DataFrame(p4_parity)
# Exact everywhere except common-panel means, whose summation order differs from wave 2's r11
# (wave 2's own r11 tolerance, 1e-14).
p4_parity_df.tolerance = [r.check == "common_panel_scores" ? 1e-14 : 0.0 for r in eachrow(p4_parity_df)]
p4_parity_df.passed = p4_parity_df.max_abs_diff .<= p4_parity_df.tolerance
CSV.write(joinpath(P4_OUT, "reference_parity.csv"), p4_parity_df)
all(p4_parity_df.passed) || error("reference parity failed; see reference_parity.csv")

# %% 6. Paired intervals. 8-week moving blocks of calendar weeks within season, B = 999,
# seed 20261009 (wave 1's bc_bootstrap, verbatim law); each replicate is an observation-
# weighted mean exactly like the point estimate.
function p4_week(id)
    d = Date(p4_match[id].match_date)
    return (String(p4_match[id].season), d - Day(dayofweek(d) - 1))
end
function p4_bootstrap(fixture_ids, sums, counts; circular::Bool, B = P4_B, block = P4_BLOCK,
                      seed = P4_SEED)
    weeks = Dict{Tuple{String,Date},Vector{Int}}()
    for (i, id) in enumerate(fixture_ids)
        push!(get!(weeks, p4_week(id), Int[]), i)
    end
    seasons = Dict{String,Vector{Vector{Int}}}()
    for key in sort!(collect(keys(weeks)))
        push!(get!(seasons, key[1], Vector{Int}[]), weeks[key])
    end
    rng = Random.Xoshiro(seed)
    stats = zeros(B)
    for b in 1:B
        total, n = 0.0, 0.0
        for season in sort!(collect(keys(seasons)))
            ws = seasons[season]
            W = length(ws)
            L = min(block, W)
            chosen = Int[]
            while length(chosen) < W
                if circular
                    s = rand(rng, 1:W)
                    append!(chosen, [mod1(s + k, W) for k in 0:(L - 1)])
                else
                    s = rand(rng, 1:(W - L + 1))
                    append!(chosen, s:(s + L - 1))
                end
            end
            for w in chosen[1:W], i in ws[w]
                total += sums[i]
                n += counts[i]
            end
        end
        stats[b] = total / n
    end
    return (; point = sum(sums) / sum(counts), mean = mean(stats), lo = quantile(stats, 0.05),
              hi = quantile(stats, 0.95))
end
classify(lo, hi) = hi < 0 ? "better" : lo > 0 ? "worse" : "no detectable difference"
function p4_interval(tier, a, b, metric, market, ids, sums, counts, fa, fb)
    nc = p4_bootstrap(ids, sums, counts; circular = false)
    cc = p4_bootstrap(ids, sums, counts; circular = true)
    cl = H._paired_bootstrap(fa, fb; B = 10_000)
    isapprox(cl.delta, nc.point; atol = 1e-12, rtol = 0) || error("point estimates differ $a/$b/$metric/$market")
    class, circular_class = classify(nc.lo, nc.hi), classify(cc.lo, cc.hi)
    return (; tier, arm = a, reference = b, metric, market, n_obs = Int(sum(counts)),
        n_fixtures = length(ids), point = nc.point, noncircular_mean = nc.mean,
        noncircular_lo90 = nc.lo, noncircular_hi90 = nc.hi, class, circular_mean = cc.mean,
        circular_lo90 = cc.lo, circular_hi90 = cc.hi, circular_class,
        circular_contradicts = circular_class != class, cluster_lo95 = cl.lo, cluster_hi95 = cl.hi,
        cluster_class = classify(cl.lo, cl.hi))
end
# The brief's pairs in order; tier 5 is context only (each other arm against the close).
const P4_PAIRS = [("1_primary", P4C.CONTROL, "market_close"),
                  ("2_vs_marketobs", P4C.CONTROL, "qs_marketobs"),
                  ("3_bias_ablation", P4C.CONTROL, "fusion_qs_nobias"),
                  ("4_secondary", P4C.CONTROL, "fusion_grw_bias"),
                  ("4_secondary", P4C.CONTROL, "qs_joint"),
                  ("5_context", "fusion_qs_nobias", "market_close"),
                  ("5_context", "fusion_grw_bias", "market_close"),
                  ("5_context", "qs_marketobs", "market_close")]
p4_ll(frame, family) = family == "all" ? frame : frame[frame.family .== family, :]
p4_rows = NamedTuple[]
for (tier, a, b) in P4_PAIRS
    # Harness per-selection binary LogLoss, every market family.
    for family in ("1X2", "OU2.5", "BTTS", "all")
        fa, fb = p4_ll(p4_frames[a], family), p4_ll(p4_frames[b], family)
        j = innerjoin(select(fa, :match_id, :selection, :ll_model => :la),
                      select(fb, :match_id, :selection, :ll_model => :lb); on = [:match_id, :selection])
        nrow(j) == nrow(fa) == nrow(fb) || error("$a/$b $family observation sets differ")
        j.d = j.la .- j.lb
        pf = combine(groupby(j, :match_id), :d => sum => :d, nrow => :n)
        push!(p4_rows, p4_interval(tier, a, b, "logloss_binary", family, pf.match_id, pf.d, pf.n, fa, fb))
    end
    # Standard 3-way categorical LogLoss and RPS, common 595 fixtures.
    for (metric, col) in (("logloss_categorical", :categorical_ll), ("rps", :rps))
        ca, cb = filter(:arm => ==(a), p4_cat), filter(:arm => ==(b), p4_cat)
        ca.match_id == cb.match_id || error("categorical panels differ $a/$b")
        fa = DataFrame(match_id = ca.match_id, selection = fill(metric, nrow(ca)), ll_model = ca[!, col])
        fb = DataFrame(match_id = cb.match_id, selection = fill(metric, nrow(cb)), ll_model = cb[!, col])
        push!(p4_rows, p4_interval(tier, a, b, metric, "1X2", ca.match_id, ca[!, col] .- cb[!, col],
            ones(nrow(ca)), fa, fb))
    end
    # Goal channels (loss orientation: negative log score), all 710 fixtures.
    b == "market_close" && continue
    j = innerjoin(filter(:arm => ==(a), p4_goal), filter(:arm => ==(b), p4_goal);
                  on = :match_id, makeunique = true)
    nrow(j) == 710 || error("goal panel for $a/$b is $(nrow(j))")
    for channel in (:joint, :total, :allocation)
        la, lb = -j[!, channel], -j[!, Symbol(channel, "_1")]
        fa = DataFrame(match_id = j.match_id, selection = fill(String(channel), nrow(j)), ll_model = la)
        fb = DataFrame(match_id = j.match_id, selection = fill(String(channel), nrow(j)), ll_model = lb)
        push!(p4_rows, p4_interval(tier, a, b, "neg_goal_logscore", String(channel), j.match_id,
            la .- lb, ones(nrow(j)), fa, fb))
    end
end
p4_intervals = DataFrame(p4_rows)
CSV.write(joinpath(P4_OUT, "paired_intervals.csv"), p4_intervals)

# %% 7. Bias posteriors by fold, and δ_M with log κ read jointly. The market's per-side level
# is ζ + δ_M = η + log κ + δ_M, so δ_M is its offset from the goals model and δ_M + log κ its
# offset from the xG latent η; the two trade off wherever η's level is weakly pinned.
p4_draw(chain, site) = vec(Array(chain[Symbol(site)]))
const P4_SITES = ("obs.ν", "obs.log_κ", "obs.σ_obs", "obs.κ_D", "obs.δ_D", "obs.δ_M")
p4_post, p4_level = NamedTuple[], NamedTuple[]
for arm in P4_ARMS[1:3], fold in p4_fits[arm].folds
    present = Set(string.(names(fold.chain)))
    for site in P4_SITES
        site in present || continue
        x = p4_draw(fold.chain, site)
        q = quantile(x, [0.05, 0.5, 0.95])
        ref = site == "obs.κ_D" ? 1.0 : 0.0
        push!(p4_post, (; arm, fold = fold.fold, site, q05 = q[1], median = q[2], q95 = q[3],
            mean = mean(x), sd = std(x), ref, prob_below_ref = mean(x .< ref)))
    end
    bias = all(s -> s in present, ("obs.κ_D", "obs.δ_D", "obs.δ_M"))
    bias == (arm != "fusion_qs_nobias") || error("bias sites wrong for $arm fold $(fold.fold)")
    bias || continue
    lk, dm = p4_draw(fold.chain, "obs.log_κ"), p4_draw(fold.chain, "obs.δ_M")
    s = dm .+ lk
    q = quantile(s, [0.05, 0.5, 0.95])
    push!(p4_level, (; arm, fold = fold.fold, cor_deltaM_logkappa = cor(dm, lk),
        sd_deltaM = std(dm), sd_logkappa = std(lk), sd_sum = std(s), level_q05 = q[1],
        level_median = q[2], level_q95 = q[3], prob_level_below0 = mean(s .< 0)))
end
p4_post_df, p4_level_df = DataFrame(p4_post), DataFrame(p4_level)
nrow(p4_level_df) == 80 || error("expected 80 bias-arm folds")
CSV.write(joinpath(P4_OUT, "posterior_bias_by_fold.csv"), p4_post_df)
CSV.write(joinpath(P4_OUT, "posterior_level_joint_by_fold.csv"), p4_level_df)

# Posterior r by fold for the QS arms (wave 2's table; rₖ is absent in season-first folds).
p4_r = NamedTuple[]
for arm in P4_ARMS[1:2], fold in p4_fits[arm].folds
    present = Set(string.(names(fold.chain)))
    for scale in ("r₀", "rₛ", "rₖ")
        "dyn.$scale" in present || continue
        x = p4_draw(fold.chain, "dyn.$scale")
        q = quantile(x, [0.05, 0.5, 0.95])
        push!(p4_r, (; arm, fold = fold.fold, scale, r_q05 = q[1], r_median = q[2], r_q95 = q[3],
            rho_median = median((x .^ 2 .- 1) ./ (x .^ 2 .+ 1))))
    end
end
CSV.write(joinpath(P4_OUT, "posterior_r_by_fold.csv"), DataFrame(p4_r))

# %% 8. Native SVG of the bias posteriors (frozen CSV values only)
function p4_bias_svg(post, level, path)
    cols = [("obs.κ_D", "κ_D", 1.0), ("obs.δ_D", "δ_D", 0.0), ("obs.δ_M", "δ_M", 0.0),
            ("level", "δ_M + log κ", 0.0), ("obs.σ_obs", "σ_obs", NaN)]
    arms = (P4C.CONTROL, "fusion_grw_bias")
    panel(arm, key) = key == "level" ?
        sort(select(filter(:arm => ==(arm), level), :fold, :level_q05 => :q05,
                    :level_median => :median, :level_q95 => :q95), :fold) :
        sort(filter(r -> r.arm == arm && r.site == key, post), :fold)
    f(x) = @sprintf("%.2f", x)
    width, height = 200, 170
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560">
<title>Market bias posteriors by fold</title>
<desc>Each point is a posterior median; whiskers are 90 percent intervals. Dashed lines mark no bias.</desc>
<rect width="1200" height="560" fill="white"/>
<g font-family="sans-serif" fill="#222">
<text x="30" y="28" font-size="18">Market bias posteriors by fold: median and 90% interval</text>
<text x="30" y="48" font-size="12">Dashed: no bias (κ_D = 1, δ = 0). δ_M is the market level relative to the goals model; δ_M + log κ relative to the xG latent.</text>""")
        for (c, (key, label, ref)) in enumerate(cols)
            panels = [panel(arm, key) for arm in arms]
            all(p -> nrow(p) == 40, panels) || error("incomplete $key panel")
            lo = minimum(minimum(p.q05) for p in panels)
            hi = maximum(maximum(p.q95) for p in panels)
            isnan(ref) || (lo = min(lo, ref); hi = max(hi, ref))
            pad = 0.06 * (hi - lo)
            lo, hi = lo - pad, hi + pad
            for (a, arm) in enumerate(arms)
                left, top = 70 + 230 * (c - 1), 95 + 235 * (a - 1)
                x(fold) = left + (fold - 1) / 39 * width
                y(v) = top + height * (hi - v) / (hi - lo)
                color = a == 1 ? "#2266aa" : "#b54b20"
                println(io, "<text x=\"$left\" y=\"$(top - 12)\" font-size=\"14\">$arm: $label</text>")
                println(io, "<rect x=\"$left\" y=\"$top\" width=\"$width\" height=\"$height\" fill=\"none\" stroke=\"#ccc\"/>")
                for v in (lo + pad, (lo + hi) / 2, hi - pad)
                    println(io, "<text x=\"$(left - 6)\" y=\"$(f(y(v) + 4))\" text-anchor=\"end\" font-size=\"10\">$(f(v))</text>")
                end
                isnan(ref) || println(io, "<line x1=\"$left\" y1=\"$(f(y(ref)))\" x2=\"$(left + width)\" y2=\"$(f(y(ref)))\" stroke=\"#555\" stroke-dasharray=\"5 4\"/>")
                for r in eachrow(panels[a])
                    xx = f(x(r.fold))
                    println(io, "<line x1=\"$xx\" x2=\"$xx\" y1=\"$(f(y(r.q05)))\" y2=\"$(f(y(r.q95)))\" stroke=\"$color\" stroke-opacity=\"0.5\"/>")
                    println(io, "<circle cx=\"$xx\" cy=\"$(f(y(r.median)))\" r=\"2.4\" fill=\"$color\"/>")
                end
                for tick in (1, 20, 40)
                    println(io, "<text x=\"$(f(x(tick)))\" y=\"$(top + height + 15)\" text-anchor=\"middle\" font-size=\"10\">$tick</text>")
                end
            end
        end
        println(io, "<text x=\"600\" y=\"550\" text-anchor=\"middle\" font-size=\"12\">Fold</text>")
        println(io, "</g></svg>")
    end
end
p4_bias_svg(p4_post_df, p4_level_df, joinpath(P4_OUT, "posterior_bias_by_fold.svg"))

p4_primary = only(filter(r -> r.tier == "1_primary" && r.metric == "logloss_binary" &&
                         r.market == "1X2", p4_intervals))
println("PHASE4_SCORE_DONE pairs=$(nrow(p4_intervals)) contradictions=$(count(p4_intervals.circular_contradicts)) ",
        "primary_1x2=$(p4_primary.point) [$(p4_primary.noncircular_lo90), $(p4_primary.noncircular_hi90)] ",
        "class=$(p4_primary.class)")
