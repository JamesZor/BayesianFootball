module QSPaper2627Sampling

# Phase 1 (TODO 045): in-memory walk-forward sampling of the new 26/27 positions for
# qs_marketobs and m12, then held-out probabilities and log loss. Nothing is written to any
# database: the fold set is serialised under Q.OUT_DIR and small CSVs go to results/.
# The sampling path is `Training.extend_fit` minus its transaction (c4_extend_inmemory, copied),
# with wave 2's per-chain task-local seed loop (06_.../l05_grid.jl sample_pending!, copied).
# Comments only below; the static checker scans every non-comment line for write verbs.

import BayesianFootball as BF
import CSV
import DataFrames as DF
import Dates
import Random
import Serialization
import Statistics
import UUIDs

const Q = Main.QSPaper2627
const TR = BF.Training                    # sample_fold, FoldFit, load_fit
const TI = TR.Inference                   # audit_convergence, _extension_extract_latents, _inf_narrow
const FEAT = BF.Features
const CAL = BF.Calibration

# Pins recorded at Phase 0 (DECISIONS_LOG D3, D4, D8).
const DS_SHA = "f407a1b4c3188472b803804245dd7dbb0011aafc6471390e67038bb8c426c2b6"
const OB_SHA = "2adb12715f9bfd73b06ae0a71bda5724fdd4b298776789670b5e4bd9f8a5443a"
const EXT_FEATURE_SHA = "17b3ceb3d1cf9f99a49a80531c722cb50db79be328f5d557f8960f59001cb2f6"
const RHAT_RERUN = 1.05

# Seed rules. qs: the fit's own wave-2 schedule, arm 4 (REPRODUCIBILITY.md, l05_grid.jl).
# m12: the run records no seed rule (its DB extensions ran extend_fit on the task RNG), so a new
# schedule is frozen here before sampling.
const SEED_RULES = Dict(
    "qs_marketobs" => (chain = (fold, attempt, c) -> 202700000 + 100000 * 4 + 100fold + 10attempt + c,
                       post = attempt -> 202700000 + 100000 * 4 + 9000 + attempt,
                       source = "wave-2 l05_grid.jl chain_seed/post_seed, arm 4"),
    "m12" => (chain = (fold, attempt, c) -> 2026101100 + 100fold + 10attempt + c,
              post = attempt -> 2026101100 + 9000 + attempt,
              source = "new frozen schedule 2026101100 + 100 fold + 10 attempt + chain"))

# ===================================================================
# 1. Pinned inputs and the two plans
# ===================================================================
function inputs()
    ds, _ = Q.datastore_2627(; expected_sha = DS_SHA)
    fx, clubs = Q.fixtures_2627(ds)
    ob, _ = Q.orderbook_2627(ds, fx.match_id; expected_sha = OB_SHA)
    closebook, _ = Q.close_book(ob.snaps)
    pinned = Q.pinned_datastore()
    return (; ds, fx, clubs, ob, closebook, pinned)
end

function qs_setup(inp)
    _, table = Q.rates_2627(inp.ds, inp.closebook)
    ref = Q.committed_table()
    ext = Q.table_with_2627(ref, table)
    fit = TR.load_fit(TR.PostgresStorage(Q.QS.ns), UUIDs.UUID(Q.QS.run))
    model = Q.qs_model_2627(fit, ref, ext)
    FEAT.market_rates_digest(model.observation.feature) == EXT_FEATURE_SHA || error("extended feature digest differs")
    scope = Q.scope_extended(fit.config.splitter.scope)
    plan = Q.walkforward_plan("qs_marketobs", fit, Q.Data.apply_scope(inp.ds, scope), Q.Data.ScopedWalkForwardCV(scope),
                              Set(Int.(fit.latents.match_ids)), Q.fitted_heldout(fit, inp.pinned))
    filt = Q.filtration(plan, Set(scope.train_tournaments); full_book = Set(table.match_id[table.full_book]))
    return (; label = "qs_marketobs", fit, model, plan, filt, sampler = fit.config.sampler)
end

function m12_setup(inp)
    fit = TR.load_fit(TR.PostgresStorage(Q.M12.ns), UUIDs.UUID(Q.M12.run))
    by_fold = Q.fitted_heldout(fit, inp.pinned)
    m = inp.pinned.matches
    old = Set(Int.(m.match_id[in.(coalesce.(m.season, ""), Ref(["24/25", "25/26"])) .& in.(m.tournament_id, Ref([56, 57]))]))
    plan = Q.walkforward_plan("m12", fit, inp.ds, fit.config.splitter, intersect(Set(Int.(fit.latents.match_ids)), old),
                              Dict(k => v for (k, v) in by_fold if k <= Q.N_PERSISTED))
    filt = Q.filtration(plan, Set(Q.TOURNAMENTS))
    # extend_fit's rule: the draw count of the existing folds (800 = the config's n_samples here).
    size(fit.folds[1].chain, 1) == fit.config.sampler.n_samples || error("m12 sampler draw count differs from its folds")
    return (; label = "m12", fit, model = fit.config.model, plan, filt, sampler = fit.config.sampler)
end

# ===================================================================
# 2. Sampling (c4_extend_inmemory + wave-2 sample_pending!, copied; no checkpoints, no DB)
# ===================================================================
function sample_positions(setup, positions; attempt::Int = 0)
    rule = SEED_RULES[setup.label]
    plan, model, sampler = setup.plan, setup.model, setup.sampler
    local_idx = [findfirst(==(p), plan.delta) for p in positions]
    selected = [plan.boundaries[p] for p in positions]
    feature_sets = FEAT.create_features(selected, plan.ds, model, plan.splitter)
    nc = sampler.n_chains
    buffers = [Vector{Any}(nothing, nc) for _ in positions]
    seeds = [rule.chain(p, attempt, c) for p in positions, c in 1:nc]
    sem, lk = Base.Semaphore(16), ReentrantLock()
    started = time()
    @sync for (k, p) in enumerate(positions), c in 1:nc
        Threads.@spawn begin
            Base.acquire(sem)
            try
                Random.seed!(rule.chain(p, attempt, c))
                part = TR.sample_fold(model, sampler, feature_sets[k][1], p; chain_id = c)
                part === nothing && error("empty chain: fold=$p chain=$c")
                lock(lk) do
                    buffers[k][c] = part
                end
            finally
                Base.release(sem)
            end
        end
    end
    sampled_s = time() - started
    folds = TI._inf_narrow(TR.FoldFit[TR.FoldFit(p, cat(buffers[k]...; dims = 3), feature_sets[k][2])
                                       for (k, p) in enumerate(positions)])
    diagnostics = TI.audit_convergence(folds; thresholds = setup.fit.diagnostics.thresholds,
                                       max_depth = TI.sampler_max_depth(sampler))
    Random.seed!(rule.post(attempt))
    oos = Any[plan.oos[i] for i in local_idx]
    per_fold, _ = TI._extension_extract_latents(model, folds, feature_sets, oos)
    println("SAMPLED model=", setup.label, " positions=", join(positions, ","), " attempt=", attempt,
            " minutes=", round(sampled_s / 60; digits = 2), " seeds=", join(vec(seeds), ";"))
    flush(stdout)
    return (; positions, attempt, folds, diagnostics, per_fold, feature_sets, oos, seeds, sampled_s)
end

# c4_convergence_table (copied); the rerun trigger is R-hat > 1.05 only (ruling 2).
function convergence_table(label, diagnostics, attempt)
    rows = NamedTuple[]
    for f in diagnostics.folds
        push!(rows, (; model = label, fold = f.fold, attempt, max_rhat = f.max_rhat, worst_rhat = String(f.worst_rhat_param),
                     min_ess_bulk = f.min_ess_bulk, min_ess_tail = f.min_ess_tail, n_divergent = f.n_divergent,
                     n_transitions = f.n_transitions, divergence_pct = 100 * f.divergence_rate,
                     treedepth_pct = 100 * f.treedepth_rate, min_bfmi = f.min_bfmi,
                     rerun_due = f.max_rhat > RHAT_RERUN))
    end
    return DF.DataFrame(rows)
end

# Filtration on what was actually sampled: the feature set's ordered training IDs.
function sampled_filtration(setup, s, ds)
    date_of = Dict(Int(r.match_id) => Dates.Date(r.match_date) for r in eachrow(ds.matches))
    rows = NamedTuple[]
    for (k, p) in enumerate(s.positions)
        fs = s.feature_sets[k][1]
        train = Int.(fs.data[:ordered_match_ids])
        bd, _ = setup.plan.boundaries[p]
        planned = Set(Int.(vcat(bd.history_match_ids, bd.target_match_ids)))
        ho = Int.(s.oos[k].match_id)
        lat = Set(Int.(s.per_fold[p].match_ids))
        last_train = maximum(date_of[id] for id in train)
        first_ho = minimum(date_of[id] for id in ho)
        mask = get(fs.data, :flat_market_available, nothing)
        n_obs = mask === nothing ? missing : Int(sum(mask))
        n_obs_2627 = mask === nothing ? missing :
                     Int(sum(mask[i] for i in eachindex(train) if date_of[train[i]] >= Q.FIRST_DAY; init = 0.0))
        ok = Set(train) == planned && isempty(intersect(Set(train), Set(ho))) && last_train < first_ho && lat == Set(ho)
        push!(rows, (; model = setup.label, fold = p, attempt = s.attempt, n_train = length(train),
                     train_equals_plan = Set(train) == planned, train_last = last_train, heldout_first = first_ho,
                     n_heldout = length(ho), latents_equal_heldout = lat == Set(ho),
                     n_market_obs = n_obs, n_market_obs_2627 = n_obs_2627, pass = ok))
        println("SAMPLED_FILTRATION model=", setup.label, " fold=", p, " attempt=", s.attempt, " n_train=", length(train),
                " train_last=", last_train, " heldout_first=", first_ho, " market_obs=", n_obs, " market_obs_2627=", n_obs_2627,
                " pass=", ok)
    end
    tab = DF.DataFrame(rows)
    all(tab.pass) || error("$(setup.label): sampled filtration failed")
    return tab
end

# m12 held-out XI (manager condition): which teamsheet the lineup pillar sees for each held-out
# fixture. The feature builder maps every store match's STARTING rows of ds.lineups (the played
# teamsheet) through the fold's history-fitted RAPM ratings; positions 1-40 used the same builder.
function m12_xi_table(s, ds)
    lu = ds.lineups
    starters = filter(r -> !coalesce(r.is_substitute, false) && !ismissing(r.player_id), lu)
    nstart = Dict{Tuple{Int,String},Int}()
    for r in eachrow(starters)
        k = (Int(r.match_id), String(r.team_side))
        nstart[k] = get(nstart, k, 0) + 1
    end
    rows = NamedTuple[]
    for (k, p) in enumerate(s.positions)
        fs = s.feature_sets[k][1]
        lmap = get(fs.data, :player_lineup_ratings_map, Dict())
        for id in Int.(s.oos[k].match_id)
            push!(rows, (; fold = p, match_id = id, home_starters = get(nstart, (id, "home"), 0),
                         away_starters = get(nstart, (id, "away"), 0), in_lineup_map = haskey(lmap, id),
                         xi_source = "ds.lineups starting rows (played teamsheet)"))
        end
    end
    return DF.DataFrame(rows)
end

# One model: attempt 0 on every new position, one seeded rerun (attempt 1) for R-hat > 1.05.
function run_extension(setup, ds, results_dir)
    s0 = sample_positions(setup, setup.plan.delta)
    conv = convergence_table(setup.label, s0.diagnostics, 0)
    filt = sampled_filtration(setup, s0, ds)
    final = Dict(p => (fold = s0.folds[k], latents = s0.per_fold[p], attempt = 0) for (k, p) in enumerate(s0.positions))
    xi = setup.label == "m12" ? m12_xi_table(s0, ds) : nothing
    due = conv.fold[conv.rerun_due]
    if !isempty(due)
        s1 = sample_positions(setup, due; attempt = 1)
        c1 = convergence_table(setup.label, s1.diagnostics, 1)
        conv = vcat(conv, c1)
        filt = vcat(filt, sampled_filtration(setup, s1, ds))
        for (k, p) in enumerate(s1.positions)
            final[p] = (fold = s1.folds[k], latents = s1.per_fold[p], attempt = 1)
        end
        println("RERUN model=", setup.label, " folds=", join(due, ","), " max_rhat_after=", maximum(c1.max_rhat))
    end
    for r in eachrow(conv)
        println("CONVERGENCE model=", r.model, " fold=", r.fold, " attempt=", r.attempt, " rhat=", round(r.max_rhat; digits = 4),
                " (", r.worst_rhat, ") ess_bulk=", round(r.min_ess_bulk; digits = 0), " ess_tail=", round(r.min_ess_tail; digits = 0),
                " divergent=", r.n_divergent, "/", r.n_transitions, " rerun_due=", r.rerun_due)
    end
    positions = sort!(collect(keys(final)))
    latents = TI.merge_latents(Any[final[p].latents for p in positions])
    used = DF.DataFrame([(; model = setup.label, fold = p, attempt_used = final[p].attempt) for p in positions])
    ext = (; label = setup.label, run = setup.label == "qs_marketobs" ? Q.QS.run : Q.M12.run,
           seed_rule = SEED_RULES[setup.label].source, sampler = setup.sampler, positions,
           folds = [final[p].fold for p in positions], attempt_used = Dict(p => final[p].attempt for p in positions),
           per_fold_latents = Dict(p => final[p].latents for p in positions), latents,
           diagnostics = conv, filtration = filt, plan_filtration = setup.filt)
    path = joinpath(Q.OUT_DIR, "ext_$(setup.label)_2627.jls")
    Serialization.serialize(path, ext)
    println("EXTENSION_FILE model=", setup.label, " path=", path, " sha256=", Q.sha_file(path),
            " heldout=", length(latents.match_ids))
    CSV.write(joinpath(results_dir, "phase1_$(setup.label)_convergence.csv"), conv)
    CSV.write(joinpath(results_dir, "phase1_$(setup.label)_filtration_sampled.csv"), filt)
    CSV.write(joinpath(results_dir, "phase1_$(setup.label)_attempts.csv"), used)
    xi === nothing || CSV.write(joinpath(results_dir, "phase1_m12_heldout_xi.csv"), xi)
    return ext
end

end # module
