# Read-only leakage audit: reconstruct likelihood inputs and price SAVED latents only.
# No inversion optimisation, sampling, persistence updates, or reference refits.
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, Statistics, ThreadPinning
import SHA
pinthreads(:cores); BLAS.set_num_threads(1)
const LA_C = Module(:LeakageCandidates)
Base.include(LA_C, joinpath(@__DIR__, "candidates.jl"))
const LA_H = BayesianFootball.Harness
const LA_API = BayesianFootball.Models.PreGame.Builder
const LA_OUT = "/root/BF_runs/qs_experiment_w2_out/leakage_audit"
mkpath(LA_OUT)

# %% 1. Frozen snapshot/table and each fixture's OWN close provenance
let
    ds = Main.QSMarketArm.checked_datastore()
    config = Main.QSMarketArm.panel_config()
    panel_path = "/root/BF_runs/qs_experiment_out/market_panel"
    read(joinpath(panel_path, "inversion_manifest.txt"), String) ==
        "$(Main.QSMarketArm.SNAPSHOT_SHA)\n$(repr(config))\n" || error("inversion manifest mismatch")
    rates = CSV.read(joinpath(panel_path, "inverted_rates.csv"), DataFrame; types=Dict(:season=>String))
    table = CSV.read(joinpath(@__DIR__, "results", "market_rates.csv"), DataFrame)
    length(unique(rates.match_id)) == nrow(rates) == nrow(table) || error("duplicate/missing rates")
    byrate = Dict(Int(r.match_id)=>r for r in eachrow(rates))
    kickoff = LA_H._match_kickoffs(ds.matches)
    book, _ = Main.QSMarketArm.MM.gated_close(ds, config)
    bybook = Dict(Int(first(g.match_id))=>DataFrame(g) for g in groupby(book, :match_id))
    rawgroups = Dict(Int(first(g.match_id))=>g for g in groupby(ds.betfair_odds, :match_id))
    provenance = NamedTuple[]
    for row in eachrow(table)
        id = Int(row.match_id)
        r = byrate[id]
        full = r.accepted && r.n_selections >= config.min_selections_ladder
        row.full_book == full || error("full-book mismatch $id")
        expected = full ? (log(r.lambda_h), log(r.lambda_a)) : (0.0, 0.0)
        isequal((row.log_lambda_h, row.log_lambda_a), expected) || error("rate value mismatch $id")
        Date(kickoff[id]) == Date(r.date) || error("rate date mismatch $id")
        own = get(bybook, id, book[1:0, :])
        # Remove ALL other fixtures before close construction: identical own-book prices.
        isolated = if haskey(rawgroups, id)
            close = Main.QSMarketArm.MM.closing_book((; betfair_odds=DataFrame(rawgroups[id])), config)
            first(Main.QSMarketArm.MM.gate_book(close))
        else
            book[1:0, :]
        end
        columns = [:match_id, :market_name, :market_line, :selection, :odds_close, :prob_fair_close]
        sortkeys = [:market_name, :market_line, :selection]
        isolated_equal = isequal(sort(select(own, columns), sortkeys), sort(select(isolated, columns), sortkeys))
        isolated_equal || error("cross-fixture close dependence $id")
        cross_match_rows = count(!=(id), own.match_id)
        post_kickoff_ticks_used = haskey(rawgroups, id) ? count(x ->
            config.price_window[1] <= x <= config.price_window[2] && x > 0,
            rawgroups[id].minutes_to_kickoff) : 0
        kl_gap = isfinite(r.kl) ? abs(Main.QSMarketArm.MM.kl_loss(log.([r.lambda_h, r.lambda_a]), own) - r.kl) : 0.0
        kl_gap <= 1e-12 || error("frozen own-book KL mismatch $id: $kl_gap")
        push!(provenance, (; match_id=id, full_book=full, own_book_rows=nrow(own),
            cross_match_rows, post_kickoff_ticks_used, isolated_book_exact=isolated_equal, kl_gap))
        (cross_match_rows == 0 && post_kickoff_ticks_used == 0) || error("close leakage $id")
    end
    CSV.write(joinpath(LA_OUT, "leakage_provenance.csv"), DataFrame(provenance))

    # %% 2. EVERY fold: actual ordered likelihood rows and binary observation mask
    # SplitBoundary.target_match_ids are fitted past-season rows, NOT held-out targets.
    # Held-out targets come from get_next_matches, also checked against saved fold metadata.
    db = Training.PostgresStorage(LA_C.EXPERIMENT)
    runs = CSV.read(joinpath(@__DIR__, "results", "phase3", "RUNS.csv"), DataFrame)
    saved = CSV.read(joinpath(@__DIR__, "results", "phase4", "observation_frames.csv"), DataFrame)
    headline = CSV.read(joinpath(@__DIR__, "results", "phase4", "headline.csv"), DataFrame)
    counts = NamedTuple[]
    score_rows = NamedTuple[]
    common_ids = Set(Int.(filter(r -> r.model == "market_close" && r.family == "1X2", saved).match_id))
    length(common_ids) == 595 || error("common panel is not 595")
    for arm in ("grw_marketobs", "qs_marketobs")
        c = only(filter(c -> c.name == arm, LA_C.CANDIDATES))
        id = string(only(filter(r -> r.label == arm, runs)).run_id)
        fit = Training.load_fit(db, id)
        LA_H._structural_equal(fit.config.model, c.model) || error("saved model mismatch $arm")
        inputs = LA_H._fold_inputs(c, ds; stage=:grid)
        length(inputs.feature_sets) == length(fit.folds) == 40 || error("fold coverage $arm")
        fitted_model = fit.config.model
        all_targets = Int[]
        for k in 1:40
            item = inputs.feature_sets[k]
            fs = item isa Tuple ? first(item) : item
            ids = Int.(fs.data[:ordered_match_ids])
            mask = fs.data[:flat_market_available]
            design = LA_API.observation_design(fitted_model.observation, fs, length(ids), ones(length(ids)))
            design.mask_weights == mask || error("likelihood mask mismatch $arm/$k")
            target_ids = Set(Int.(inputs.oos[k].match_id))
            fold = only(filter(f -> f.fold == k, fit.folds))
            persisted_targets = Set(Int.(Data.get_next_matches(inputs.ds, fold.meta,
                Data.ScopedWalkForwardCV(c.scope)).match_id))
            target_ids == persisted_targets || error("saved target mismatch $arm/$k")
            cutoff = minimum(kickoff[m] for m in target_ids)
            active_ids = ids[mask .== 1.0]
            in_target = count(in(target_ids), active_ids)
            at_or_after = count(m -> kickoff[m] >= cutoff, active_ids)
            push!(counts, (; arm, run_id=id, fold=k, training_rows=length(ids),
                active_market_rows=length(active_ids), target_rows=length(target_ids),
                first_target_kickoff=string(cutoff), active_in_target=in_target,
                active_at_or_after_target=at_or_after,
                any_likelihood_target_rows=count(in(target_ids), ids)))
            CSV.write(joinpath(LA_OUT, "leakage_audit.csv"), DataFrame(counts))
            (in_target == 0 && at_or_after == 0) || error("LEAKAGE DETECTED $arm fold $k: target=$in_target time=$at_or_after; STOP")
            isempty(intersect(Set(ids), target_ids)) || error("target enters likelihood $arm/$k")
            append!(all_targets, collect(target_ids))
        end
        length(all_targets) == length(unique(all_targets)) == 710 || error("target panel mismatch")

        # %% 3. Reprice saved posterior latents, match saved probabilities, recompute binary LL
        odds = LA_H._betfair_closing_odds(ds)
        panel = LA_H._season_panel(ds, fit, ["24/25", "25/26"])
        ctx = LA_H._context(LA_H._restrict(fit, panel), odds, ds)
        frame = filter(r -> r.family == "1X2", LA_H._observation_frame(arm, ctx, odds))
        Set(Int.(frame.match_id)) == common_ids || error("arm-specific common panel $arm")
        old = filter(r -> r.model == arm && r.family == "1X2", saved)
        keys = [:match_id, :selection]
        # Runtime selections are Symbols; the committed CSV stores their strings.
        frame.selection = String.(frame.selection)
        old.selection = String.(old.selection)
        sort!(frame, keys); sort!(old, keys)
        nrow(frame) == nrow(old) == 1785 || error("selection count $arm")
        isequal(select(frame, keys), select(old, keys)) || error("prediction keys mismatch")
        probability_gap = maximum(abs.(frame.p_model .- old.p_model))
        probability_gap == 0 || error("saved prediction mismatch $arm: $probability_gap")
        ll = mean(-r.y * log(r.p_model) - (1-r.y) * log1p(-r.p_model) for r in eachrow(frame))
        reported = only(filter(r -> r.model == arm, headline)).ll_1x2
        abs(ll - reported) <= 1e-14 || error("reported LogLoss mismatch $arm")
        push!(score_rows, (; arm, fixtures=595, selections=1785, logloss=ll,
            reported_logloss=reported, logloss_gap=abs(ll-reported), saved_probability_gap=probability_gap))
    end
    CSV.write(joinpath(LA_OUT, "leakage_logloss.csv"), DataFrame(score_rows))
    println("LEAKAGE_AUDIT_PASS folds=80 target_overlap=0 at_or_after_cutoff=0 own_close_provenance=PASS common_fixtures=595 saved_predictions_exact=PASS")
end
