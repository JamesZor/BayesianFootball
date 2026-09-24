# ==============================================================================
# l01 — MultiScaleGRW on the pooled Scottish pyramid (+ SPFL cup bridges)
# ==============================================================================
#
# Definitions only; `r01_smoke.jl` and `r02_overnight.jl` execute.
#
# QUESTION. Does training the Scottish Lower GRW on the whole SPFL (G1), and on the
# SPFL plus the cup ties that bridge its divisions (G2), improve pricing of the
# canonical League One/Two book? G3 repeats G2 under the two-arm joint likelihood.
#
#   g1_grw_all_spfl            leagues 54–57                         Poisson
#   g2_grw_all_spfl_cups       54–57 + SPFL-vs-SPFL cup ties         Poisson
#   g3_grw_joint_all_spfl_cups 54–57 + SPFL-vs-SPFL cup ties         Joint Gamma-Poisson (pxG)
#
# Control (already persisted): Task 007 `m00_baseline_grw` f64a00a2 — the same
# model on 56/57 rows only. G1 − control = pooling; G2 − G1 = the cups.
#
# COMPARABILITY CONTRACT
# * Held-out fixtures and fold boundaries are the CANONICAL 40-fold / 710-fixture
#   56/57 grid (`gph_splitter`), unchanged. Only the TRAINING rows widen.
# * Every added row kicks off strictly before its fold's first held-out fixture.
# * Added rows join the GRW on the canonical 56/57 calendar: history rows by season
#   (macro step), target-season rows by the 56/57 match-biweek clock (micro step).
#   July cup ties before the first league week are clamped into step 1.
# * No league offset: TODO 029 measured tier goal levels flat across the SPFL
#   (g_T within ±0.06), so the model is the unchanged `m00_baseline_grw` recipe.
#
# CUP ROWS. Only ties between two clubs that are in the SPFL that season, played at
# a club ground (semis/finals at Hampden etc. removed), enter — from
# `data/cup_bridge_allowlist.csv`, derived from the TODO 029 r01 panel
# (point-in-time tier membership). B-teams, guests and non-league clubs are
# excluded tonight: ~250 thinly observed extra teams would destabilise the walk.
# League Cup season labels ("2025") are rewritten to the football season ("25/26")
# so they cannot create a phantom GRW macro step.
# ==============================================================================

include(joinpath(@__DIR__, "..", "grw_player_hybrid", "l01_loader.jl"))   # components + helpers

using CSV

const PCX_DATA = BayesianFootball.Data
const PCX_ALLOWLIST = joinpath(@__DIR__, "data", "cup_bridge_allowlist.csv")
const PCX_LEAGUES = [54, 55, 56, 57]
const PCX_CUPS = [73, 982, 1520]

# ------------------------------------------------------------------------------
# 1. Segment and data preparation
# ------------------------------------------------------------------------------

struct ScottishPyramidCups <: PCX_DATA.DataTournemantSegment end
PCX_DATA.tournament_ids(::ScottishPyramidCups) = vcat(PCX_LEAGUES, PCX_CUPS)

pcx_fs_label(d::Date) = (y = month(d) >= 7 ? year(d) : year(d) - 1;
                         @sprintf("%02d/%02d", y % 100, (y + 1) % 100))

"""
    pcx_load_data(; max_age_hours) -> DataStore

All seven tournaments from `sofascore.matches`, then in place:
cup season labels → football season; cup rows not on the allowlist dropped.
"""
function pcx_load_data(; max_age_hours::Int = 12)
    ds = PCX_DATA.load_datastore_cached(ScottishPyramidCups(); max_age_hours = max_age_hours)
    allow = Set(Int.(CSV.read(PCX_ALLOWLIST, DataFrame).match_id))
    m = ds.matches
    is_cup = in.(m.tournament_id, Ref(PCX_CUPS))
    n_cup_raw = count(is_cup)
    # football-season labels for every cup row (982 uses calendar years)
    seasons = String.(m.season)
    for i in findall(is_cup)
        seasons[i] = pcx_fs_label(Date(m.match_date[i]))
    end
    m.season = seasons
    keep = .!is_cup .| in.(Int.(m.match_id), Ref(allow))
    deleteat!(m, findall(.!keep))
    @info "pcx_load_data" leagues = count(in.(m.tournament_id, Ref(PCX_LEAGUES))) cup_rows_raw = n_cup_raw cup_rows_kept = count(in.(m.tournament_id, Ref(PCX_CUPS))) latest = maximum(m.match_date)
    return ds
end

# ------------------------------------------------------------------------------
# 2. Splitter: canonical 56/57 folds, widened training rows
# ------------------------------------------------------------------------------

Base.@kwdef struct PyramidGRWCV <: PCX_DATA.AbstractSplitter
    target_seasons::Vector{String} = ["24/25", "25/26"]
    history_seasons::Int = 2
    dynamics_col::Symbol = :match_biweek
    extra_tournaments::Vector{Int} = [54, 55]
end

pcx_lower(s::PyramidGRWCV) = gph_splitter(s.target_seasons)

function PCX_DATA.create_id_boundaries(ds::PCX_DATA.DataStore, s::PyramidGRWCV)
    lower = pcx_lower(s)
    out = Vector{Tuple{PCX_DATA.SplitBoundary, PCX_DATA.AbstractSplitMetaData}}()
    m = ds.matches
    mid = Int.(m.match_id)
    mdate = Date.(m.match_date)
    extra = in.(m.tournament_id, Ref(s.extra_tournaments))
    for (b, meta) in PCX_DATA.create_id_boundaries(ds, lower)
        held = PCX_DATA.get_next_matches(ds, meta, lower)
        if nrow(held) == 0
            push!(out, (b, meta))          # keep fold alignment; nothing to predict
            continue
        end
        cutoff = minimum(Date.(held.match_date))
        hist_set = Set(Int.(b.history_match_ids))
        hist_seasons = Set(String.(m.season[in.(mid, Ref(hist_set))]))
        add_hist = mid[extra .& in.(String.(m.season), Ref(hist_seasons)) .& (mdate .< cutoff)]
        add_tgt = mid[extra .& (String.(m.season) .== meta.target_season) .& (mdate .< cutoff)]
        # the canonical target rows must already precede the cutoff
        tgt_set = Set(Int.(b.target_match_ids))
        isempty(tgt_set) || maximum(mdate[in.(mid, Ref(tgt_set))]) < cutoff ||
            error("canonical target rows reach the held-out bin (fold $(b.fold_id))")
        nb = PCX_DATA.SplitBoundary(b.fold_id, b.target_step,
                                    vcat(Int.(b.history_match_ids), add_hist),
                                    vcat(Int.(b.target_match_ids), add_tgt))
        push!(out, (nb, meta))
    end
    return out
end

PCX_DATA.get_next_matches(ds::PCX_DATA.DataStore, meta::PCX_DATA.GroupedSplitMetaData, s::PyramidGRWCV) =
    PCX_DATA.get_next_matches(ds, meta, pcx_lower(s))
PCX_DATA.get_next_matches(ds::PCX_DATA.DataStore, t::Tuple{Any, <:PCX_DATA.AbstractSplitMetaData}, s::PyramidGRWCV) =
    PCX_DATA.get_next_matches(ds, t[2], s)

"""
    pcx_align_time!(fs, boundary, meta, ds, s)

The pooled counterpart of the framework's `_align_splitter_time!`: history rows index
by season; target rows by the 56/57 calendar biweek, with the SAME anchor and formula
(`Data._effective_step_map`), extended to the added tiers' rows.
"""
function pcx_align_time!(fs, boundary, meta, ds, s::PyramidGRWCV)
    hist_ids = Set(Int.(boundary.history_match_ids)); tgt_ids = Set(Int.(boundary.target_match_ids))
    all_ids = union(hist_ids, tgt_ids)
    mdf = DataFrames.subset(ds.matches, :match_id => ByRow(id -> Int(id) in all_ids))
    hdf = DataFrames.subset(mdf, :match_id => ByRow(id -> Int(id) in hist_ids))
    tdf = DataFrames.subset(mdf, :match_id => ByRow(id -> Int(id) in tgt_ids))
    ordered = Int.(vcat(hdf.match_id, tdf.match_id))
    ordered == Int.(fs.data[:ordered_match_ids]) ||
        error("pcx_align_time!: row order differs from the feature builder's (fold $(boundary.fold_id))")

    hsteps = sort(unique(String.(hdf.season)))
    hstate = Dict(x => i for (i, x) in enumerate(hsteps))
    hidx = Int[hstate[String(x)] for x in hdf.season]

    # 56/57 anchor for the target season — identical to Data._effective_step_map
    lowseason = ds.matches[in.(ds.matches.tournament_id, Ref([56, 57])) .&
                           (String.(ds.matches.season) .== meta.target_season), :]
    anchor = minimum(PCX_DATA._week_ending_sunday.(Date.(lowseason.match_date)))
    width = PCX_DATA.CALENDAR_DYNAMICS_WIDTH_WEEKS[s.dynamics_col]
    rawstep(d) = max(1, cld(1 + div(Dates.value(PCX_DATA._week_ending_sunday(Date(d)) - anchor), 7), width))
    raw = Dict(Int(r.match_id) => rawstep(r.match_date) for r in eachrow(tdf))
    # cross-check: for 56/57 rows this must equal the framework's own clock
    canon = PCX_DATA._effective_step_map(ds.matches, [56, 57], meta.target_season, s.dynamics_col)
    for r in eachrow(tdf)
        r.tournament_id in (56, 57) || continue
        raw[Int(r.match_id)] == canon[Int(r.match_id)] ||
            error("pcx_align_time!: clock mismatch for match $(r.match_id)")
    end
    tsteps = sort(unique(values(raw)))
    tstate = Dict(x => i for (i, x) in enumerate(tsteps))
    tidx = Int[length(hsteps) + tstate[raw[Int(id)]] for id in tdf.match_id]

    fs.data[:time_indices] = vcat(hidx, tidx)
    fs.data[:n_history_steps] = length(hsteps)
    fs.data[:n_target_steps] = length(tsteps)
    fs.data[:n_rounds] = length(hsteps) + length(tsteps)
    fs.data[:effective_target_steps] = raw
    return fs
end

function GPH_FEATURES.create_features(splits::Vector{<:Tuple{PCX_DATA.SplitBoundary, <:Any}},
                                      ds::PCX_DATA.DataStore, model, s::PyramidGRWCV)
    items = [(let fs = GPH_FEATURES.create_features(b, ds, model, s.dynamics_col)
                  pcx_align_time!(fs, b, meta, ds, s)
                  (fs, meta)
              end) for (b, meta) in splits]
    return BayesianFootball.TypesInterfaces.FeatureCollection(items)
end

# ------------------------------------------------------------------------------
# 3. Arms, sampler, configs
# ------------------------------------------------------------------------------

Base.@kwdef struct PCXConfig
    experiment::String = "scottish_pyramid_grw_cups"
    save_root::String = joinpath(@__DIR__, "results")
    target_seasons::Vector{String} = ["24/25", "25/26"]
    expected_folds::Int = 40
    expected_oos::Int = 710
    gph::GPHConfig = GPHConfig(save_root = joinpath(@__DIR__, "results"))
end

const PCX_ARMS = ["g1_grw_all_spfl", "g2_grw_all_spfl_cups", "g3_grw_joint_all_spfl_cups"]
const PCX_DESCRIPTIONS = Dict(
    "g1_grw_all_spfl" =>
        "m00_baseline_grw recipe trained on all four SPFL leagues; scored on the canonical 56/57 grid.",
    "g2_grw_all_spfl_cups" =>
        "g1 plus Scottish Cup / League Cup / Challenge Cup ties between SPFL clubs at club grounds.",
    "g3_grw_joint_all_spfl_cups" =>
        "g2 rows under the two-arm joint Gamma-Poisson (commentary pxG) likelihood; no wealth.",
)
const PCX_TAGS = ["scottish-lower", "24/25", "25/26", "multiscale-grw", "cross-tier", "cups", "todo029"]

pcx_splitter(arm::AbstractString, c::PCXConfig) = PyramidGRWCV(
    target_seasons = copy(c.target_seasons),
    extra_tournaments = arm == "g1_grw_all_spfl" ? [54, 55] : [54, 55, 73, 982, 1520])

function pcx_model(arm::AbstractString)
    b = CountModelBuilder(Symbol(arm)) |> add(GlobalInterception()) |> add(gph_dynamics()) |>
        add(GlobalHomeAdvantage())
    obs = arm == "g3_grw_joint_all_spfl_cups" ? gph_joint_observation() : PoissonObservation()
    return b |> add(obs) |> build
end

function pcx_fit_config(arm, c::PCXConfig, sampler)
    return FitConfig(name = arm, model = pcx_model(arm), splitter = pcx_splitter(arm, c),
                     sampler = sampler, execution = gph_execution(c.gph),
                     tags = copy(PCX_TAGS), description = PCX_DESCRIPTIONS[arm],
                     save_dir = joinpath(c.save_root, arm))
end

function pcx_register!(db, arm, fc)
    save_model(db, arm, fc.model; description = PCX_DESCRIPTIONS[arm], tags = PCX_TAGS)
    save_config(db, arm * "_fit", fc; description = PCX_DESCRIPTIONS[arm], tags = PCX_TAGS)
end

"Per-fold counts of added rows — the filtration evidence for the report."
function pcx_widening_report(ds, inputs)
    rows = NamedTuple[]
    tid = Dict(Int(r.match_id) => Int(r.tournament_id) for r in eachrow(ds.matches))
    for (i, (b, meta)) in enumerate(inputs.boundaries)
        ids = vcat(Int.(b.history_match_ids), Int.(b.target_match_ids))
        t = [tid[x] for x in ids]
        push!(rows, (fold = i, season = meta.target_season, n_train = length(ids),
                     n_lower = count(in((56, 57)), t), n_upper = count(in((54, 55)), t),
                     n_cup = count(in((73, 982, 1520)), t), n_oos = nrow(inputs.oos[i]),
                     n_target_steps = Int(inputs.feature_sets[i][1].data[:n_target_steps]),
                     n_teams = Int(inputs.feature_sets[i][1].data[:n_teams])))
    end
    return DataFrame(rows)
end
