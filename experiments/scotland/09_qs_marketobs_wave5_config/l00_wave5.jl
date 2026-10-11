module QSWave5

# Wave 5 (TODO 044): basket-filtered scenario Kelly, lambda sweeps and the 24/25 seal.
# Read-only: wave 4 loaders (saved fits by UUID, the saved datastore cache). No fitting,
# no database writes. Comments only below; the static checker scans every non-comment line
# for write verbs, so prose lives in comments, not docstrings.

import BayesianFootball as BF
import CSV
import DataFrames as DF
import Dates
import Serialization
import SHA
import Statistics

const PF = BF.Portfolio
const MD = BF.MatchDay
const P0 = Main.QSBacktestPhase0          # read_query, audit_snapshot, snapshot
const SC = Main.QSBacktestScenario        # scenario_problem (joint draws + scoreline rows)
const P3 = Main.QSBacktestPhase3          # goal_model, close_index, cell_tables, block_bootstrap

const SPLIT_DATE = Dates.Date(2025, 5, 3)
const W4_DIR = normpath(joinpath(@__DIR__, "..", "08_qs_marketobs_backtest"))
const PHASE0_DIR = joinpath(W4_DIR, "results", "phase0_historical")
const LAMBDA_GRID = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 23.0, 26.0, 30.0]
const BUDGET_TOL = 1e-3
const MODEL_SPECS = Dict(m.label => m for m in P3.GOAL_MODELS)

# ===================================================================
# 1. The seal (ruling 7)
# ===================================================================
# 25/26 stays sealed until the Phase 2 runner calls release!, which requires FREEZE.md to be
# tracked and committed. The full period is reachable only with gate = true (Phase 0 wave-4
# reproduction of already-public cells).
const RELEASED = Ref(false)

function release!(freeze_path::AbstractString)
    isfile(freeze_path) || error("SEAL: $(freeze_path) does not exist")
    dir, name = dirname(freeze_path), basename(freeze_path)
    Base.run(`git -C $dir ls-files --error-unmatch $name`)
    last = readchomp(`git -C $dir log -1 --format=%H -- $name`)
    isempty(last) && error("SEAL: FREEZE.md is not committed")
    println("SEAL_RELEASED freeze_commit=", last)
    RELEASED[] = true
    return last
end

half_of(d::Dates.Date) = d <= SPLIT_DATE ? :selection : :evaluation

function half_ids(ctx, half::Symbol; gate::Bool = false)
    if half === :full
        gate || error("SEAL: the full period is for the Phase 0 wave-4 gate only")
        return ctx.ids
    end
    half in (:selection, :evaluation) || error("unknown half $half")
    half === :evaluation && !RELEASED[] && error("SEAL: 25/26 is sealed until FREEZE.md is released")
    return Set(id for id in ctx.ids if half_of(ctx.date_of[id]) === half)
end

half_odds(ctx, odds, half::Symbol; gate::Bool = false) =
    filter(r -> Int(r.match_id) in half_ids(ctx, half; gate = gate), odds)

function assert_half(books, half::Symbol)
    half === :full && return nothing
    all(b -> half_of(b.date) === half, books) || error("SEAL: a built book lies outside the $half half")
    return nothing
end

# ===================================================================
# 2. Inputs: cache, the two goal models, snapshot books (parity with wave 4 Phase 0)
# ===================================================================
function load_inputs(snaps; labels = ("qs_marketobs", "m12"))
    cache = joinpath(pkgdir(BF), ".cache", "datastore_ScottishLower.jls")
    isfile(cache) || error("wave 5 requires the saved datastore cache; no SQL fallback")
    println("DATASTORE_CACHE sha256=", bytes2hex(open(SHA.sha256, cache)))
    ds = Serialization.deserialize(cache)
    ids = Set(Int.(ds.matches.match_id[in.(ds.matches.season, Ref(P3.SEASONS))]))
    models = Dict{String,Any}(l => P3.goal_model(MODEL_SPECS[l], ds, ids) for l in labels)
    ref_ids = Int.(CSV.read(joinpath(PHASE0_DIR, "phase0_fixture_universe.csv"), DF.DataFrame).match_id)
    for l in labels
        universe = sort!(collect(Set(Int.(BF.Models.latent_match_ids(models[l].latents)))))
        universe == sort(ref_ids) && length(universe) == 710 || error("$l: universe differs from Phase 0")
        println("MODEL ", l, " run=", models[l].run, " draws=", models[l].nd, " layout=", models[l].layout)
    end
    season_of = Dict{Dates.Date,String}()
    date_of = Dict{Int,Dates.Date}()
    for r in eachrow(ds.matches)
        Int(r.match_id) in ids || continue
        d = Dates.Date(r.match_date)
        get!(season_of, d, String(r.season)) == String(r.season) || error("date $d spans two seasons")
        date_of[Int(r.match_id)] = d
    end
    # The split must coincide with the season boundary.
    all(d -> (d <= SPLIT_DATE) == (season_of[d] == "24/25"), keys(season_of)) ||
        error("SPLIT_DATE is not the 24/25 | 25/26 boundary")
    ctx = (; ds, ids, date_of, season_of)
    books = Dict{String,DF.DataFrame}()
    for s in snaps
        book, _ = P0.snapshot(ctx, s.minutes; staleness = s.staleness)
        ref = CSV.read(joinpath(PHASE0_DIR, s.stem * "_book.csv"), DF.DataFrame)
        ref.selection = Symbol.(ref.selection)
        isequal(ref, book) || error("snapshot $(s.label) differs from its committed Phase 0 book")
        books[s.label] = book
        println("INPUT_PARITY_PASS snapshot=", s.label, " selections=", DF.nrow(book))
    end
    return (; ctx, models, books)
end

# ===================================================================
# 3. Books and policies
# ===================================================================
canonical_spec() = PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb())
with_lambda(policy::PF.PolicySpec, lam::Real) = PF._with_lambda(policy, Float64(lam))
basket_policy(trust, lam::Real) = PF.PolicySpec(trust = trust, risk = PF.SlateDrawdown(Float64(lam)),
                                               cap = PF.FixedCap(0.25))

# Build canonical books for one half; the odds are restricted before any book exists.
function half_books(inputs, model_label, snap_label, half::Symbol; gate::Bool = false)
    odds = half_odds(inputs.ctx, inputs.books[snap_label], half; gate = gate)
    books, report = PF.build_books_reported(canonical_spec(), inputs.models[model_label].source, odds,
        inputs.ctx.ds; require_result = true, require_converged = false, quiet = true)
    assert_half(books, half)
    return books, report
end

simulate_books(policy, books, report) =
    PF.simulate_portfolio(policy, books, report; initial_bankroll = 1.0, bootstrap = false)

simulate_slates(policy, slates, report) =
    PF.simulate_portfolio(policy, slates; initial_bankroll = 1.0, bootstrap = false,
                          converged = report.converged, failed_gates = report.failed_gates)

# ===================================================================
# 4. Baskets (ruling 4)
# ===================================================================
# A basket is a set of (group, line, direction) keys in TieredTrust spelling. Canonical tiers
# keep their weight; any other direction gets 0.25. Legs outside the basket never enter the
# joint solve.
const CANON_TIERS = Dict(("1x2", 0.0, :home) => 0.35, ("over_under", 2.5, :under) => 0.35,
                         ("1x2", 0.0, :draw) => 0.25, ("1x2", 0.0, :away) => 0.25)

direction_key(s::PF.Selection) = PF._tiered_key(s.group, s.line, s.selection)
direction_label(k) = k[1] == "1x2" ? string(k[3]) :
                     k[1] == "over_under" ? string(k[3] === :over ? "O" : "U", k[2]) :
                     string(k[1], "_", k[3])

function basket(name::AbstractString, directions)
    keys_ = [PF._tiered_key(d...) for d in directions]
    length(unique(keys_)) == length(keys_) || error("basket $name repeats a direction")
    table = Dict(k => get(CANON_TIERS, k, 0.25) for k in keys_)
    return (; name = String(name), directions = keys_, trust = PF.TieredTrust(table; default = 0.0))
end

in_basket(b, s::PF.Selection) = PF.book_trust_for(b.trust, s) > 0.0

# ===================================================================
# 5. Joint scenario problems: built once per day over every canonical leg
# ===================================================================
function day_problems(model, books; spec = canonical_spec(), grouping = PF.DailySlate())
    slates = PF.group(grouping, books)
    w = PF.BookWorkspace(spec, model.latents; quiet = true)
    row = Dict(Int(id) => i for (i, id) in enumerate(BF.Models.latent_match_ids(model.latents)))
    alloc = PF.ScenarioKelly()
    days = NamedTuple[]
    for sl in slates
        grids = Array{Float64,3}[]
        for b in sl.books
            PF.price_fixture!(w, model.latents, row[b.m_id])
            g = w.S[:, :, :]
            p = vec(Statistics.mean(g, dims = 3)[:, :, 1])
            p ./= sum(p)
            p == b.p_grid || error("draw grid does not reproduce book p_grid: $(b.m_id)")
            push!(grids, g)
        end
        groups = [model.witness[b.m_id] for b in sl.books]
        problem = SC.scenario_problem(alloc, sl, grids, groups; native = true)
        push!(days, (; slate = sl, problem, grids, groups))
    end
    return days
end

# Leg-filtered book: same p_grid, settlement and per-match shrink; only the leg set shrinks.
function filtered_book(b::PF.MatchBook, idx::Vector{Int}, a::Vector{Float64}, kkt, converged)
    settle = b.settle === nothing ? nothing : b.settle[idx]
    return PF.MatchBook(b.m_id, b.date, b.sels[idx], b.p_grid, b.R[:, idx], settle, a, b.k_shrink,
                        kkt, converged)
end

# Proof that a column subset equals the problem rebuilt on leg-filtered books.
function subset_equivalence(days, bk; n_days = 5)
    rows = NamedTuple[]
    for d in days[1:min(n_days, length(days))]
        keep_books, keep_grids, keep_groups, cols = PF.MatchBook[], Array{Float64,3}[], String[], Int[]
        for (i, b) in enumerate(d.slate.books)
            idx = findall(s -> in_basket(bk, s), b.sels)
            isempty(idx) && continue
            push!(keep_books, filtered_book(b, idx, b.a_kelly[idx], b.kkt, b.converged))
            push!(keep_grids, d.grids[i])
            push!(keep_groups, d.groups[i])
            append!(cols, d.problem.columns[i][idx])
        end
        rebuilt = SC.scenario_problem(PF.ScenarioKelly(), PF.Slate(d.slate.window, keep_books),
                                      keep_grids, keep_groups; native = true)
        same = rebuilt.R == d.problem.R[:, cols] && rebuilt.weights.values == d.problem.weights.values
        push!(rows, (; basket = bk.name, date = d.slate.window, n_legs = length(cols), equal = same))
    end
    return rows
end

# keep(sel) -> Bool chooses the solve's legs; the policy's trust then stakes them.
function solve_days(days, keep; exec = canonical_spec().exec)
    alloc = PF.ScenarioKelly()
    slates, diag = PF.Slate[], NamedTuple[]
    for d in days
        sl, pb = d.slate, d.problem
        picked, cols = Tuple{Int,Vector{Int},UnitRange{Int}}[], Int[]
        for (i, b) in enumerate(sl.books)
            idx = findall(keep, b.sels)
            isempty(idx) && continue
            span = (length(cols) + 1):(length(cols) + length(idx))
            append!(cols, pb.columns[i][idx])
            push!(picked, (i, idx, span))
        end
        if isempty(cols)
            push!(slates, PF.Slate(sl.window, PF.MatchBook[]))
            push!(diag, (; date = sl.window, n_books = 0, n_legs = 0, raw_total_joint = 0.0,
                         budget_binding = false, kkt = 0.0, converged = true))
            continue
        end
        sol = PF.allocate(alloc, pb.weights, pb.R[:, cols], exec)
        sol.converged || error("joint scenario allocation did not converge on $(sl.window)")
        all(isfinite, sol.a) && all(>=(0.0), sol.a) &&
            sum(sol.a) <= min(1.0, exec.budget) + 1e-10 &&
            all(<=(exec.max_selection_stake + 1e-10), sol.a) || error("scenario constraints failed")
        books = PF.MatchBook[filtered_book(sl.books[i], idx, sol.a[span], sol.kkt, sol.converged)
                             for (i, idx, span) in picked]
        push!(slates, PF.Slate(sl.window, books))
        total = sum(sol.a)
        push!(diag, (; date = sl.window, n_books = length(books), n_legs = length(cols),
                     raw_total_joint = total, budget_binding = total >= exec.budget - BUDGET_TOL,
                     kkt = sol.kkt, converged = sol.converged))
    end
    return slates, DF.DataFrame(diag)
end

all_legs(::PF.Selection) = true

# ===================================================================
# 6. Metrics
# ===================================================================
function drawdown_pct(bank::AbstractVector{<:Real})
    rm = accumulate(max, bank)
    return minimum((bank .- rm) ./ rm .* 100)
end

# MDD within each season, bankroll reset to 1.0 at the season's first slate (opening counts).
function season_mdd(result, season_of)
    t = result.trajectory
    rows = NamedTuple[]
    for s in sort!(unique(season_of[d] for d in t.dates))
        idx = findall(d -> season_of[d] == s, t.dates)
        bank = vcat(t.bankroll[first(idx)], t.bankroll[idx .+ 1]) ./ t.bankroll[first(idx)]
        push!(rows, (; season = s, n_slates = length(idx), wealth = bank[end], mdd_pct = drawdown_pct(bank),
                     bets = count(d -> season_of[d] == s, t.bets.date)))
    end
    return rows
end

function curve_row(key, result, diag)
    s = result.summary
    st = PF.states_frame(result)
    return (; key..., final_wealth = result.trajectory.bankroll[end], g = s.growth_per_slate,
            mdd_pct = s.mdd, n_bets = s.n_bets, n_slates = s.n_slates,
            mean_exposure = Statistics.mean(st.exposure), max_exposure = maximum(st.exposure),
            cap_share = Statistics.mean(st.capped), risk_active_share = Statistics.mean(st.k_risk .< 1.0),
            budget_share = diag === nothing ? missing : Statistics.mean(diag.budget_binding),
            roi_frac_pct = s.roi, sharpe = s.sharpe, calmar = s.calmar)
end

# ===================================================================
# 7. Audit frame: static read check, SHOW once, before/after audit around every runner
# ===================================================================
function with_audit(f, out::AbstractString, label::AbstractString, files)
    mkpath(out)
    rel = [relpath(joinpath(@__DIR__, x), W4_DIR) for x in files]
    Base.run(`python3 $(joinpath(W4_DIR, "check_read_paths.py")) $rel`)
    conn = BF.Training.Inference._db_connect(BF.Training.PostgresStorage(MODEL_SPECS["m12"].ns))
    try
        value = String(P0.read_query(conn, "SHOW default_transaction_read_only")[1, 1])
        println("SESSION_DEFAULT_TRANSACTION_READ_ONLY=", value)
        before = P0.audit_snapshot(conn)
        CSV.write(joinpath(out, "$(label)_db_before.csv"), before)
        println("DB_AUDIT_BEFORE tables=", length(unique(before.table_name)))
        flush(stdout)
        try
            f()
        finally
            after = P0.audit_snapshot(conn)
            CSV.write(joinpath(out, "$(label)_db_after.csv"), after)
            tables = sort!(unique(vcat(String.(before.table_name), String.(after.table_name))))
            checks = DF.DataFrame([(; table_name = t,
                unchanged = isequal(filter(r -> r.table_name == t, before),
                                    filter(r -> r.table_name == t, after))) for t in tables])
            CSV.write(joinpath(out, "$(label)_db_audit.csv"), checks)
            println("DB_AUDIT_AFTER unchanged=", all(checks.unchanged), " tables=", length(tables))
            flush(stdout)
            all(checks.unchanged) || error("before/after database audit changed; stop")
        end
    finally
        close(conn)
    end
    return nothing
end

include(joinpath(@__DIR__, "l01_phase0_gate.jl"))
include(joinpath(@__DIR__, "l02_phase1_select.jl"))
include(joinpath(@__DIR__, "l03_phase2_eval.jl"))

end # module
