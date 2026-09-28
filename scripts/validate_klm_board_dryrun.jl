#!/usr/bin/env julia
# Read-only rehearsal of runbook §2 + §4 against production: score the W0 re-score CSV in
# memory, apply `write_scores!`'s replace-by-key to a copy of the production v1.2 scores,
# render the board and gate it. It never calls `write_scores!` or an `ensure_*schema!`;
# production is only SELECTed (`read_scores`) and loaded (`load_fit`).
include(joinpath(@__DIR__, "score_runs.jl"))

const DRYRUN_CSV = joinpath(@__DIR__, "..", "docs", "architecture", "harness_klm_W0_rescore_v12.csv")
const DRYRUN_V11_SHA = "090db11e"
const DRYRUN_M12 = "132df5c2-c742-4e95-8693-3aeb2b2cbaef"
const DRYRUN_TD_LOWER_JOINT = "97c7a3d9-a05a-4029-90cb-e34279b8c791"
const DRYRUN_COLUMNS = [:run_id, :model, :stage, :scorecard_version, :panel, :subset, :market,
                        :metric, :value, :lo, :hi, :n_obs, :n_fixtures, :reference, :control_run_id]

_dryrun_id(id) = ismissing(id) || id === nothing ? missing : string(id)

"String-typed IDs, so database (UUID) and in-memory rows group and key identically."
function _dryrun_normalise(df)
    out = select(df, DRYRUN_COLUMNS)
    out.run_id = string.(out.run_id)
    out.control_run_id = Union{Missing,String}[_dryrun_id(id) for id in out.control_run_id]
    out.model = String.(out.model)
    return out
end

"`write_scores!`: delete every row sharing (run, version, control, stage, subset) with `new`."
function _dryrun_replace(existing, new)
    key(r) = (r.run_id, r.scorecard_version, coalesce(r.control_run_id, ""), r.stage, r.subset)
    replaced = Set(key(r) for r in eachrow(new))
    kept = filter(r -> key(r) ∉ replaced, existing)
    return vcat(kept, new), nrow(existing) - nrow(kept)
end

"(label, Target LL, UUID) of every row in the first table of the v1.1 board."
function _dryrun_v11_rows()
    root = joinpath(@__DIR__, "..")
    markdown = read(`git -C $root show $DRYRUN_V11_SHA:experiments/LEADERBOARD.md`, String)
    rows = Tuple{String,Float64,String}[]
    for line in split(first(split(markdown, "\n## ")), '\n')
        m = match(r"^\| `([^`]+)` \| ([0-9.]+) \|.*\| `([0-9a-f-]{36})` \|$", line)
        m === nothing || push!(rows, (String(m[1]), parse(Float64, m[2]), String(m[3])))
    end
    return rows
end

"Headline rows of the posterior-grid tables whose Target LL renders as \"—\"."
function _dryrun_dash_target_ll(markdown)
    in_grid = false
    n = 0
    for line in split(markdown, '\n')
        startswith(line, "## Posterior grid") && (in_grid = true)
        startswith(line, "### ") && (in_grid = false)
        in_grid && occursin(r"^\| `[^`]+` \| — \|", line) && (n += 1)
    end
    return n
end

function dryrun_main()
    groups = load_run_groups(DRYRUN_CSV)
    refs = unique!(reduce(vcat, (group.refs for group in groups); init = Harness.RunRef[]))
    include_run_loaders(refs)
    ds = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours = 10_000)

    t0 = time()
    (; scores, failures) = score_csv_groups(groups, ds; target_seasons = ["24/25", "25/26"],
                                            expected_fixtures = 710)
    score_seconds = round(time() - t0; digits = 1)
    isempty(failures) || error("scoring failures: $(failures)")
    new = _dryrun_normalise(scores)
    length(unique(new.run_id)) == 26 || error("expected 26 scored runs, got $(length(unique(new.run_id)))")

    production = _dryrun_normalise(Harness.read_scores(PostgresStorage("harness");
                                                        scorecard_version = "v1.2"))
    merged, n_replaced = _dryrun_replace(production, new)
    grid = filter(:stage => ==("grid"), merged)
    board = Harness.leaderboard(grid)

    headline = [:target_logloss_all, :target_logloss_1x2, :target_ece_all, :compression_slope]
    nan_rows = filter(r -> any(c -> !isfinite(r[c]), headline), board)
    for r in eachrow(nan_rows)
        println("NaN_ROW model=$(r.model) run=$(r.run_id) control=$(r.control_name)")
    end

    v11 = _dryrun_v11_rows()
    length(v11) == 32 || error("expected 32 v1.1 first-table pairs, got $(length(v11))")
    missing_pairs = 0
    ll_mismatch = 0
    for (label, ll, run_id) in v11
        rows = filter(r -> r.model == label && string(r.run_id) == run_id, board)
        if nrow(rows) == 0
            missing_pairs += 1
            println("MISSING_V11_PAIR label=$label run=$run_id")
        elseif !all(r -> abs(r.target_logloss_all - ll) <= 5e-6, eachrow(rows))
            ll_mismatch += 1
            println("V11_TARGET_LL_CHANGED label=$label v11=$ll v12=$(rows.target_logloss_all)")
        end
    end

    register = only(eachrow(filter(:id => ==("scottish_dynamics_scope_matrix_w1"),
        Harness.read_experiments(PostgresStorage("harness")))))
    w1 = Set(strip.(split(String(register.run_ids), r"[,;]")))
    length(w1) == 12 || error("expected 12 W1 run UUIDs in the register, got $(length(w1))")
    w1_rows = filter(r -> string(r.run_id) in w1, board)
    controlled(id) = filter(r -> _dryrun_id(r.control_run_id) == id, w1_rows)
    n_m12 = length(unique(string.(controlled(DRYRUN_M12).run_id)))
    n_tdj = length(unique(string.(controlled(DRYRUN_TD_LOWER_JOINT).run_id)))
    tdj_names = unique(controlled(DRYRUN_TD_LOWER_JOINT).control_name)

    script = Module(:DryrunBoard)
    Base.include(script, joinpath(@__DIR__, "leaderboard.jl"))
    markdown = script.leaderboard_markdown(board, DataFrame())
    out = get(ENV, "KLM_DRYRUN_BOARD", "")
    isempty(out) || write(out, markdown)
    dash_ll = _dryrun_dash_target_ll(markdown)

    println("KLM_BOARD_DRYRUN scored=$(length(unique(new.run_id))) score_seconds=$score_seconds " *
            "replaced_rows=$n_replaced board_rows=$(nrow(board)) nan_headline_rows=$(nrow(nan_rows)) " *
            "markdown_dash_target_ll=$dash_ll v11_pairs=$(length(v11) - missing_pairs)/$(length(v11)) " *
            "v11_target_ll_mismatch=$ll_mismatch w1_m12=$n_m12 w1_tdj=$n_tdj " *
            "tdj_control_name=$(join(tdj_names, '|'))")
    # The v1.1 Target LL comparison is diagnostic: v1.2 may legitimately move a headline.
    ok = nrow(nan_rows) == 0 && dash_ll == 0 && missing_pairs == 0 && n_m12 == 12 &&
         n_tdj == 12 && tdj_names == ["td_lower_joint"]
    ok || error("board dry-run gate failed")
    println("KLM_BOARD_DRYRUN_PASS")
end

dryrun_main()
