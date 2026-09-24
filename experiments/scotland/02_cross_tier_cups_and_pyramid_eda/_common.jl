# Shared paths, constants and helpers for the TODO 029 pyramid EDA suite.
#
# Every rNN runner `include`s this file.  It loads no BayesianFootball code (the
# suite is pure descriptive / econometric work) so a warm REPL can re-include any
# runner in seconds.  The only DB entry point is `db_connect()`, which never
# prints the DSN.

using DataFrames, CSV, LibPQ, Dates, Statistics, StatsBase, Printf, JSON3, DotEnv
using LinearAlgebra, SparseArrays, Random

const SUITE   = @__DIR__
const REPO    = normpath(joinpath(SUITE, "..", "..", ".."))
const DATA    = joinpath(SUITE, "data")
const RESULTS = joinpath(SUITE, "results")
const FIGS    = joinpath(RESULTS, "figures")
foreach(d -> mkpath(d), (DATA, RESULTS, FIGS))

const AS_OF         = Date(2026, 9, 23)   # last calendar day included (UTC)
const PRIMARY_START = Date(2021, 7, 1)    # primary window: football seasons 21/22 .. 26/27
const LONG_START    = Date(2008, 7, 1)    # robustness window: all sofascore.events history

const LEAGUE_TIER = Dict(54 => 1, 55 => 2, 56 => 3, 57 => 4)
const CUP_IDS     = (73, 982, 1520)
const TOURNAMENTS = [54, 55, 56, 57, 73, 982, 1520]
const COMP_LABEL  = Dict(54 => "Premiership", 55 => "Championship", 56 => "League One",
                         57 => "League Two", 73 => "Scottish Cup", 982 => "League Cup",
                         1520 => "Challenge Cup")
const SENIOR   = ("T1", "T2", "T3", "T4", "T5")
const OLD_FIRM = ("celtic", "rangers")

"Football season start year: July..June blocks (2021 ⇒ 21/22)."
fseason(d::Date) = month(d) >= 7 ? year(d) : year(d) - 1
fs_label(y::Integer) = @sprintf("%02d/%02d", y % 100, (y + 1) % 100)

# ── database ────────────────────────────────────────────────────────────────
function db_connect()
    dsn = get(ENV, "BF_DB_URL", "")
    if isempty(dsn)
        envfile = joinpath(REPO, ".env")
        isfile(envfile) && DotEnv.load!(envfile)
        dsn = get(ENV, "BF_DB_URL", "")
    end
    isempty(dsn) && error("BF_DB_URL is not set; no extraction attempted.")
    conn = try
        LibPQ.Connection(dsn)
    catch err
        # libpq's message can embed the DSN; never echo it.
        error("Could not connect to betdb ($(typeof(err))); DSN suppressed.")
    end
    execute(conn, "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
    execute(conn, "SET statement_timeout = '300s'")
    return conn
end

query(conn, sql, params = []) = DataFrame(execute(conn, sql, params))

# ── fixture panel ───────────────────────────────────────────────────────────
function load_fixtures(window::Symbol = :primary)
    df = CSV.read(joinpath(DATA, "r01_pyramid_fixtures.csv"), DataFrame; missingstring = "")
    window === :primary && (df = df[df.in_primary_window, :])
    return df
end

senior_only(df) = df[in.(df.home_cat, Ref(SENIOR)) .& in.(df.away_cat, Ref(SENIOR)), :]

"""
    oriented_cross_tier(df)

Senior cross-tier fixtures re-expressed from the higher-tier club's side.
`venue ∈ {hi_home, lo_home, neutral}`; every paired home/away column is swapped
when the higher-tier club is the away side.
"""
function oriented_cross_tier(df)
    d = senior_only(df)
    d = d[coalesce.(d.home_tier .!= d.away_tier, false), :]
    hh = d.home_tier .< d.away_tier
    pick(a, b) = ifelse.(hh, a, b)
    out = DataFrame(match_id = d.match_id, tournament_id = d.tournament_id,
                    competition = d.competition, fs = d.fs, match_date = d.match_date,
                    round_name = d.round_name,
                    hi_tier = min.(d.home_tier, d.away_tier), lo_tier = max.(d.home_tier, d.away_tier),
                    hi_team = pick(d.home_team, d.away_team), lo_team = pick(d.away_team, d.home_team),
                    venue = ifelse.(d.neutral, "neutral", ifelse.(hh, "hi_home", "lo_home")))
    out.gap = out.lo_tier .- out.hi_tier
    for (h, a, nm) in (("home_goals", "away_goals", "goals"), ("home_shots", "away_shots", "shots"),
                       ("home_sot", "away_sot", "sot"), ("home_pxg", "away_pxg", "pxg"),
                       ("p_home_mkt", "p_away_mkt", "p_mkt"), ("odds_home", "odds_away", "odds"),
                       ("lam_home_mkt", "lam_away_mkt", "lam_mkt"),
                       ("bf_p_home", "bf_p_away", "bf_p"))
        hasproperty(d, h) || continue
        out[!, nm * "_hi"] = pick(d[!, h], d[!, a])
        out[!, nm * "_lo"] = pick(d[!, a], d[!, h])
    end
    for c in ("p_draw_mkt", "odds_draw", "bf_p_draw")
        hasproperty(d, c) && (out[!, c] = d[!, c])
    end
    out.gd_hi  = out.goals_hi .- out.goals_lo
    out.res_hi = Int.(sign.(out.gd_hi))       # +1 higher-tier win, 0 draw, −1 upset
    return out
end

# ── small numerics ──────────────────────────────────────────────────────────
nanmean(x) = (v = collect(skipmissing(x)); isempty(v) ? NaN : mean(v))
nanstd(x)  = (v = collect(skipmissing(x)); length(v) < 2 ? NaN : std(v))
nnz_(x)    = count(!ismissing, x)
se_mean(x) = (v = collect(skipmissing(x)); length(v) < 2 ? NaN : std(v) / sqrt(length(v)))

# ── output helpers ──────────────────────────────────────────────────────────
save_csv(name, df) = CSV.write(joinpath(RESULTS, name), df)
save_json(name, obj) = open(io -> JSON3.pretty(io, obj), joinpath(RESULTS, name), "w")

"GitHub-markdown table (keeps README tables generated, never hand-typed)."
function md_table(df::AbstractDataFrame; digits::Int = 3)
    f(v) = v isa AbstractFloat ? (isfinite(v) ? string(round(v; digits = digits)) : "") :
           ismissing(v) ? "" : string(v)
    cols = names(df)
    io = IOBuffer()
    println(io, "| ", join(cols, " | "), " |")
    println(io, "|", join(fill("---", length(cols)), "|"), "|")
    for r in eachrow(df)
        println(io, "| ", join((f(r[c]) for c in cols), " | "), " |")
    end
    return String(take!(io))
end
save_md(name, df; digits = 3) = write(joinpath(RESULTS, name), md_table(df; digits = digits))
