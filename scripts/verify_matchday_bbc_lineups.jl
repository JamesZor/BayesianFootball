# Verify the native MatchDay BBC lineup source against the 2026-09-05 Scottish Lower slate.
#
# This is a network/database integration check, not a unit test. It reads credentials only via
# `MatchDay.paper_connection()` / `BF_DB_URL`, fetches BBC's public CDN, and intentionally
# persists the self-healed event crosswalk, player mappings, and full lineups in `betdb.bbc`.
#
# Usage:
#   julia --project -t 8 scripts/verify_matchday_bbc_lineups.jl

# %%
# ===================================================================
# 1. Packages and frozen verification instant
# ===================================================================
using BayesianFootball
using DataFrames, Dates
import LibPQ

const BBCV_MD = BayesianFootball.MatchDay
const BBCV_DD = BayesianFootball.Data
const BBCV_DAY = Date(2026, 9, 5)
const BBCV_AS_OF = DateTime(2026, 9, 5, 13, 35)
const BBCV_TOURNAMENTS = [56, 57]
const BBCV_EXPECTED_FIXTURES = 10

# %%
# ===================================================================
# 2. Fixture inventory
# ===================================================================
function bbcv_fixtures()
    lo = Int(round(datetime2unix(DateTime(BBCV_DAY))))
    hi = Int(round(datetime2unix(DateTime(BBCV_DAY + Day(1)))))
    conn = BBCV_MD.paper_connection()
    try
        frame = DataFrame(LibPQ.execute(conn, """
            SELECT match_id, home_team, away_team, start_timestamp, tournament_id
            FROM sofascore.events
            WHERE tournament_id = ANY(\$1)
              AND start_timestamp >= \$2 AND start_timestamp < \$3
            ORDER BY start_timestamp, match_id
        """, (BBCV_TOURNAMENTS, lo, hi)))
        return BBCV_MD.Fixture[
            BBCV_MD.Fixture(Int(row.match_id), String(row.home_team), String(row.away_team),
                            unix2datetime(row.start_timestamp), Int(row.tournament_id))
            for row in eachrow(frame)
        ]
    finally
        close(conn)
    end
end

# %%
# ===================================================================
# 3. BBC fetch, three-tier resolution, and strict 11v11 gate
# ===================================================================
function bbcv_verify()
    fixtures = bbcv_fixtures()
    length(fixtures) == BBCV_EXPECTED_FIXTURES || error(
        "BBC verification expected $BBCV_EXPECTED_FIXTURES fixtures on $BBCV_DAY; " *
        "found $(length(fixtures))")

    ds = BBCV_DD.load_datastore_cached(BBCV_DD.ScottishLower(); max_age_hours = 72)
    source = BBCV_MD.BBCLineupSource(ds = ds)
    row_type = NamedTuple{(:match_id, :fixture, :source, :confirmed, :home_starters,
                           :away_starters, :synthetic),
                          Tuple{Int,String,Symbol,Bool,Int,Int,Int}}
    rows = row_type[]
    for fixture in fixtures
        selected = BBCV_MD.lineup(source, fixture, BBCV_AS_OF)
        home_starters = selected === nothing ? 0 : count(player -> !player.substitute, selected.home)
        away_starters = selected === nothing ? 0 : count(player -> !player.substitute, selected.away)
        synthetic = selected === nothing ? -1 :
                    count(player -> player.player_id < 0, vcat(selected.home, selected.away))
        push!(rows, (match_id = fixture.m_id,
                     fixture = "$(fixture.home) v $(fixture.away)",
                     source = selected === nothing ? :none : selected.source,
                     confirmed = selected !== nothing && selected.confirmed,
                     home_starters, away_starters, synthetic))
    end
    report = DataFrame(rows)
    show(stdout, MIME"text/plain"(), report; allrows = true, allcols = true)
    println()

    complete = (report.source .== :bbc) .& report.confirmed .&
               (report.home_starters .>= 11) .& (report.away_starters .>= 11)
    all(complete) || error(
        "BBC verification failed for match IDs $(join(report.match_id[.!complete], ", "))")
    println("BBC MatchDay verification passed: $(sum(complete))/$(nrow(report)) complete confirmed lineups")
    return report
end

bbcv_verify()
