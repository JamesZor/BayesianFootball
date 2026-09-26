# l17 — Put an unplayed card into the DataStore so the walk-forward splitter can see it.
#
# WHY. `extend_fit` and `MatchDay.select_split` both derive folds from `ds.matches`, and the
# SQL fetcher loads FINISHED matches only. So the newest fold a live slate can condition on is
# the one whose held-out block was the LAST PLAYED biweek — the chain is always one full
# biweek (plus the gap to the card) staler than the data. Appending the card's fixtures,
# unscored, makes the splitter emit one more fold: trained on every played match, held out =
# the card. `extend_fit` samples it, and `select_split` rule 1 then picks it positively
# (`get_next_matches` of that fold IS the card).
#
# SAFETY.
# * Scores are `missing`, never a placeholder: anything that tries to train on an injected
#   row fails loudly instead of learning a fake 0-0.
# * Only `status_type = 'notstarted'` fixtures, inside an explicit [from, to) window.
# * The same call must run in the EXTENSION job and in the SLATE job, or the slate rebuilds
#   one boundary fewer and falls back to the stale fold (select_split warns when this happens).
#
# Requires BayesianFootball, DataFrames, Dates, LibPQ loaded by the caller; BF_DB_URL set.

"""
    inject_upcoming_fixtures!(ds, from::DateTime, to::DateTime; tournaments = [56, 57]) -> Vector{Int}

Append every not-started fixture of `tournaments` kicking off in `[from, to)` to `ds.matches`,
with missing scores. Returns the injected match ids. Idempotent: ids already present are skipped.
"""
function inject_upcoming_fixtures!(ds, from::DateTime, to::DateTime; tournaments = [56, 57])
    conn = LibPQ.Connection(ENV["BF_DB_URL"])
    rows = try
        DataFrame(LibPQ.execute(conn, """
            SELECT e.match_id, e.tournament_id, e.season_id, s.year AS season,
                   e.home_team, e.away_team, e.start_timestamp,
                   (e.raw_data #>> '{roundInfo,round}')::int AS round
            FROM sofascore.events e JOIN sofascore.seasons s ON s.season_id = e.season_id
            WHERE e.tournament_id = ANY(\$1) AND e.status_type = 'notstarted'
              AND e.start_timestamp >= \$2 AND e.start_timestamp < \$3
            ORDER BY e.start_timestamp, e.match_id""",
            (tournaments, Int(round(datetime2unix(from))), Int(round(datetime2unix(to))))))
    finally
        close(conn)
    end
    m = ds.matches
    have = Set(Int.(m.match_id))
    rows = rows[[!(Int(id) in have) for id in rows.match_id], :]
    isempty(rows) && return Int[]
    allowmissing!(m, [:home_score, :away_score])
    for r in eachrow(rows)
        ko = unix2datetime(r.start_timestamp)
        d = Date(ko)
        # match_week/biweek are tournament-local compatibility columns; pooled 56/57 folds use the
        # calendar clock from match_date. Continue the tournament-season's local count.
        same = (m.tournament_id .== r.tournament_id) .& coalesce.(m.season .== r.season, false)
        wk = any(same) ? maximum(m.match_week[same]) + 1 : 1
        push!(m, (tournament_id = Int32(r.tournament_id), season_id = Int32(r.season_id),
                  season = String(r.season), match_id = Int32(r.match_id), tournament_slug = missing,
                  home_team = r.home_team, away_team = r.away_team,
                  home_score = missing, away_score = missing, home_score_ht = missing,
                  away_score_ht = missing, winner_code = missing,
                  round = ismissing(r.round) ? missing : Int32(r.round),
                  injury_time1 = missing, injury_time2 = missing, has_xg = missing, has_stats = missing,
                  match_hour = hour(ko), match_month = month(ko), match_dayofweek = dayofweek(ko) - 1,
                  match_date = d, match_week = wk, match_biweek = cld(wk, 2));
              cols = :subset)
    end
    return Int.(rows.match_id)
end
