#!/usr/bin/env python3
"""Read-only extraction for Scottish pedigree EDA.

Requires BF_DB_URL in the environment.  It never prints the DSN, uses a
read-only transaction, and writes only suite-local CSV fixtures.  Run from the
repository root:

    python experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/r01_extract_empirical.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import psycopg

SUITE = Path("experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda")
DATA = SUITE / "data"
AS_OF = "2026-09-23"
TIERS = (54, 55, 56, 57)


def write_rows(path: Path, columns: list[str], rows: Iterable[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def columns(conn: psycopg.Connection, schema: str, table: str) -> list[str]:
    return [row[0] for row in conn.execute(
        """SELECT column_name FROM information_schema.columns
           WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position""",
        (schema, table),
    )]


def main() -> None:
    # The repository convention stores this operational credential in a git-ignored
    # root .env.  Read only the one required key and never echo its value.
    dsn = os.environ.get("BF_DB_URL")
    if not dsn:
        dotenv = Path(".env")
        if dotenv.exists():
            for line in dotenv.read_text(encoding="utf-8").splitlines():
                if line.startswith("BF_DB_URL="):
                    dsn = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not dsn:
        raise SystemExit("BF_DB_URL is not set; no extraction was attempted.")

    try:
        conn = psycopg.connect(dsn, connect_timeout=10)
    except psycopg.Error as error:
        # Never echo libpq's exception text: it may embed a credential-bearing DSN.
        raise SystemExit(f"Could not connect to betdb ({type(error).__name__}); DSN suppressed.") from None

    with conn:
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SET statement_timeout = '120s'")

        inventory = []
        for schema, table in (("sofascore", "events"), ("sofascore", "matches"), ("sofascore", "tournaments"),
                              ("sofascore", "seasons"), ("betfair", "match_meta"),
                              ("betfair", "markets"), ("betfair", "odds_history"),
                              ("bbc", "match_meta"), ("bbc", "match_stats"),
                              ("sofascore", "match_player_lineups")):
            inventory.append((schema, table, "|".join(columns(conn, schema, table))))
        write_rows(DATA / "r01_schema_inventory.csv", ["schema", "table", "columns"], inventory)

        # This verifies the supplied Scottish tier IDs against source metadata and
        # exposes their observed date extent.  Tournament IDs are data, not a prompt assumption.
        cur = conn.execute("""
            SELECT t.tournament_id, t.name AS tournament_name, COUNT(m.match_id) AS finished_matches,
                   MIN(m.start_timestamp::date) AS first_finished_date,
                   MAX(m.start_timestamp::date) AS last_finished_date
            FROM sofascore.tournaments AS t
            LEFT JOIN sofascore.matches AS m
              ON m.tournament_id = t.tournament_id AND m.status_type = 'finished'
            WHERE t.tournament_id = ANY(%s)
            GROUP BY t.tournament_id, t.name ORDER BY t.tournament_id
        """, (list(TIERS),))
        write_rows(DATA / "r01_tournament_inventory.csv", [d.name for d in cur.description], cur.fetchall())

        match_columns = columns(conn, "sofascore", "matches")
        required = {"match_id", "tournament_id", "start_timestamp", "home_team", "away_team",
                    "home_score", "away_score"}
        missing = required - set(match_columns)
        if missing:
            raise RuntimeError(f"sofascore.matches missing required columns: {sorted(missing)}")
        # `matches` is an enrichment table with materially incomplete historic
        # coverage. `events` is the primary fixture universe; its raw SofaScore
        # score object supplies normal-time scores, while seasons supplies actual
        # source season labels. This is essential for complete first-20 windows.
        event_date_expr = "to_timestamp(e.start_timestamp)::date"
        event_time_expr = "to_timestamp(e.start_timestamp)"
        event_score_h = "NULLIF(e.raw_data #>> '{homeScore,normaltime}', '')::float8"
        event_score_a = "NULLIF(e.raw_data #>> '{awayScore,normaltime}', '')::float8"
        fixtures_sql = f"""
            SELECT e.match_id, e.tournament_id, s.year AS season,
                   {event_date_expr} AS match_date, e.round,
                   e.home_team, e.away_team, {event_score_h} AS home_score,
                   {event_score_a} AS away_score, 'events_raw_normaltime' AS score_source
            FROM sofascore.events AS e
            JOIN sofascore.seasons AS s ON s.season_id = e.season_id
            WHERE e.tournament_id = ANY(%s) AND e.status_type = 'finished'
              AND {event_date_expr} >= DATE '2021-01-01' AND {event_date_expr} <= %s::date
              AND {event_score_h} IS NOT NULL AND {event_score_a} IS NOT NULL
            ORDER BY {event_date_expr}, e.match_id
        """
        fixture_cursor = conn.execute(fixtures_sql, (list(TIERS), AS_OF))
        fixture_columns = [d.name for d in fixture_cursor.description]
        fixture_rows = fixture_cursor.fetchall()
        write_rows(DATA / "r01_tier_league_fixtures.csv", fixture_columns, fixture_rows)

        # Enrichment domains key to the narrower `matches` table; their coverage is
        # always reported against the events fixture universe below.
        date_expr = "m.start_timestamp::date"
        # BBC has no provider xG field in the extracted statistics schema. Retain
        # shot-total coverage as a proxy-input diagnostic, never relabel it xG.
        shot_sql = f"""
            SELECT m.match_id, m.tournament_id, {date_expr} AS match_date,
                   s.home_value AS bbc_home_shots, s.away_value AS bbc_away_shots
            FROM sofascore.matches AS m
            JOIN bbc.match_stats AS s ON s.match_id = m.match_id
            WHERE m.tournament_id = ANY(%s) AND {date_expr} >= DATE '2021-01-01'
              AND {date_expr} <= %s::date AND s.stat_cat = 'basic' AND s.stat_type = 'shotsTotal'
        """
        cur = conn.execute(shot_sql, (list(TIERS), AS_OF))
        write_rows(DATA / "r01_bbc_shot_totals.csv", [d.name for d in cur.description], cur.fetchall())

        # Proposed market values are a sparse provider field. Aggregating non-null
        # rows documents coverage, but not a point-in-time squad valuation: lineups
        # and scrape timing may post-date kickoff.
        value_sql = f"""
            SELECT m.match_id, m.tournament_id, {date_expr} AS match_date,
                   l.is_home_team, COUNT(*) FILTER (WHERE l.proposed_market_value IS NOT NULL) AS players_with_value,
                   SUM(l.proposed_market_value) FILTER (WHERE l.proposed_market_value IS NOT NULL) AS proposed_value_sum,
                   MIN(l.proposed_market_value_currency) FILTER (WHERE l.proposed_market_value IS NOT NULL) AS currency
            FROM sofascore.matches AS m
            JOIN sofascore.match_player_lineups AS l ON l.match_id = m.match_id
            WHERE m.tournament_id = ANY(%s) AND {date_expr} >= DATE '2021-01-01'
              AND {date_expr} <= %s::date
            GROUP BY m.match_id, m.tournament_id, {date_expr}, l.is_home_team
            ORDER BY m.match_id, l.is_home_team DESC
        """
        cur = conn.execute(value_sql, (list(TIERS), AS_OF))
        write_rows(DATA / "r01_lineup_value_proxy.csv", [d.name for d in cur.description], cur.fetchall())

        odds_sql = f"""
            SELECT m.match_id, m.tournament_id, {date_expr} AS match_date, m.start_timestamp,
                   oh.odds_data
            FROM sofascore.matches AS m
            JOIN betfair.match_meta AS mm ON mm.match_id = m.match_id
            JOIN betfair.markets AS mk ON mk.match_id = m.match_id
            JOIN betfair.odds_history AS oh ON oh.market_id = mk.market_id
            WHERE m.tournament_id = ANY(%s) AND {date_expr} >= DATE '2021-01-01'
              AND {date_expr} <= %s::date AND m.status_type = 'finished'
              AND mm.status = 'SUCCESS' AND mm.is_verified = TRUE AND mk.market_type = 'MATCH_ODDS'
        """
        price_rows = []
        for match_id, tier, match_date, kickoff, odds_data in conn.execute(odds_sql, (list(TIERS), AS_OF)):
            if not isinstance(odds_data, dict):
                continue
            timestamps = odds_data.get("timestamps", [])
            home, draw, away = odds_data.get("home", []), odds_data.get("draw", []), odds_data.get("away", [])
            if not (len(timestamps) == len(home) == len(draw) == len(away)):
                continue
            kickoff_ms = kickoff.timestamp() * 1000.0
            valid = [i for i, ts in enumerate(timestamps) if ts <= kickoff_ms and all(x[i] is not None and x[i] > 1 for x in (home, draw, away))]
            if not valid:
                continue
            i = valid[-1]
            odds = [float(home[i]), float(draw[i]), float(away[i])]
            inverse = [1.0 / price for price in odds]
            total = sum(inverse)
            price_rows.append((match_id, tier, match_date, timestamps[i], (timestamps[i] - kickoff_ms) / 60000.0,
                               *odds, *(value / total for value in inverse), total - 1.0))
        write_rows(DATA / "r01_betfair_1x2_last_coherent_preko.csv",
                   ["match_id", "tournament_id", "match_date", "snapshot_unix_ms", "minutes_to_kickoff",
                    "odds_home", "odds_draw", "odds_away", "fair_p_home", "fair_p_draw", "fair_p_away", "overround"],
                   price_rows)

        membership_sql = f"""
            WITH event_sides AS (
                SELECT e.tournament_id, s.year AS season, {event_date_expr} AS match_date, e.home_team AS club
                FROM sofascore.events AS e JOIN sofascore.seasons AS s ON s.season_id = e.season_id
                WHERE e.tournament_id = ANY(%s) AND e.status_type = 'finished'
                  AND {event_date_expr} >= DATE '2021-01-01' AND {event_date_expr} <= %s::date
                UNION ALL
                SELECT e.tournament_id, s.year AS season, {event_date_expr} AS match_date, e.away_team AS club
                FROM sofascore.events AS e JOIN sofascore.seasons AS s ON s.season_id = e.season_id
                WHERE e.tournament_id = ANY(%s) AND e.status_type = 'finished'
                  AND {event_date_expr} >= DATE '2021-01-01' AND {event_date_expr} <= %s::date
            )
            SELECT tournament_id, season, club, MIN(match_date) AS first_match_date,
                   MAX(match_date) AS last_match_date, COUNT(*) AS league_matches
            FROM event_sides GROUP BY tournament_id, season, club
            ORDER BY season, tournament_id, club
        """
        cur = conn.execute(membership_sql, (list(TIERS), AS_OF, list(TIERS), AS_OF))
        write_rows(DATA / "r01_club_tier_membership.csv", [d.name for d in cur.description], cur.fetchall())

        # Retrospective same-season tier labels are correct for descriptive cup
        # bridges (not pre-match feature availability). Candidates exclude league
        # IDs and retain tournament name, reserve flag, and normal-time scores for
        # an explicit whitelist/review rather than silently treating non-league
        # opponents as League Two.
        cross_tier_sql = f"""
            WITH membership AS (
                SELECT e.tournament_id AS tier_id, s.year AS season, e.home_team AS club
                FROM sofascore.events e JOIN sofascore.seasons s ON s.season_id = e.season_id
                WHERE e.tournament_id = ANY(%s)
                UNION
                SELECT e.tournament_id, s.year, e.away_team
                FROM sofascore.events e JOIN sofascore.seasons s ON s.season_id = e.season_id
                WHERE e.tournament_id = ANY(%s)
            )
            SELECT e.match_id, e.tournament_id AS cup_tournament_id, t.name AS cup_tournament_name,
                   s.year AS season, {event_date_expr} AS match_date, e.home_team, e.away_team,
                   {event_score_h} AS home_normaltime_score, {event_score_a} AS away_normaltime_score,
                   h.tier_id AS home_tier, a.tier_id AS away_tier,
                   (e.home_team ~ '(b|c|u)[ -]?team|reserves|academy') AS home_possible_reserve,
                   (e.away_team ~ '(b|c|u)[ -]?team|reserves|academy') AS away_possible_reserve
            FROM sofascore.events e
            JOIN sofascore.seasons s ON s.season_id = e.season_id
            JOIN sofascore.tournaments t ON t.tournament_id = e.tournament_id
            LEFT JOIN membership h ON h.season = s.year AND h.club = e.home_team
            LEFT JOIN membership a ON a.season = s.year AND a.club = e.away_team
            WHERE e.status_type = 'finished' AND e.tournament_id <> ALL(%s)
              AND {event_date_expr} >= DATE '2021-01-01' AND {event_date_expr} <= %s::date
              AND {event_score_h} IS NOT NULL AND {event_score_a} IS NOT NULL
              AND h.tier_id IS NOT NULL AND a.tier_id IS NOT NULL AND h.tier_id <> a.tier_id
            ORDER BY match_date, e.match_id
        """
        cur = conn.execute(cross_tier_sql, (list(TIERS), list(TIERS), list(TIERS), AS_OF))
        write_rows(DATA / "r01_cross_tier_cup_candidates.csv", [d.name for d in cur.description], cur.fetchall())

        print(f"Wrote membership extract: {DATA / 'r01_club_tier_membership.csv'}")
        print(f"Wrote {len(fixture_rows)} completed league fixture rows and schema inventory.")


if __name__ == "__main__":
    main()
