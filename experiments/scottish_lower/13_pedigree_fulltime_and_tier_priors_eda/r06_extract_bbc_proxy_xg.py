#!/usr/bin/env python3
"""Extract the project BBC commentary proxy-xG for Scottish tiers 54--57.

This is deliberately a bounded, read-only EDA extraction, with same-window empirical-
Bayes conversion estimation rather than predictive model training. It adapts the measurement part of
`Features.pxg_match_observations`: parse BBC shot commentary, fit the global empirical-
Bayes zone/body/context table, and sum predicted attempt values by match and side.

Outputs are fixture-grain so a later worker can join them to its full-time status fixture
CSV on `match_id`. The fixture universe is r01's finished, normal-time-score event panel,
not only matches-table enrichment; absent commentary is observable as coverage=0.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
R01_FIXTURES_PATH = DATA_DIR / "r01_tier_league_fixtures.csv"
START = "2021-01-01 00:00:00+00"
END_EXCLUSIVE = "2026-09-24 00:00:00+00"  # includes the requested 2026-09-23 UTC date
TIERS = (54, 55, 56, 57)
PSEUDO_COUNT = 25.0
DEFAULT_BASE_RATE = 0.1
DEFAULT_PENALTY_XG = 0.76
SHOT_EVENTS = (
    "goal", "attempt_missed", "attempt_saved", "attempt_blocked", "post",
    "penalty_missed", "penalty_saved",
)

# Exact ordered vocabulary and free-kick remap from src/features/plus_minus/shot_parser.jl.
ZONE_PATTERNS = (
    ("the left side of the six yard box", "six_yard_side"),
    ("the right side of the six yard box", "six_yard_side"),
    ("a difficult angle and long range", "difficult_long"),
    ("the centre of the box", "box_centre"),
    ("the left side of the box", "box_side"),
    ("the right side of the box", "box_side"),
    ("a difficult angle on the left", "difficult_angle"),
    ("a difficult angle on the right", "difficult_angle"),
    ("very close range", "six_yard_centre"),
    ("more than 35 yards", "very_long_range"),
    ("more than 40 yards", "very_long_range"),
    ("long range on the left", "long_range"),
    ("long range on the right", "long_range"),
    ("outside the box", "outside_box"),
    ("a free kick", "free_kick_zone"),
)
BODY_PATTERNS = (("header", "header"), ("right footed", "right_foot"),
                 ("left footed", "left_foot"))
CONTEXT_PATTERNS = (("from a direct free kick", "direct_free_kick"),
                    ("following a set piece situation", "set_piece"),
                    ("following a corner", "corner"), ("following a fast break", "fast_break"))

# r01's events-based normal-time fixture panel is canonical for this EDA. `sofascore.matches`
# is only an enrichment domain, not fixture truth; the audit query below quantifies its mismatch.
MATCH_ENRICHMENT_SQL = """
SELECT m.match_id, m.tournament_id, s.name AS season, m.start_timestamp,
       m.status_type AS matches_status, e.status_type AS events_status,
       e.raw_data #>> '{homeScore,normaltime}' AS events_normal_home,
       e.raw_data #>> '{awayScore,normaltime}' AS events_normal_away,
       m.home_team, m.away_team, m.home_score, m.away_score
FROM sofascore.matches AS m
LEFT JOIN sofascore.events AS e ON e.match_id = m.match_id
JOIN sofascore.seasons AS s
  ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
WHERE m.tournament_id = ANY(%s)
  AND m.start_timestamp >= %s
  AND m.start_timestamp < %s
  AND m.home_score IS NOT NULL
  AND m.away_score IS NOT NULL
ORDER BY m.start_timestamp, m.match_id
"""

EVENT_SQL = """
SELECT lt.match_id, lt.post_index, lt.event_type, lt.text,
       CASE
         WHEN regexp_replace(lt.team, '-fc$', '') =
              regexp_replace(mm.bbc_home_slug, '-fc$', '') THEN true
         WHEN regexp_replace(lt.team, '-fc$', '') =
              regexp_replace(mm.bbc_away_slug, '-fc$', '') THEN false
         ELSE NULL
       END AS is_home_event
FROM bbc.live_text AS lt
JOIN bbc.match_meta AS mm ON mm.match_id = lt.match_id
WHERE lt.match_id = ANY(%s) AND lt.event_type = ANY(%s)
ORDER BY lt.match_id, lt.post_index
"""


def first_label(text: str, patterns: Iterable[tuple[str, str]], default: str) -> str:
    for needle, label in patterns:
        if needle in text:
            return label
    return default


def parse_shot(event_type: str, text: str | None) -> dict[str, Any]:
    """Faithful Python counterpart of Features.parse_shot."""
    event_type = event_type or ""
    if text is None:
        return {"zone": "unknown", "body_part": "unknown", "context": "open_play",
                "is_penalty": event_type.startswith("penalty"), "parsed": False}
    lower = text.lower()
    is_penalty = event_type.startswith("penalty") or "penalty" in lower
    zone = first_label(lower, ZONE_PATTERNS, "unknown")
    body = first_label(lower, BODY_PATTERNS, "unknown")
    context = first_label(lower, CONTEXT_PATTERNS, "open_play")
    if zone == "free_kick_zone":
        zone = "outside_box"
        context = "direct_free_kick"
    return {"zone": zone, "body_part": body, "context": context,
            "is_penalty": is_penalty, "parsed": zone != "unknown" or is_penalty}


def read_rows(conn: psycopg.Connection, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [column.name for column in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def round_or_blank(value: float | None, digits: int = 6) -> str:
    return "" if value is None or not math.isfinite(value) else str(round(value, digits))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    ss_y = sum((y - mean_y) ** 2 for y in ys)
    if ss_x == 0.0 or ss_y == 0.0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / math.sqrt(ss_x * ss_y)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fixture_reconciliation(r01_rows: list[dict[str, str]],
                                 matches_rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Audit source-universe disagreement without allowing it to drop r01 fixtures."""
    r01_by_id = {int(row["match_id"]): row for row in r01_rows}
    matches_by_id = {int(row["match_id"]): row for row in matches_rows}
    r01_ids, matches_ids = set(r01_by_id), set(matches_by_id)
    audit_rows: list[dict[str, Any]] = []
    for status, ids in (
        ("matched", sorted(r01_ids & matches_ids)),
        ("r01_events_only", sorted(r01_ids - matches_ids)),
        ("matches_only", sorted(matches_ids - r01_ids)),
    ):
        for match_id in ids:
            r01 = r01_by_id.get(match_id, {})
            enriched = matches_by_id.get(match_id, {})
            audit_rows.append({
                "reconciliation_status": status,
                "match_id": match_id,
                "tournament_id": r01.get("tournament_id", enriched.get("tournament_id", "")),
                "season": r01.get("season", enriched.get("season", "")),
                "match_date": r01.get("match_date", enriched.get("start_timestamp", "")),
                "home_team": r01.get("home_team", enriched.get("home_team", "")),
                "away_team": r01.get("away_team", enriched.get("away_team", "")),
                "score_source": r01.get("score_source", "sofascore_matches_scores"),
                "matches_status": enriched.get("matches_status", ""),
                "events_status": enriched.get("events_status", "finished" if r01 else ""),
                "events_normal_home": r01.get("home_score", enriched.get("events_normal_home", "")),
                "events_normal_away": r01.get("away_score", enriched.get("events_normal_away", "")),
            })
    write_csv(RESULTS_DIR / "r06_bbc_proxy_xg_fixture_reconciliation.csv", audit_rows,
              ["reconciliation_status", "match_id", "tournament_id", "season", "match_date",
               "home_team", "away_team", "score_source", "matches_status", "events_status",
               "events_normal_home", "events_normal_away"])
    return len(r01_ids & matches_ids), len(r01_ids - matches_ids), len(matches_ids - r01_ids)


def main() -> None:
    dsn = os.environ.get("BF_DB_URL")
    if not dsn:
        raise RuntimeError("BF_DB_URL must be set; it is intentionally neither printed nor stored.")

    # default_transaction_read_only applies before the first query.  The explicit SET below is a
    # second guard and statement_timeout bounds accidental DB stalls.
    with psycopg.connect(dsn, options="-c default_transaction_read_only=on") as conn:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL statement_timeout = '120s'")
        matches_rows = read_rows(conn, MATCH_ENRICHMENT_SQL, (list(TIERS), START, END_EXCLUSIVE))
        with R01_FIXTURES_PATH.open(newline="", encoding="utf-8") as source:
            r01_rows = list(csv.DictReader(source))
        r01_ids = [int(row["match_id"]) for row in r01_rows]
        events = read_rows(conn, EVENT_SQL, (r01_ids, list(SHOT_EVENTS)))

    # r01's raw event scores and calendar window define the primary fixture denominator. Preserve
    # its season labels and exact normal-time goals; r06's matches-table row is never required.
    fixture_by_id = {int(row["match_id"]): {
        "match_id": int(row["match_id"]),
        "tournament_id": int(row["tournament_id"]),
        "season": row["season"],
        "match_date": row["match_date"],
        "home_team": row["home_team"], "away_team": row["away_team"],
        "home_score": float(row["home_score"]), "away_score": float(row["away_score"]),
    } for row in r01_rows}
    reconciled_counts = write_fixture_reconciliation(r01_rows, matches_rows)
    parsed_events: list[dict[str, Any]] = []
    for event in events:
        parsed = parse_shot(str(event["event_type"]), event["text"])
        parsed_events.append({**event, **parsed, "is_goal": event["event_type"] == "goal"})

    # Persist descriptors (not raw commentary) for an independent row-level Julia parser check.
    write_csv(DATA_DIR / "r06_bbc_proxy_xg_shot_descriptors.csv", parsed_events,
              ["match_id", "post_index", "zone", "body_part", "context", "is_penalty", "parsed"])

    # This exactly matches fit_shot_xg: the cell table uses every parsed BBC shot, including
    # a rare row whose side cannot be mapped. Side attribution is required only when values
    # are subsequently summed into a home/away match observation.
    usable = [event for event in parsed_events if event["is_home_event"] is not None]
    open_play = [event for event in parsed_events if not event["is_penalty"] and event["parsed"]]
    base_rate = safe_div(sum(event["is_goal"] for event in open_play), len(open_play))
    base_rate = DEFAULT_BASE_RATE if base_rate is None else base_rate
    penalties = [event for event in parsed_events if event["is_penalty"]]
    penalty_xg = safe_div(sum(event["is_goal"] for event in penalties), len(penalties))
    penalty_xg = DEFAULT_PENALTY_XG if penalty_xg is None else penalty_xg

    cell_shots: Counter[tuple[str, str, str]] = Counter()
    cell_goals: Counter[tuple[str, str, str]] = Counter()
    for event in open_play:
        cell = (event["zone"], event["body_part"], event["context"])
        cell_shots[cell] += 1
        cell_goals[cell] += int(event["is_goal"])
    cell_xg = {cell: (cell_goals[cell] + PSEUDO_COUNT * base_rate) /
               (count + PSEUDO_COUNT) for cell, count in cell_shots.items()}

    side_xg: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    side_events: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    side_parsed: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    raw_events = Counter(int(event["match_id"]) for event in parsed_events)
    unresolved_events = Counter(int(event["match_id"]) for event in parsed_events
                                if event["is_home_event"] is None)
    for event in usable:
        match_id = int(event["match_id"])
        side = 0 if event["is_home_event"] else 1
        side_events[match_id][side] += 1
        side_parsed[match_id][side] += int(event["parsed"])
        if event["is_penalty"]:
            value = penalty_xg
        else:
            cell = (event["zone"], event["body_part"], event["context"])
            value = cell_xg.get(cell, base_rate) if event["parsed"] else base_rate
        side_xg[match_id][side] += value

    match_rows: list[dict[str, Any]] = []
    for match_id, fixture in fixture_by_id.items():
        xg_h, xg_a = side_xg[match_id]
        resolved_h, resolved_a = side_events[match_id]
        parsed_h, parsed_a = side_parsed[match_id]
        home_goals, away_goals = int(fixture["home_score"]), int(fixture["away_score"])
        covered_h, covered_a = resolved_h > 0, resolved_a > 0
        match_rows.append({
            "match_id": match_id,
            "tournament_id": int(fixture["tournament_id"]),
            "season": fixture["season"],
            "match_date": fixture["match_date"],
            "home_team": fixture["home_team"], "away_team": fixture["away_team"],
            "home_goals": home_goals, "away_goals": away_goals,
            "bbc_shot_events_raw": raw_events[match_id],
            "bbc_shot_events_unresolved_side": unresolved_events[match_id],
            "bbc_shot_events_home_resolved": resolved_h,
            "bbc_shot_events_away_resolved": resolved_a,
            "bbc_parsed_home_resolved": parsed_h,
            "bbc_parsed_away_resolved": parsed_a,
            "proxy_xg_home": round(xg_h, 6) if covered_h else "",
            "proxy_xg_away": round(xg_a, 6) if covered_a else "",
            "proxy_xg_available_home": int(covered_h),
            "proxy_xg_available_away": int(covered_a),
            "proxy_xg_available_both": int(covered_h and covered_a),
            "proxy_minus_goals_home": round(xg_h - home_goals, 6) if covered_h else "",
            "proxy_minus_goals_away": round(xg_a - away_goals, 6) if covered_a else "",
        })
    match_rows.sort(key=lambda row: (row["match_date"], row["match_id"]))

    match_columns = list(match_rows[0]) if match_rows else ["match_id", "tournament_id", "season"]
    write_csv(DATA_DIR / "r06_bbc_proxy_xg_match.csv", match_rows, match_columns)
    # r06_bbc_proxy_xg_match is now already the 4,068-row status-ready r01 fixture panel.
    write_csv(DATA_DIR / "r06_bbc_proxy_xg_joined_fulltime_fixture.csv", match_rows, match_columns)

    def summary_rows(groups: dict[tuple[Any, ...], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        output = []
        for key, rows in sorted(groups.items()):
            both = [row for row in rows if row["proxy_xg_available_both"] == 1]
            pairs_xg = [float(row["proxy_xg_home"]) for row in both] + [float(row["proxy_xg_away"]) for row in both]
            pairs_goals = [float(row["home_goals"]) for row in both] + [float(row["away_goals"]) for row in both]
            diffs = [xg - goal for xg, goal in zip(pairs_xg, pairs_goals)]
            record = {"tournament_id": key[0]}
            if len(key) > 1:
                record["season"] = key[1]
            record.update({
                "completed_fixtures": len(rows),
                "fixtures_with_any_raw_bbc_shot_event": sum(row["bbc_shot_events_raw"] > 0 for row in rows),
                "fixtures_proxy_available_both": len(both),
                "fixture_coverage_both": round_or_blank(safe_div(len(both), len(rows))),
                "home_side_coverage": round_or_blank(safe_div(sum(row["proxy_xg_available_home"] for row in rows), len(rows))),
                "away_side_coverage": round_or_blank(safe_div(sum(row["proxy_xg_available_away"] for row in rows), len(rows))),
                "unresolved_side_event_share": round_or_blank(safe_div(
                    sum(row["bbc_shot_events_unresolved_side"] for row in rows),
                    sum(row["bbc_shot_events_raw"] for row in rows))),
                "mean_proxy_xg_per_side": round_or_blank(safe_div(sum(pairs_xg), len(pairs_xg))),
                "mean_goals_per_side_covered": round_or_blank(safe_div(sum(pairs_goals), len(pairs_goals))),
                "mean_proxy_minus_goals_per_side": round_or_blank(safe_div(sum(diffs), len(diffs))),
                "mae_proxy_vs_goals_per_side": round_or_blank(safe_div(sum(abs(diff) for diff in diffs), len(diffs))),
                "pearson_proxy_vs_goals_per_side": round_or_blank(pearson(pairs_xg, pairs_goals)),
            })
            output.append(record)
        return output

    by_tier_season: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_tier: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in match_rows:
        by_tier_season[(row["tournament_id"], row["season"])].append(row)
        by_tier[(row["tournament_id"],)].append(row)
    coverage_rows = summary_rows(by_tier_season)
    tier_rows = summary_rows(by_tier)
    coverage_columns = list(coverage_rows[0]) if coverage_rows else ["tournament_id", "season"]
    tier_columns = list(tier_rows[0]) if tier_rows else ["tournament_id"]
    write_csv(RESULTS_DIR / "r06_bbc_proxy_xg_coverage_by_tier_season.csv", coverage_rows, coverage_columns)
    write_csv(RESULTS_DIR / "r06_bbc_proxy_xg_coverage_by_tier.csv", tier_rows, tier_columns)

    coefficient_rows = [{"term_type": "global", "zone": "", "body_part": "", "context": "",
                         "shots": len(open_play), "goals": sum(event["is_goal"] for event in open_play),
                         "raw_goal_rate": round(base_rate, 8), "proxy_xg_per_shot": round(base_rate, 8),
                         "pseudo_count": PSEUDO_COUNT},
                        {"term_type": "penalty", "zone": "penalty", "body_part": "", "context": "",
                         "shots": len(penalties), "goals": sum(event["is_goal"] for event in penalties),
                         "raw_goal_rate": round(penalty_xg, 8), "proxy_xg_per_shot": round(penalty_xg, 8),
                         "pseudo_count": PSEUDO_COUNT}]
    for cell in sorted(cell_shots):
        count = cell_shots[cell]
        coefficient_rows.append({"term_type": "cell", "zone": cell[0], "body_part": cell[1],
                                 "context": cell[2], "shots": count, "goals": cell_goals[cell],
                                 "raw_goal_rate": round(cell_goals[cell] / count, 8),
                                 "proxy_xg_per_shot": round(cell_xg[cell], 8),
                                 "pseudo_count": PSEUDO_COUNT})
    write_csv(RESULTS_DIR / "r06_bbc_proxy_xg_conversion_coefficients.csv", coefficient_rows,
              ["term_type", "zone", "body_part", "context", "shots", "goals", "raw_goal_rate",
               "proxy_xg_per_shot", "pseudo_count"])

    matched_count, r01_only_count, matches_only_count = reconciled_counts
    print(f"Wrote {len(match_rows)} r01 normal-time fixtures and {len(parsed_events)} BBC shot events.")
    print(f"Source-universe audit: {matched_count} matched, {r01_only_count} r01 events-only, "
          f"{matches_only_count} matches-only.")
    print("Outputs: data/r06_bbc_proxy_xg_match.csv and results/r06_bbc_proxy_xg_*.csv")


if __name__ == "__main__":
    main()
