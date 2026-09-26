#!/usr/bin/env python3
"""All-fixture descriptive transition panel from the r01 events fixture universe.

A transition is an observed change of league tier between consecutive observed
club seasons. It is not a claimed administrative promotion/relegation: this
fixture data does not encode the mechanism. All league fixtures in the new
season are counted before selection of matches 1--5, 6--10 and 11--20.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

SUITE = Path("experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda")
DATA, RESULTS = SUITE / "data", SUITE / "results"


def read(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(name: str, rows: list[dict], fields: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as file:
        out = csv.DictWriter(file, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)


def window(match_number: int) -> str:
    return "1-5" if match_number <= 5 else "6-10" if match_number <= 10 else "11-20"


def num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.3f}" if values else ""


def main() -> None:
    fixtures = read("r01_tier_league_fixtures.csv")
    # Expected SPFL league fixture totals. The coverage indicator is a source
    # archive test, not proof a missing event was postponed/cancelled or lacked a
    # normal-time score. Current/as-of season and initial partial source season
    # are censored rather than judged incomplete.
    expected_fixtures = {54: 228, 55: 180, 56: 180, 57: 180}
    observed_by_tier_season = defaultdict(int)
    for fixture in fixtures:
        observed_by_tier_season[(int(fixture["tournament_id"]), fixture["season"])] += 1
    prices = {r["match_id"]: r for r in read("r01_betfair_1x2_last_coherent_preko.csv")}
    shots = {r["match_id"]: r for r in read("r01_bbc_shot_totals.csv")}
    pxg = {r["match_id"]: r for r in read("r06_bbc_proxy_xg_match.csv")}

    # One record for each club's participation in every fixture. The score and all
    # aligned measures are oriented to that club, so both home and away fixtures
    # contribute to its new-season match count.
    sides: list[dict] = []
    for r in fixtures:
        hg, ag = float(r["home_score"]), float(r["away_score"])
        for is_home, club, opponent, gf, ga in (
            (True, r["home_team"], r["away_team"], hg, ag),
            (False, r["away_team"], r["home_team"], ag, hg),
        ):
            price = prices.get(r["match_id"])
            shot = shots.get(r["match_id"])
            xg = pxg.get(r["match_id"])
            market_p = num(price["fair_p_home"] if is_home else price["fair_p_away"]) if price else None
            shot_gf = num(shot["bbc_home_shots"] if is_home else shot["bbc_away_shots"]) if shot else None
            shot_ga = num(shot["bbc_away_shots"] if is_home else shot["bbc_home_shots"]) if shot else None
            xg_gf = num(xg["proxy_xg_home"] if is_home else xg["proxy_xg_away"]) if xg and xg["proxy_xg_available_both"] == "1" else None
            xg_ga = num(xg["proxy_xg_away"] if is_home else xg["proxy_xg_home"]) if xg and xg["proxy_xg_available_both"] == "1" else None
            sides.append({"match_id": r["match_id"], "season": r["season"], "tier": int(r["tournament_id"]),
                          "match_date": r["match_date"], "club": club, "opponent": opponent, "is_home": is_home,
                          "gf": gf, "ga": ga, "goal_diff": gf - ga, "win": int(gf > ga),
                          "market_probability": market_p, "shot_diff": shot_gf - shot_ga if shot_gf is not None and shot_ga is not None else None,
                          "proxy_xg_diff": xg_gf - xg_ga if xg_gf is not None and xg_ga is not None else None})

    by_club_season: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for side in sides:
        by_club_season[(side["club"], side["season"])].append(side)
    for group in by_club_season.values():
        group.sort(key=lambda r: (r["match_date"], int(r["match_id"])))

    # Season chronology is fixture-date based, while membership itself uses real
    # season IDs/names carried by the events-to-seasons join.
    club_seasons: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for (club, season), group in by_club_season.items():
        tiers = {r["tier"] for r in group}
        if len(tiers) == 1:
            club_seasons[club].append((season, group))
    transition_rows = []
    for club, season_groups in club_seasons.items():
        season_groups.sort(key=lambda item: (item[1][0]["match_date"], item[0]))
        for index, (season, group) in enumerate(season_groups):
            if index == 0:
                continue  # beginning-of-source / first-observed-season censoring
            previous_season, previous_group = season_groups[index - 1]
            previous_tier, current_tier = previous_group[0]["tier"], group[0]["tier"]
            if previous_tier == current_tier:
                continue
            direction = "upward_tier_move" if current_tier < previous_tier else "downward_tier_move"
            total_matches = len(group)
            for match_number, row in enumerate(group, start=1):
                if match_number > 20:
                    break
                transition_rows.append({"club": club, "season": season, "previous_observed_season": previous_season,
                                        "previous_tier": previous_tier, "current_tier": current_tier,
                                        "movement_direction": direction, "match_id": row["match_id"], "match_date": row["match_date"],
                                        "opponent": row["opponent"], "is_home": int(row["is_home"]), "new_season_match_number": match_number,
                                        "window": window(match_number), "matches_observed_in_new_season": total_matches,
                                        "tier_season_observed_fixtures": observed_by_tier_season[(current_tier, season)],
                                        "tier_season_expected_fixtures": expected_fixtures[current_tier],
                                        "tier_season_fixture_coverage": f"{observed_by_tier_season[(current_tier, season)] / expected_fixtures[current_tier]:.3f}",
                                        "tier_season_archive_complete": int(observed_by_tier_season[(current_tier, season)] >= expected_fixtures[current_tier]),
                                        "has_complete_20_match_window": int(total_matches >= 20), "goals_for": row["gf"], "goals_against": row["ga"],
                                        "goal_difference": row["goal_diff"], "win": row["win"],
                                        "market_team_probability": "" if row["market_probability"] is None else f"{row['market_probability']:.6f}",
                                        "bbc_shot_difference": "" if row["shot_diff"] is None else f"{row['shot_diff']:.3f}",
                                        "bbc_proxy_xg_difference": "" if row["proxy_xg_diff"] is None else f"{row['proxy_xg_diff']:.3f}",
                                        "transition_definition": "consecutive observed seasons changed tier; mechanism unverified"})
    fields = list(transition_rows[0]) if transition_rows else ["club"]
    write("r08_transition_all_fixture_panel.csv", transition_rows, fields)

    summaries = []
    for (direction, label), group in sorted(_group(transition_rows, lambda r: (r["movement_direction"], r["window"])).items()):
        market = [float(r["market_team_probability"]) for r in group if r["market_team_probability"]]
        shot_diff = [float(r["bbc_shot_difference"]) for r in group if r["bbc_shot_difference"]]
        pxg_diff = [float(r["bbc_proxy_xg_difference"]) for r in group if r["bbc_proxy_xg_difference"]]
        summaries.append({"movement_direction": direction, "window": label, "club_fixture_rows": len(group),
                          "distinct_transition_club_seasons": len({(r["club"], r["season"]) for r in group}),
                          "rows_from_complete_20_match_windows": sum(int(r["has_complete_20_match_window"]) for r in group),
                          "rows_from_archive_complete_tier_seasons": sum(int(r["tier_season_archive_complete"]) for r in group),
                          "mean_goal_difference": mean([float(r["goal_difference"]) for r in group]),
                          "win_rate": mean([float(r["win"]) for r in group]),
                          "market_probability_coverage": f"{len(market)/len(group):.3f}", "mean_market_team_probability": mean(market),
                          "bbc_shot_difference_coverage": f"{len(shot_diff)/len(group):.3f}", "mean_bbc_shot_difference": mean(shot_diff),
                          "bbc_proxy_xg_difference_coverage": f"{len(pxg_diff)/len(group):.3f}", "mean_bbc_proxy_xg_difference": mean(pxg_diff)})
    write("r08_transition_window_descriptives.csv", summaries, list(summaries[0]) if summaries else ["movement_direction"])


def _group(rows: list[dict], key):
    output = defaultdict(list)
    for row in rows:
        output[key(row)].append(row)
    return output


if __name__ == "__main__":
    main()
