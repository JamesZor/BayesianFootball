#!/usr/bin/env python3
"""Dependency-free descriptive EDA for r01 Scottish tier fixtures.

Produces transparent CSV tables, exact binomial intervals for outcome rates, and
club-season bootstrap intervals for means.  The bootstrap resamples clubs within
each tier-season, retaining every fixture row for a sampled club; it is an
uncertainty sensitivity, not a claim of independent club observations.
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

SUITE = Path("experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda")
DATA, RESULTS = SUITE / "data", SUITE / "results"
N_BOOT, SEED = 2_000, 20260923
TIER_NAMES = {54: "Premiership", 55: "Championship", 56: "League One", 57: "League Two"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lower, upper = int(pos), math.ceil(pos)
    return values[lower] if lower == upper else values[lower] + (values[upper] - values[lower]) * (pos - lower)


def bootstrap_cluster_mean(rows: list[dict], value: str, rng: random.Random) -> tuple[float, float]:
    # Clubs appear home and away; cluster membership is deliberately the home club
    # for this fixture-oriented sensitivity. It avoids pretending 4,066 matches
    # are i.i.d., but does not solve schedule or opponent dependence.
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["home_team"]].append(float(row[value]))
    clubs = list(groups)
    draws = []
    for _ in range(N_BOOT):
        sampled = [groups[rng.choice(clubs)] for _ in clubs]
        values = [value for cluster in sampled for value in cluster]
        draws.append(sum(values) / len(values))
    return percentile(draws, .025), percentile(draws, .975)


def mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.3f}" if values else ""


def wilson(successes: int, n: int) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return centre - radius, centre + radius


def main() -> None:
    raw = read_csv("r01_tier_league_fixtures.csv")
    rows = []
    for row in raw:
        try:
            home, away = float(row["home_score"]), float(row["away_score"])
        except ValueError:
            continue
        # Correct result in this source is stored regulation final score. It cannot
        # distinguish a score amended after a later ruling; that limitation remains.
        rows.append({**row, "home_score": home, "away_score": away,
                     "goal_diff": home - away, "total_goals": home + away,
                     "home_win": int(home > away), "draw": int(home == away)})

    rng = random.Random(SEED)
    tier_rows = []
    for tier in sorted(TIER_NAMES):
        group = [row for row in rows if int(row["tournament_id"]) == tier]
        n = len(group)
        mean_diff = sum(row["goal_diff"] for row in group) / n
        mean_total = sum(row["total_goals"] for row in group) / n
        home_wins, draws = sum(row["home_win"] for row in group), sum(row["draw"] for row in group)
        gd_lo, gd_hi = bootstrap_cluster_mean(group, "goal_diff", rng)
        tg_lo, tg_hi = bootstrap_cluster_mean(group, "total_goals", rng)
        hw_lo, hw_hi = wilson(home_wins, n)
        dr_lo, dr_hi = wilson(draws, n)
        tier_rows.append({"tier_id": tier, "tier_name": TIER_NAMES[tier], "matches": n,
                          "home_goals_per_match": f"{sum(r['home_score'] for r in group)/n:.3f}",
                          "away_goals_per_match": f"{sum(r['away_score'] for r in group)/n:.3f}",
                          "mean_goal_difference_home_minus_away": f"{mean_diff:.3f}",
                          "cluster_bootstrap_95pct_low": f"{gd_lo:.3f}", "cluster_bootstrap_95pct_high": f"{gd_hi:.3f}",
                          "goals_per_match": f"{mean_total:.3f}", "total_goals_cluster_bootstrap_95pct_low": f"{tg_lo:.3f}",
                          "total_goals_cluster_bootstrap_95pct_high": f"{tg_hi:.3f}",
                          "home_win_rate": f"{home_wins/n:.3f}", "home_win_wilson_95pct_low": f"{hw_lo:.3f}",
                          "home_win_wilson_95pct_high": f"{hw_hi:.3f}", "draw_rate": f"{draws/n:.3f}",
                          "draw_wilson_95pct_low": f"{dr_lo:.3f}", "draw_wilson_95pct_high": f"{dr_hi:.3f}"})
    write_csv("r02_tier_goal_descriptives.csv", tier_rows, list(tier_rows[0]))

    season_rows = []
    for (season, tier), group in sorted(((key, group) for key, group in _group(rows, lambda r: (r['season'], int(r['tournament_id']))).items())):
        n = len(group)
        season_rows.append({"season": season, "tier_id": tier, "tier_name": TIER_NAMES[tier], "matches": n,
                            "goals_per_match": f"{sum(r['total_goals'] for r in group)/n:.3f}",
                            "home_goal_difference": f"{sum(r['goal_diff'] for r in group)/n:.3f}"})
    write_csv("r02_tier_season_goal_descriptives.csv", season_rows, list(season_rows[0]))

    prices = {row["match_id"]: row for row in read_csv("r01_betfair_1x2_last_coherent_preko.csv")}
    market_rows = []
    for tier in sorted(TIER_NAMES):
        paired = [(row, prices[row["match_id"]]) for row in rows if int(row["tournament_id"]) == tier and row["match_id"] in prices]
        n = len(paired)
        if not n:
            continue
        supremacies = [math.log(float(p["fair_p_home"]) / float(p["fair_p_away"])) for _, p in paired]
        minutes = [float(p["minutes_to_kickoff"]) for _, p in paired]
        market_rows.append({"tier_id": tier, "tier_name": TIER_NAMES[tier], "coherent_preko_1x2_matches": n,
                            "fixture_coverage": f"{n / len([r for r in rows if int(r['tournament_id']) == tier]):.3f}",
                            "mean_log_fair_p_home_over_away": f"{sum(supremacies)/n:.3f}",
                            "median_minutes_to_kickoff": f"{percentile(minutes, .5):.2f}",
                            "p10_minutes_to_kickoff": f"{percentile(minutes, .1):.2f}",
                            "p90_minutes_to_kickoff": f"{percentile(minutes, .9):.2f}"})
    write_csv("r02_betfair_1x2_coverage_and_supremacy.csv", market_rows, list(market_rows[0]))

    fixture_ids = {row["match_id"] for row in rows}
    shots = [row for row in read_csv("r01_bbc_shot_totals.csv") if row["match_id"] in fixture_ids]
    shot_rows = []
    for tier in sorted(TIER_NAMES):
        group = [row for row in shots if int(row["tournament_id"]) == tier and row["bbc_home_shots"] and row["bbc_away_shots"]]
        n = len(group)
        fixture_n = len([row for row in rows if int(row["tournament_id"]) == tier])
        if n:
            diffs = [float(row["bbc_home_shots"]) - float(row["bbc_away_shots"]) for row in group]
            shot_rows.append({"tier_id": tier, "tier_name": TIER_NAMES[tier], "matches_with_basic_bbc_shot_totals": n,
                              "fixture_coverage": f"{n/fixture_n:.3f}", "mean_home_minus_away_shots": f"{sum(diffs)/n:.3f}",
                              "measurement_note": "shotsTotal is not xG; BBC schema contains no provider xG field"})
    write_csv("r02_bbc_shot_coverage.csv", shot_rows, list(shot_rows[0]))

    values = read_csv("r01_lineup_value_proxy.csv")
    value_rows = []
    for tier in sorted(TIER_NAMES):
        group = [row for row in values if int(row["tournament_id"]) == tier]
        covered = [row for row in group if row["players_with_value"] and int(row["players_with_value"]) > 0]
        currencies = sorted(set(row["currency"] for row in covered))
        value_rows.append({"tier_id": tier, "tier_name": TIER_NAMES[tier], "team_match_rows": len(group),
                           "rows_with_any_proposed_value": len(covered), "row_coverage": f"{len(covered)/len(group):.3f}" if group else "",
                           "currencies": "|".join(currencies),
                           "measurement_note": "lineup-derived proposed values; no scrape-time guarantee, unsuitable as point-in-time resource measure"})
    write_csv("r02_lineup_value_proxy_coverage.csv", value_rows, list(value_rows[0]))

    # Status contrast: strict uses Verified statuses only; sensitivity admits
    # Inferred statuses. Both require a direct season/tier/slug panel match and
    # exclude Hybrid/Unknown rather than treating them as PT.
    status_rows = read_csv("spfl_club_operational_status.csv")
    statuses = {(r["season"], r["tournament_id"], r["club_name"]): r for r in status_rows}
    status_out = []
    for panel, accepted_evidence in (("strict_verified", {"Verified"}), ("continuity_inferred_sensitivity", {"Verified", "Inferred"})):
        buckets = defaultdict(list)
        exclusion = defaultdict(int)
        for match in rows:
            if int(match["tournament_id"]) != 56:
                continue
            key_base = (match["season"], match["tournament_id"])
            home = statuses.get((*key_base, match["home_team"]))
            away = statuses.get((*key_base, match["away_team"]))
            if home is None or away is None:
                exclusion["unmatched_slug_or_panel"] += 1
                continue
            if home["evidence_level"] not in accepted_evidence or away["evidence_level"] not in accepted_evidence:
                exclusion["not_accepted_evidence"] += 1
                continue
            hs, aws = home["operational_status"], away["operational_status"]
            if hs not in {"Full-Time", "Part-Time"} or aws not in {"Full-Time", "Part-Time"}:
                exclusion["hybrid_or_unknown"] += 1
                continue
            # Orient every mixed-status result to the FT team, including away FT.
            if hs == "Full-Time" and aws == "Part-Time":
                category, ft_gd, ft_win = "FT_vs_PT", match["goal_diff"], int(match["goal_diff"] > 0)
            elif hs == "Part-Time" and aws == "Full-Time":
                category, ft_gd, ft_win = "FT_vs_PT", -match["goal_diff"], int(match["goal_diff"] < 0)
            elif hs == aws == "Full-Time":
                category, ft_gd, ft_win = "FT_vs_FT", match["goal_diff"], match["home_win"]
            else:
                category, ft_gd, ft_win = "PT_vs_PT", match["goal_diff"], match["home_win"]
            buckets[category].append((ft_gd, ft_win))
        for category in ("FT_vs_PT", "FT_vs_FT", "PT_vs_PT"):
            values = buckets[category]
            if values:
                n = len(values)
                wins = sum(v[1] for v in values)
                lo, hi = wilson(wins, n)
                status_out.append({"panel": panel, "match_category": category, "matches": n,
                                   "mean_oriented_goal_difference": f"{sum(v[0] for v in values)/n:.3f}",
                                   "oriented_win_rate": f"{wins/n:.3f}",
                                   "win_rate_wilson_95pct_low": f"{lo:.3f}", "win_rate_wilson_95pct_high": f"{hi:.3f}",
                                   "excluded_unmatched_slug_or_panel": exclusion["unmatched_slug_or_panel"],
                                   "excluded_not_accepted_evidence": exclusion["not_accepted_evidence"],
                                   "excluded_hybrid_or_unknown": exclusion["hybrid_or_unknown"]})
    write_csv("r02_league_one_ft_pt_status_descriptives.csv", status_out, list(status_out[0]))

    # Supplement status cells with same-match, oriented market / BBC / lineage
    # measures. Value is explicitly non-PIT and only compared when currencies match.
    pxg = {r["match_id"]: r for r in read_csv("r06_bbc_proxy_xg_match.csv")}
    value_sides = {(r["match_id"], r["is_home_team"].lower()): r for r in read_csv("r01_lineup_value_proxy.csv")}
    status_metrics = []
    for panel, accepted_evidence in (("strict_verified", {"Verified"}), ("continuity_inferred_sensitivity", {"Verified", "Inferred"})):
        buckets = defaultdict(list)
        for match in rows:
            if int(match["tournament_id"]) != 56:
                continue
            home = statuses.get((match["season"], match["tournament_id"], match["home_team"]))
            away = statuses.get((match["season"], match["tournament_id"], match["away_team"]))
            if not home or not away or home["evidence_level"] not in accepted_evidence or away["evidence_level"] not in accepted_evidence:
                continue
            hs, aws = home["operational_status"], away["operational_status"]
            if hs not in {"Full-Time", "Part-Time"} or aws not in {"Full-Time", "Part-Time"}:
                continue
            if hs == "Full-Time" and aws == "Part-Time":
                category, orient = "FT_vs_PT", 1.0
            elif hs == "Part-Time" and aws == "Full-Time":
                category, orient = "FT_vs_PT", -1.0
            elif hs == aws == "Full-Time":
                category, orient = "FT_vs_FT", 1.0
            else:
                category, orient = "PT_vs_PT", 1.0
            price = prices.get(match["match_id"])
            market_diff = (float(price["fair_p_home"]) - float(price["fair_p_away"])) * orient if price else None
            shot = next((s for s in shots if s["match_id"] == match["match_id"]), None)
            shot_diff = (float(shot["bbc_home_shots"]) - float(shot["bbc_away_shots"])) * orient if shot and shot["bbc_home_shots"] and shot["bbc_away_shots"] else None
            proxy = pxg.get(match["match_id"])
            proxy_diff = (float(proxy["proxy_xg_home"]) - float(proxy["proxy_xg_away"])) * orient if proxy and proxy["proxy_xg_available_both"] == "1" else None
            home_val, away_val = value_sides.get((match["match_id"], "true")), value_sides.get((match["match_id"], "false"))
            value_diff = None
            if home_val and away_val and home_val["proposed_value_sum"] and away_val["proposed_value_sum"] and home_val["currency"] == away_val["currency"]:
                value_diff = (float(home_val["proposed_value_sum"]) - float(away_val["proposed_value_sum"])) * orient
            market_logratio = math.log(float(price["fair_p_home"]) / float(price["fair_p_away"])) * orient if price else None
            buckets[category].append((market_diff, shot_diff, proxy_diff, value_diff, market_logratio))
        for category, values in buckets.items():
            each = lambda index: [v[index] for v in values if v[index] is not None]
            market, shot_d, proxy_d, value_d = each(0), each(1), each(2), each(3)
            status_metrics.append({"panel": panel, "match_category": category, "matches": len(values),
                                   "market_supremacy_difference_coverage": f"{len(market)/len(values):.3f}", "mean_fair_probability_difference": mean(market),
                                   "mean_log_fair_probability_ratio": mean(each(4)),
                                   "bbc_shot_difference_coverage": f"{len(shot_d)/len(values):.3f}", "mean_bbc_shot_difference": mean(shot_d),
                                   "bbc_proxy_xg_difference_coverage": f"{len(proxy_d)/len(values):.3f}", "mean_bbc_proxy_xg_difference": mean(proxy_d),
                                   "nonpit_lineup_value_difference_coverage": f"{len(value_d)/len(values):.3f}", "mean_nonpit_value_difference_eur": mean(value_d)})
    write_csv("r02_league_one_ft_pt_aligned_metrics.csv", status_metrics, list(status_metrics[0]))

    # Transition candidates are an observable fixture-history construct, not proof
    # of promotion/relegation. A club's first fixture after a tier change is counted;
    # first observed seasons and incomplete 20-match windows retain explicit flags.
    by_club = _group(rows, lambda r: r["home_team"])
    transitions = []
    for club, home_rows in by_club.items():
        ordered = sorted(home_rows, key=lambda r: (r["match_date"], int(r["match_id"])))
        previous_tier = None
        match_number = 0
        for row in ordered:
            tier = int(row["tournament_id"])
            if tier != previous_tier:
                prior = previous_tier
                match_number = 1
                previous_tier = tier
            else:
                match_number += 1
                prior = previous_tier
            if match_number <= 20:
                transitions.append({"club": club, "match_id": row["match_id"], "match_date": row["match_date"],
                                    "season": row["season"], "current_tier": tier, "previous_observed_tier": prior or "",
                                    "home_entry_match_number": match_number,
                                    "window": "1-5" if match_number <= 5 else "6-10" if match_number <= 10 else "11-20",
                                    "home_score": row["home_score"], "away_score": row["away_score"],
                                    "goal_difference": row["goal_diff"],
                                    "method_note": "home fixtures only; tier-change candidate, not verified promotion/relegation"})
    write_csv("r02_transition_home_fixture_candidates.csv", transitions, list(transitions[0]))


def _group(rows: list[dict], key):
    result = defaultdict(list)
    for row in rows:
        result[key(row)].append(row)
    return result


if __name__ == "__main__":
    main()
