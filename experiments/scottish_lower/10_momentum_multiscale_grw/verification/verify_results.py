"""Read-only sign-off audit of committed CSV artifacts; Python standard library only."""
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
NAMES = ("m01_poisson_time_decay", "m02_poisson_grw_1st_order", "m03_poisson_momentum_grw")


def rows(path):
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def audit():
    production = {r["model"]: r for r in rows("production/production_runs.csv")}
    headlines = {r["model"]: r for r in rows("evaluation/headlines.csv")}
    scores = {(r["model"], r["scope"]): r for r in rows("evaluation/proper_scores.csv")}
    assert set(production) == set(headlines) == set(NAMES)
    panel_rows = rows("evaluation/portfolio_panel.csv")
    panel = {r["match_id"] for r in panel_rows}
    assert len(panel) == len(panel_rows) == 622
    refusals = rows("evaluation/portfolio_refusals.csv")
    common_outcomes = None
    common_refusals = None
    for name in NAMES:
        p, h = production[name], headlines[name]
        assert (int(p["folds"]), int(p["oos"]), int(p["draws"])) == (40, 710, 800)
        assert int(p["persistence_stride"]) == 4
        assert p["passed"] == p["gate_pass"] == "true"
        assert float(p["max_rhat"]) <= 1.05
        assert min(float(p["min_ess_bulk"]), float(p["min_ess_tail"])) >= 200
        assert int(p["n_divergent"]) == 0 and int(p["n_transitions"]) == 128000
        assert float(p["min_bfmi"]) >= 0.30 and float(p["treedepth_rate"]) <= 0.05
        assert float(p["worst_partition"]) <= 1e-12
        assert p["run_id"] == h["run_id"] and h["portfolio_id"]
        filtration = rows(f"production/{name}_filtration.csv")
        assert len(filtration) == 40 and sum(int(r["n_oos"]) for r in filtration) == 710
        assert all(r["ordered"] == "true" for r in filtration)

        obs = rows(f"evaluation/{name}_observations.csv")
        labels = {(r["match_id"], r["selection"]): (r["y"], r["p_market"]) for r in obs}
        assert len(labels) == len(obs) == 2899
        assert len({r["match_id"] for r in obs}) == 627
        if common_outcomes is not None:
            assert labels == common_outcomes
        common_outcomes = labels
        for scope, count in (("all", 2899), ("1X2", 1785), ("OU2.5", 758), ("BTTS", 356)):
            selected = obs if scope == "all" else [r for r in obs if r["family"] == scope]
            assert len(selected) == count
            close(scores[name, scope]["logloss"], mean(float(r["ll_model"]) for r in selected))
            close(scores[name, scope]["market_logloss"], mean(float(r["ll_market"]) for r in selected))

        omitted = {(r["match_id"], r["reason"]) for r in refusals if r["model"] == name}
        assert Counter(reason for _, reason in omitted) == {"no closing quotes": 75, "no usable selection": 13}
        if common_refusals is not None:
            assert omitted == common_refusals
        common_refusals = omitted
        assert not panel.intersection(mid for mid, _ in omitted)
        bets = rows(f"evaluation/{name}_bets.csv")
        assert len(bets) == int(h["n_bets"])
        assert all(r["match_id"] in panel for r in bets)
        for r in bets:
            close(r["pnl"], float(r["stake"]) * float(r["payoff"]))
        stake = sum(float(r["stake"]) for r in bets)
        close(h["flat_roi_pct"], 100 * sum(float(r["pnl"]) for r in bets) / stake)
        for field, predicate in (("capital_ge4", lambda x: x >= 4), ("capital_le18", lambda x: x <= 1.8)):
            close(h[field], sum(float(r["stake"]) for r in bets if predicate(float(r["odds"]))) / stake)
        favourites = rows(f"evaluation/{name}_favourites.csv")
        assert len(favourites) == int(h["n_favourites"]) == 18
        close(h["favourite_model"], mean(float(r["p_model"]) for r in favourites))
        close(h["favourite_market"], mean(float(r["prob_fair_close"]) for r in favourites))
        close(h["favourite_realized"], mean(float(r["realized"]) for r in favourites))

    rates = rows("evaluation/market_inversions.csv")
    assert len(rates) == 710 and sum(r["accepted"] == "true" for r in rates) == 623
    paired = rows("evaluation/paired_logloss.csv")
    assert len(paired) == 28
    for r in paired:
        model_score = float(scores[r["model"], r["scope"]]["logloss"])
        other = scores[r["model"], r["scope"]]["market_logloss"] if r["comparator"] == "market" else scores[r["comparator"], r["scope"]]["logloss"]
        close(r["delta"], model_score - float(other))
        assert float(r["lo"]) <= float(r["hi"])
    posterior = rows("evaluation/momentum_posterior.csv")
    assert len(posterior) == 144  # 36 active-velocity folds x 2 sides x 2 parameters
    print("PASS: production gates, matched scoring rows, common portfolio panel, ledger arithmetic,")
    print("      favourite summaries, bootstrap point estimates, and posterior artifact coverage.")


if __name__ == "__main__":
    audit()
