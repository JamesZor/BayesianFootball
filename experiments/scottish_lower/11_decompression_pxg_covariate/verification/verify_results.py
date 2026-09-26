#!/usr/bin/env python3
"""Read-only sign-off checks for TODO 024's committed CSV evidence."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


production = rows("production/production_runs.csv")
assert len(production) == 3
for row in production:
    assert int(row["folds"]) == 40
    assert int(row["oos"]) == 710
    assert int(row["draws"]) == 3200
    assert int(row["n_divergent"]) == 0
    assert row["gate_pass"] == "true"
    assert float(row["max_rhat"]) <= 1.05
    assert float(row["min_ess_bulk"]) >= 200
    assert float(row["min_ess_tail"]) >= 200
    assert float(row["worst_partition"]) <= 1e-12

headlines = {row["model"]: row for row in rows("evaluation/headlines.csv")}
assert set(headlines) == {
    "m01_poisson_time_decay",
    "m02_joint_gamma_poisson",
    "m03_negbin_pxg_covariate",
}
for row in headlines.values():
    assert int(row["n_inverted"]) == 623
    assert int(row["n_inversion_refused"]) == 87
    assert int(row["n_portfolio_fixtures"]) == 622

candidate = headlines["m03_negbin_pxg_covariate"]
joint = headlines["m02_joint_gamma_poisson"]
assert float(candidate["market_on_model_slope"]) < float(joint["market_on_model_slope"])
assert float(candidate["market_on_model_slope"]) > 1.15  # target was not reached

posterior = rows("evaluation/pxg_posterior.csv")
assert len(posterior) == 40
assert all(0.40 <= float(row["mean"]) <= 0.80 for row in posterior)
assert all(float(row["p_positive"]) >= 0.99 for row in posterior)

scores = rows("evaluation/proper_scores.csv")
assert len(scores) == 12
assert {int(row["n_obs"]) for row in scores if row["scope"] == "all"} == {2899}

panel = rows("evaluation/portfolio_panel.csv")
assert len(panel) == 622
assert len({row["match_id"] for row in panel}) == 622

print("TODO 024 verification PASS")
