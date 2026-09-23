#!/usr/bin/env python3
"""Read-only provenance probe for existing m12 relational artefacts.

It intentionally does not deserialize fit blobs or price mean lambdas. The
match_latents table has rate summaries plus compressed draws; extracting true
posterior-predictive 1X2 probabilities requires approved Julia artefact loading
and draw-wise score-grid pricing on the common point-in-time fixture cohort.
"""
from __future__ import annotations
import csv
from pathlib import Path
import psycopg

OUT = Path("experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/results/r08_saved_m12_provenance.csv")
RUN_ID = "928dad3b-ccaf-4909-b6b7-4f1a815e1cab"

with psycopg.connect("host=mcmc-beast port=5432 dbname=mcmc_experiments user=postgres connect_timeout=10") as conn:
    conn.execute("SET TRANSACTION READ ONLY")
    row = conn.execute("""
        SELECT r.id, r.run_id, r.name, r.experiment_name, r.created_at,
               COUNT(ml.match_id) AS latent_rows,
               MIN(ml.match_id) AS first_match_id, MAX(ml.match_id) AS last_match_id,
               pr.portfolio_run_id, pr.n_bets, pr.max_drawdown_pct, pr.metadata
        FROM runs r
        LEFT JOIN fold_results fr ON fr.run_id = r.run_id
        LEFT JOIN match_latents ml ON ml.fold_id = fr.fold_id
        LEFT JOIN portfolio_runs pr ON pr.model_run_id = r.run_id
        WHERE r.run_id = %s
        GROUP BY r.id, r.run_id, r.name, r.experiment_name, r.created_at,
                 pr.portfolio_run_id, pr.n_bets, pr.max_drawdown_pct, pr.metadata
    """, (RUN_ID,)).fetchone()
OUT.parent.mkdir(exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["run_integer_id", "run_id", "run_name", "experiment_name", "created_at", "latent_rows", "first_match_id", "last_match_id", "portfolio_run_id", "portfolio_n_bets", "portfolio_max_drawdown_pct", "portfolio_metadata"]); w.writerow(row)
print(f"Wrote {OUT}")
