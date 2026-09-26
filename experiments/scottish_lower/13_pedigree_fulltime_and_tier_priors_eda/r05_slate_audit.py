#!/usr/bin/env python3
"""Read-only provenance audit for the 2026-09-19 paper_runbook slate.

Writes bounded, reproducible extracts only. Connection strings and driver exceptions are
never emitted, because either may contain credentials. The paper ledger is the source of
operationally priced probabilities; mcmc_experiments proves the immutable run lineage.
"""
from __future__ import annotations

import csv
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

SLATE_DATE = "2026-09-19"
CONNECT_TIMEOUT_SECONDS = 5
STATEMENT_TIMEOUT_MILLISECONDS = 5_000
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / f"slate_{SLATE_DATE}.csv"
RESULTS_PATH = ROOT / "results" / f"slate_{SLATE_DATE}.csv"

LEDGER_SQL = """
SELECT
    s.slate_id::text, s.account_id, s.slate_window::text, s.as_of::text,
    s.model_run_id::text, s.run_name, s.fold_idx, s.bankroll, s.total_risk,
    s.slate_exposure, s.exposure_cap, s.k_risk, s.batch_status,
    o.order_id::text, o.match_id, e.home_team, e.away_team, o.kickoff::text, o.market_group,
    o.market_line, o.selection, o.side, o.venue_odds, o.effective_odds,
    o.p_model, o.p_market, o.edge, o.stake_fraction, o.risk, o.venue_stake,
    o.quote_ts::text, o.state,
    ca.close_prob, ca.close_ts::text, ca.close_source, ca.clv, ca.clv_pct,
    ps.outcome, ps.gross_return, ps.commission, ps.net_pnl, ps.settled_at::text,
    COALESCE(f.fill_count, 0) AS fill_count,
    COALESCE(f.filled_size, 0) AS filled_size,
    COALESCE(f.risk_filled, 0) AS risk_filled
FROM paper_runbook.paper_slates AS s
JOIN paper_runbook.paper_orders AS o ON o.slate_id = s.slate_id
LEFT JOIN sofascore.events AS e ON e.match_id = o.match_id
LEFT JOIN paper_runbook.clv_audit AS ca ON ca.order_id = o.order_id
LEFT JOIN paper_runbook.paper_settlements AS ps ON ps.order_id = o.order_id
LEFT JOIN LATERAL (
    SELECT count(*) AS fill_count, sum(size) AS filled_size, sum(risk_filled) AS risk_filled
    FROM paper_runbook.paper_fills WHERE order_id = o.order_id
) AS f ON true
WHERE s.slate_window = %s
ORDER BY s.as_of, o.kickoff, o.order_id;
"""

RUN_SQL = """
SELECT r.run_id::text, r.id, r.name, r.experiment_name, r.status,
       r.git_commit, r.git_branch, r.created_at::text, r.finished_at::text,
       c.config_hash, c.model_config::text
FROM runs AS r
LEFT JOIN configs AS c ON c.config_id = r.run_id
WHERE r.run_id = %s::uuid;
"""


def _connect(dsn: str):
    try:
        import psycopg  # type: ignore
        return psycopg.connect(dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS)
    except ImportError:
        import psycopg2  # type: ignore
        return psycopg2.connect(dsn, connect_timeout=CONNECT_TIMEOUT_SECONDS)


def _enable_read_only(conn: Any) -> None:
    """Enable server-side read-only transactions for psycopg 2 or 3."""
    if hasattr(conn, "set_session"):
        conn.set_session(readonly=True, autocommit=True)
    else:
        conn.autocommit = True
        conn.execute("SET default_transaction_read_only = on")
    with conn.cursor() as cursor:
        cursor.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT_MILLISECONDS}ms'")


def _rows(conn: Any, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def _all_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Preserve columns that occur only on later rows (e.g. resolved run provenance)."""
    return list(dict.fromkeys(column for row in rows for column in row))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _text(row.get(column)) for column in columns})


def main() -> int:
    betdb_dsn = os.environ.get("BF_DB_URL")
    if not betdb_dsn:
        print("BLOCKED: BF_DB_URL is not configured; no database query was attempted.", file=sys.stderr)
        return 2

    try:
        ledger = _connect(betdb_dsn)
        _enable_read_only(ledger)
        orders = _rows(ledger, LEDGER_SQL, (SLATE_DATE,))
        ledger.close()
    except Exception:
        print("BLOCKED: unable to read paper_runbook; connection details suppressed.", file=sys.stderr)
        return 3

    if not orders:
        print(f"BLOCKED: no paper_runbook orders found for slate date {SLATE_DATE}.", file=sys.stderr)
        return 4

    order_columns = _all_columns(orders)
    _write_csv(DATA_PATH, orders, order_columns)

    run_ids = sorted({row["model_run_id"] for row in orders if row["model_run_id"]})
    provenance: list[dict[str, Any]] = []
    experiments_dsn = os.environ.get(
        "BF_EXPERIMENTS_DB_URL", "postgresql://postgres@mcmc-beast:5432/mcmc_experiments"
    )
    try:
        experiments = _connect(experiments_dsn)
        _enable_read_only(experiments)
        for run_id in run_ids:
            provenance.extend(_rows(experiments, RUN_SQL, (run_id,)))
        experiments.close()
    except Exception:
        # Preserve the ledger extraction: this is an explicit provenance gap, not a reason to
        # replace exact operational probabilities with an unverified reconstructed prediction.
        provenance = [{"run_id": run_id, "provenance_status": "UNRESOLVED"} for run_id in run_ids]

    provenance_by_id = {row["run_id"]: row for row in provenance}
    results: list[dict[str, Any]] = []
    for order in orders:
        row = dict(order)
        p_model = Decimal(str(order["p_model"]))
        p_market = Decimal(str(order["p_market"]))
        edge = Decimal(str(order["edge"]))
        bankroll = Decimal(str(order["bankroll"]))
        edge_difference = edge - (p_model - p_market)
        row["edge_probability_points"] = edge * Decimal("100")
        row["edge_difference"] = edge_difference
        # Each stored probability and edge is rounded to six decimal places, so independent
        # subtraction may differ by one final unit. Larger differences are audit failures.
        row["edge_equals_probability_difference"] = abs(edge_difference) <= Decimal("0.000001")
        row["risk_share_of_slate_total_pct"] = (
            Decimal(str(order["risk"])) / Decimal(str(order["total_risk"])) * Decimal("100")
            if Decimal(str(order["total_risk"])) else None
        )
        row["risk_share_of_bankroll_pct"] = Decimal(str(order["risk"])) / bankroll * Decimal("100") if bankroll else None
        row.update({f"run_{key}": value for key, value in provenance_by_id.get(order["model_run_id"], {}).items()})
        results.append(row)

    result_columns = _all_columns(results)
    _write_csv(RESULTS_PATH, results, result_columns)
    print(f"Wrote {len(orders)} ledger rows to {DATA_PATH.relative_to(ROOT)}")
    print(f"Wrote {len(results)} audited rows to {RESULTS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
