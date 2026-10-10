# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Phase 4 derived tables. Local, file-only: reads the committed results/phase3 CSVs and
writes results/phase4. No Julia, no database, no fits.

Run: uv run experiments/scotland/08_qs_marketobs_backtest/x04_phase4_tables.py

Uncertainty follows the Phase 3 scheme (8-week moving blocks within season, B = 999,
seed 20261009, 90% percentile interval) re-implemented with numpy's RNG, so the draws
differ from the Julia run. `check_vs_julia` reports how close the re-implementation lands on
the committed paired-vs-m12 intervals. The circular variant is a sensitivity only.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P3 = HERE / "results" / "phase3"
OUT = HERE / "results" / "phase4"
B, BLOCK, SEED = 999, 8, 20261009


def week_index(dates: pd.Series, seasons: pd.Series):
    """Season -> list of weeks (Monday start, sorted), each a list of row positions."""
    monday = dates - pd.to_timedelta(dates.dt.weekday, unit="D")
    keys = pd.DataFrame({"season": seasons.values, "week": monday.values, "pos": np.arange(len(dates))})
    out = {}
    for season, g in keys.groupby("season", sort=True):
        out[season] = [w.pos.to_numpy() for _, w in g.groupby("week", sort=True)]
    return out


def block_bootstrap(num, den, weeks, circular=False, seed=SEED):
    """Ratio statistic sum(num)/sum(den) over resampled weeks; den = 1 per day gives a mean."""
    rng = np.random.default_rng(seed)
    wn = {s: np.array([num[w].sum() for w in ws]) for s, ws in weeks.items()}
    wd = {s: np.array([den[w].sum() for w in ws]) for s, ws in weeks.items()}
    stats = np.empty(B)
    for b in range(B):
        tn = td = 0.0
        for s in sorted(weeks):
            W = len(weeks[s])
            L = min(BLOCK, W)
            chosen = []
            while len(chosen) < W:
                if circular:
                    start = rng.integers(0, W)
                    chosen.extend((start + k) % W for k in range(L))
                else:
                    start = rng.integers(0, W - L + 1)
                    chosen.extend(range(start, start + L))
            idx = np.array(chosen[:W])
            tn += wn[s][idx].sum()
            td += wd[s][idx].sum()
        stats[b] = tn / td
    return {"point": num.sum() / den.sum(), "boot_mean": stats.mean(),
                "lo90": np.quantile(stats, 0.05), "hi90": np.quantile(stats, 0.95)}


def paired(daily, a, b):
    """Daily log-growth difference a - b over the union of days (missing day = 0)."""
    x = daily[a].set_index("date")[["log_growth", "season"]]
    y = daily[b].set_index("date")[["log_growth", "season"]]
    days = x.index.union(y.index)
    season = x.season.combine_first(y.season).reindex(days)
    diff = x.log_growth.reindex(days, fill_value=0.0) - y.log_growth.reindex(days, fill_value=0.0)
    unmatched = int((~days.isin(x.index) | ~days.isin(y.index)).sum())
    return pd.Series(days), season, diff.to_numpy(), unmatched


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(P3 / "phase3_daily.csv", parse_dates=["date"])
    julia = pd.read_csv(P3 / "phase3_paired_vs_m12.csv")
    sel = lambda m, s, p: (daily.model == m) & (daily.snapshot == s) & (daily.policy == p)
    models = sorted(daily.model.unique())
    snapshots = sorted(daily.snapshot.unique())
    policies = sorted(daily.policy.unique())

    rows = []
    def contrast(kind, m_a, p_a, m_b, p_b, snap):
        days, season, diff, unmatched = paired(daily, sel(m_a, snap, p_a), sel(m_b, snap, p_b))
        weeks = week_index(days, season)
        ones = np.ones_like(diff)
        for circ in (False, True):
            bs = block_bootstrap(diff, ones, weeks, circular=circ)
            rows.append(dict(contrast=kind, snapshot=snap, a=f"{m_a}|{p_a}", b=f"{m_b}|{p_b}",
                             scheme="circular" if circ else "noncircular", n_days=len(diff),
                             unmatched_days=unmatched, **bs))

    for snap in snapshots:
        for m in models:
            contrast("canonical_minus_plain_kelly", m, "canonical", m, "plain_kelly", snap)
            contrast("scenario_minus_canonical", m, "scenario_kelly", m, "canonical", snap)
        for p in policies:
            for m in models:
                if m != "m12":
                    contrast("model_minus_m12", m, p, "m12", p, snap)
    con = pd.DataFrame(rows)
    con.to_csv(OUT / "phase4_contrasts.csv", index=False)

    # How far the numpy re-implementation lands from the committed Julia intervals.
    mine = con[(con.contrast == "model_minus_m12") & (con.scheme == "noncircular")].copy()
    mine["model"] = mine.a.str.split("|").str[0]
    mine["policy"] = mine.a.str.split("|").str[1]
    chk = mine.merge(julia, on=["model", "snapshot", "policy"], suffixes=("_py", "_jl"))
    assert len(chk) == len(julia) and np.allclose(chk.point_py, chk.point_jl, atol=1e-12, rtol=0)
    chk["dlo"] = chk.lo90_py - chk.lo90_jl
    chk["dhi"] = chk.hi90_py - chk.hi90_jl
    chk[["model", "snapshot", "policy", "point_jl", "lo90_jl", "lo90_py", "hi90_jl", "hi90_py", "dlo", "dhi"]] \
        .to_csv(OUT / "phase4_check_vs_julia.csv", index=False)
    print(f"check_vs_julia: {len(chk)} pairs, points equal; max |dlo| {chk.dlo.abs().max():.5f}, "
          f"max |dhi| {chk.dhi.abs().max():.5f} (log-growth units)")

    # CLV: bets with a valid close only; resampled by the same week blocks.
    bets = pd.read_csv(P3 / "phase3_bets.csv.gz", parse_dates=["date"])
    seasons = daily[["date", "season"]].drop_duplicates().set_index("date").season
    clv_rows = []
    for (m, s, p), g in bets.groupby(["model", "snapshot", "policy"], sort=True):
        g = g[g.clv_pp.notna()].sort_values("date").reset_index(drop=True)
        weeks = week_index(g.date, g.date.map(seasons))
        pp = block_bootstrap(g.clv_pp.to_numpy(), np.ones(len(g)), weeks)
        pct = block_bootstrap(g.clv_pct.to_numpy(), np.ones(len(g)), weeks)
        beat = block_bootstrap((g.clv_pp > 0).to_numpy(float) * 100, np.ones(len(g)), weeks)
        stake_w = block_bootstrap((g.clv_pct * g.stake_unit).to_numpy(), g.stake_unit.to_numpy(), weeks)
        clv_rows.append({"model": m, "snapshot": s, "policy": p, "clv_n": len(g),
                             "clv_pp": pp["point"], "clv_pp_lo90": pp["lo90"], "clv_pp_hi90": pp["hi90"],
                             "clv_pct": pct["point"], "clv_pct_lo90": pct["lo90"], "clv_pct_hi90": pct["hi90"],
                             "beat_pct": beat["point"], "beat_lo90": beat["lo90"], "beat_hi90": beat["hi90"],
                             "clv_pct_stake_weighted": stake_w["point"],
                             "clv_pct_sw_lo90": stake_w["lo90"], "clv_pct_sw_hi90": stake_w["hi90"]})
    pd.DataFrame(clv_rows).to_csv(OUT / "phase4_clv_intervals.csv", index=False)

    # Same fixtures, two timings: the early snapshot's own bets against the T-25m bets on
    # the fixtures that early book prices. ROI = sum(pnl_frac) / sum(stake_frac) (experiment 07's
    # convention); growth cannot be split by fixture without re-staking the slates.
    books = {"T-6h_s90": "phase0_t360_book.csv", "T-24h_s90": "phase0_t1440_book.csv",
             "T-24h_s360": "phase0_t1440_sensitivity360_book.csv"}
    same = []
    for snap, f in books.items():
        ids = set(pd.read_csv(HERE / "results" / "phase0_historical" / f, usecols=["match_id"]).match_id)
        for timing, s in (("early", snap), ("T-25m_same_fixtures", "T-25m_s90")):
            g = bets[(bets.snapshot == s) & bets.match_id.isin(ids)]
            for (m, p), h in g.groupby(["model", "policy"], sort=True):
                h = h.sort_values("date").reset_index(drop=True)
                weeks = week_index(h.date, h.date.map(seasons))
                roi = block_bootstrap(h.pnl_frac.to_numpy() * 100, h.stake_frac.to_numpy(), weeks)
                c = h[h.clv_pp.notna()]
                same.append({"early_snapshot": snap, "timing": timing, "model": m, "policy": p,
                                 "n_fixtures_priced": len(ids), "n_bet_fixtures": h.match_id.nunique(),
                                 "n_bets": len(h), "roi_frac_pct": roi["point"], "roi_lo90": roi["lo90"],
                                 "roi_hi90": roi["hi90"], "clv_n": len(c), "clv_pp": c.clv_pp.mean()})
    pd.DataFrame(same).to_csv(OUT / "phase4_same_fixtures.csv", index=False)
    print(f"wrote {OUT}/phase4_contrasts.csv ({len(con)} rows), phase4_clv_intervals.csv "
          f"({len(clv_rows)} rows), phase4_same_fixtures.csv ({len(same)} rows)")


if __name__ == "__main__":
    main()
