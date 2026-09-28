# Napkin brief 04 — grw_spfl_joint on 26/27 Scottish League One/Two, priced from the live order book

The same rules as `BRIEF.md` and `BRIEF_03.md` apply:
- solo Claude CLI, no subagents;
- commit only in `current_development/clv_napkin/` and push `proto/clv-napkin`;
- Julia only on mcmc-beast (tmux `claude_clv_napkin`);
- databases are read-only, and never print credentials.

Use the loader/runner pair `l04_*.jl` / `r04_*.jl`.

**New in this brief:** the human has approved **one sampling job**, a walk-forward extension of
one fit. That job is the only sampling allowed.

**The question.** How would `grw_spfl_joint` (W1, run `82e01a3b-d686-4e29-a269-3702b96fc5ba`)
have done this season (26/27) in **Scottish League One and Two (56/57) only**, entering early?
Price every entry at the **executable back price from the live 1-minute order book** that the
human has been collecting for 26/27 in betdb (schema `betfair_live`), not at LTP.

## Step 1: read-only discovery (do this first and report it briefly)

- **Find the order-book tables in `betfair_live`.** Record:
  - the columns and snapshot cadence;
  - whether the best 3 back/lay prices and sizes and the total matched volume are present;
  - how the rows map to fixtures and runners.
- **Coverage for 56/57 26/27 MATCH_ODDS.** Report:
  - the number of fixtures played so far;
  - how many of them have order-book data;
  - the first snapshot's lead time (median and range);
  - gaps in the series.

  Keep queries light, because the collector is live.
- **Check that betdb has 26/27 results and the BBC stats** needed for the joint model's pxG, for
  those fixtures.
- **If the order book is unusable** for 56/57, fall back to `betfair.odds_history` LTP and say so.
  If the results or the model's inputs are missing, print `CLV_R04_BLOCKED` with the reason.

## Step 2: walk-forward extension of `grw_spfl_joint` over 26/27

- **Held-out predictions.** Extend the fit so that every played 26/27 League One/Two match day
  gets held-out predictions, the same way the W1 folds did: train on everything strictly before
  the match day's calendar date, then predict that day. Mind the memory note on
  `LastHistorical` leakage: filter on a strictly earlier calendar day.
- **Reuse existing code.** Use the `Training.extend_fit`/MatchDay live-extension path, and keep the
  same model, recipe and sampler settings as the original run. Rebuild the DataStore from the DB,
  since the cached `.jls` predates most of 26/27.
- **Do NOT modify any production run in `mcmc_experiments`.** Prefer an in-memory extension
  serialised to `/root/BF_runs/clv_napkin_dev/out/` on the beast. If persistence is unavoidable,
  save it as a brand-new run under experiment `clv_napkin_2627_scratch` and say so.
- **One heavy job at a time.**
  - Check `pgrep -af julia` first. The only other process should be your own idle REPL (pid
    3812829); stop it if it is yours.
  - Run with `-t 16`, and log to `/root/BF_runs/logs/clv_napkin/`.
- **Convergence:** report R̂ and the divergence rate per fold. The rerun trigger is R̂ > 1.05 or
  divergences > 0.1%.

## Step 3: evaluate (1X2, back only, edge ≥ 2 pp, flat 1 unit, commission 2%)

- **Entry instants:** earliest order-book snapshot, T−24h, T−6h, T−3h, T−60m, T−25m.
  - Price each entry at the **best available back price at that minute**, and record the size
    available at it and the back–lay spread in ticks.
  - For comparison, also compute the same bets at LTP, to measure what the spread costs.
- **Model edge** is measured against the de-vigged mid-price book at the entry minute.
- **Close:** the de-vigged **mid-price** TWA over (−20, 0] minutes. Also show the §2 LTP-based
  close as a cross-check.
- **Report per instant:**
  - n bets and mean `clv_pp` [slate-bootstrap CI];
  - EV@close at 2% [CI] and realised ROI at 2% [CI];
  - the median back size available, and the share of bets with ≥ £20, ≥ £50 and ≥ £100 at the
    taken price;
  - the median spread.
- **Also report, descriptively (not as a test):**
  - the §12 AGAINST/WITH split at T−60m (direction since T−3h, on the mid-price);
  - a market-only null (back everything) at each instant.
- **Be blunt about sample size.** About two months of 56/57 is roughly 100 bets per instant.
  State the CI width and what effect size it could detect. Draw no conclusions beyond that.

## Output

Append **§13 "26/27 live season: grw_spfl_joint on the order book"** to `REPORT.md`. It should
contain:
- a one-paragraph answer;
- the coverage table;
- the per-instant table;
- a short "spread and size" paragraph, i.e. what the LTP-based numbers of §10–12 lose once the
  entry is an executable back price;
- caveats.

Put CSVs in `out/r04_*`.

When done: commit, push, print exactly `CLV_R04_DONE` on its own line, and stop.
