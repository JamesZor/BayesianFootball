# Napkin brief 03 — lead time on 54/55, and Betfair price movement as a signal

This follows the same rules as `BRIEF.md`:
- solo Claude CLI, no subagents;
- commit only in `current_development/clv_napkin/`, and push `proto/clv-napkin`;
- Julia only on mcmc-beast in tmux `claude_clv_napkin`, with `-t 8` and `nice` if another Julia job is busy;
- **no sampling or new fits**;
- the databases are read-only, and never print credentials.

The convention is a loader/runner pair: `l03_*.jl` and `r03_*.jl`. The context is REPORT §7 (the
54/55 monitor at T−25) and §10 (the lead-time curve, which covered 56/57 only).

## Part A: lead-time curve on the 54/55 monitor (about an hour)

**The question:** does the early-entry 1X2 edge seen in League 1/2 (EV@close about +2% at the
earliest book, crossing 0 around T−45m to T−25m) also exist in the more liquid **Premiership (54)**
and **Championship (55)**? If yes, it is a candidate we can bet at size. If no, the 56/57 early
edge is probably a thin-market artefact.

- Reuse `l02`/`r02` unchanged, applied to the held-out monitor latents of the pooled fits:
  - `td_spfl_cups_joint`, `grw_spfl_joint` (W1);
  - `td_base`, `grw_step_a2_carry_jump` (W2).

  Reuse whatever r01 already priced for §7 (`out/table_monitor.csv` inputs). Do not re-price if
  the probabilities are saved.
- Keep the same instants, staleness gate, close definition, e = 2, 1X2, 2% commission and
  slate bootstrap. Add a floor/null row (back everything).
- Report per league (54 and 55 separately):
  - coverage per instant;
  - clv_pp, EV@close [CI] and β given placebo per instant;
  - a direct side-by-side with the 56/57 §10 numbers;
  - tick count per fixture per instant, as a liquidity proxy, so the three tiers can be compared.
- Note that the Championship archive covers only 22/23–24/25 (TODO 033).

## Part B: EDA on price movement as a signal (the main new work)

The earlier finding: a **market-only placebo predicts the entry→close move 3–5× better than our
models** (REPORT §1). Is the price path itself exploitable, alone or combined with the model?

**Data:** `betfair.odds_history` LTP series for MATCH_ODDS (and OVER_UNDER_25 if cheap), all
Scottish leagues 54–57 with archive coverage. The earlier findings to take into account:
- there is no volume or spread field;
- the tick count is the only liquidity proxy;
- early ticks are stale.

**B1: describe the price path.** For each league, show the typical path of the de-vigged
probability from the earliest book to the off:
- the absolute move per lead-time bucket;
- the share of the total move (earliest→close) that has happened by T−24h, T−6h, T−60m and T−25m;
- tick frequency over time;
- differences between favourites and longshots, and between home, draw and away.

A few ASCII charts are welcome.

**B2: momentum vs reversion.**
- Does the move over an early window predict the move over a later window to the close? Examples:
  earliest→T−6h predicting T−6h→close; T−3h→T−60m predicting T−60m→close.
- Give the sign and size of the regression slope, with slate-clustered CIs, per league.
- **Separate genuine drift from LTP bounce.** LTP flips between the back and lay sides, so a
  "reversion" can be pure bid/ask bounce that cannot be traded (you pay the spread). Test it:
  - use multi-tick smoothed prices (e.g. a TWA over 30–60 min) rather than a single LTP;
  - drop the last tick before the window boundary;
  - report how much of the effect survives.

**B3: is the close efficient with respect to movement?** Given the close, does the earlier
movement predict **outcomes** (a result-level log-loss or Brier gain over the close)? If not, the
movement is only a *timing* signal (take the price before it moves), not information the close
misses.

**B4: combine with the model.** Among the model's edge ≥ 2 bets at an early instant, does the
recent price direction sort CLV and EV@close? Compare two groups:
- model edge **and** the price moving towards the model;
- model edge with the price moving against the model.

Is "the model agrees with the steam" better than either signal alone?
- Use m12_td/m05_joint_td on the 710 panel (56/57);
- use one pooled fit for 54/55.

**B5: a market-only rule.** State the best simple market-only rule in words, e.g. "back when the
T−6h→T−60m drift exceeds X". Then give its EV@close and ROI, with CIs.

## Honesty rules for Part B (important: this is a data-snooping minefield)

- **Split by time.**
  - **Discovery:** seasons up to and including 23/24. Explore freely there.
  - **Confirmation:** 24/25–25/26, used once. Freeze every threshold, window and rule from
    discovery before touching confirmation.
  - Report both numbers, and say plainly which results only exist in discovery.
- **Budget:** at most about 6 confirmatory hypotheses. List them in REPORT before running
  confirmation, and apply Holm across them.
- **Tradability:**
  - With LTP only, any reversion or bounce edge must be marked **"not tradeable without ladder
    data"** unless it survives the smoothing tests in B2.
  - Say what `betfair_live` ladder data (real back/lay prices and sizes) would be needed to
    confirm it.

## Output

Append to `REPORT.md`:
- **§11 "Lead time on 54/55"** (Part A);
- **§12 "Price movement as a signal"** (Part B).

Each section opens with a **one-paragraph answer**, followed by tables, then caveats. Keep it
short. CSVs go in `out/`.

Close §12 with a **"What this means for betting"** paragraph of 3–5 lines, answering three
questions:
1. Is there a candidate rule worth a pre-registered forward paper test?
2. Which league, which entry time, model or market-only?
3. What ladder data does it need?

When done: commit, push, print exactly `CLV_R03_DONE` on its own line, and stop. If blocked, print
`CLV_R03_BLOCKED` with the reason.
