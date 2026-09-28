# Napkin brief 07 — half-time re-pricing: does the first half tell us more than the in-play market?

The same rules as the earlier briefs apply:
- solo; commit only in `current_development/clv_napkin/` on `proto/clv-napkin`, and push;
- Julia only on mcmc-beast;
- databases are read-only, and never print credentials;
- **no MCMC**: small MLE/GLM fits only.

Use the loader/runner pair `l07_*.jl` / `r07_*.jl`. Leave the untracked `l06/r06` files and
`BRIEF_06.md` alone; they are the human's.

## The question

At half-time, does a model that combines

- the **pre-match market** view (the anchor),
- the **HT score / red cards**,
- **first-half performance** from BBC live text (shots, shots on target, big chances, pxG if derivable,
  corners), measured against what the pre-match anchor expected,

predict the **full-time result** better than the **in-play Betfair price at half-time**?

Score it on the *outcome*, with a proper scoring rule. CLV is not the headline here.

## Data (see the manager's coverage check)

- **In-play prices:** the `betfair.odds_history` MATCH_ODDS archive has ticks during matches. About
  950, 490, 864 and 716 matches in 54, 55, 56 and 57 have ticks in the [KO + 45, KO + 70] min window.
  - Also check the OVER_UNDER_25 in-play coverage.
  - Define the **HT market price** robustly: the de-vigged last-traded prices during the half-time
    break. KO + 45 + first-half added time (`sofascore.matches.injury_time1`) through KO + 60 is a
    reasonable window; take the TWA or median of ticks inside it, and state the choice.
  - The archive is LTP only.
- **The 26/27 order book** (`betfair_live.order_book_1m`; ×10⁴ units, bid = back) has in-play
  back/lay prices with sizes. Use it for the tradability step.
- **HT state:** `sofascore.matches` (`home_score_ht`, `away_score_ht`, `injury_time1`,
  `start_timestamp`). Red cards come from BBC live text or sofascore incidents
  (`sofascore.match_incidents`).
- **First-half performance:** `bbc.live_text`, with event rows by `time`, `event_type` and `team`.
  - Reconstruct first-half counts per team.
  - Check `sofascore.match_statistics` and `match_shotmap`: they may have per-period stats or
    shot-level xG, which would be better than live text. Use the best source available and report
    the coverage of each.
- **Pre-match anchor:** the de-vigged 1X2 + O/U 2.5 near the close, inverted to (λh, λa) with the
  §14 method. Reuse `l05`.

## Model (MLE; keep it small)

- For the second-half goals of each team:
  `g2 ~ Poisson(λ_pre · s₂ · exp(β'·perf_1H + γ'·state))`, where:
  - `perf_1H` is the first-half stats relative to the anchor's expectation. Examples: shots for
    minus expected shots; the pxG gap.
  - `state` covers the score difference (trailing/leading, with a cap), red cards and the total
    goals so far.
  - `s₂` is the second-half share of the rate.

  Add a DC or bivariate adjustment only if cheap.
- **Baselines:**
  - (0) the HT in-play market;
  - (1) the anchor plus score state only (no performance stats), to show what the stats add;
  - (2) the full model;
  - (3) optionally, a blend of the model and the HT market.

## Evaluation

- **Split:** fit on 21/22–23/24, test on 24/25–25/26. Report per league and pooled.
- **Metric:** the full-time 1X2 log-loss *given HT* for (0)–(3), plus O/U 2.5 (full time) if the
  in-play O/U coverage allows. Use a paired, match-day-clustered bootstrap on the differences.
- **Split by game state** (level / one-goal lead / two or more), and by "first-half dominance
  without a goal lead", where the edge would plausibly live.
- **Tradability** (descriptive only):
  - On the test seasons, apply the rule: back or lay the full-time 1X2 where the model beats the HT
    market by more than 2 pp. Report EV and ROI against the HT LTP.
  - On 26/27, use the order book's actual **back and lay** prices and sizes during the HT break.
    Report the spread and size in-play at HT, and how much of the edge (if any) survives.
- **Hedging view:** for a pre-match back bet on a team, compare three HT decisions — hold, close
  (lay at the HT lay price), or add — under the model. Report whether "close when the model says
  the position is worth less than the lay price" beats "always hold", on the test seasons.
- **Power:** state the smallest log-loss difference the test sample can detect.

## Output

Append **§16 "Half-time re-pricing"** to `REPORT.md`. Include:
- a one-paragraph answer:
  - does the first half tell us more than the HT market?
  - where (game state, league)?
  - does it survive the in-play spread?
  - does model-guided closing beat holding?
- coverage tables, results tables and caveats.

Put CSVs in `out/r07_*`. Commit, push, print exactly `CLV_R07_DONE` on its own line, and stop. If
blocked, print `CLV_R07_BLOCKED` with the reason.
