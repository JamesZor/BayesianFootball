# Napkin brief 05 — learn from the market: how it rates teams, and what its score grid looks like

The same rules as the earlier briefs apply:
- solo, no subagents;
- commit only in `current_development/clv_napkin/` and push `proto/clv-napkin`;
- Julia only on the beast;
- databases are read-only, and never print credentials;
- **no sampling or MCMC fits.** Small MLE/least-squares fits of a few parameters on price data are
  fine.

Use the loader/runner pair `l05_*.jl` / `r05_*.jl`.

**The purpose.** The human wants to *learn how the market prices matches*, so as to improve the
model's structure. This is not a betting test. It has two parts.

**Data:**
- the `betfair.odds_history` archive: `MATCH_ODDS`, `OVER_UNDER_*` and `CORRECT_SCORE`, de-vigged
  near the close, e.g. the (−20, 0] TWA as in §2;
- the `betfair_live.order_book_1m` 26/27 book (prices and volumes ×10⁴, bid = back; see §13.1).

The primary scope is Scottish 54–57 across all archive seasons. If it is cheap, repeat Part B on one
big liquid league as a reference.

## Part A: the market's team ratings, and what explains them

1. **Invert each match's close into (λ_home, λ_away)** using 1X2 plus O/U 2.5 (plus other O/U lines
   if present). Use the grid model chosen in Part B, or independent Poisson first, and say which.
2. **Fit market-implied ratings**
   `log λ_home = μ + home + att_h − def_a`, `log λ_away = μ + att_a − def_h`
   per team, with a slow time variation, e.g. a rolling window or a random walk by match day. This
   gives a market rating path per team.
3. **Compare with our model.** Take `m12_td` and `grw_spfl_joint` held-out latents (from
   `out/probs.jls`, and the §13 extension for 26/27) and set their λ against the market λ. Where do
   they disagree most, by team, by season stage (first 5 matches vs later), and by
   promoted/relegated status?
4. **What explains the market's ratings?** Regress the market ratings at season start, and the
   first-weeks change, on what we have in betdb:
   - the wealth feature(s) used by `m05_joint_production_wealth_grw` (find them in `src/`);
   - previous-season tier and final position;
   - full-time/part-time status, if available;
   - sofascore player or lineup data (e.g. player ratings, market values, squad turnover), if it
     exists;
   - last season's goal difference.

   Report which features carry weight and how much variance they explain. Then report **how fast
   the market's rating of a club that changed tier converges**: the first match's price vs its
   settled level. Show 26/27's four transition clubs (Ross County included) as worked examples next
   to our model's λ.

## Part B: the market's score grid vs Poisson

1. **Take the de-vigged `CORRECT_SCORE` close** (all score runners plus "Any Other"), with the
   1X2/O-U close of the same match.
2. **Compare candidate grids.** For each match, fit these to the market's 1X2 + O/U, then compare
   the implied correct-score grid with the market's CS grid:
   - (a) independent Poisson;
   - (b) Dixon–Coles, with ρ estimated globally and per league;
   - (c) bivariate Poisson (λ₃);
   - (d) negative binomial or another over-dispersion form;
   - (e) a simple diagonal/draw inflation.
3. **Report:**
   - which grid matches the market best (mean KL or log-loss of the market's CS grid);
   - the fitted ρ / λ₃ / dispersion values with CIs;
   - the systematic residual pattern (market minus Poisson) on 0-0, 1-0, 0-1, 1-1, 2-2 and the draw
     total;
   - whether that pattern differs by league (54–57) or by expected total goals.

   An ASCII heatmap of the mean residual grid would help.
4. **Then check our model's grid.** Does `m12_td` (and the joint model) show the same residual
   pattern vs the market's CS, or a different one? That tells us whether swapping our observation
   model for the best grid would close part of the model–market gap.
5. **Market consistency (short).** Are thinner markets (CS, BTTS, O/U 1.5/3.5) priced consistently
   with the grid implied by 1X2 + O/U 2.5? Report the size of typical deviations. This is
   descriptive only; a betting test would come later.

## Output

Append **§14 "How the market prices matches"** to `REPORT.md`. Open with a one-paragraph answer
covering:
- what the market's ratings track;
- how fast it adapts to promoted/relegated clubs;
- which score grid it uses;
- what that means for our model (a concrete recommendation, e.g. "use DC with ρ ≈ …", "add feature
  X", "seed transition clubs from …").

Then tables and caveats. CSVs go in `out/r05_*`.

When done: commit, push, print exactly `CLV_R05_DONE` on its own line, and stop.
