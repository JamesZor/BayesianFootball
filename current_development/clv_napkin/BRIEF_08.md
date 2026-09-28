# Napkin brief 08 — is our score grid mis-weighted? (observed vs expected over all folds)

The same rules as the earlier briefs apply:
- solo; commit only in `current_development/clv_napkin/` on `proto/clv-napkin`, and push;
- Julia only on mcmc-beast;
- databases are read-only;
- **no MCMC**: only 1-parameter MLE fits on training folds.

Use the loader/runner pair `l08_*.jl` / `r08_*.jl`. Leave the untracked `l06/r06` files and
`BRIEF_06.md` alone.

**The human's hypothesis.** Our Poisson-based score grid is wrongly weighted across scorelines
compared with what actually happens. Good RQRs on each team's goals separately would not catch this.

## Models

Use held-out, walk-forward score grids from persisted fits only. Reuse the `out/probs.jls`
machinery; if it holds only 1X2/O-U, rebuild each fit's held-out λ/score grids the same way.

- `m12_td` `132df5c2` and `m05_joint_td` `ed541a7c` (W0);
- `grw_spfl_joint` `82e01a3b` (W1);
- `td_lower_a2full_carry_jump` and `grw_step_a2_carry_jump` (W2 co-finalists);
- the floor, `s12_m01_td_poisson` `de7fa956`.

Cover League One and Two (56/57) held-out targets on each run's panel, and do not pool panels. Add
the 54/55 monitor if cheap.

## 1. Observed vs expected scorelines

- **Per scoreline:** compute `E(i,j) = Σ_m p_m(i,j)` over all held-out matches and all folds, and
  `O(i,j)` = the actual count, for i, j ∈ 0..4 plus a 5+ bucket.
- **Per model, report:**
  - the O and E tables;
  - the standardised residuals `(O−E)/√E` as a 5×5 ASCII table;
  - an overall chi-square with pooled sparse cells, plus a parametric bootstrap p-value, since
    cells are dependent through the grid.
- **Summaries:**
  - draws: expected vs observed [CI];
  - total goals 0, 1, 2, 3, 4, 5+: expected vs observed;
  - **dispersion**: the observed variance of total goals vs the model-implied variance (mean of
    per-match variances + variance of per-match means);
  - goal difference −3..+3.
- **Side by side:** put the same residual pattern computed for the **market's** CS grid (§14:
  market-minus-Poisson) next to it, so it is visible whether our grid errs in the same direction as
  "Poisson vs market".

## 2. Keep our rates, change only the shape

- **For each model, keep each match's (λh, λa) fixed.** Fit ONE shape parameter by maximising the
  exact-score log-likelihood on **training folds only**. Train on 23/24 held-out predictions where
  they exist, test on 24/25–25/26; if a panel has no earlier season, use a two-way season split and
  say so. The shapes are:
  - (a) Dixon–Coles ρ;
  - (b) bivariate Poisson λ₃, keeping the marginal means equal to (λh, λa), i.e. U = λh − λ₃;
  - (c) COM-Poisson ν on both marginals, holding the means;
  - (d) the §14 best combination (bivariate + COM), as 2 parameters.
- **Held-out scores:**
  - exact-score log-loss;
  - 1X2 log-loss;
  - O/U 2.5 log-loss;
  - the draw rate.

  Each is compared with the unmodified Poisson grid, using paired, match-day-clustered bootstrap
  CIs.
- **Also show the fitted parameter per model** next to the market's §14 values (ρ ≈ −0.04,
  λ₃ ≈ 0.08–0.09, ν ≈ 1.08).

## 3. Does it matter for betting?

- For the best reshaped grid, re-run the T−25 and T−60 edge ≥ 2 1X2 and O/U 2.5 bet selection
  (§3/§10 method, LTP) and the §13 26/27 order-book view (back price).
- Report how many bets change, and the change in CLV and EV@close. This is descriptive only.

## Output

Append **§17 "Is our score grid mis-weighted?"** to `REPORT.md`. Include:
- a one-paragraph answer:
  - which scorelines are over- or under-weighted, and by how much;
  - does it match the market's pattern?
  - which shape fixes it, and by how much out of sample?
  - does it change the betting picture?
- the tables;
- caveats.

Put CSVs in `out/r08_*`. Commit, push, print exactly `CLV_R08_DONE` on its own line, and stop.
