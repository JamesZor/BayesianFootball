# Work package — TODO 039 Phase B3: copula/tail check on team steps, and a score-grid comparison

You are the implementer. Claude is the manager and reviews your work against §5. **Do Phase B3 only,
then stop.**

**Night mode.** The human is asleep. The manager answers `ask_user_question` prompts, and only
within this brief's scope. Prefer making an in-scope choice yourself and documenting it. Ask only
when the brief is genuinely ambiguous. Anything outside scope (`src/`, thresholds, data, other
phases): write it under "Blocked" and stop.

## 0. Rules (B2's rules, unchanged)

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-b3`, branch
  `pi/market-model-phase-b3`, already created. Work only there.
  - Commit in small logical steps.
  - You may `git push origin pi/market-model-phase-b3`, and no other branch.
  - Never force-push, merge, rebase or touch the stash.
- **Julia runs only through a persistent REPL in tmux**, never one-shot. Every file prints an
  anchored end marker (`R04_DONE`, `T04_DONE`, …).
- **Address panes only by ID.**
  - Create each session with `tmux new -d -P -F '#{pane_id}' -s <name> …`.
  - Use only the returned `%N`.
  - Never address a pane by session name or `session:window`.
- **Laptop:** your own new session `pi_julia_mm_b3`. Never touch any other session, in particular
  `pi_julia_mm_b`, `agents`, `0` and `library` (the human's) or any `claude_*` session (the
  manager's).
- **mcmc-beast: every sampling run.**
  - Your own checkout:
    `git -C /root/BayesianFootball worktree add --detach /root/BF_runs/market_model_b3 origin/pi/market-model-phase-b3`.
  - In it, link `.env` and `Manifest.toml` from `/root/BayesianFootball/`.
  - Copy the pinned cache from `/root/BF_runs/market_model_b/.cache/` and check its SHA256 is
    `c786e2fc…b423b4`.
  - Your own session `pi_mm_b3` (pane ID only):
    `JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16`, then
    `BLAS.set_num_threads(1)`.
  - Logs go in `/root/BF_runs/logs/market_model_b3/`.
  - If the load average exceeds 20, print `PHASEB3_BLOCKED` with the numbers.
  - Don't touch other sessions.
- **Data, packages, scope:** pinned cache with `max_age_hours = 10^6`. No new or updated packages.
  No `src/` changes, no database writes. Everything lives in `current_development/market_model/`
  and is driven by `MarketModelConfig`.
- **Context handover:** if your context passes ~75%, commit, write
  `current_development/market_model/results/B3/HANDOVER.md` (state, what's done, next steps,
  open issues), print `PHASEB3_HANDOVER` on its own line, and stop. A fresh session will continue
  from it.
- **Honesty:** if a number doesn't reproduce or a gate fails, say so with the numbers. Don't loosen
  thresholds or pick subsets.

## 1. Read first

1. `current_development/market_model/README.md` (the process) and `DESIGN.md` §5 (B2, Phase C
   revisions) and §8.
2. `results/B2/PHASE_B2_REPORT.md`, especially R6, `rotation_quality_style.csv` and the addendum.
3. `results/A/PHASE_A_REPORT.md`: the draw/BTTS misfit and the held-out-line test.
4. TODO 039's Verification section: the 1X2-only totals bias (log-total −0.103, from 517 books
   re-inverted on their 1X2 alone).
5. Loaders `l01`/`l02`/`l03` and their configs; `src/features/market_inverse_utils.jl` (grid
   builders, `dixon_coles_tau`); `src/MyDistributions` (`FrankCopulaNegBin`, for reference only).
6. Literature in the human's library, via the `research-library` skill tools:
   - `mchale2011-modelling-dependence-goals-scored-opposing` (Frank copula, negative
     dependence, pp. 9–13);
   - `genest2007-primer-copulas-count-data` (discrete margins: copulas aren't unique, p. 1);
   - `ruppert2010-copulas` (t-copula tail dependence p. 4; Kendall's τ pp. 9–10).

   Cite doc_id and page for every use.

## 2. What to build

New loader `l04_copula_grid.jl`, runner `r04_copula_grid.jl`, tests `t04_copula_grid_tests.jl`.
Outputs go to `results/B3/`; figures are `results/figures/B3_*.png`.

### 2.1 (a) Is the Gaussian dependence of team steps adequate?

- Refit B2's **R6** (10a and 10b, same priors and budgets) at the B3 SHA on the beast, and draw
  FFBS paths. Rotate the steps to quality Δq = (Δα − Δβ)/2 and style Δs = (Δα + Δβ)/2.
- Use only steps **in active weeks**: weeks with an observation of that team, or between its
  matches within a season. Say exactly which.
- **Posterior predictive check:** FFBS paths are shrunk toward the Gaussian prior where data are
  thin, so comparing them with Gaussian theory is biased. Instead, for each of ≥ 200 posterior
  draws:
  1. compute each statistic on that draw's FFBS steps;
  2. simulate a replicate panel from the fitted R6 model at that draw: same fixtures, same
     weeks, Gaussian steps and noise;
  3. refilter the replicate, draw its FFBS steps, and compute the same statistic.

  Report the posterior predictive p-value P(T_rep ≥ T_obs) and both distributions.
- **Statistics T**, on (Δq, Δs) and on (Δα, Δβ):
  1. excess kurtosis of Δq and of Δs;
  2. Kendall's τ of (Δα, Δβ), against the Gaussian-copula value (2/π)·arcsin ρ;
  3. joint 95% exceedances: the frequency with which a large improvement in attack coincides
     with a large fall in concessions, and the same for collapses, split by direction for
     asymmetry;
  4. lag-1 autocorrelation of Δq² (volatility clustering).
- Also list the 15 largest |Δq| (team, week, date, size) as a table for the human to check
  against manager changes. **Don't** fetch outside data.
- Outputs: `tail_ppc.csv`, `largest_quality_steps.csv`, `B3_tail_ppc.png`.
- Only if a PPC p-value is extreme (< 0.01 or > 0.99): fit **one** alternative, a Student-t
  step on Δq alone, via the existing Student-t scale-mixture machinery adapted to the state
  equation. Gate it like every other rung. If no p-value is extreme, fit nothing and say so.

### 2.2 (b) Which score grid reproduces the book?

Four grids, each with **one global dependence parameter** shared by all matches:

| Grid | P(x, y) | Rates reported as | Reduces to double Poisson when |
|---|---|---|---|
| G0 double Poisson | Pois(x; λ_h)·Pois(y; λ_a) | λ_h, λ_a | (baseline) |
| G1 Dixon–Coles | G0 × τ(x, y; λ_h, λ_a, ρ), renormalised; `dixon_coles_tau` | λ_h, λ_a | ρ = 0 |
| G2 bivariate Poisson | Σ_k Pois(x−k; λ₁)·Pois(y−k; λ₂)·Pois(k; λ₃), λ_h = λ₁+λ₃, λ_a = λ₂+λ₃ (expected goals) | λ_h, λ_a | λ₃ = 0 |
| G3 Frank copula, Poisson margins | C(F_h(x), F_a(y)) − C(F_h(x−1), F_a(y)) − C(F_h(x), F_a(y−1)) + C(F_h(x−1), F_a(y−1)), with Frank C(u, v; κ) | λ_h, λ_a | κ → 0 |

- Write G1–G3 **locally** in `l04` (no `src/` change), on the same 11×11 support as Phase A
  (`max_goals = 10`), renormalised.
- For G3, handle κ near 0 with the independence limit.
- **Estimating the global parameter:** profile it. For each candidate value, invert every Phase A
  accepted book (same lines, KL loss, multistart, same gates) and sum the KL. Then refine
  around the minimum with Brent or golden section. Report the profile curve.
- Per grid, report:
  1. **In-sample fit:** total and per-book KL; per-line residuals q − p with fixture-bootstrap
     CIs, as Phase A's `line_residuals.csv`. The draw and BTTS lines are the ones to watch.
  2. **Held-out-line test** (Phase A's design): invert on 1X2 + O/U 2.5, predict O/U
     0.5/1.5/3.5 and BTTS. Report the mean signed and absolute error, with CIs.
  3. **The 1X2-only totals bias:** re-invert the full books (≥ 5 selections) from their 1X2
     alone, and report the mean Δ log(λ_h + λ_a) against the full-book inversion, with
     q05/q50/q95 and a fixture-bootstrap CI. The baseline G0 value is −0.103.
  4. **Outcome log loss** on the 517 Phase B fixtures: the grid-implied 1X2 and O/U 2.5 from that
     grid's full-book rates, against results, paired with G0 and with the de-vigged close
     (fixture-bootstrap CIs). This uses Phase B's goal columns.
  5. **Rate shift:** how far each grid moves log λ_h, log λ_a, supremacy and level relative to
     G0, per fixture (quantiles).
- **R6 refit on the best grid's rates (informational):**
  - choose the grid with the lowest held-out mean absolute error;
  - refit B2's R6 on its full-book rates (the 517 set), 10a and 10b;
  - report the B2 measures side by side with R6 on G0 rates.

  If that grid's 1X2-only bias is |bias| < 0.03, also run R6 with the 78 1X2-only books
  included, as a second informational row.
- Outputs: `grid_profile.csv`, `grid_fit.csv`, `grid_line_residuals.csv`, `grid_heldout.csv`,
  `grid_1x2only_bias.csv`, `grid_outcome_logloss.csv`, `grid_rate_shift.csv`,
  `r6_on_best_grid.csv`, `B3_grid_residuals.png`, `B3_grid_profile.png`.

**The human's decision rule (already fixed):** Phase C keeps double Poisson whatever B3 finds. B3
is evidence for a later Phase C2 that the human will decide on. Don't recommend changing Phase C.

## 3. Tests (`t04`, laptop)

1. Each grid sums to 1. Its marginal means equal the reported (λ_h, λ_a) to 1e-6 on the
   truncated support (or document the truncation error).
2. The reductions: G1 at ρ = 0, G2 at λ₃ = 0 and G3 at κ = 10⁻⁸ equal G0 to ≤ 1e-10.
3. Synthetic recovery per grid: exact books from known (λ_h, λ_a, parameter) invert back to
   1e-5.
4. G3 matches a brute-force Frank-copula CDF difference on 20 random (λ_h, λ_a, κ), including
   κ < 0.
5. The PPC machinery, on a simulated panel: Gaussian steps give p-values that aren't extreme;
   planted heavy-tailed Δq (t₃) gives an extreme kurtosis p-value.
6. `t02` and `t03` still pass (counts).

## 4. Out of scope

Phase C; changing Phase C's grid; any `src/` edit; English segments; new packages.

## 5. Acceptance (the manager reruns each item)

1. `t04`, `t03` and `t02` pass in a fresh laptop REPL (commands, counts, markers).
2. All grid gates pass (reductions, recovery, brute-force Frank).
3. R6 refits meet R̂ ≤ 1.05 and bulk/tail ESS ≥ 200 for every θ, in both protocols.
4. Every §2.1 and §2.2 table and figure exists; every number in the report is copied from them.
5. `r04_copula_grid.jl` runs top to bottom in a fresh beast REPL to `R04_DONE`, and a second
   fresh run gives byte-identical CSVs (`SCIENTIFIC_SHA256SUMS.txt`, `REPRODUCIBILITY.md`).
6. The README gains a B3 section: what was tested and the results with CSV paths. Its decisions
   list records the human's "double Poisson for Phase C; C2 later".
7. TODO 039 has a dated `@pi` line; `./scripts/todo.sh check` passes.

## 6. Report and stop

- Write `results/B3/PHASE_B3_REPORT.md`:
  - evidence per acceptance item;
  - your choices where §2 left room;
  - **in plain words:** is the Gaussian dependence of team steps adequate, or is there tail
    dependence, asymmetry or heavy-tailed quality jumps? Which grid reproduces the book best,
    does any remove the 1X2-only totals bias, and does any change outcome log loss? What would
    R6 look like on the best grid?
  - anything you believe is wrong in DESIGN or this brief.
- Commit and push. Close your sessions.
- Print exactly `PHASEB3_DONE` on its own line and stop, or `PHASEB3_BLOCKED` /
  `PHASEB3_HANDOVER` as above.
