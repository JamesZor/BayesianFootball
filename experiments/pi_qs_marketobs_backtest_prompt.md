# Wave 4 brief: scenario-Kelly betting backtest of qs_marketobs at T−25m, T−6h and T−24h
Role: pi implementer on exp/qs-goal-vs-market; Claude is the manager. Scope is fixed by James on 2026-10-10.
Goal: does qs_marketobs make money against executable Betfair prices after commission, versus m12 and three other models?
DONE WHEN is §4. Stop rule: stop at any failed gate, or at a phase boundary with PHASEBT_HANDOVER.
No new model fits. Backtest only. Small src change allowed only in src/Portfolio allocators (§3 Phase 2).

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources; trust it, and open a
source only for the line ranges listed.

### What we're testing

- On 40-fold Scottish League One/Two (24/25 + 25/26, 595 common 1X2 fixtures),
  **qs_marketobs** (QualityStyleGRW observing past market rates) matched the de-vigged close:
  3-way LogLoss 1.0505 vs 1.0534, not detectable. It had the best point estimate on every score,
  and its leakage audit was clean.
- Fusing in goals and learned market bias (wave 3) didn't help, and outcomes showed no
  market compression.
- **So any edge should come from timing.** The model sees only earlier weeks' closes, so it's
  close-quality *before* the market converges.
- Earlier evidence (CLV napkin, 2026-09-28):
  - EV at the close was below 0 at T−25;
  - the early 1X2 edge appeared only in thin 56/57;
  - "the edge lives inside the spread".

### Runs to backtest (all saved; do not refit)

| Model | Run |
|---|---|
| **qs_marketobs** (wave 2) | `81fdf817-f82d-4f66-9bfd-5f456c4fe29f`, namespace `scottish_lower_qs_wave2_2426` |
| m12 (live production, TimeDecay hybrid) | `132df5c2-c742-4e95-8693-3aeb2b2cbaef`, namespace `scottish_lower_joint_player_2426` |
| grw_joint (wave 2) | `98447840-e95c-420f-92b4-2db6545506a3` |
| fusion_qs_nobias (wave 3) | `5afdfeb3-abad-4d23-82fb-1bfe6babe797`, namespace `scottish_lower_qs_wave3_2426` |
| market C0 (wave 1) | files: `experiments/scotland/05_quality_style_goal_vs_market/results/market_grid_summary.csv`, beast outputs `/root/BF_runs/qs_experiment_out/` |

If C0 can't supply draw-level latents for the engine, use its saved mixture probabilities. In
that case mark its Baker–McHale and scenario rows **NOT_AVAILABLE**, with the reason.

### Prices (`src/Calibration/book.jl`)

`PointInTimeBookConfig(; as_of_minutes, max_staleness_minutes = 90.0)` (l.76–94): the last traded
price at or before the cutoff, per selection, with staleness as a column; `assert_book_as_of`
guards it.

Snapshots:

| Snapshot | `as_of_minutes` |
|---|---:|
| T−25m | −25 |
| T−6h | −360 |
| T−24h | −1440 |

- Keep the 90-minute staleness bound at every snapshot, and report coverage per snapshot.
- If T−24h coverage is below 50%, also report a sensitivity run with `max_staleness_minutes = 360`.

### The portfolio engine (`src/Portfolio`)

- `stake.jl` header l.1–17: a_kelly → trust → shrink → risk → cap → filter. The risk factor is
  homogeneous of degree 0: once the drawdown constraint binds, only λ moves exposure.
- `stake_slate` l.27; `slates.jl` (`DailySlate` l.16 / `group` l.29).
- `pricing.jl`:
  - `BookWorkspace` l.71 builds per-draw score grids from `AbstractPosteriorLatents`;
  - `build_books_reported(spec, fit::Training.Fit, odds, fixtures)` l.497.
- `implementations/`: `allocators.jl`, `shrinkage.jl` (`NoShrinkage`, `FractionalKelly`,
  `BakerMcHale`), `risk.jl`, `caps.jl`, `commission.jl`, `trust.jl`, `filters.jl`.

### The previous study (experiment 07)

`experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/` (README §1, its verdicts):
T−25, 99 daily slates, m12; λ is "everything"; per-line pruning overfits (0/24 out of sample).
**Use it to reproduce the production stack, not to re-tune it.**

### Reading rules

- Use `read` with offset/limit, or `rg -n`, for anything not listed. Don't read a whole README,
  report or design doc.
- Don't re-read a file you've already read unless it changed or a compaction dropped it.
- Print at most 40 lines of any command. Send longer output to
  `experiments/scotland/08_qs_marketobs_backtest/logs/<name>.log`, then `rg`/`tail` it.
  Use `context_get` for stored output.
- Write decisions and their reasons to `08_.../DECISIONS_LOG.md` as you go; compaction may drop
  them from your context.
- Keep `08_.../PROGRESS.md` current: phase, done, next action, open panes and job IDs.

**After a compaction** (your context starts with `[Session Goal]` and a compaction note):
1. Re-read this brief's DONE WHEN and stop rules, then `PROGRESS.md` and `DECISIONS_LOG.md`.
2. Never guess a path, command, run ID or UUID. Take it from the summary's `[Files And Changes]`,
   from `recall` (`recall "<keyword>"`; `recall mode:touched` lists files you edited), or from
   `rg --files`. Re-read only the lines you are about to edit.
3. Don't redo work the summary or `PROGRESS.md` marks done; check that its output exists.

**Context:** compaction is automatic at about 200K and keeps your last ~24K tokens verbatim.
Don't wait for it to hand over. At a natural phase boundary, commit, write `HANDOVER.md` and stop
with the handover sentinel.

## 2. Rules

- **Git:** worktree `/home/james/bet_project/.worktrees/BayesianFootball-qs-experiment`, branch
  `exp/qs-goal-vs-market`. Push only this branch; never merge, rebase or force-push.
- **TODO 043:** create it with `scripts/todo.sh` conventions: "Wave 4: scenario-Kelly backtest
  of qs_marketobs at executable prices".
- **Julia only on mcmc-beast**, in persistent tmux REPLs by pane ID (`pi_qsbt_*`), started with
  `-t 16` and BLAS 1. Logs go in `/root/BF_runs/logs/qs_backtest/`. No laptop Julia. Never touch
  other sessions.
- **Databases:** `betdb` read-only; `mcmc_experiments` read-only (`load_fit` by UUID).
- **No tuning on the test window.** Policy parameters (λ, trust, caps, commission, Kelly fraction)
  are fixed before any backtest number is seen, from production defaults or this brief. Write
  them into `DECISIONS_LOG.md` before Phase 3.
- **Honesty:** report every arm, snapshot and market, including losers. No favourable subsets.
- **Polling:** every 5–6 minutes on Codex.

## 3. Phases (stop at the end of each with PHASEBT_HANDOVER)

### Phase 0: inventory and reproduction

1. Locate the **production policy** exactly as MatchDay's default configuration builds it:
   markets, trust (`CanonicalScottishLowerTrust`), shrinkage, risk λ, caps, commission rate,
   and **whether a calibration container is applied**. Record it.
2. Reproduce experiment 07's m12 T−25 headline (final wealth and number of bets) with that stack,
   to within its rounding, before anything new. If it doesn't reproduce, stop and report.
3. Build the point-in-time books at all three snapshots. Report coverage: fixtures and
   selections per market, and staleness quantiles.

### Phase 1: the allocator check

Read `allocators.jl` and `stake_slate`. Determine whether the allocator maximises expected log
wealth **jointly over posterior draws across the whole daily slate**, i.e. scenario Kelly over
correlated same-match markets and cross-match parameter uncertainty. If not, how does it treat
draws: per match, or marginal means?

Write the answer with line references in `DECISIONS_LOG.md`.

### Phase 2: scenario-Kelly allocator (only if Phase 1 says it's missing)

Add `ScenarioKelly` in `src/Portfolio/implementations/allocators.jl`, using the existing
allocator interface:
- for a daily slate, take S joint posterior draws (the same draw index across all of that day's
  matches);
- simulate scorelines per draw;
- choose stakes maximising mean log(1 + Σ stakes × net payoff after commission);
- constraints: stakes ≥ 0, Σ ≤ 1, per-bet caps from the policy.

Use a deterministic seed; cap S at about 1,000. It should feed into the existing trust, shrink,
risk, cap and filter steps unchanged.

**Tests:**
- a single bet reduces to the closed-form Kelly fraction;
- two independent bets match a brute-force grid search;
- deterministic seed replay;
- no negative stakes;
- the commission payoff is correct.

### Phase 3: the backtest grid

The grid is 5 models × 3 snapshots × policies, all on DailySlate, from the same 710 fixtures and
40 folds' out-of-sample predictions.

| Policy | What it is |
|---|---|
| **production** | The exact production stack from Phase 0, for every model. Apply the calibration container to every model if production uses one. |
| **plain_kelly** | Fractional Kelly 0.25. Same markets (1X2, O/U 2.5, BTTS), no trust tiers, the same caps, commission and drawdown control. |
| **scenario_kelly** | Only if Phase 2 built it: scenario Kelly with the production shrink, risk, cap and filter steps. |

**Metrics, per model × snapshot × policy:**
- final wealth, mean daily log growth, ROI on turnover, number of bets, turnover;
- max drawdown, Sharpe and Calmar (as in experiment 07);
- **CLV**: the bet price against the T−0 close, mean pp and the % of bets beating the close;
- a per-market breakdown.

**Uncertainty:**
- 8-week moving-block bootstrap of daily log growth (within season, 999 reps, 90% intervals);
- paired differences against m12 on the same days.

### Phase 4: report

`08_.../README.md` and `REPORT.md`: a summary of at most 12 lines first, then tables per
snapshot.

It must answer:
1. Does qs_marketobs beat m12 after commission, and at which snapshot?
2. Is the edge larger early (T−24h, T−6h) than at T−25?
3. What do the trust tiers add (production vs plain_kelly)?
4. Does scenario Kelly help (if built)?
5. Is CLV positive?

Label in-sample risks. These are 2 seasons and about 99 slates; say how wide the intervals are.

## 4. DONE WHEN

1. Phase 0 reproduces experiment 07's m12 T−25 headline, and all three snapshot books are
   built with coverage reported.
2. Phase 1's answer is recorded with line references. Phase 2 is built and tested, or recorded
   as not needed.
3. Every model × snapshot × policy cell is computed, or marked NOT_AVAILABLE with a reason, with
   bootstrap intervals and paired differences against m12.
4. The report answers the 5 questions with numbers traceable to committed CSVs.
5. TODO 043 is closed with follow-ups, everything is pushed, the beast panes are closed, and
   `PHASEBT_DONE` is printed alone on its line.

**Sentinels**, alone on their line: `PHASEBT_HANDOVER` (phase boundary), `PHASEBT_BLOCKED`,
`PHASEBT_DONE`.
