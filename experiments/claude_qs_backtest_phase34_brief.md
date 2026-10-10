# Wave 4, Phases 3–4 (Claude CLI): run and report the qs_marketobs betting backtest
Role: Claude Code implementer on exp/qs-goal-vs-market; the launching Claude session is the manager. James set the scope.
Goal: the backtest grid and the report. Phases 0–2 are done by pi (Codex fault plus weekly quota, so handed over).
DONE WHEN is §4. Stop at the end of each phase with QBT_HANDOVER; stop at any failed gate with QBT_BLOCKED.
No new model fits, no betdb or mcmc_experiments writes. src is frozen (ScenarioKelly is already in).

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources; trust it, and open a
source only for the line ranges listed.

### State: everything is in `experiments/scotland/08_qs_marketobs_backtest/`

Read `HANDOVER.md`, `PROGRESS.md` and `DECISIONS_LOG.md` first, by section. Phases 0–2 passed:

| Phase | Result | Commit |
|---|---|---|
| 0 | Experiment 07's m12 T−25 result reproduces exactly (wealth 2.515204600600047, 1127 bets, 99 slates) using experiment 07's own stack (`HISTORICAL_REPRODUCTION_RECIPE.md`) | `72cd5496` / `e4b95a19` |
| 1 | The stock allocator works per match and discards posterior draws; there is no joint daily optimisation (`PHASE1_ALLOCATOR_AUDIT.md`) | |
| 2 | `ScenarioKelly` (`src/Portfolio/implementations/allocators.jl`, commit `13677357`) plus the opt-in daily adapter `l01_scenario_kelly.jl`. 51/51 checks, 92/92 portfolio regression, bit-identical legacy allocations (`PHASE2_SCENARIO_KELLY.md`) | `72548977` |

### Coverage (90-minute staleness, of 710 fixtures)

| Snapshot | Fixtures | Complete 1X2 |
|---|---:|---:|
| T−25m | 611 | 545 |
| T−6h | 295 | 182 |
| T−24h | 53 | 47 |
| T−24h at 360-minute staleness | 190 | 170 |

### Manager rulings, all binding (in `DECISIONS_LOG.md`)

1. **Read-only.** A static read-path check plus a before/after audit of the fit, config and harness
   tables around every runner; stop on any change. No PGOPTIONS.
2. **Two named stacks, both reported:**
   - **canonical:** r07_serve_console (canonical markets, CanonicalScottishLowerTrust, λ 23,
     BakerMcHale, PerBetCommission 0.02, min 0.0001; measured on m12: 2.7735 / 1132);
   - **option_b_live:** FractionalKelly 0.30, min 0.001, trust 1 and 1/1.4 including Over 1.5, λ 28,
     on raw probabilities.

   A fitted calibration container per model is out of scope. **plain_kelly** at 0.25 is kept.
3. **historical_exp07:** the exact experiment 07 stack is the reproduction reference. Also report it
   as a policy.
4. **T−6h and T−24h are low-coverage subsets.** Never pool them with T−25m. Report T−24h at both
   90 and 360 minutes, labelled.

### Models (saved; never refit)

| Model | Run |
|---|---|
| **qs_marketobs** | `81fdf817-f82d-4f66-9bfd-5f456c4fe29f` (ns `scottish_lower_qs_wave2_2426`) |
| m12 | `132df5c2-c742-4e95-8693-3aeb2b2cbaef` (ns `scottish_lower_joint_player_2426`) |
| grw_joint | `98447840-e95c-420f-92b4-2db6545506a3` (wave 2 ns) |
| fusion_qs_nobias | `5afdfeb3-abad-4d23-82fb-1bfe6babe797` (ns `scottish_lower_qs_wave3_2426`) |
| market C0 | files in `05_quality_style_goal_vs_market/results/` and beast `/root/BF_runs/qs_experiment_out/`; if no draw-level latents, use the mixture probabilities and mark its BakerMcHale and scenario rows NOT_AVAILABLE |

### Reading rules

- Use `Read` with offset/limit, or `rg -n`. Never read whole long files.
- Keep command output short; send long output to `logs/` and grep it.
- Keep `PROGRESS.md` and `DECISIONS_LOG.md` current. **After a compaction, re-read both before
  acting.**

## 2. Rules

- **Git:** commit small steps and push `exp/qs-goal-vs-market` only; never merge, rebase or
  force-push. Update TODO 043 (pi created it): it's no longer BLOCKED, and its log should record
  the handover.
- **Julia only on mcmc-beast**, in persistent tmux REPLs that are yours only (`claude_qsbt_*`), by
  pane ID, started with `-t 16` and BLAS 1. Logs go in `/root/BF_runs/logs/qs_backtest/`. No laptop
  Julia.
- **Never touch other tmux panes.** In particular `%109` is pi and `%110` is James's own Claude
  session.
- **Waiting on long beast jobs:** use one background command that blocks until an end marker or
  `ERROR:` appears, with `run_in_background` and an explicit long timeout. No polling loops.
- **No tuning on the test window.** All policy parameters are fixed by the rulings above.
- **Honesty:** report every arm, snapshot and market, losers included.

## 3. Phases

### Phase 3: the backtest grid

All on DailySlate, from the 710 fixtures and the 40 folds' out-of-sample predictions.

- **Models:** the 5 above.
- **Snapshots:** T−25m, T−6h, T−24h (90 min), T−24h (360 min).
- **Policies:** historical_exp07, canonical, option_b_live, plain_kelly, and **scenario_kelly**
  (ScenarioKelly through the daily adapter, then the canonical trust, shrink, risk, cap and filter
  steps).

**Gate first:** the m12 × T−25m × historical_exp07 cell must reproduce 2.515204600600047 / 1127
again.

**Metrics, per cell:**
- final wealth, mean daily log growth, ROI on turnover, number of bets, turnover;
- max drawdown, Sharpe and Calmar (as in experiment 07);
- **CLV**: the bet price against the T−0 close, mean pp and the % of bets beating the close;
- a per-market breakdown.

**Uncertainty:** 8-week moving-block bootstrap of daily log growth (within season, 999 reps, 90%),
and paired differences against m12 on the same days and policy.

### Phase 4: report

`README.md` and `REPORT.md`: a summary of at most 12 lines first, then tables per snapshot.

It must answer:
1. Does qs_marketobs beat m12 after commission, and at which snapshot?
2. Is the edge larger at T−6h or T−24h than at T−25 (with the low-coverage caveat)?
3. What do the trust tiers add (canonical against plain_kelly)?
4. Does scenario Kelly help against per-match allocation?
5. Is CLV positive?

Label in-sample risks: 2 seasons and about 99 slates; policies were partly chosen on these seasons
in experiment 07. Say how wide the intervals are.

## 4. DONE WHEN

1. The Phase 3 gate reproduces, and every cell is computed or marked NOT_AVAILABLE with a reason,
   with bootstrap intervals and paired m12 differences, in committed CSVs.
2. The report answers the 5 questions with traceable numbers.
3. The DB audit is unchanged throughout.
4. TODO 043 is closed with follow-ups, everything is pushed, your beast panes are closed, and
   `QBT_DONE` is printed alone on its line.
