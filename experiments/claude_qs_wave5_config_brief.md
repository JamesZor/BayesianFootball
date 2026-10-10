# Wave 5 (Claude CLI): fix the qs_marketobs + scenario Kelly paper-trading config for 26/27
Role: Claude Code implementer on exp/qs-goal-vs-market; the launching Claude session is the manager. James set the scope.
Goal: pick the basket and λ on 24/25 only, freeze them, then score the frozen config on 25/26 against m12.
DONE WHEN is §5. Stop at the end of each phase with W5_HANDOVER; stop at any failed gate with W5_BLOCKED.
No model fits, no DB writes, src frozen. The 25/26 half stays sealed until FREEZE.md is pushed and the manager says continue.

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources; trust it, and open a
source only for the line ranges listed.

### Prior state (wave 4, TODO 043, `experiments/scotland/08_qs_marketobs_backtest/`)
- T−25m (90 min staleness), 99 slates (49 in 24/25, 50 in 25/26), 2% commission. Exact cells
  from `results/phase3/phase3_cells.csv`:

  | Model × policy | Wealth | Bets | MDD % (full) |
  |---|---|---:|---:|
  | m12 × canonical | 2.773493294562897 | 1132 | −16.15 |
  | qs_marketobs × scenario_kelly | 2.154029744590488 | 952 | −10.47 |
  | qs_marketobs × canonical | 1.8574969585782386 | 984 | −12.58 |
  | m12 × canonical, T−6h | 1.6540763835245025 | 351 | −6.82 |
  | qs_marketobs × scenario_kelly, T−6h | 1.2149816047771074 | 328 | −10.19 |

- **Per-season MDD**, with the bankroll reset to 1 at each season start (manager's duckdb over
  `phase3_daily.csv`): m12 canonical 24/25 **−13.60%**, 25/26 −16.15%; qs SK 24/25 −7.81%,
  25/26 −10.47%. So at λ 23, qs SK has room to run more aggressively.
- Canonical policy: `src/MatchDay/slate.jl:74-105` (`canonical_markets()` = 1X2, BTTS,
  OU0.5–3.5; `canonical_scottish_lower_policy()` = `CanonicalScottishLowerTrust`, `SlateDrawdown(23.0)`,
  `FixedCap(0.25)`). Trust table: `src/Portfolio/implementations/trust.jl:87-131` (`TieredTrust`:
  Home and U2.5 at 0.35, Draw and Away at 0.25, default 0).
- **How wave 4's scenario Kelly solved** (`08_.../l02_phase3_grid.jl:232-306`, `l01_scenario_kelly.jl:87-150`):
  it built the canonical books (every canonical market leg), ran ONE joint daily solve over all
  those legs, then the canonical trust, shrink, λ and cap overlay ran unchanged. Trust 0 zeroes
  BTTS/Over/other-line legs **after** the solve. The joint raw stake averaged 0.74 per day, and
  the 0.99 budget bound on 52% of T−25m days, so excluded legs used up budget.
- Wave 4 runner: `l02_phase3_grid.jl` (systems `:51-71`, scenario_slates `:232-261`, run_group
  `:263-306`, bootstrap `:308-`), entry `r02_phase3_grid.jl`. Beast checkout
  `/root/BF_runs/qs_backtest_phase0` (detached at `4f1d3d43`; make your own, see §2).
- `src/Portfolio/calibrate.jl:19-41` `calibrate_lambda` bisects λ to a **mean exposure**, not an
  MDD. Wave 5 needs an MDD search (ruling 3); you may reuse its `_with_lambda` / `_relambda` helpers.
- The method you follow: `experiments/scottish_lower/MARKET_LINE_EDA_REPORT.md` §0 (3 corrections),
  §2 (per-line verdict table), §2.1, §4 and §5.1 only. In that report the per-line rule chose
  OU1.5, which reversed out of sample. The A/B re-simulation decides, not the per-line rule.

### Manager rulings (binding; copy into `DECISIONS_LOG.md` at Phase 0)
1. **Halves.** Selection = slates dated ≤ 2025-05-03 (24/25). Evaluation = slates after it (25/26).
   Each half starts at bankroll 1.0; MDD is measured within the half.
2. **Comparison cells (James):** the candidate **qs_marketobs + scenario Kelly at its MDD-matched λ**,
   and the benchmark **m12 × canonical × λ 23** (unchanged: per-match allocator, canonical basket).
   You also compute the **fallback**: qs SK, canonical basket, λ 23, with legs restricted as in
   ruling 4. It is needed by the decision rule.
3. **λ target = m12 canonical's 24/25 MDD at T−25m** (recompute; expect −13.60%). Search grid
   λ ∈ {4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 30}. Choose the **smallest λ such that the 24/25 MDD
   ≤ target at that λ AND at every larger grid λ**: a monotone envelope, so one lucky dip can't
   choose it. Report the full λ curve: MDD, g, mean exposure, budget-binding share, cap-binding share.
   If the budget (0.99) or the 0.25 slate cap binds on most days at the chosen λ, say so: then
   λ is no longer the active dial.
4. **A basket is the set of directions eligible for the joint solve.** Filter each book's legs to
   trust > 0 **before** the scenario solve, not after. Directions outside the canonical trust table
   get trust **0.25** (the lower canonical tier). The canonical tiers are otherwise fixed: wave 4
   found the tiers add nothing, and λ is the dial. Sensitivity, on 24/25 only: the canonical basket
   at λ 23 solved wave-4 style (all legs, then trust) vs basket-only legs.
5. **Candidate baskets (James):** B1 canonical (H/D/A + U2.5); B2 1X2 only; B3 1X2 + OU2.5 both
   sides; B4 a data-driven basket from the 24/25 per-direction table (BTTS-no, OU1.5, OU3.5 and
   others are candidates). **James asked you to "play around"**: on 24/25 you may try any other
   baskets or directions. Log every basket you try in `DECISIONS_LOG.md` with its 24/25 g and MDD;
   the report counts them as the number of looks.
6. **Selection rule, written down before any basket is scored:** the basket with the highest 24/25
   mean daily log growth at its own MDD-matched λ. If a smaller basket comes within
   0.5 ×10⁻³/day of it, take the smaller one.
7. **Seal.** No wave-5 code path may compute any 25/26 slate before FREEZE.md is committed,
   pushed and released by the manager. Phase 1 code filters by date before simulating, and asserts
   it. The wave-4 full-period numbers are already public. So 25/26 is unseen *by the selection
   rule*, not unseen by anyone; say so in the report.
8. **Decision rule (James):** the frozen config stands unless its 25/26 g ≤ 0 (point estimate) or its
   25/26 MDD is worse than 1.5 × target. In either case, paper-trade the fallback instead. d vs m12 is
   reported, but descriptively: about 50 slates can't rank configs, and 26/27 is the real test.
9. **T−6h (James):** apply the T−25m-frozen config and m12 canonical at T−6h on both halves.
   Report only; no separate calibration.
10. **Uncertainty:** as in wave 4 (8-week noncircular blocks within season, 999 reps, 90%,
    seed 20261009). Add the circular-block column (wave 4 follow-up 2). Give paired d vs m12 on
    the same days.

### Reading rules
- Use `Read` with offset/limit, or `rg -n`. Never read whole long files.
- At most 40 lines of command output; send longer output to `logs/` and grep it.
- Keep `PROGRESS.md` and `DECISIONS_LOG.md` current in the new folder.

**After a compaction:**
1. Re-read this brief's §5 and stop rules, then `PROGRESS.md` and `DECISIONS_LOG.md`.
2. Never guess a path, run ID or pane ID; take it from those files or `rg --files`.
3. Don't redo work marked done; check that its output exists.

## 2. Rules
- **Folder:** `experiments/scotland/09_qs_marketobs_wave5_config/`. Reuse wave 4 code by `include`.
  Copy a function only if you must change it, and say so in DECISIONS_LOG.
- **TODO 044:** create it from `todos/template.md` (assignee claude, P2), add it to `todos/README.md`,
  and keep its work log current. Close it at the end.
- **Git:** commit small steps and push `exp/qs-goal-vs-market` only; never merge, rebase or
  force-push. src is frozen. Ask the manager (stop with W5_BLOCKED) if you think src must change.
- **Julia only on mcmc-beast** (`ssh root@mcmc-beast`). Make your own checkout:
  `/root/BF_runs/qs_wave5` at your pushed commit. Use persistent tmux REPLs named `claude_w5_*`,
  addressed by the pane ID you record at creation (`tmux new -d -P -F '#{pane_id}'`), with
  `julia --project -t 16` and BLAS 1. Logs go in `/root/BF_runs/logs/qs_wave5/`. No laptop Julia.
- **Never touch other tmux panes or sessions**, locally or on the beast. Locally, `%109` is pi and
  `%110` is James's own Claude. On the beast, leave `julia`, `claude_clv_napkin`, `pi_klm`,
  `w2_watch` and `mbtop` alone. Keep `/root/BF_runs/*`.
- **Read-only DB:** the same before/after audit of the fit, config and harness tables as wave 4,
  around every runner. Stop on any change.
- **Waiting on beast jobs:** one background command that blocks until an end marker or `ERROR:`,
  with `run_in_background` and an explicit long timeout. No polling loops.
- **Honesty:** report every basket and cell you computed, losers included.

## 3. Phases

### Phase 0: setup and gate
- Folder, TODO 044, DECISIONS_LOG (rulings 1–10), PROGRESS, beast checkout.
- **Gate:** your wave-5 runner must reproduce, bit-identical, m12 × canonical × T−25m
  (2.773493294562897 / 1132) and qs × scenario_kelly (wave-4 all-legs style) × T−25m
  (2.154029744590488 / 952).
- Recompute the per-season MDDs above. They must match to 0.01 pp.
- Stop with W5_HANDOVER.

### Phase 1: selection half (24/25 only)
1. **Per-direction breakdown** for qs_marketobs on 24/25. Use one ungated scenario-Kelly run (every
   canonical-markets direction eligible at trust 0.25, λ 23) plus the canonical SK run. Use the
   EDA §2 columns: bets, Kelly ROI (rescaled by slate opening bankroll), flat ROI, capital
   efficiency, cap share, standalone MDD, and CLV. Add m12 canonical as context.
2. **λ curve per basket** (ruling 3) for B1–B4 and anything you explore. Then the ruling-4
   sensitivity.
3. Apply the ruling-6 selection and write **FREEZE.md**. It holds:
   - the basket's trust table, λ, and the full PolicySpec/BookSpec in Julia;
   - the scenario count and seed;
   - the MDD target;
   - the number of baskets looked at;
   - the fallback config.

   Commit and push it. Print the commit hash.
4. Stop with W5_HANDOVER. **Do not touch 25/26.** The manager reviews FREEZE.md and releases Phase 2.

### Phase 2: evaluation, decision, paper-trading spec
1. Score on 25/26, and on 24/25 for completeness:
   - the frozen config, m12 canonical, and the fallback;
   - every Phase-1 candidate basket at its own λ, reported only.

   Columns as wave 4: wealth, g with both bootstrap intervals, paired d vs m12, MDD, Sharpe,
   Calmar, ROI, turnover, bets, CLV, per-direction table, budget- and cap-binding share.
2. T−6h rows (ruling 9).
3. Apply ruling 8 and state the decision in one line.
4. `REPORT.md`: a summary of at most 12 lines, then §§ selection, λ curve, freeze, evaluation,
   T−6h, decision, caveats. The caveats cover:
   - the number of looks;
   - about 50 slates per half;
   - m12's experiment-07 in-sample advantage on both halves;
   - last-traded prices, not executable.
5. `PAPER_TRADING_CONFIG.md`: the exact frozen config, plus a short list of what the later
   src/MatchDay integration needs (weekly market-rate inversion feed, live QS fits, scenario
   adapter, basket-filtered legs). No code.

## 4. Sentinels
Print alone on its own line: `W5_HANDOVER` at the end of Phases 0 and 1, `W5_BLOCKED` with a
one-line reason, `W5_DONE` at the end.

## 5. DONE WHEN
1. The Phase 0 gate reproduces bit-identical, and the DB audit is unchanged throughout.
2. FREEZE.md was pushed before any 25/26 wave-5 number existed; the git log shows the order.
3. The Phase 2 CSVs are committed, and REPORT.md and PAPER_TRADING_CONFIG.md answer §3 Phase 2
   with traceable numbers.
4. TODO 044 is closed with follow-ups, everything is pushed, your beast panes are closed, and
   `W5_DONE` is printed.
