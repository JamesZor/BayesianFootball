# 043 — Wave 4: scenario-Kelly backtest of qs_marketobs at executable prices

| Field | Value |
|---|---|
| ID | 043 |
| Title | Wave 4: scenario-Kelly backtest of qs_marketobs at executable prices |
| Status | COMPLETED |
| Priority | P2 |
| Assignee | claude |
| Created | 2026-10-10 |
| Updated | 2026-10-10 |
| Related Files / Commits / PRs | `experiments/pi_qs_marketobs_backtest_prompt.md`; `experiments/scotland/08_qs_marketobs_backtest/` |

## Context & Problem Statement

Backtest five saved models against T−25m/T−6h/T−24h Betfair archive prices after commission. Claude manages the package. Work phase by phase, stopping at every boundary or failed gate. No new fits; databases read-only; Julia only on mcmc-beast.

## Acceptance Criteria

- [x] Phase 0: canonical and option_b_live inventories, historical experiment 07 m12 wealth/bet parity, unchanged DB audits, three snapshot coverage reports.
- [x] Phase 1 allocator audit with source line references; Phase 2 scenario allocator tested or unnecessary.
- [x] All five-model × snapshot × policy cells computed or NOT_AVAILABLE with reasons, block-bootstrap intervals and paired m12 differences.
- [x] Reports answer all five brief questions with numbers traceable to committed CSVs.
- [x] Follow-ups recorded, branch pushed, owned beast panes closed, QBT_DONE (the Claude brief's sentinel; it replaces PHASEBT_DONE).

## Ideas & Candidate Solutions

Freeze policy before viewing results. Distinguish canonical MatchDay defaults from the newer Option B stack and experiment 07's historical book settings. Do not force parity by tuning.

## Work Log & Progress

- [2026-10-10 @pi] Claimed Phase 0 in `BayesianFootball-qs-experiment`, branch `exp/qs-goal-vs-market`. Recorded exact canonical defaults and experiment 07 historical target before running any reproduction; Option B is a distinct configuration. Next: test parity on beast and stop at failure or boundary.
- [2026-10-10 @pi] STOP at Phase 0 read-only-session assertion on corrected startup (`9249fbcd`). No fit loaded or numerical backtest computed. Startup world-age failure preserved separately. Dependency: Claude authorises verified read-only connection setup and resolves canonical/Option B/historical policy discrepancy. Owned beast pane `%451` closed; reports/logs handed over. No acceptance criterion closed.

- [2026-10-10 @pi] Manager authorised Phase 0 resume: static read-path verification plus before/after table-wide DB audit replaces session-default assertion; actual SHOW value logged once, no PGOPTIONS debugging. Canonical alone targets 1127 bets / 2.5152 ±0.00005. Both canonical and option_b_live inventories reported; raw probabilities, no fitted calibration; plain_kelly 0.25 retained. Updated loader to audit even failure exits and to deserialize cache with no SQL fallback. Static gate passes, four injected negative fixtures rejected.

- [2026-10-10 @pi] Resumed runner `8018dacf` FAIL canonical parity: wealth 2.773493294562897 / 1132 bets versus 2.5152 ±0.00005 / 1127. 710 saved OOS IDs, 99 slates, 611 books, converged=true, zero build errors. Static gate PASS and finally before/after DB audit unchanged 10/10 (byte-identical); SHOW default off logged once. No retry/tuning/fit/write; numerical Option B and remaining snapshot inventory stopped. Pane %452 closed. Dependency: Claude reviews canonical target/stack mismatch before authorising further Phase 0 work.

- [2026-10-10 @pi] Ruling 3: manager corrected historical target association. Reconstructed runner-3 m12/raw/B0_canonical/prod from runner/loader/CSV with line citations; FractionalKelly0.30/min0.001/tradeable markets, λ23/cap0.25/canonical geometric trust. It is now the sole reproduction gate. Canonical measurement retained separately, no rerun/tuning. New outputs preserve prior audits; same cached inputs/710 IDs/T−25 book required. Static checker PASS; next execute historical gate, fixed Option B reference and snapshot inventory if accepted.

- [2026-10-10 @pi] Source72cd5496 PASS exact historical reproduction: unit wealth2.515204600600047 /1127 bets/99 slates/608 books; committed CSV headline matches exactly, zero build errors, converged=true. Same710 IDs/T−25 book byte-identical to canonical; static gate PASS, SHOW off logged once, finally DB audit10/10 unchanged/byte-identical. OptionB raw reference2.207736087761667 /1172. All snapshot coverage/staleness complete: T−25 any=611/1X2=545; T−6h any=295/1X2=182; T−24h any=53/1X2=47; 360-minute sensitivity any=190/1X2=170 (denominator710). Canonical2.773493294562897 /1132 retained separately, wrong-target failure superseded, no rerun. Pane%453 closed; phase0 criterion met. STOP at boundary, no Phase1+. Overall task remains IN_PROGRESS pending manager continuation; PHASEBT_HANDOVER.

- [2026-10-10 @pi] Manager accepted Phase0 and authorised allocator check/conditional Phase2. Phase1 audit complete with line references: per-match posterior-mean joint score-grid Kelly, coherent same-match payoff correlations; BakerMcHale per-match re-solves/scalar shrink; risk/cap do not jointly optimise daily expected log wealth. Shared posterior daily objective missing; Phase2 required. MatchBook/stake_slate do not retain/receive draw-level inputs, so pending confirmation of an opt-in experiment daily-slate adapter within src-allocator-only scope. No src changes/tests/DB/panes yet. T−6h and both labelled T−24h bounds remain low-coverage unpooled strata. Audit in PHASE1_ALLOCATOR_AUDIT.md and DECISIONS_LOG.md.

- [2026-10-10 @claude] Handover: pi (Codex fault + weekly quota) handed Phases 3–4 to a Claude CLI agent; brief `experiments/claude_qs_backtest_phase34_brief.md` (`d83cca9e`). Not BLOCKED; Phases 0–2 accepted. Phase 3 runner `l02_phase3_grid.jl`/`r02_phase3_grid.jl` (`f059662b`) running on beast in owned REPL `jl-claude_qsbt_p3`. Fresh-REPL cold-load smoke of the native ScenarioKelly checks: 51/51 PASS. C0 has genuine draw-level latents (512 = 128θ × 4 state draws per fold), so its BakerMcHale and scenario cells are computed, not NOT_AVAILABLE.

- [2026-10-10 @claude] Phase 3 PASS (runner `4f1d3d43`). Gate m12 × T−25m × historical_exp07 = 2.515204600600047 / 1127, bit-identical; canonical 2.773493294562897 / 1132 and option_b_live 2.207736087761667 / 1172 also reproduce Phase 0 exactly. 100/100 cells computed (none NOT_AVAILABLE), 0 build errors, 1,435 scenario solves converged, DB audit unchanged 10/10. First attempt stopped at CSV write (C0 `converged = nothing`); fixed and rerun. Results in `results/phase3/`. Stopped at the phase boundary.

- [2026-10-10 @claude] Phase 4 done; TODO closed. README.md and REPORT.md answer the five questions from `results/phase3` plus local derived tables (`x04_phase4_tables.py` → `results/phase4/`).
  - qs_marketobs trails m12 at T−25m (canonical 1.857 vs 2.773, d −4.05 ×10⁻³/day [−11.95, 1.27]) and at T−6h, and leads only at T−24h (78–289 bets, no committed interval excluding 0).
  - On the same fixtures, the T−25m ROI per stake is at least the early ROI, so the T−24h lead is a fixture-subset effect. T−24h/360 prices are up to 6 h old and not executable.
  - Trust tiers add nothing consistent. Scenario Kelly gives +1.50 [0.14, 2.66] for qs_marketobs at T−25m and does not change the ranking.
  - CLV is positive in 100/100 cells, at +0.17 to +0.40 pp at T−25m.
  - Everything is in-sample (2 seasons, 99 slates, experiment 07 policy selection with m12), and the intervals are wide.
  - No owned beast session remains.

## Verification & Findings

### Current ruling 3 attempt — Phase0 accepted

- Actual beast run72cd5496, persistent Julia1.12.4/16 threads/core-pinned/BLAS1, owned pane%453 now closed.
- Exact historical recipe and source lines: HISTORICAL_REPRODUCTION_RECIPE.md. CSV asserts unique target/config and runtime accepted fixed tolerance/bets; local DuckDB compares headline to committed experiment07 row to1e-12.
- Historical £1000 reporting scale normalised to unitwealth; 99 daily rows sum1127 bets and end2.515204600600047. Canonical source8018dacf preserved without rerun, no longer a reproduction gate.
- Before/after audit byte-identical,10/10 unchanged; same input IDs/T−25 book byte-identical; static positive check passes (unchanged checker previously rejected four negative fixtures).
- Four snapshot/bound settings,36 per-market rows including zeros; staleness maxima respect bounds. Last-traded price observations are NOT verified executable touch/depth; no profitability claim for qs_marketobs.
- New evidence: results/phase0_historical/ and logs/phase0_historical.log, policy-reference comparison and result checks. REPORT/HANDOVER current, previous failures preserved in chronology. No src changes or later phases; no broad test suite run.

### Preceding wrong-target attempt — superseded by ruling3

- Executed `8018dacf` in beast persistent Julia 1.12.4, 16 threads, pinned cores, BLAS 1, owned pane %452 (now closed).
- Static read-path checker PASS; four negative fixtures rejected. Actual session default off recorded. DB counts/maxima unchanged across all 10 audited fit/config/harness tables, including failure exit; snapshots byte-identical.
- m12 UUID loaded once, restricted to 710 OOS IDs. Canonical parity FAIL: wealth 2.773493294562897 / 1132 bets; no build errors, converged=true. No alternative stack or relaxed threshold tested.
- T−25 book and 99 canonical daily rows committed; all aggregate snapshot coverage and remaining cutoffs pending. Option B inventory reported, numerical arm NOT_RUN. No later phase.
- Current evidence/report/handover under `experiments/scotland/08_qs_marketobs_backtest/`. Static/tracker/diff checks pass; all owned panes/jobs closed.

### Earlier stops (historical, superseded safety assertion)

- Beast `julia --project -t 16 -i`, pinned cores, BLAS 1; include `r00_phase0_inventory.jl` in owned pane `%451`.
- Attempt 1 `8c4fccec`: startup MethodError/world-age warning, no gate reached; corrected module include once.
- Attempt 2 `9249fbcd`: FAIL `assert_readonly` (`SHOW default_transaction_read_only`), before m12 load_fit. No parity/coverage/wealth/CLV or bootstrap evidence.
- At those earlier stops, target2.5152/1127 was committed experiment07 evidence, not yet freshly reproduced (ruling3 run now reproduces it).
- No fits, source allocator changes, or DB writes. At those earlier stops, m12 UUID `132df5c2-c742-4e95-8693-3aeb2b2cbaef` had not been loaded.
- Evidence and resume requirements: `experiments/scotland/08_qs_marketobs_backtest/{REPORT,HANDOVER,PROGRESS,DECISIONS_LOG}.md`, committed attempt logs and execution-status CSV.
- `scripts/todo.sh check` and `git diff --check` passed before the source commits; final checks recorded in handoff commit. Owned session closed and verified absent.

### Phase 4 findings and follow-ups (Claude CLI)

Report: `experiments/scotland/08_qs_marketobs_backtest/REPORT.md`. Follow-ups, none started:

1. **`kkt_residual` budget tolerance** (`src/Portfolio/implementations/allocators.jl`). It treats the budget as binding only within 1e-6, while the barrier stops 1e-5 to 1e-3 short. That flags 255 healthy scenario solve-days. Scale the tolerance with `barrier_mu`, or use μ/(B − Σa) as ν.
2. **Bootstrap centring.** Noncircular 8-week blocks under-sample the edge weeks, so intervals can be lopsided. Adopt circular or stationary blocks experiment-wide, or report both.
3. **Executable prices.** Re-score the T−25m and T−6h cells on back/lay touch and depth from the 26/27 order-book archive, not last-traded prices.
4. **CLV coverage.** At T−25m the close covers 96% of 1X2 bets but only 61% of OU2.5, 43% of BTTS and 34% of OU1.5 bets. Use a fixed complete-close fixture set or a wider close window; never impute.
5. **Out-of-sample season.** Freeze one policy per model and score 26/27 prospectively; every Phase 3 policy is in-sample through experiment 07.
6. **Scenario Kelly for the market-informed models.** It helped qs_marketobs, fusion and C0 at T−25m. Confirm out of sample before any production change.
