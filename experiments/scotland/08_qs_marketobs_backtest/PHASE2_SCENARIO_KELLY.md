# Phase 2 — opt-in joint daily scenario Kelly

Phase1 found the missing joint daily objective. User approved the experiment-adapter
path, without wider src plumbing. Prototype sourceb39db972 passed44/44 checks before
src graduation. Native source13677357 passed51/51 checks and existing portfolio
regression92/92. **PHASEBT_HANDOVER** at the Phase2 boundary; no Phase3 backtest.

## Implementation contract

- `Portfolio.ScenarioKelly(n_scenarios=1000, seed=20261010)` and
  `ScenarioWeights` are defined in the sole changed src file,
  `src/Portfolio/implementations/allocators.jl`.
- Existing allocate(p,R,exec) seam: explicit joint-scenario weights, one flat matrix
  with every day's bet column, commission-net payoffs in each scenario row. Objective
  is Σ_s p_s log(1+Σ_j R_sj a_j). Uniform weights give Monte Carlo mean log wealth.
- Nonnegative/per-bet box bounds and one entire-slate ExecutionConfig budget<=1;
  same log-barrier and tiny-stake trimming as the stock Kelly backend. This is an
  approximate barrier solution, not an exact unbarriered optimum certificate.
- Shared backend preserves legacy1e-3 initial stakes. Native scenario solves use
  min(1e-3, per-bet-cap/2, slate-budget/(2n)) for a feasible interior start. No stored
  layouts, production defaults, objective/gradient or existing policy steps changed.
- Ordinary per-match vectors are rejected, so selecting ScenarioKelly in a stock
  BookSpec cannot silently claim daily posterior integration. Tags mark the caller's
  contract; they cannot independently prove provenance from numeric payoffs alone.

## Adapter and posterior alignment

`l01_scenario_kelly.jl` constructs scenarios, not a new production engine. Inputs:
existing daily Slate, one home×away×draw grid per fixture, explicit shared fitted
posterior/fold/chain-layout witnesses, and the existing execution spec.

It rejects mismatched witnesses, unequal draw axes, duplicate fixtures, mixed dates,
invalid grids or inconsistent payoff dimensions. Equal column counts alone do not
prove aligned posterior draws. Witnesses must come from real saved-fit provenance
in Phase3, not be fabricated. Missing C0 draw-level latents remain NOT_AVAILABLE
for scenario rows; a singleton mixture masquerading as a posterior is not a fallback.

For each scenario, one posterior column is used across **all** matches. Independent
scoreline uniforms conditional on that draw are keyed by day/fixture. All a match's
legs use the same scoreline row of its existing payoff matrix, retaining within-match
correlations and the original commission/push/loss rules. Draw indices are balanced
shuffled passes: uniform subset without replacement if ndraws>S, repeated shuffled
passes otherwise. Score distributions are explicitly normalised within each
truncated draw grid; verify real-fit grid support/tail mass before financial claims.

The adapter solves once, returns new books replacing only a_kelly and allocation
quality fields, and leaves original objects untouched. Per-match shrink factors are
preserved from the production base books. They remain that production overlay, NOT
new joint-slate Baker–McHale factors. Existing risk retains its sequential/marginal
law, NOT a risk constraint integrating shared posterior scenarios. Final policy
stakes therefore are overlays on the raw joint optimum, not themselves claimed
unconstrained joint Kelly optima.

## Use (no backtest executed here)

```julia
PF = BayesianFootball.Portfolio
joint = QSBacktestScenario.scenario_books(
    PF.ScenarioKelly(), daily_slate, per_draw_grids, saved_draw_group_witnesses,
    base_book_spec.exec; native = true)
allocation = PF.stake_slate(policy, joint.slate, ctx)
```

For simulation, construct each day's replacement slate/books and use the existing
simulation path. The base books use the frozen production pricing/shrink settings,
not BookSpec(allocator=ScenarioKelly()). No change to ordinary callers/defaults.
Provenance extraction and real-fit grid assembly are Phase3 wiring, not completed here.

## Verification

Pure prototype/native checks: commission-net single-bet closed form; two independent
bets versus a separately re-derived brute-force objective/grid; shared posterior
perfect-correlation counterexample; deterministic seed/stake replay and fixture
ordering; nonnegative/budget/per-bet constraints; rejection of ordinary means and
mismatched provenance; input immutability; delegation to stock trust/shrink/risk/cap.
Native also checks a many-leg/small-budget initial point and bit-identical stock
allocation against three pre-src golden legs, plus existing portfolio_tests.jl.
All fixtures synthetic, no SQL/cache/fits/MCMC. DB audit/SHOW N/A because no tables
are touched; not represented as a new database-safety pass.

Actual beast run: Julia1.12.4/16 threads/core-pinned/BLAS1; prototype44/44,
native51/51, existing portfolio regression92/92, all passed. Native checked all
three pre-src legacy legs bit-identically. Five quantitative rows in
`results/phase2/phase2_native_tests.csv` pass; deterministic stake replay difference0.
Logs: `logs/phase2_{prototype,native}.log`; source13677357 remains the executed
native version, distinct from the evidence commit. Owned pane%454 is closed.

Revise emitted two binding world-age warnings for ScenarioKelly/ScenarioWeights
outside the redirected test log; preserved in `logs/phase2_repl_capture.log`.
They did not fail the checks on Julia1.12.4, but future stricter Julia/Revise use
requires attention. Start a fresh owned REPL and verify cold-load behaviour before
any authorised Phase3 use; no warning fix or additional phase ran at this boundary.

## Coverage remains four distinct analysis strata

T−25m/90-minute reference; T−6h/90-minute low-coverage subset (complete1X2=182/710);
T−24h/90-minute primary low-coverage subset (47/710); T−24h/360-minute sensitivity
low-coverage subset (170/710). Never pool early subsets with T−25m or replace the
90-minute T−24h primary with its sensitivity. Actual figures/tables remain in
PHASE1_ALLOCATOR_AUDIT.md and results/phase0_historical/phase0_snapshot_overview.csv.
