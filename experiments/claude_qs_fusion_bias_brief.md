# Work package (Claude CLI): wave 3, fusion of goals+xG and market rates with learned market bias

**Roles:** you are the implementer, a Claude Code agent in tmux. The manager is the Claude
session that launched you; it checks in, compacts your context between phases, and reviews.
The human set the scope on 2026-10-10. **Do this package only.**

**Why.** In wave 2, `qs_marketobs` (QualityStyleGRW observing past market rates only) tied the
de-vigged close on 1X2 LogLoss: 0.61192 against 0.61312, not detectable, with a clean leakage
audit. A market-only model can at best reproduce the market. To *beat* it, the model needs the
outcomes, so it can learn where the market is systematically wrong.

**This wave's model** puts both observations on one latent strength, with **learned static market
bias**. Write D = log λ_h − log λ_a (supremacy) and M = (log λ_h + log λ_a)/2 (level):

    goals + xG:  the unchanged JointGammaPoissonObservation on the TRUE rates
    market:      D̃ = δ_D + κ_D · D_true + e_D,   M̃ = δ_M + M_true + e_M

The market part is equivalent to observing the log-rates:

    log λ̃_h = δ_M + δ_D/2 + M_true + κ_D·D_true/2,   log λ̃_a = δ_M − δ_D/2 + M_true − κ_D·D_true/2

with i.i.d. Normal noise σ_obs on each log-rate, exactly as wave 2's `MarketRateObservation`.

- κ_D < 1 means the market compresses favourites.
- δ_D is the market's home tilt beyond the true home advantage. Goals identify true γ; the market
  sees γ + δ_D.
- δ_M is the market's totals offset.

**Priors:** κ_D ~ LogNormal(0, 0.2), δ_D ~ Normal(0, 0.1), δ_M ~ Normal(0, 0.1). σ_obs uses the
wave-2 prior.

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources; trust it, and open a
source only for the line ranges listed.

### Branch and worktree

`/home/james/bet_project/.worktrees/BayesianFootball-qs-experiment`, branch
`exp/qs-goal-vs-market`. It holds waves 1–2 (TODO 040 and TODO 041).

### Components that exist (tested; zero-allocation gradients)

| Component | Where |
|---|---|
| `QualityStyleGRW` | `src/models/pregame/components/dynamics/team_level/quality_style.jl` (struct l.14). Per-side scale τ_k; weak-r prior LogNormal(0, 1). |
| `MarketRateObservation{F<:MarketRatesFeature}` | `src/models/pregame/builder/components.jl` l.996 |
| Its likelihood | `engine.jl` l.166 (`_observe(o::MarketRateObservation, η_h, η_a, …)`) |
| `MarketRatesFeature(table)` | `src/features/market_rates.jl`: sorted arrays plus a canonical digest in its display, so the recipe identity includes the table |
| Tests | `test/test_market_rate_observation.jl`, `test/test_quality_style_grw.jl` |
| `JointGammaPoissonObservation` | `components.jl` l.1139; joint likelihood `engine.jl` `_observe(o::SharedKappaJoint…)` (≈ l.330) |
| Joint config used in wave 2 | `experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl` l.17–25, `make_joint_observation()` with no league deltas |

### Wave-2 folder (`experiments/scotland/06_qs_joint_and_market_observation/`)

Reuse its code; don't modify wave-2 outputs.
- `candidates.jl`: scope `lower`, W2 sampler 4×(500+1000), the frozen market table plus its
  SHA256.
- `l05_grid.jl`: **control forwarding**, so non-control arms get an explicit control `RunRef`.
- Phase-gate launchers; `r06_score.jl` (with the parse fix); the leakage audit
  `r10_leakage_audit.jl`; `r07_reproduce.jl`.
- `REPORT.md` (summary at the top) and `DECISIONS_LOG.md`, which holds every manager ruling.

**Read the wave-2 manager rulings and follow them:**
- the C2-pending t05 child set is excluded from acceptance;
- the finite-difference audit is a step-size convergence check with Richardson ≤ 1e-8, plus
  exact AD ≤ 1e-10;
- recovery uses 3 seeds: no parameter misses on all 3, and pooled coverage ≥ 80%;
- control forwarding;
- deterministic control rescoring with an exact-equality audit;
- the namespace provenance rule (load wave 1's market module at top level).

### Reference runs, reused and never refitted

| Arm | Run |
|---|---|
| grw_joint (wave 2) | `98447840-e95c-420f-92b4-2db6545506a3` |
| qs_joint | `2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea` |
| grw_marketobs | `e47a71a7-52eb-4aa5-b747-e64099ea9977` |
| qs_marketobs | `81fdf817-f82d-4f66-9bfd-5f456c4fe29f` |

Plus wave 1's runs and market C0 (`05_.../results/RUNS.csv`) and the market close.

### Snapshot and data

- Snapshot: pinned `c786e2fc…423b4` (published ScottishLower). T03 needs its original mtime:
  `touch -d "2026-09-25 12:57:15.480765468 UTC"`.
- The frozen market-rates table: as wave 2. Check its SHA256.

### Docs, by section only

- AD rules: `docs/turing_ad_performance_guide.md`, §3 l.161–289, §4 binary masking l.290–370,
  §10 l.696–858.
- Harness: `src/harness/stages.jl` (`smoke` l.104, `grid` l.225) and `scoring.jl`.

### Reading rules

- Use `Read` with offset/limit, or `rg -n`. Never read whole reports, design docs or long files.
- Don't re-read a file you've already read unless it changed.
- Keep command output short (`| tail -40`). Send long output (Julia logs, test runs) to
  `07_.../logs/<name>.log` and grep it.
- Keep `07_.../PROGRESS.md` (phase, done, next action, open beast panes) and `DECISIONS_LOG.md`
  current. **After a compaction, re-read both before acting.**

## 2. Rules

- **Git:** commit in small steps and push `exp/qs-goal-vs-market` only. Never merge, rebase or
  force-push.
- **TODO 042:** create it with `scripts/todo.sh` conventions: "Wave 3: fusion of goals+xG and
  market rates with learned market bias".
- **`src/` changes:** only the new fusion observation (component, likelihood seam, export,
  tests). The existing components stay unchanged.
- **Julia only on mcmc-beast**, in persistent REPLs in tmux, never one-shot.
  - Checkout: `/root/BF_runs/qs_experiment`, updated to your pushed commit.
  - Sessions are yours only, named `claude_qsx3_*`, by pane ID.
  - Start Julia with `JULIA_PKG_PRECOMPILE_AUTO=0 /root/.juliaup/bin/julia --project -t 16`, then
    `BLAS.set_num_threads(1)`.
  - Logs go in `/root/BF_runs/logs/qs_experiment_w3/`.
  - No Julia on the laptop. Never touch other sessions.
- **Databases:** `betdb` read-only. The new harness namespace is `scottish_lower_qs_wave3_2426`.
- **Waiting on long beast jobs:** don't poll in a loop. Run one background command that blocks
  until the job's end marker or an `ERROR:` appears in its log, for example:

      until grep -aqE 'MARKER|ERROR:' LOG; do sleep 60; done

  Use `run_in_background`. You're notified on completion, at no token cost while you wait.
- **Honesty:** no loosened thresholds, favourable subsets, prior tuning after seeing scores, or
  refits of reference runs. If a gate fails, stop and report with the numbers.

## 3. Phases

**At the end of each phase:** commit, push, update `PROGRESS.md`, then print
`QF3_HANDOVER` alone on its line and **stop**. The manager compacts your context and tells you to
continue.

### Phase 0: branch health

On the beast, in fresh REPLs: the QS, market-rate-observation, MultiScaleGRW, tape-allocation,
builder and harness tests, and market-model t01–t06 (C2-pending excluded). Confirm the frozen
table SHA.

### Phase 1: the fusion component

`JointMarketFusionObservation`, or a name that fits the code style. It wraps the existing joint
goals+xG observation and the market-rate observation with the bias terms above, on the same
linear predictors. Every bias term needs an "off" switch (κ_D = 1, δ_D = δ_M = 0 fixed) for the
ablation arm. Predictions use the TRUE rates through the double-Poisson grid.

**Tests:**
1. Wiring and masking (matches without a full book add 0 market log density).
2. The likelihood equals the hand-written joint + biased-Normal density to 1e-12, compared at the
   same accumulation seam (wave 2's lesson).
3. With bias off, the market part equals `MarketRateObservation` and the goals part equals the
   joint observation, exactly.
4. Exact AD (compiled and fresh ReverseDiff vs ForwardDiff) ≤ 1e-10, plus the FD convergence
   audit, on a real fold, for both dynamics.
5. Zero-allocation compiled gradients (add cases to `tape_allocation_tests.jl`).
6. Synthetic recovery, 3 pre-declared seeds per arm. Truths include κ_D = 0.85, δ_D = 0.03,
   δ_M = −0.05. The wave-2 gate applies.

### Phase 2: candidates and smoke

`experiments/scotland/07_qs_fusion_market_bias/candidates.jl`, scope `lower`, W2 sampler. The
arms:

| Arm | Model |
|---|---|
| `fusion_qs_bias` | QualityStyleGRW (weak r) + fusion with bias. **This wave's control** for harness scoring, using control forwarding. |
| `fusion_qs_nobias` | QualityStyleGRW (weak r) + fusion, bias fixed off |
| `fusion_grw_bias` | MultiScaleGRW + fusion with bias |

Run `smoke` on every arm. Report R̂, ESS, divergences, the hard checks, wall times and the full
ETA. **Then ask the manager with AskUserQuestion before the grid.**

### Phase 3: 40-fold grid

Order: fusion_qs_bias → fusion_qs_nobias → fusion_grw_bias. One seeded rerun for any fold with
R̂ > 1.05; divergences ≤ 0.1%; tail ESS reviewed against 400. Wait using background completion
notifications.

### Phase 4: scoring, leakage audit and report

Use wave 2's metric set and intervals. Classify on the noncircular 8-week interval, and flag any
class the circular interval contradicts. Report the standard 3-way 1X2 LogLoss alongside the
harness's per-selection LogLoss.

**Pairs:**
1. **Primary:** fusion_qs_bias − market close.
2. fusion_qs_bias − qs_marketobs.
3. fusion_qs_bias − fusion_qs_nobias (is any gain from the bias terms?).
4. Secondary: fusion_qs_bias − fusion_grw_bias; fusion_qs_bias − qs_joint.

**Bias posteriors by fold:** κ_D, δ_D, δ_M and σ_obs. Does the market compress (κ_D < 1)? Is its
home tilt or totals offset non-zero?

**Leakage audit** as wave 2: zero target or after-cutoff market rows, and own-close provenance.

**Write** `README.md`, `REPORT.md` (a summary of at most 12 lines first), harness leaderboard
rows, and the TODO 042 log.

### Phase 5: reproduction

Refit 1 fold per new arm from the frozen seed in a fresh REPL and compare byte-for-byte. Record
the result in `REPRODUCIBILITY.md`.

**Finish:** commit, push, close your beast panes, and print `QF3_DONE` alone on its line. If
blocked, print `QF3_BLOCKED` with a short reason.
