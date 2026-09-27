# Work package — TODO 031 Wave 2, Phase A: pyramid tier components (build only)

You are the implementer; Claude is the manager and reviews against §4 before anything runs on the
grid. **Do Phase A only, then stop.** Phase A builds and tests components and the W2 candidates
file. It does **not** run smokes or grids: the manager runs those after review.

## 0. Rules (unchanged from the harness phases — re-read `experiments/pi_harness_phase1_prompt.md` §1)

- Worktree `/home/james/bet_project/.worktrees/BayesianFootball-experiment-harness`. Create branch
  **`feat/w2-tier-components` from `59769c4a`** and work only there. Commit in small logical steps.
  Push at the end.
- No credentials in files, prompts or logs. `BF_DB_URL` comes from `.env`.
- The laptop runs **unit tests only**. Anything that loads a real fold or samples runs on
  **mcmc-beast**, from a `scripts/beast_checkout.sh <sha>` checkout of a pushed SHA, with logs under
  `/root/BF_runs/logs/<sha>/`, never inside the checkout. Use `-t 16`, `pinthreads(:cores)` and
  `BLAS.set_num_threads(1)`. Before starting anything heavy, check `pgrep -af julia`. The Kaimon
  process in tmux session `julia` is expected; leave it alone.
- **AD rules, AGENTS.md §7:**
  - no `if`/`for`/`findall`/`isnan` inside `@model`;
  - mask, don't subset; `A[idx]`, never `view`; no mutation;
  - all conditional logic lives in the design/feature layer, outside `@model`;
  - optional parts are selected by **type dispatch** on config structs.
- Every new option defaults to **off**. Existing recipes must keep their `string(model)` and
  therefore their `recipe_hash`. Test this.

## 1. Read first

1. `experiments/scotland/04_pyramid_tier_transition_arms/README.md`: **the contract**. §2 has the
   exact math and priors. If anything below seems to disagree with it, the README wins; say so in
   your report.
2. TODO 031 Work Log, the 2026-09-27 decision block.
3. `src/models/pregame/components/dynamics/team_level/multiscale.jl` (`grw_accumulators`,
   `_grw_pair`, `grw_step_counts`, `_grw_reconstruct_trajectory`, `extract_dynamics`);
   `src/models/pregame/builder/grw_dynamics.jl` (design, `_cb_oos_dynamics`);
   `src/models/pregame/builder/engine.jl` (`composable_count_engine`, `cb_design`,
   `extract_parameters`, predictor and OOS hooks, the `SharedKappaJoint` / `HierarchicalKappaJoint`
   observation and `_cb_rates`); `src/models/pregame/builder/components.jl` (`HierarchicalKappa`,
   `JointGammaPoissonObservation`, the covariate/predictor-term pattern); `builder.jl` (validation,
   chain-site widths).
4. `src/Data/splitting/scoped.jl` (`DataScope`, `apply_scope`, `club_season_tiers`,
   `:senior_spfl_ties`); `src/features/builder.jl` (history/target steps, `:history_match_ids`).
5. `src/harness/` (`candidate.jl`, `checks.jl`, `stages.jl`) and
   `experiments/scotland/03_dynamics_scope_matrix/candidates.jl` (the W1 recipe shapes to copy).
6. `docs/turing_ad_performance_guide.md`, `docs/guides/julia_coding_context_for_agents.md`.

## 2. What to build

### 2.1 `MultiScaleGRW(target_season_step::Bool = false)`

- **Today:** `n_history − 1` macro (σₛ) transitions, all inside the history. The first target state
  is the last history state plus micro step 1 (σₖ).
- **With the flag and `n_target ≥ 1`:** add one more macro transition, present from state
  `n_history + 1` onward (the summer into the target season). Separate innovations for α and β,
  with their own σₛ.
- **With the flag and `n_target == 0`** (the first fold of a season, so no fitted target state):
  the OOS forecast must integrate the boundary step. Per posterior draw, add `σₛ·z`, with a fresh
  `z ~ N(0, 1)` for each (team, side). Use a **deterministic RNG** seeded from the fold, so
  `save_fit`/`load_fit` latent parity still holds. With `n_target ≥ 1`, OOS keeps today's rule
  (final fitted state carried forward).
- Extraction, `grw_step_counts`, and the chain-site widths must be unambiguous with the flag on.
  Read the flag from the config; don't infer it from site counts.

### 2.2 `PyramidTiers` (a predictor-side component; name it as you see fit and document it)

For side c ∈ {home, away} of a match in season s, B_c = τ_anchor + κ·(τ_s − τ_anchor) + o·OF_c,
with τ_T = −s_step·(T − 1) and T = 1..4 for tournaments 54..57. Then:

```
η_home += 0.48·B_home + ηα_home(active events)  −  0.52·B_away + ηβ_away(active events)
η_away += 0.48·B_away + ηα_away(active events)  −  0.52·B_home + ηβ_home(active events)
```

| Field | Default / options |
|---|---|
| `tier_term::Bool` | `true` (`false` gives B1-literal: η only) |
| `anchor` | `:first_in_window`: tier in the club's first season with an SPFL tier among the fold's window seasons (history seasons + target season). `:full_pyramid`: the same rule, but read from the **full** SPFL club-season tier table even when the scope trains on 56/57 only (see 2.4) |
| `step_prior` | `truncated(Normal(0.47, 0.16), 0, Inf)`, one shared step |
| `attack_share` | `0.48` (defence share = 1 − 0.48 = 0.52), fixed |
| `old_firm_prior` | `truncated(Normal(1.15, 0.10), 0, Inf)` on Celtic and Rangers (verify their team keys in the data; the first team, not B-sides); `nothing` disables it |
| `carry` | `NoCarry()` (κ ≡ 0), `EstimatedCarry(Beta(1.5, 4))`, and `FixedCarry(x)` (for tests) |
| `jump` | `NoJump()`, `EstimatedJump(Gamma(2, 0.08))` (η ~ N(0, σ_j²), non-centred, separate α and β draws per event), `FixedTransitionPriors(table)` with the TODO 029 table below |

**Change events.** An event is (club, s): the club's tier in s differs from its tier in s − 1, both
in the tier table, and s is a window season **after** the club's first window season. An event at
the club's first window season is already covered by the anchor. It persists: its η applies to
every match of that club in seasons ≥ s. Entered-SPFL clubs (no tier in s − 1) get no event.

Build the event incidence as dense 0/1 design matrices (matches × events, home and away) outside
`@model`, so the shift is a matrix–vector product.

**B1-literal table** (fixed means and sd per direction, on α and β; other directions get no event):

| Direction | α | β |
|---|---|---|
| 55→56 relegated into L1 | N(+0.12, 0.20²) | N(−0.16, 0.26²) |
| 56→57 relegated into L2 | N(+0.03, 0.20²) | N(−0.04, 0.24²) |
| 57→56 promoted into L1 | N(0.00, 0.21²) | N(+0.01, 0.31²) |

**Cup ties:** each side uses its own club's tier that season. `:senior_spfl_ties` guarantees both
clubs have one.

### 2.3 League observation deltas (for `JointGammaPoissonObservation`)

- **Intercept deltas:** δ_int[T], zero-sum over the 4 tiers, non-centred:
  `δ = 0.10·(raw − mean(raw))`, with `raw ~ N(0, 1)`. Added to each side's log-intensity by **the
  attacking side's** club tier that season.
- **Kappa mode `CompetitionKappa`:** δ_κ[c] for the **match's competition** c ∈ {54, 55, 56, 57,
  cups} (all senior cup tournaments share one level). It is shared by **both sides** of the match,
  zero-sum over the competitions present in the scope, non-centred with a fixed scale **0.25**
  (`δ = 0.25·(raw − mean(raw))`).
- It enters the **goals arm only**: `y ~ Poisson(κ·exp(δ_κ[c])·μ)`. The Gamma pxG arm and ν are
  unchanged. README §2.3 has the evidence (goals ÷ pxG is 0.85 in the Premiership vs 1.23 in L1/L2).
- No learned scales. Refuse either delta with a clear error in a scope with fewer than 2 tiers or
  competitions.

### 2.4 Tier table and features

- Features need the club-season tier table for **all** SPFL seasons, even in the lower scope
  (`:full_pyramid`). `scripts/run_candidates.jl` loads `ScottishPyramid` and `apply_scope` then
  drops 54/55 for the lower scope.
- Carry the full table through scoping, e.g. computed in `apply_scope` from the unscoped league
  matches, or an equivalent you justify. It must **not** enter `string(model)` or the
  `recipe_hash`, and must not change existing recipes.
- A club's tier in a season is the league it is scheduled in (a schedule fact, known before the
  season). Use it for the target season too.
- Features must use no information from after the fold's cutoff beyond that scheduled tier.

### 2.5 OOS prediction path

All three components must reach the held-out fixtures through `extract_parameters` / the latents
path: tier shift, active events, league deltas and κ. Use the fold's posterior draws and the
held-out fixture's features (target-season tier, anchor, active events).

### 2.6 W2 candidates file

Write `experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl`:
- `EXPERIMENT = "scottish_pyramid_tier_transition_w2"`, `TODO = 31`;
- the joint observation exactly as W1 (`make_joint_observation`), `GlobalInterception` +
  `GlobalHomeAdvantage`;
- the W1 sampler (4 × (500 + 1000), acceptance 0.65, max depth 10).

**Scopes:** `spfl_cups` (train 54–57, `:senior_spfl_ties`, monitor 54/55) and `lower` (56/57), both
with **`target_seasons = ["23/24", "24/25", "25/26"]`**, `history_seasons = 2`, clock 56/57,
`dynamics_col = :match_biweek`.

The 15 cells, with these names:

| # | Name | Recipe |
|---|---|---|
| 1 | `td_base` | TD(180), spfl_cups, joint (control) |
| 2 | `td_a2` | + tiers, `NoCarry`, `NoJump` |
| 3 | `td_a2_carry` | + tiers, `EstimatedCarry` |
| 4 | `td_a2_carry_jump` | + tiers, `EstimatedCarry`, `EstimatedJump` |
| 5 | `td_ldelta` | + intercept deltas + `CompetitionKappa` |
| 6 | `td_a2_carry_jump_ldelta` | 4 + 5 |
| 7 | `grw_base` | `MultiScaleGRW()`, spfl_cups, joint (control) |
| 8 | `grw_step` | `MultiScaleGRW(target_season_step = true)` |
| 9 | `grw_step_a2_carry_jump` | 8 + tiers, carry, jump |
| 10 | `grw_step_ldelta` | 8 + league deltas |
| 11 | `grw_step_a2_carry_jump_ldelta` | 9 + 10 |
| 12 | `td_lower_base` | TD(180), lower, joint (control) |
| 13 | `td_lower_a2full_carry_jump` | + tiers `:full_pyramid`, carry, jump |
| 14 | `td_lower_b1lit` | + `tier_term = false`, `FixedTransitionPriors` |
| 15 | `m12_td_ref` | the **exact** m12 recipe of run `132df5c2-c742-4e95-8693-3aeb2b2cbaef` (reconstruct it from its persisted model/config), lower scope, new panel |

- Controls: 1 for cells 2–6, 7 for 8–11, 12 for 13–14.
- Encode the paired control **per candidate** if the harness allows it. If it doesn't, say so: the
  manager scores with `scripts/score_runs.jl` per base.
- Keep `STATUS`/`DECISION` as "spec agreed, not run".

## 3. Tests (a new `test/pyramid_tier_tests.jl`, plus additions where natural; add to `run_parallel_tests.jl`)

1. **GRW accumulators:**
   - pin the current state composition for (n_history, n_target) ∈ {(2, 0), (2, 3), (3, 2)}
     exactly as matrices;
   - with the flag on, exactly one extra macro column, present from `n_history + 1`;
   - flag-on OOS in an `n_target == 0` fold adds zero-mean noise with sd ≈ σₛ (statistical test on
     many draws) and is deterministic under reload.
2. **Regression guard:** with every new option off (and the flag off), `string(model)`,
   `recipe_hash` and the log density at 3 random θ equal the W1 recipes', to machine precision. Use
   a real fold for TD and GRW (beast) plus a synthetic league (laptop).
3. **Reductions:**
   - `FixedCarry(0)` equals `NoCarry`;
   - `FixedCarry(1)` moves a changer's log-rates by exactly +0.48·Δτ (attack) / −0.52·Δτ
     (concession) vs anchor-only;
   - an Old Firm club is shifted by exactly 0.48·o / −0.52·o.
4. **Features:**
   - anchor = first window tier;
   - `:full_pyramid` anchors a relegated club with no lower history at T2;
   - event construction: persisting incidence, no event at the first window season, none for
     entered clubs;
   - B1 table directions mapped correctly;
   - cup ties use per-side tiers;
   - **no leakage:** removing every fixture after a fold's cutoff leaves that fold's features
     unchanged.
5. **Priors:** the default configs reproduce every number in README §2 and the B1 table.
6. **AD:** a ReverseDiff compiled-tape gradient vs finite differences (≤ 1e-6 relative) at 3
   perturbed points, for TD + tiers + carry + jump + league δ, and for GRW + step + the same, on a
   synthetic league. Grep-level check: no `if`, `view(`, `for` or `push!` inside any new `@model`.
7. **OOS parity:** for training matches, the extraction path's per-draw log-rates equal the
   in-model log-rates for the same draw (tiers, events, league δ, κ).
8. **Existing suites green:**
   - `test/harness_runner_tests.jl` (55/55), `test/harness_scoring_tests.jl` (29/29),
     `test/scoped_splitter_tests.jl` (48/48), the portfolio harness tests (25/25);
   - `test/run_parallel_tests.jl` no worse than the known T007 failure.

**On the beast:** instantiate every one of the 15 candidates' Turing models on fold 1 **and** on the
first fold of 24/25 (`n_target == 0` for GRW). Report parameter counts, tape compile time and
gradient latency per candidate. **Do not sample.**

## 4. Acceptance (the manager reruns each item)

1. Tests 1–7 pass on the laptop (synthetic) and on the beast (real-fold parts). Report the commands
   and counts.
2. Test 8 is green, with counts.
3. The beast instantiation table for all 15 × 2 folds (no errors; parameter counts; gradient
   latency).
4. The regression guard holds: W1 recipe hashes unchanged (print them next to W1's
   `harness_checks.recipe_hash` values).
5. `./scripts/todo.sh check` passes; TODO 031 Work Log has a dated `@pi` line.

## 5. Report and stop

- Write `docs/architecture/w2_phaseA_report.md`:
  - evidence per acceptance item (commands, SHAs, numbers);
  - design choices you made where §2 left room (component names, tier-table plumbing, RNG seeding);
  - anything in the README you believe is wrong or ambiguous.
- Leave the beast tidy.
- Commit, push, print exactly `PHASEA_DONE` and stop. If blocked on a human decision: write it
  under "Blocked" in the report, print `PHASEA_BLOCKED`, and stop.
