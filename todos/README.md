# TODO Tracking

`todos/` is the repository-local tracker for actionable work. Task files are
canonical; this registry is a human-maintained index of their metadata. It does
not replace experiment records, issue trackers, or `docs/tickets/`.

## Registry

<!-- TASKS:START -->
| ID | Title | Status | Priority | Assignee | Last Updated |
|---|---|---|---|---|---|
| [001](001_sampling_budget_benchmark.md) | Benchmark MCMC Sampling Budget and Target Acceptance on Fold 1 | COMPLETED | P1 | claude | 2026-09-10 |
| [002](002_complete_experiment_08_goal_decomposition_production_grid.md) | Complete Experiment 08 Goal Decomposition Production Grid | IN_PROGRESS | P1 | claude | 2026-09-10 |
| [003](003_benchmark_julia_gc_tuning_and_heap_size_hint_on_multi_core_sampling.md) | Benchmark Julia GC Tuning and Heap Size Hint on Multi-Core Sampling | COMPLETED | P1 | claude | 2026-09-10 |
| [004](004_scope_turing_jl_v0_46_and_dynamicppl_v0_42_upgrade.md) | Scope Turing.jl v0.46 and DynamicPPL v0.42 Upgrade | ACTIVE | P1 | claude | 2026-09-10 |
| [005](005_rerun_scottish_lower_experiment_01_poisson_grid_with_reversediff_and_verify_post.md) | Rerun Scottish Lower Experiment 01 Poisson grid with ReverseDiff and verify posterior parity | COMPLETED | P1 | pi | 2026-09-10 |
| [006](006_rerun_scottish_lower_historical_paradigms_with_reversediff_and_unified_benchmark.md) | Rerun Scottish Lower historical paradigms with ReverseDiff and unified benchmark | COMPLETED | P1 | pi | 2026-09-10 |
| [007](007_prototype_gaussian_random_walk_state_space_dynamics_with_reversediff.md) | Prototype Gaussian Random Walk State Space Dynamics with ReverseDiff | COMPLETED | P2 | pi | 2026-09-11 |
| [008](008_hierarchical_scottish_pitch_type_and_match_timing_home_advantage.md) | Hierarchical Scottish Pitch Type and Match Timing Home Advantage | COMPLETED | P2 | claude | 2026-09-18 |
| [009](009_hierarchical_multi_tournament_pooling_across_scottish_segments.md) | Hierarchical Multi Tournament Pooling Across Scottish Segments | BACKLOG | P2 | unassigned | 2026-09-10 |
| [010](010_design_and_prototype_autonomous_agent_bayesian_model_search_loop.md) | Design and Prototype Autonomous Agent Bayesian Model Search Loop | BACKLOG | P2 | unassigned | 2026-09-10 |
| [011](011_fuse_goal_decomposition_with_proxy_xg_and_player_lineup_dynamics.md) | Fuse Goal Decomposition with Proxy xG and Player Lineup Dynamics | BACKLOG | P2 | unassigned | 2026-09-10 |
| [012](012_standardise_model_attribution_and_capture_ratio_in_portfolio_engine.md) | Standardise Model Attribution and Capture Ratio in Portfolio Engine | COMPLETED | P2 | pi | 2026-09-18 |
| [013](013_fuse_multiscalegrw_with_player_lineup_dynamics.md) | Fuse MultiScaleGRW with Player Lineup Dynamics | IN_PROGRESS | P1 | claude | 2026-09-11 |
| [014](014_prototype_jointgammanegbinobservation_for_totals_and_btts_calibration.md) | Prototype JointGammaNegBinObservation for Totals and BTTS Calibration | COMPLETED | P1 | claude | 2026-09-12 |
| [015](015_prototype_multiscalegrw_with_market_smile_and_supremacy_anchoring.md) | Prototype MultiScaleGRW with Market Smile and Supremacy Anchoring | COMPLETED | P1 | claude | 2026-09-13 |
| [016](016_prototype_1_parameter_smile_spine_with_multiscalegrw.md) | Prototype 1-Parameter Smile Spine with MultiScaleGRW | COMPLETED | P1 | claude | 2026-09-17 |
| [017](017_graduate_smilescoregrid_anti_diagonal_reweighting_and_close_t011.md) | Graduate SmileScoreGrid Anti-Diagonal Reweighting and Close T011 | COMPLETED | P1 | pi | 2026-09-18 |
| [018](018_abstract_and_automate_portfolio_market_trust_pruning_sweeps.md) | Abstract and automate portfolio market trust pruning sweeps | COMPLETED | P1 | pi | 2026-09-18 |
| [019](019_slim_agents_md_master_index_and_modularize_operational_guides.md) | Slim AGENTS.md master index and modularize operational guides | COMPLETED | P1 | claude | 2026-09-18 |
| [020](020_sweep_slatedrawdown_lambda_risk_budgets_on_grw_models.md) | Sweep SlateDrawdown lambda risk budgets on GRW models | COMPLETED | P1 | pi | 2026-09-18 |
| [021](021_prototype_fast_slow_grw_rate_pooling_and_decompression.md) | Prototype fast-slow GRW rate pooling and decompression | COMPLETED | P1 | claude | 2026-09-21 |
| [022](022_prototype_momentum_multiscale_grw_dynamics.md) | Prototype momentum multiscale GRW dynamics | COMPLETED | P1 | pi | 2026-09-22 |
| [023](023_prototype_market_inverse_grw_dynamics.md) | Prototype market-inverse state-space and dynamic GRW volatility models | COMPLETED | P1 | claude | 2026-09-22 |
| [024](024_prototype_negbin_with_pxg_form_supremacy_covariate.md) | Prototype Negative Binomial with Linear Proxy-xG Form Covariate | COMPLETED | P1 | pi | 2026-09-22 |
| [025](025_prototype_decoupled_generative_xg_primary_model.md) | Prototype Decoupled Generative xG-Primary Model with Subordinate Goals | COMPLETED | P1 | pi | 2026-09-23 |
| [026](026_prototype_two_tier_fast_slow_latent_state_space_model.md) | Prototype Two-Tier Fast-Slow Latent State-Space Dynamics | BACKLOG | P2 | unassigned | 2026-09-22 |
| [027](027_eda_scottish_club_pedigree_full_time_status_and_tier_priors.md) | EDA: Scottish club pedigree, full-time status, and tier priors | BLOCKED | P1 | pi | 2026-09-23 |
| [028](028_cross_tier_scottish_pyramid_and_informative_priors_time_decay_models.md) | Cross-Tier Scottish Pyramid and Informative Priors Time-Decay Models | COMPLETED | P1 | claude | 2026-09-27 |
| [029](029_cross_tier_scottish_cup_and_pyramid_hierarchy_eda.md) | Cross-tier Scottish Cup and pyramid hierarchy EDA | COMPLETED | P1 | claude | 2026-09-24 |
| [030](030_shared_experiment_harness_recipes_gates_runner_scorecard_leaderboard.md) | Shared experiment harness: recipes, gates, runner, scorecard, leaderboard | COMPLETED | P1 | pi | 2026-09-27 |
| [031](031_scottish_pyramid_dynamics_x_scope_x_observation_matrix.md) | Scottish pyramid dynamics x scope x observation matrix | ACTIVE | P1 | claude | 2026-09-28 |
| [032](032_fixed_scale_map_screen_for_non_centred_models.md) | Fixed-scale MAP screen for non-centred models | BACKLOG | P2 | unassigned | 2026-09-27 |
| [033](033_backfill_championship_betfair_archive_for_the_t55_monitor.md) | Backfill Championship Betfair archive for the T55 monitor | BACKLOG | P2 | unassigned | 2026-09-27 |
| [034](034_zero_allocation_compiled_gradients_in_the_builder_engine.md) | Zero-allocation compiled gradients in the builder engine | COMPLETED | P1 | claude | 2026-09-27 |
| [035](035_sampler_side_allocation_and_grw_multi_thread_contention.md) | Sampler-side allocation and GRW multi-thread contention | BACKLOG | P2 | unassigned | 2026-09-27 |
| [036](036_harness_persistence_and_reporting_fixes_k_l_m_j_gate_alignment.md) | Harness persistence and reporting fixes (k, l, m, j, gate alignment) | ACTIVE | P1 | pi | 2026-09-28 |
| [037](037_context_cards_short_linked_docs_for_harness_persistence_w2_components_and_fit_pi.md) | Context cards: short linked docs for harness, persistence, W2 components and fit pipeline | ACTIVE | P2 | pi | 2026-09-28 |
| [038](038_market_evaluation_layer_typed_recipe_axes_pluggable_evaluators_unified_clv.md) | Market evaluation layer: typed recipe axes, pluggable evaluators, unified CLV | ACTIVE | P1 | claude | 2026-09-28 |
| [039](039_market_model_inversion_and_team_strength_eda.md) | Market model: inversion and team-strength EDA | IN_PROGRESS | P1 | pi | 2026-10-08 |
| [040](040_quality_style_goal_model_vs_market_model_40_fold_scottish_lower_experiment.md) | Quality/style goal model vs market model: 40-fold Scottish Lower experiment | BLOCKED | P1 | pi | 2026-10-09 |
<!-- TASKS:END -->

**039:** Revision8 HANDOVER / IN_PROGRESS (2026-10-06): source70336b07 fast
C0/H1/H2 BOTH protocols accepted,107/107 gates,20/20 convergence; TWO fresh
full pipelines26CSV+3PNG byte-identical. Final reports22/22,scalar/native17/17,
t05 305/305,workflow85/85,t04 92/92,t03/t02 each131/131. Manager-approved scalar
callback and explicitly absent club panels; dense/plot-failed attempts preserved.
C0≈R6; H1 away suppression unidentified,H2kappa spanszero/no gain. Static HA
mixesall12000draws; honesttrainingonly. PHASE_C_PROGRESS_REPORT/HANDOVER current.
Recovery%265 at6335316f, C1 10b%267/10a%268 at41e28736 LIVE/frozen/unpromoted.
At07:46:59Z uncertain conservative remaining15.6h/36.1h forC1,54.6h recovery
(censored16.2–54.6h). No fullPhaseC DONE: recovery+C1inference/reproduction remain;
C2/PhaseD deferred, settings unchanged, no duplicate sampling. Ownidlepanesclosed.
Historical revision7:
Revision7 HANDOVER / IN_PROGRESS (2026-10-06): both exact regressions PASS,
new fixture12476686 mode/marginal gaps0.0/0.0; unchanged tolerances. Fresh t05
305/305, t04 92/92,t03/t02 131/131, full Gate1 12/12,state53/53. Prepared workflow
85/85, core C0/C1 runner parsed NOT executed. Frozen6335316f recovery ACTIVE in
beast%265, all chains warmup200 checkpoint; no completed fit/interval/convergence
or production promotion. Do not restart/kill/check out code while it runs. Local
panes closed; active recovery ownership transferred in results/C/HANDOVER.md.
Recovery then C0/C1/10b and measuredR6gap, H1/H2, figures/full reproduction remain.
C2 remains deferred. Historical revision6 BLOCKED (2026-10-06): exact-case polish now passes
(mode2.00e-13/marginal1.16e-11), fresh t05 289/289, full Gate1 12/12,
engines53/53. Prescribed recovery at d9e6c003 hits NEW Newton100-iteration
exhaustion, root seed4961 initial target; peers cancelled warmup1. No retained
fit or promotion. Observer-only trace: fixture12476686, calls7–100 zero movement,
δ1.035864e-12 (>1e-12), step4.189253e-8; Armijo accepts unchanged x and never
triggers polishing. Review zero-motion stall recognition; no solver retry/fix or
tolerance change. Owned %56/%264 closed. C0/C1/H1/H2 authorised; C2 deferred.
Historical revision5 BLOCKED checkpoint, source120ac906: authorised stopping rule/accounting
implemented, but exact seed4964-coordinate regression fails twice identically.
Fixture12476625: δ3.3821e-12 passes; mode gap4.0160e-7 exceeds1e-8,
Laplace marginal gap6.6586e-7 exceeds1e-9. Tighter Newton residual7.67e-12;
legacy gradient reproduced exactly. Fresh t05 282pass/2fail/284; no third
variation, beast sampling or promotion. Review accuracy/termination compatibility.
C0/C1/H1/H2 remains authorised; C2 deferred. Prior revision4 Gate1 and engine
results remain historical, not regenerated under revision5. Recovery, fits,
measures/figures and two-run production reproduction are unfinished. Thresholds,
seeds, priors, data and grid unchanged; revision4 evidence preserved.
Evidence: `current_development/market_model/results/C/PHASE_C_REPORT.md`.
Double Poisson stays for C; later score-grid C2 and Phase D remain unauthorised.

## Commands and Task Files

Use `NNN_lowercase_slug.md` task names. The CLI is implemented:

```bash
./scripts/todo.sh list
./scripts/todo.sh new "Task title"
./scripts/todo.sh view 001
./scripts/todo.sh check
```

`list` is the default command; colors appear only on a terminal (`NO_COLOR=1`
disables them). `new` creates the next ID using the UTC date with `BACKLOG`, `P2`,
and `unassigned`, and updates the index automatically. Begin manually created
tasks from [`template.md`](template.md), replace its `{{ID}}`, `{{TITLE}}`, and
`{{DATE}}` placeholders, and preserve required metadata and sections. IDs are
`001`–`999`; use the largest existing ID plus one, never recycle completed IDs.
Keep metadata on single-line table rows without literal `|` inside values.
`check` flags a stale registry index or invalid task contract. When creating or
closing a task, update the canonical metadata, matching README row, `Updated`
date, and dated work log together. Close only after acceptance criteria are met:
record verification/findings, set `COMPLETED`, and retain the file in this index.
If work cannot proceed, set `BLOCKED` with the dependency and next action instead
of closing it. Run `check` before handing off or committing.

`new` serializes allocation in this worktree with `todos/.todo-new.lock`. That
lock does not cover separate clones or manual edits: coordinate ID reservations
and resolve collisions before merge. If the lock is stale, first verify no writer
is running, inspect any staging files and reconcile task/index changes left by an
interrupted writer, then remove those staging files and
`rmdir todos/.todo-new.lock`. Existing locks are never auto-cleared. Humans and
agents update README rows with metadata; there is no sync command.

Run the isolated CLI regression tests with `bash test/test_todo.sh`; they create
throwaway tasks in a temporary copy, not in this registry.

## Workflow and Ownership

Statuses move `BACKLOG -> ACTIVE -> IN_PROGRESS -> COMPLETED`, or become
`BLOCKED`. `BACKLOG` is unprioritized/unclaimed work; `ACTIVE` is the prioritized
ready queue; `IN_PROGRESS` is claimed and executing; `COMPLETED` is accepted
with findings recorded; `BLOCKED` awaits an external dependency or decision.
Unblock into `ACTIVE` for a new claim or `IN_PROGRESS` when the owner resumes;
reopen a completed task only with a logged reason and new acceptance criteria.

Priority means `P0` immediate operational/data-integrity risk, `P1` high-value or
near-term work, `P2` normal planned work, and `P3` deferred/nice-to-have work.
Assignee values are `unassigned`, `human`, `antigravity`, `claude`, or `pi`.
Claiming changes metadata and the registry row to `IN_PROGRESS`, records the
assignee plus session/worktree in a dated work-log entry, and is announced to
collaborators. Do not silently steal a task; coordinate a hand-off or unassign it
first.

Add concrete proposed approaches and trade-offs to **Ideas & Candidate Solutions**
before substantial implementation; preserve rejected ideas where they explain a
decision. `docs/tickets/` remains the separate `Txxx` namespace for durable
defects and tracked engineering tickets. Cross-link related TODOs and tickets;
do not migrate or renumber either namespace.
