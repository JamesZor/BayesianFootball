# Work package — TODO 037: context cards (short, linked docs for agents)

You are a **solo documentation writer**; Claude is the manager.
- **No subagents.** A separate Claude CLI (Opus 5.5) reviews your work when you finish.
- The design below was agreed with the human in a grill-me session on 2026-09-28. **Do not
  re-open it.** If something is impossible, stop and report (§7).

## 0. Purpose

Agents lose context re-reading large files to find one fact. Build a set of **short context
cards**:
- one concept per file, small enough to load on its own;
- cards link to each other instead of repeating each other;
- they explain how the modules and functions fit together (ASCII flows, interface tables, call
  graphs), so a future agent can modify or extend the code safely without reading everything.

## 1. Rules

- **Worktree** `/home/james/bet_project/.worktrees/BayesianFootball-context-cards`, branch
  **`docs/context-cards`**, cut from the `feat/w2-tier-components` merge commit the manager names.
  - Commit in small steps, and push only this branch. Never merge.
- **You may write only:** `docs/context/**`, the four guides in §4, one row in `AGENTS.md` §1,
  `scripts/context_stale.sh`, the hook in `scripts/todo.sh`, `docs/README.md` (index pointer),
  and TODO 037 (Work Log).
  - **Never edit `src/`, `test/` or `experiments/`**, other than your report.
- **No Julia, no ssh, no database access.** Everything comes from reading the code in the
  worktree. `rg`/`grep`/`git` are your verification tools.
- **Accuracy rules.** The reviewer checks every one of them, and earlier Gemini documentation in
  this repo cited wrong line numbers and wrong defaults.
  1. **No line numbers, anywhere.** Cite `path/to/file.jl` plus the function, type or constant
     name.
  2. Copy every signature, keyword default, threshold, constant, table name, column name and CLI
     flag **verbatim from the code**. After writing a card, grep for each such token and confirm it
     exists as written.
  3. Never infer behaviour you have not read. If you are unsure, write `UNVERIFIED:` and the
     question. The reviewer resolves these.
  4. Describe the code as it is at `verified_sha`. Plans and TODOs go only in the "Why" section,
     as links.
- **Credentials:** none, ever. Refer to `BF_DB_URL` / `BF_EXPERIMENTS_DB_URL` by name only.
- **Budget:** `AGENTS.md` must stay under 22,000 bytes, and `./scripts/todo.sh check` must pass.

## 2. Read first

1. `AGENTS.md`, `docs/README.md`, `docs/guides/julia_coding_context_for_agents.md` (for vocabulary).
2. The four guides in §4 in full. You are slimming them, so you must know everything in them.
3. The design decisions the cards must cite in their "Why" sections:
   - TODO 030 (the harness);
   - TODO 031 and `experiments/scotland/04_pyramid_tier_transition_arms/README.md` (W2 components
     and the decision rule);
   - TODO 034 (the zero-allocation engine);
   - TODO 036, `docs/architecture/harness_klm_report.md` and `harness_klm_review.md` (per-fold
     artefacts, the gate, the nonce, the leaderboard);
   - `docs/turing_ad_performance_guide.md` §10.5.
4. The code in §3's source lists. Read each function before you document it.

## 3. Structure and card list

```
docs/context/
  INDEX.md                  top map: one ASCII diagram of how the four areas connect + one line per area INDEX
  harness/INDEX.md          one line per card + the area's ASCII flow
  harness/<card>.md
  persistence/INDEX.md, persistence/<card>.md
  models/INDEX.md, models/<card>.md
  pipeline/INDEX.md, pipeline/<card>.md
  SLIMMING_LEDGER.md        (§4)
```

Suggested cards follow. Split, merge or rename them if the code argues for it, and say so in
your report. File names are short `snake_case`.

- **harness** (`src/harness/*.jl`, `scripts/run_candidates.jl`, `score_runs.jl`, `leaderboard.jl`,
  `seed_register.jl`):
  - `lifecycle` (screen → smoke → grid → score → board);
  - `candidates_and_recipes` (`Candidate`, `recipe_hash`, `DataScope`, `candidates.jl`
    conventions);
  - `stage_screen`, `stage_smoke`, `stage_grid`, `stage_portfolio`;
  - `checks_and_gates` (hard, review, diagnostic and info checks; `DiagnosticFailure`; the
    convergence rule constants);
  - `scorecard` (subsets, metrics, bootstrap, `SCORECARD_VERSION`, panel labels);
  - `transition_cohorts` (tiers, cohorts, bias and clustered CIs);
  - `controls_and_pairing` (the CSV `control` column, `control_run_id`);
  - `leaderboard_and_register`;
  - `cli_flags` (every flag of the four scripts, including `--test-db`);
  - `harness_tables` (`harness_scores`, `harness_checks`, `harness_experiments`: columns, keys,
    merge semantics);
  - `perf_and_tape_monitors` (`perf_monitor.jl`, `tape_profile.jl`).
- **persistence** (`src/training/inference/db_storage.jl`, `extension.jl`, `io.jl`, `types.jl`,
  `db/schema.sql`):
  - `two_databases` (a short pointer card);
  - `schema_map` (an ASCII entity diagram of every table and its keys, including
    `fit_fold_artifacts` and the `layout` marker);
  - `save_fit`;
  - `load_fit_layouts` (per-fold vs legacy);
  - `extend_fit`;
  - `config_hash_and_duplicates` (`on_duplicate`, the smoke nonce, recipe tags);
  - `blob_limits` (binary binding, the 1 GiB field limit, stride history);
  - `latents_storage`;
  - `file_and_dual_storage`;
  - `connection_resolution` (password-safe resolution, by name only).
- **models** (W2 components; `src/models/pregame/builder/pyramid_tiers.jl`, `league_deltas.jl`,
  `grw_dynamics.jl`, `components.jl`, and
  `src/models/pregame/components/dynamics/team_level/multiscale.jl`):
  - `pyramid_tier_component` (anchor, step s, Old Firm, carry κ, jump η);
  - `tier_features` (anchor rules, `:full_pyramid`, change events, cup ties, leakage guard);
  - `b1_literal`;
  - `league_deltas` (intercept by attacking tier, κ by match competition);
  - `multiscale_grw` (the macro/micro accumulators);
  - `grw_target_season_step` (including the season-opening out-of-sample integration).
- **pipeline** (`src/models/pregame/builder/builder.jl`, `engine.jl`, `components.jl`;
  `src/training/`; `src/models/latents/`; `src/evaluation/`):
  - `builder_overview` (`CountModelBuilder`, `add`, `build`, validation);
  - `component_slots` (which slot each component type fills, and the dispatch hooks every dynamics
    component implements);
  - `engine_and_design` (`composable_count_engine`, `cb_design`; features built outside
    `@model`);
  - `observations` (`JointGammaPoissonObservation` and friends);
  - `fitconfig_and_fit_model` (`FitConfig`, splitters, `QueuedNUTSConfig`, execution);
  - `latents_and_prediction`;
  - `evaluation`;
  - `ad_rules` (a pointer card to the AD guide rules, including 0 bytes per gradient).

### Card template (every card, ≤ 150 lines including frontmatter)

````markdown
---
id: harness/stage_grid
area: harness
title: Grid stage
sources: [src/harness/stages.jl, scripts/run_candidates.jl]
symbols: [grid, save_fit, has_passing_smoke]        # names that must exist in `sources`
verified_sha: <the commit you read>
related: [harness/checks_and_gates, persistence/save_fit, harness/scorecard]
---
# Grid stage
**Purpose.** 1–3 lines.
**Why it is like this.** 2–5 lines, each point linked to the deciding TODO/README/report.
## Flow
```text
(ASCII: boxes/arrows from caller to callees, data in/out, DB tables touched)
```
## Interface
| Symbol | Signature (verbatim) | Inputs → outputs | Side effects / DB | Invariants |
## Calls / called by
## Gotchas
## Tests
(which test files and testsets cover it)
## How to change it
(the steps and the tests to extend)
## See also
(links to related cards and to the long-form source doc, if any)
````

- Keep diagrams ASCII inside ```` ```text ```` fences, at most 100 columns wide.
- Each area INDEX carries the area's module-interface diagram: which file calls which, across
  modules.
- The top `INDEX.md` carries the whole-system map, e.g.
  `candidates.jl → run_candidates → stages → fit_model → save_fit → score_runs → harness_scores → leaderboard`.

## 4. Slim the four guides onto the cards (no information loss)

Guides:
- `docs/guides/experiment_harness_guide.md`;
- `docs/guides/experiment_database_and_config_truth_guide.md`;
- `docs/architecture/composable_model_builder_specification.md`;
- `docs/architecture/unified_v2_architecture.md`.

1. Each guide becomes a **short overview**: purpose, the reading order of the cards, and whatever
   genuinely spans cards. Detail moves to the cards, and the guide links them.
2. **No information loss.** Every paragraph, table, code block and rule removed from a guide must
   appear on a card, or stay in the guide. Record each move in `docs/context/SLIMMING_LEDGER.md`:
   | guide | section heading | disposition (kept / moved → card id / merged into card id) |
3. **Preserve every anchor that other files link to.** Check with
   `rg -n 'experiment_database_and_config_truth_guide.md#|experiment_harness_guide.md#|unified_v2_architecture.md#|composable_model_builder_specification.md#' --glob '!docs/context/**'`.
   - AGENTS.md links `#0-agent-protocol--the-seven-rules` and `#2-the-two-databases`: those
     headings and their content (**the seven rules stay verbatim in the DB guide**) must remain.
   - List each preserved anchor in the ledger.
4. **Do not change a rule's meaning while moving it.** If a guide contradicts the code, the code
   wins on the card. Record the discrepancy in the ledger and in your report; don't silently fix
   it.

## 5. Stale check

Write `scripts/context_stale.sh` (bash, no new dependencies). For every `docs/context/**/*.md`
card with frontmatter, it warns (`STALE <card>: <reason>`) when:
- any listed `sources` file has changed between `verified_sha` and `HEAD`
  (`git diff --quiet <sha> HEAD -- <file>`);
- any listed `sources` file no longer exists;
- any listed `symbols` entry is not found (`grep -w`) in any of that card's `sources`.

Requirements:
- It exits 0 whether or not anything is stale.
- It prints a one-line summary.
- `--strict` makes it exit 1 on a stale card, for manual use only.
- Hook it into `./scripts/todo.sh check` as a **warning only**: the check must still pass when
  cards are stale.
- Test it by hand: make a card stale in a scratch commit, show the warning, then revert. Put the
  transcript in your report.

## 6. Wiring and report

- Add one `AGENTS.md` §1 table row:
  `docs/context/INDEX.md` | "Before reading any large guide: short cards for harness, persistence,
  W2 components, fit pipeline". Keep AGENTS.md under 22,000 bytes.
- Add a pointer in `docs/README.md`.
- Write `docs/architecture/context_cards_report.md` with:
  - the card list with line counts (all ≤ 150);
  - the slimmed guides with before/after line counts;
  - the anchor checks;
  - the list of `UNVERIFIED:` items;
  - code-vs-guide discrepancies found;
  - the stale-check transcript;
  - `./scripts/todo.sh check` output.
- Add a dated `@pi` line to the TODO 037 Work Log. Commit, push, print the head SHA, then print
  exactly `CTX_BUILD_DONE` on its own line and stop.
- **Review loop:** the manager sends findings (`docs/architecture/context_cards_review.md`). Fix
  them, append "Fix round N" to your report, push, print `CTX_FIX<N>_DONE`, and stop.

## 7. If blocked

Write the problem under "Blocked" in your report, push, print `CTX_BLOCKED` with a one-line
reason, and stop.
