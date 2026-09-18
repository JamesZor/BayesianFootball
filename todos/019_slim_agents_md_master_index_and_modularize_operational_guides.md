# 019 — Slim AGENTS.md master index and modularize operational guides

| Field | Value |
|---|---|
| ID | 019 |
| Title | Slim AGENTS.md master index and modularize operational guides |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-18 |
| Updated | 2026-09-18 |
| Related Files / Commits / PRs | `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `docs/guides/`, `docs/setup/`, `docs/architecture/` |

## Context & Problem Statement

Investigation into live context windows revealed that the Antigravity (AGY) rule injector imposes a
hard limit (~24 KiB) per rule file. `AGENTS.md` has grown to 45.8 KiB (823 lines). Consequently, the
rule injector silently truncates **21,863 bytes** starting at line 390.

This silent truncation means Sections 6.1 through 13—including MatchDay live & replay consoles, Turing
AD-safety, extension recipes, test tiers, compute topology, remote execution protocols, and experiment
database conventions—are completely omitted from the agent's working prompt. Furthermore, for Claude
Code and Pi, loading an oversized 45 KiB context file consumes ~12,000–15,000 tokens on every single turn.

We must slim `AGENTS.md` down to a lean, punchy master index (<20 KiB, strictly <22,000 bytes) and
modularize detailed operational sections into dedicated `docs/guides/` files that agents read on-demand.

## Acceptance Criteria

- [ ] **Byte-Size Ceiling**: Slim `AGENTS.md` to strictly under 22,000 bytes (target ~15–19 KiB) so it
  injects 100% cleanly without truncation in Antigravity, while minimizing prompt token overhead in
  Claude Code and Pi.
- [ ] **Retained Core Index**: Preserve essential high-frequency context in `AGENTS.md`:
  - Quick Reference table & task tracking rules
  - Unified V2 Architecture layer table
  - The Two Databases topology overview (`betdb` vs `mcmc_experiments`)
  - Unified V2 production pipeline call structure
  - Prototyping rules (`current_development/` vs `experiments/`)
  - Model generations summary table (Scottish Lower 24/26)
  - Non-negotiable rules (AD safety summary, credentials, database separation)
- [ ] **Modular Operational Guides**: Extract detailed operational instructions into dedicated files
  and cross-link them from `AGENTS.md`:
  - Section 7 (MatchDay consoles, ports 8085/8086, Gödel terminal workspace, dynamic slate re-solver) ->
    [`docs/guides/matchday_console_guide.md`](../docs/guides/matchday_console_guide.md)
  - Section 8 (AD-safety & ReverseDiff tape rules) -> consolidate into
    [`docs/turing_ad_performance_guide.md`](../docs/turing_ad_performance_guide.md)
  - Section 9 (Extension recipes for leagues, components, features, metrics) ->
    [`docs/guides/extension_recipes.md`](../docs/guides/extension_recipes.md)
  - Section 10 (Test execution tiers, unit, parallel, replay suites, T007) ->
    [`docs/guides/testing_and_verification_guide.md`](../docs/guides/testing_and_verification_guide.md)
  - Section 11 & 12 (Infrastructure, compute nodes, remote REPL & tmux protocols) -> consolidate with
    [`docs/setup/agy_remote_execution_guide.md`](../docs/setup/agy_remote_execution_guide.md) and
    [`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](../docs/architecture/ai_agent_infrastructure_and_execution_context.md)
  - Section 13 (Experiment DB & config truth protocol) -> consolidate with
    [`docs/guides/experiment_database_and_config_truth_guide.md`](../docs/guides/experiment_database_and_config_truth_guide.md)
- [ ] **Single Source of Truth**: Keep `CLAUDE.md` and `GEMINI.md` as minimal pointers (<10 lines) to
  `AGENTS.md` to eliminate drift across agent harnesses.
- [ ] **Automated Regression Guard**: Add an automated check (e.g. in `./scripts/todo.sh check` or a test)
  verifying `filesize(AGENTS.md) < 22_000` bytes so future edits never silently exceed the rule injector
  truncation boundary.
- [ ] `./scripts/todo.sh check` passes.

## Ideas & Candidate Solutions

- **Pointer Architecture**: Every modular guide should have a clear 1-2 sentence description and link
  in Section 1 (Quick Reference) of `AGENTS.md`.
- **Automated Size Gate**: Add a lightweight shell check in `./scripts/todo.sh check` or
  `test/test_todo.sh` asserting `AGENTS.md` file size.

## Work Log & Progress

- [2026-09-18 @antigravity] Aligned task design via `/grill-me`. Created Task 019, initialized branch
  `refactor/slim-agents-md-index` and worktree `BayesianFootball-docs-refactor`.
- [2026-09-18 @claude] Claimed in session `claude_docs_refactor`, worktree
  `/home/james/bet_project/.worktrees/BayesianFootball-docs-refactor`.

## Verification & Findings

Not run yet.
