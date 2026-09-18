# 019 — Slim AGENTS.md master index and modularize operational guides

| Field | Value |
|---|---|
| ID | 019 |
| Title | Slim AGENTS.md master index and modularize operational guides |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-18 |
| Updated | 2026-09-18 |
| Related Files / Commits / PRs | `AGENTS.md`, `scripts/todo.sh`, `docs/guides/`, `docs/architecture/unified_v2_architecture.md`, branch `refactor/slim-agents-md-index` |

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

- [x] **Byte-Size Ceiling**: Slim `AGENTS.md` to strictly under 22,000 bytes (target ~15–19 KiB) so it
  injects 100% cleanly without truncation in Antigravity, while minimizing prompt token overhead in
  Claude Code and Pi.
- [x] **Retained Core Index**: Preserve essential high-frequency context in `AGENTS.md`:
  - Quick Reference table & task tracking rules
  - Unified V2 Architecture layer table
  - The Two Databases topology overview (`betdb` vs `mcmc_experiments`)
  - Unified V2 production pipeline call structure
  - Prototyping rules (`current_development/` vs `experiments/`)
  - Model generations summary table (Scottish Lower 24/26)
  - Non-negotiable rules (AD safety summary, credentials, database separation)
- [x] **Modular Operational Guides**: Extract detailed operational instructions into dedicated files
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
- [x] **Single Source of Truth**: Keep `CLAUDE.md` and `GEMINI.md` as minimal pointers (<10 lines) to
  `AGENTS.md` to eliminate drift across agent harnesses.
- [x] **Automated Regression Guard**: Add an automated check (e.g. in `./scripts/todo.sh check` or a test)
  verifying `filesize(AGENTS.md) < 22_000` bytes so future edits never silently exceed the rule injector
  truncation boundary.
- [x] `./scripts/todo.sh check` passes.

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
- [2026-09-18 @claude] Moved content out of `AGENTS.md` verbatim (moved, not rewritten):
  §7 + L5 note → `docs/guides/matchday_console_guide.md` (old §7.N = new §N); §9 →
  `docs/guides/extension_recipes.md`; §10 → `docs/guides/testing_and_verification_guide.md`;
  §6.1–6.2 + Exp 06 table → `docs/guides/model_generations_guide.md`; §2 per-layer narrative →
  `docs/architecture/unified_v2_architecture.md`; §13's seven rules → new §0 of the experiment-DB
  guide; §11's `-t 8` / one-grid rules → infra doc §2; §8 reference points → AD guide header.
  §3.1–3.3 detail was already duplicated in the experiment-DB guide §2, so `AGENTS.md` keeps only
  the comparison table. Rewrote `AGENTS.md` as a 9-section index with a guide table and the
  `todos/` protocol in §1. Added `check_agents_size` (limit `< 22000`) to
  `./scripts/todo.sh check`. Repointed inbound `AGENTS.md §7.x` / `§13.2` references in `src/`,
  `current_development/`, `experiments/` and `docs/` (§3 kept its number). Indexed the new guides
  in `docs/README.md`.

## Verification & Findings

- `wc -c AGENTS.md` → **19,673 bytes** (was 45,010; 322 lines, was 809). Under the 22,000-byte
  gate and the 20 KiB target, leaving ~2.3 KB headroom.
- `./scripts/todo.sh check` → `OK: 19 task(s); metadata, template and registry agree; AGENTS.md
  19673 bytes (< 22000).`
- Gate failure path exercised: padding `AGENTS.md` to 41,673 bytes made `check` exit 1 with
  `todo: AGENTS.md is 41673 bytes (limit: < 22000); ...`; file restored afterwards.
- Every relative markdown link (and `#anchor`) in `AGENTS.md`, the five new guide files, the
  experiment-DB guide and the AD guide resolves. One pre-existing broken `file:///` link in
  `docs/turing_ad_performance_guide.md` §11 (absolute path to the main checkout) was left as is.
- `CLAUDE.md` / `GEMINI.md` are unchanged 8-line pointers (< 10 lines).
- **Correction made while consolidating §8:** the old `AGENTS.md` bullet "use `findall(!isnan, …)`
  to split the xG vs goals routes" contradicted Rule 3 of the AD guide (mask, don't subset) and
  the quoted "~0.64 ms" target was superseded by the guide's < 0.1 ms bar. The new §7 digest
  follows the guide; the AD guide's header records both reference points.
