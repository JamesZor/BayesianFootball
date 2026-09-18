# WORK PACKAGE — Slim AGENTS.md Master Index and Modularize Operational Guides (Task 019)

> **Assignee:** `claude` (Claude Code / Opus 5)  
> **Session Target:** `claude_docs_refactor` in tmux  
> **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-docs-refactor`  
> **Branch:** `refactor/slim-agents-md-index`  
> **Canonical Guide:** `AGENTS.md`

---

## 1. Executive Summary & Problem Statement

Investigation into live agent context windows revealed that the Antigravity (AGY) rule injector imposes a hard limit (~24 KiB) per rule file. `AGENTS.md` has grown to **45.8 KiB (823 lines)**. Consequently, the rule injector silently truncates **21,863 bytes** starting at line 390!

This silent truncation means Sections 6.1 through 13—including MatchDay live & replay consoles, Turing AD-safety, extension recipes, test tiers, compute topology, remote execution protocols, and experiment database conventions—are completely omitted from the agent's working prompt. Furthermore, for Claude Code and Pi, loading an oversized 45 KiB context file wastes ~12,000–15,000 tokens on every turn.

**Objective:**
Slim `AGENTS.md` down to a lean, punchy master index (<20 KiB, strictly <22,000 bytes) and modularize detailed operational sections into dedicated `docs/guides/` files that agents read on-demand. Ensure full alignment across the three agent harnesses (Claude Code, Antigravity, Pi).

---

## 2. Structural Blueprint & Refactoring Plan

### A. What Remains in `AGENTS.md` (Target: 15–18 KiB, strictly <22,000 bytes)
Keep high-frequency, non-negotiable architectural anchors:
1. **Quick Reference (§1)**: Fast links to all specialized guides and the `todos/` task tracking protocol.
2. **Unified Architecture (§2)**: Multi-tier layer table (L0 through L5), Generative Rate Calibration summary, and module map.
3. **The Two Databases (§3)**: Clear demarcation between `betdb` (operational, port 5433) and `mcmc_experiments` (experimental, port 5432) with credentials rules.
4. **Unified V2 Pipeline (§4)**: The canonical 5-step production Julia invocation.
5. **Prototyping & Experiment Layout (§5)**: Rules for `current_development/` (loader/runner pairs) vs `experiments/` (completed suites).
6. **Model Generations Summary (§6)**: High-level table of Scottish Lower Gen 1–4 paradigms (keep brief summary, point to experiment READMEs for mathematical derivations).
7. **Core Non-Negotiable Rules**:
   - ReverseDiff AD safety rules (summary of no-branching / no-missing, pointing to guide).
   - Credential safety (never commit/print URLs).
   - Database separation (never write paper trading to `mcmc_experiments`).

### B. What Moves to Dedicated Modular Guides
Extract detailed reference material into clean Markdown guides under `docs/guides/` or `docs/setup/`:
1. **Section 7 (MatchDay live & replay consoles)**:
   - Extract into [`docs/guides/matchday_console_guide.md`](docs/guides/matchday_console_guide.md).
   - Covers: ports 8085/8086, schema isolation, Gödel terminal workspace panels, dynamic slate re-solver, and API surface.
   - Leave a punchy 1-paragraph summary and link in `AGENTS.md`.
2. **Section 8 (Turing AD-Safety & ReverseDiff Tape Optimization)**:
   - Ensure all rules from §8 are consolidated into [`docs/turing_ad_performance_guide.md`](docs/turing_ad_performance_guide.md).
   - Keep a concise 5-bullet summary in `AGENTS.md`.
3. **Section 9 (Extension Recipes)**:
   - Extract into [`docs/guides/extension_recipes.md`](docs/guides/extension_recipes.md).
   - Covers recipes for new tournament segments, components, feature extractors, metrics, calibration laws, and MatchDay sources.
4. **Section 10 (Test Execution Tiers)**:
   - Extract into [`docs/guides/testing_and_verification_guide.md`](docs/guides/testing_and_verification_guide.md).
   - Covers: the 5 test tiers, `run_parallel_tests.jl`, ticket T007 known issue, and 4-tier replay verification.
5. **Sections 11 & 12 (Infrastructure, Compute Nodes, Remote Execution & Tmux Protocols)**:
   - Consolidate into [`docs/setup/agy_remote_execution_guide.md`](docs/setup/agy_remote_execution_guide.md) and [`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](docs/architecture/ai_agent_infrastructure_and_execution_context.md).
   - Keep a compact topology table and remote REPL send-keys cheat-sheet in `AGENTS.md`.
6. **Section 13 (Experiment Database & Config Truth Protocol)**:
   - Consolidate into [`docs/guides/experiment_database_and_config_truth_guide.md`](docs/guides/experiment_database_and_config_truth_guide.md).

### C. Single Source of Truth for Agent Harnesses
- **`CLAUDE.md`**: Must remain a strict, minimal pointer (<10 lines) referencing `AGENTS.md`.
- **`GEMINI.md`**: Must remain a strict, minimal pointer (<10 lines) referencing `AGENTS.md`.
- Ensure all relative markdown links in `AGENTS.md` and the extracted guides are valid and clickable.

### D. Automated Regression Guard
- Add an automated file-size check to `./scripts/todo.sh check` (or a dedicated check function) that asserts:
  ```bash
  AGENTS_SIZE=$(wc -c < AGENTS.md)
  if [ "$AGENTS_SIZE" -ge 22000 ]; then
      echo "ERROR: AGENTS.md exceeds 22,000 bytes ($AGENTS_SIZE bytes). It will be truncated by the rule injector!"
      exit 1
  fi
  ```
- This ensures no future pull request or agent commit can silently bloat `AGENTS.md` beyond the rule injector limit.

---

## 3. Acceptance Criteria Checklist

- [ ] `AGENTS.md` file size is strictly `< 22,000 bytes` (target ~15–18 KiB).
- [ ] The extracted guides exist and contain complete, high-fidelity content:
  - `docs/guides/matchday_console_guide.md`
  - `docs/guides/extension_recipes.md`
  - `docs/guides/testing_and_verification_guide.md`
  - (Consolidated existing guides in `docs/` updated as needed).
- [ ] `CLAUDE.md` and `GEMINI.md` remain valid, clean pointers (<10 lines) to `AGENTS.md`.
- [ ] Automated size check added to `./scripts/todo.sh check`.
- [ ] `./scripts/todo.sh check` passes with all 19 tasks agreed.
- [ ] Update `todos/019_slim_agents_md_master_index_and_modularize_operational_guides.md` with work log and verification details.

---

## 4. Execution Steps

1. Review `AGENTS.md` and identify exact line ranges for each section.
2. Create the target guide files in `docs/guides/`.
3. Refactor `AGENTS.md` down to the high-level master index, replacing long narrative sections with punchy summaries and markdown links.
4. Measure `wc -c AGENTS.md` and verify it is well under 22,000 bytes.
5. Update `./scripts/todo.sh check` to enforce the size limit.
6. Verify `./scripts/todo.sh check`.
7. Commit changes cleanly to `refactor/slim-agents-md-index`.
