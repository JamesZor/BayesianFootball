# 037 — Context cards: short linked docs for harness, persistence, W2 components and fit pipeline

| Field | Value |
|---|---|
| ID | 037 |
| Title | Context cards: short linked docs for harness, persistence, W2 components and fit pipeline |
| Status | ACTIVE |
| Priority | P2 |
| Assignee | pi |
| Created | 2026-09-28 |
| Updated | 2026-09-28 |
| Related Files / Commits / PRs | TODO 036 (must merge first), TODO 030, TODO 031; brief `experiments/pi_context_cards_prompt.md`; branch `docs/context-cards` |

## Context & Problem Statement

Agents re-derive the harness, persistence, W2 component and fit-pipeline structure from large
files every session. The big guides (harness guide, experiment DB guide, builder specification,
unified V2 architecture) each cost a lot of context to load, even when only one part is needed.

The human wants **short, linked context cards**:
- one concept per file, loadable on its own;
- ASCII flow diagrams and interface tables between modules and functions;
- enough detail to modify or extend the code safely.

The existing guides are slimmed onto these cards, and nothing may be lost.

## Acceptance Criteria

Design agreed with the human in a grill-me session on 2026-09-28.

- [ ] Starts only after TODO 036 (`fix/harness-klm`) is merged into `feat/w2-tier-components`,
      and documents that merged state.
- [ ] `docs/context/<area>/` for four areas: `harness`, `persistence`, `models` (W2 components) and
      `pipeline` (builder → fit → latents → evaluation).
  - [ ] Each area has an `INDEX.md`, and there is a top-level `docs/context/INDEX.md` with the
        module-interface ASCII map.
  - [ ] Every card is **≤ 150 lines**, uses the fixed template, and has YAML frontmatter:
        `id`, `area`, `sources`, `symbols`, `verified_sha`, `related`.
  - [ ] Every card has a short "Why" section that links the deciding TODO or README decision.
  - [ ] No line numbers. Every signature, default and threshold is copied from the code.
- [ ] Four guides are slimmed onto the cards with no information loss, recorded in a slimming
      ledger:
  - `docs/guides/experiment_harness_guide.md`;
  - `docs/guides/experiment_database_and_config_truth_guide.md` (the anchors AGENTS.md links must
    survive);
  - `docs/architecture/composable_model_builder_specification.md`;
  - `docs/architecture/unified_v2_architecture.md`.
- [ ] `scripts/context_stale.sh`, wired into `./scripts/todo.sh check` as a **warning only**. It
      warns when a card's source changed since its `verified_sha`, or when a listed symbol no
      longer exists.
- [ ] One AGENTS.md §1 row, keeping AGENTS.md under 22,000 bytes.
- [ ] Built by pi on `ag/gemini-3.8-flash-high` (solo, no subagents).
- [ ] Reviewed by Claude CLI (Opus 5.5) against the code, with at most 2 fix rounds; the manager
      merges.

## Ideas & Candidate Solutions

- Rejected:
  - one large doc per area, which defeats cheap loading;
  - a stale check that fails `todo.sh check`;
  - per-function hash tracking, which is too much machinery.

## Work Log & Progress

- [2026-09-28 @claude] Created. The design was agreed with the human (grill-me). Launch waits for
  the TODO 036 merge.

## Verification & Findings

Not run yet.
