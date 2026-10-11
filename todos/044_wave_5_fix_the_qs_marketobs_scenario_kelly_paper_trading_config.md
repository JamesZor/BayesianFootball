# 044 — Wave 5: fix the qs_marketobs + scenario Kelly paper-trading config for 26/27

| Field | Value |
|---|---|
| ID | 044 |
| Title | Wave 5: fix the qs_marketobs + scenario Kelly paper-trading config for 26/27 |
| Status | COMPLETED |
| Priority | P2 |
| Assignee | claude |
| Created | 2026-10-10 |
| Updated | 2026-10-11 |
| Related Files / Commits / PRs | `experiments/claude_qs_wave5_config_brief.md` (`5c868bc8`); `experiments/scotland/09_qs_marketobs_wave5_config/`; wave 4 = TODO 043 |

## Context & Problem Statement

Wave 4 (TODO 043) found scenario Kelly helps qs_marketobs at T−25m but left the basket and λ at
the canonical production values (λ 23, H/D/A + U2.5), which were chosen with m12. Wave 5 picks
the basket and λ on 24/25 only (λ matched to m12 canonical's 24/25 MDD), freezes them in
FREEZE.md, then scores the frozen config on 25/26 against m12 × canonical × λ 23 and writes the
paper-trading config for 26/27. No fits, no DB writes, src frozen. Julia on mcmc-beast only.

## Acceptance Criteria

- [x] Phase 0: wave-5 runner reproduces bit-identical m12 × canonical × T−25m (2.773493294562897 / 1132) and qs × scenario_kelly × T−25m (2.154029744590488 / 952); per-season MDDs match to 0.01 pp; DB audit unchanged.
- [x] Phase 1: per-direction table, λ curves for B1–B4 (+ explored baskets), ruling-4 sensitivity, FREEZE.md committed and pushed before any 25/26 wave-5 number exists.
- [x] Phase 2: 25/26 (and 24/25) scoring of frozen, m12, fallback and every candidate; T−6h rows; ruling-8 decision; REPORT.md and PAPER_TRADING_CONFIG.md with traceable numbers; CSVs committed.
- [x] DB audit unchanged around every runner; beast panes closed; everything pushed.

## Ideas & Candidate Solutions

- Baskets filter legs before the joint solve (ruling 4). The joint scenario matrix is built once per day over every canonical leg; a basket solves on a column subset (same draws, same scorelines), so baskets and λ values are cheap to sweep. λ only changes the overlay, so each basket is solved once and simulated at 12 λ.

## Work Log & Progress

- [2026-10-10 @claude] Claimed as the wave-5 Claude CLI implementer in worktree `.worktrees/BayesianFootball-qs-experiment`, branch `exp/qs-goal-vs-market`; manager is the launching Claude session. Phase 0 started.
- [2026-10-10 @claude] Phase 0 PASS (`21f3514f`, beast pane %458): both wave-4 T−25m cells bit-identical, per-season MDDs within 0.01 pp, 24/25-only runs equal the full runs' first season, basket subset equivalence holds, seal refuses 25/26, DB audit unchanged. Stopped with W5_HANDOVER.

- [2026-10-10 @claude] Phase 1 done on 24/25 only (runner `6d779be2`; attempt 1 at `ccddaa79` stopped on a B4-filter bug before any basket was scored). 6 baskets looked at; ruling 6 selects B1_canonical at λ 12 (g 17.520×10⁻³, MDD −13.07% vs target −13.60%). FREEZE.md pushed in `04fdba71`; stopped with W5_HANDOVER.
- [2026-10-11 @claude] Manager released FREEZE. Phase 2 runner `716c84a1` (only `release!` caller), results `0f600b92`; REPORT.md and PAPER_TRADING_CONFIG.md written; beast pane %458 closed. Closed.

## Verification & Findings

- 25/26 at T−25m: frozen B1 λ 12 wealth 1.624, g 9.70×10⁻³ [1.90, 21.89] (circular [−0.11, 19.53]), MDD −19.26%; m12 canonical 1.327, g 5.66, MDD −16.15%; paired d +4.04 [−10.52, 14.54]. Fallback B1 λ 23 g 5.57, MDD −11.22%.
- **Ruling 8: paper-trade the frozen config** (g > 0; MDD inside the −20.39% bound by 1.13 pp).
- The selected basket also ranked first on 25/26 (B1 > B3 > B5 > B6 > B2 > B4). The 24/25 per-direction pattern flipped (away lost, draw and U2.5 earned), so the EDA KEEP rule (B4) was worst.
- λ 12's MDD match was in-sample: on 25/26 it ran 3.1 pp deeper than m12.
- At T−6h the config trails m12 on both halves; it is a T−25m config.
- DB audit unchanged around every runner. FREEZE (`04fdba71`) precedes the first `release!` (`716c84a1`) in git order.

## Follow-ups

- src/MatchDay integration for paper trading (PAPER_TRADING_CONFIG.md §integration): weekly market-rate inversion feed, live qs_marketobs fits, ScenarioKelly day-problem adapter in src, legs filtered by basket before the joint solve, and daily SlateDrawdown bookkeeping. Needs a new TODO.
- Paper-trade alongside m12 canonical. Watch drawdown against −20.39%; the λ 23 fallback stays ready.
- A finer λ grid or a continuous match would remove the B1/B3 grid artefact (REPORT §selection). This only matters at the next re-selection, on 25/26 + 26/27 data.
- T−6h needs its own selection if an earlier bet time is wanted.
- Phase 0/1 runner logs are untracked (`logs/` is gitignored). Track them, or leave them on the beast at `/root/BF_runs/logs/qs_wave5/`.
