# Wave 5 decisions (TODO 044)

Brief: `experiments/claude_qs_wave5_config_brief.md` (`5c868bc8`). Implementer: Claude CLI; manager: the
launching Claude session. Scope set by James.

## 2026-10-10 — Phase 0: manager rulings (binding, copied verbatim in substance)

1. **Halves.** Selection = slates dated ≤ 2025-05-03 (24/25). Evaluation = slates after it (25/26).
   Each half starts at bankroll 1.0; MDD is measured within the half.
2. **Comparison cells (James):** the candidate **qs_marketobs + scenario Kelly at its MDD-matched λ**,
   and the benchmark **m12 × canonical × λ 23** (unchanged: per-match allocator, canonical basket).
   Also computed: the **fallback** = qs SK, canonical basket, λ 23, legs restricted as in ruling 4.
   The decision rule needs it.
3. **λ target = m12 canonical's 24/25 MDD at T−25m** (recomputed; expected −13.60%). Grid
   λ ∈ {4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 30}. Choose the **smallest λ such that the 24/25 MDD
   ≤ target (in magnitude) at that λ AND at every larger grid λ** (monotone envelope). Report the full
   λ curve: MDD, g, mean exposure, budget-binding share, cap-binding share. If the 0.99 budget or the
   0.25 slate cap binds on most days at the chosen λ, say so: λ is then no longer the active dial.
4. **A basket is the set of directions eligible for the joint solve.** Each book's legs are filtered to
   trust > 0 **before** the scenario solve. Directions outside the canonical trust table get trust
   **0.25**; canonical tiers otherwise fixed (Home, U2.5 0.35; Draw, Away 0.25). λ is the dial.
   Sensitivity on 24/25 only: canonical basket at λ 23 solved wave-4 style (all legs, then trust) vs
   basket-only legs.
5. **Candidate baskets (James):** B1 canonical (H/D/A + U2.5); B2 1X2 only; B3 1X2 + OU2.5 both sides;
   B4 data-driven from the 24/25 per-direction table. James asked for "play around": other baskets may
   be tried on 24/25. Every basket tried is logged here with its 24/25 g and MDD; the report counts them
   as the number of looks.
6. **Selection rule (written before any basket is scored):** the basket with the highest 24/25 mean
   daily log growth at its own MDD-matched λ. If a smaller basket (fewer directions) comes within
   0.5 ×10⁻³/day of it, take the smaller one.
7. **Seal.** No wave-5 code path computes any 25/26 slate before FREEZE.md is committed, pushed and
   released by the manager. Phase 1 code filters by date before building books, and asserts it. The
   wave-4 full-period numbers are already public, so 25/26 is unseen *by the selection rule*, not
   unseen by anyone; the report says so.
8. **Decision rule (James):** the frozen config stands unless its 25/26 g ≤ 0 (point estimate) or its
   25/26 MDD is worse than 1.5 × target. In either case, paper-trade the fallback. d vs m12 is reported
   descriptively only.
9. **T−6h (James):** apply the T−25m-frozen config and m12 canonical at T−6h on both halves. Report only;
   no separate calibration.
10. **Uncertainty:** as wave 4 (8-week noncircular blocks within season, 999 reps, 90%, seed 20261009),
    plus a circular-block column; paired d vs m12 on the same days.

## 2026-10-10 — Phase 0: implementation decisions (before any number)

- **Reuse.** `l00_wave5.jl` includes wave 4's `l00_phase0_inventory.jl`, `l01_scenario_kelly.jl` and
  `l02_phase3_grid.jl` unchanged (model loading by UUID, snapshot builder, scenario adapter, ledgers,
  bootstrap). No wave-4 function is copied. New code: date seal, basket leg filter, λ sweep, per-half
  MDD.
- **Phase 0 gate runs the full period for the two wave-4 configs only.** The brief's gate numbers are
  full-period (99 slates); reproducing them computes 25/26 slates for already-public wave-4 configs.
  This is the only full-period path, guarded by an explicit `gate = true` flag; no wave-5 candidate
  config or basket touches 25/26 before release. The per-season MDDs (including 25/26) come from the
  gate's full-period daily path with the bankroll reset to 1.0 at each season start (opening 1.0
  counts as the first peak, the `portfolio_summary` convention).
- **Half filter.** The odds snapshot is filtered to the half's fixtures (match date from the saved
  datastore) *before* books are built, and every built book's date is asserted to lie in the half.
  The raw-price snapshot itself is rebuilt from the cache over all 710 fixtures and compared with
  wave 4's committed Phase 0 CSV, as wave 4 did. It holds quoted prices only (no model price, stake
  or result), so building it computes no 25/26 slate.
- **Seal.** `QSWave5.half_odds(..., :evaluation)` errors unless `release!` was called, and `release!`
  requires `FREEZE.md` to exist in the committed tree. Only the Phase 2 runner may call it.
- **Basket mechanics.** Books are built once per (model, snapshot, half) with the canonical BookSpec
  (all canonical markets, no excision), exactly as wave 4. So the per-match BakerMcHale `k_shrink` is
  the wave-4 value for every basket; the basket changes only which legs enter the joint solve and are
  staked. Per day, the joint scenario problem is built once over every canonical leg
  (`SC.scenario_problem`, unchanged); a basket solves `ScenarioKelly` on the column subset of its legs.
  This is identical to building the problem on leg-filtered books, because draws and scoreline rows do
  not depend on the legs (verified in Phase 0 on several days: the R matrices must be `==`). Books
  with no eligible leg are dropped from the day; a day with no eligible leg is kept as an empty slate
  so every basket is scored on the same days (g is per slate).
- **λ sweep.** The joint solve does not depend on λ, so each basket is solved once and simulated at
  every grid λ with `PF._with_lambda` (internal helper the brief allows).
- **Binding shares.** Budget-binding day: joint raw total ≥ 0.99 − 1e-3 (the barrier stops 1e-5 to
  1e-3 short of the bound; wave 4 REPORT §joint diagnostics). Cap-binding day: `capped` in the daily
  state (the 0.25 FixedCap rescaled the slate). Also reported: risk-active day, `k_risk < 1`.
