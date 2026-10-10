# Wave 5 FREEZE: qs_marketobs + scenario Kelly paper-trading config (TODO 044)

Frozen on 2026-10-10. Only the 24/25 selection half (49 slates, 2024-08-03 to 2025-05-03) was used,
at T−25m (90 min staleness), with 2% commission. No wave-5 code has computed a 25/26 slate. The
wave-4 full-period numbers were already public, so 25/26 is unseen *by the selection rule*, not
unseen by anyone. Source: `results/phase1/phase1_selection.csv`, `phase1_lambda_curve.csv`; runner
`6d779be2`; decisions in `DECISIONS_LOG.md`.

## Frozen config: B1_canonical at λ 12

- **Model:** qs_marketobs, namespace `scottish_lower_qs_wave2_2426`, run
  `81fdf817-f82d-4f66-9bfd-5f456c4fe29f` (saved fit, loaded via wave 4 `P3.goal_model`).
- **Basket (ruling 4):** the legs that enter the joint daily solve are the directions with trust > 0:

  | Direction | Trust |
  |---|---:|
  | 1X2 Home | 0.35 |
  | Under 2.5 | 0.35 |
  | 1X2 Draw | 0.25 |
  | 1X2 Away | 0.25 |
  | anything else | 0 (never enters the solve) |

- **λ = 12** (SlateDrawdown).
- **MDD target** = m12 × canonical × λ 23 24/25 MDD at T−25m = **−13.595231252994628%**.
  B1's 24/25 MDD at λ 12 is −13.066%; the target holds at λ 12 and at every larger grid λ.

```julia
import BayesianFootball as BF
const PF = BF.Portfolio
const MD = BF.MatchDay

# Books: built once per (model, snapshot) with every canonical market, exactly as wave 4.
# The per-match BakerMcHale k_shrink comes from these books; the basket does not change it.
book_spec = PF.BookSpec(markets = MD.canonical_markets(),   # 1X2, BTTS, OU0.5–3.5
                        price = PF.DeArb())
# Defaults that BookSpec fills in: allocator = KellyLogUtility(), shrink = BakerMcHale(),
# exec = ExecutionConfig(commission = PerBetCommission(0.02), max_selection_stake = 0.50,
#        min_selection_stake = 1e-4, budget = 0.99, barrier_mu = 1e-6, require_complete_markets = true)

trust = PF.TieredTrust(Dict(("1X2", 0.0, :home) => 0.35,
                            ("OverUnder", 2.5, :under) => 0.35,
                            ("1X2", 0.0, :draw) => 0.25,
                            ("1X2", 0.0, :away) => 0.25); default = 0.0)

policy = PF.PolicySpec(trust = trust,
                       risk = PF.SlateDrawdown(12.0),
                       cap = PF.FixedCap(0.25),
                       filter = PF.KeepAll(),
                       grouping = PF.DailySlate())

# Joint daily solve (wave 4 scenario adapter, unchanged):
alloc = PF.ScenarioKelly()            # n_scenarios = 1000, seed = 20261010 (defaults)
# per DailySlate: problem = QSBacktestScenario.scenario_problem(alloc, slate, grids, groups; native = true)
# legs kept = those with PF.book_trust_for(trust, sel) > 0; columns = problem.columns[i][kept]
# a = PF.allocate(alloc, problem.weights, problem.R[:, columns], book_spec.exec).a
# stake path: a -> ×trust -> ×k_shrink -> ×k_risk (SlateDrawdown λ 12) -> FixedCap(0.25) -> KeepAll
```

Implementation: `QSWave5.day_problems`, `QSWave5.solve_days(days, s -> in_basket(B1, s))` and
`QSWave5.simulate_slates(basket_policy(B1.trust, 12.0), slates, report)` in `l00_wave5.jl`.

24/25 result at λ 12: final wealth 2.35957, g 17.520×10⁻³/day, MDD −13.07%, 529 bets, mean
exposure 7.6%. Budget binds on 12% of days, the 0.25 cap on 0%, and the drawdown constraint is
active on 65%, so λ is the active dial.

## Fallback config (ruling 2; paper-traded if ruling 8 fires)

The same model, books, basket and trust table as above, at **λ 23** (`PF.SlateDrawdown(23.0)`), with
legs restricted to the basket before the solve (ruling 4). 24/25: wealth 1.62456, g 9.903×10⁻³,
MDD −7.14%, 529 bets.

## Benchmark (ruling 2)

m12 × canonical × λ 23 (`MD.canonical_scottish_lower_policy()`, per-match allocator, canonical
books): namespace `scottish_lower_joint_player_2426`, run `132df5c2-c742-4e95-8693-3aeb2b2cbaef`.
24/25: wealth 2.08953, g 15.040×10⁻³, MDD −13.60%, 575 bets.

## Looks and selection

- **Baskets looked at: 6** on 24/25 (B1–B6, listed with g and MDD in DECISIONS_LOG), each at
  12 grid λ values. B4 came from the EDA §2 KEEP rule (Home, Away). B5 (all 13 directions) and B6
  (Home + U2.5) were explorations fixed before any basket was scored.
- **Ruling 6:** B1 had the highest g at its matched λ (17.520 vs B3 17.087, B2 14.742, B5 13.826,
  B4 12.943, B6 6.062 ×10⁻³). No smaller basket came within 0.5×10⁻³/day.
- **Caveat:** the grid decides between B1 and B3. B3 beats B1 at every common λ, but it misses the
  target at λ 12 by 0.19 pp (−13.78%).

## Decision rule for Phase 2 (ruling 8, fixed)

The frozen config stands unless its 25/26 g ≤ 0 (point estimate) or its 25/26 MDD is worse than
1.5 × target = **−20.39%**. In either case, paper-trade the fallback. d vs m12 is descriptive only.
