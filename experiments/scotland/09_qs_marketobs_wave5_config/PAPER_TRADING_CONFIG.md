# Paper-trading config for 26/27 (wave 5, TODO 044)

**Decision (ruling 8, `results/phase2/phase2_decision.csv`): paper-trade the frozen config.** On 25/26 it scored g 9.70×10⁻³ > 0 with MDD −19.26%, inside the −20.39% bound. The config is frozen in `FREEZE.md` (`04fdba71`) and copied exactly below. See `REPORT.md` for the evidence.

## The config

| Item | Value |
|---|---|
| Goal model | qs_marketobs (QS goal model with the market-rate observation). The backtest fit is run `81fdf817-f82d-4f66-9bfd-5f456c4fe29f`, namespace `scottish_lower_qs_wave2_2426`. Live use needs fresh fits (below). |
| Bet time | **T−25m** before kick-off (backtest snapshot `T-25m_s90`: last-traded price, at most 90 min stale). Not T−6h: the config trails m12 there (REPORT §T−6h). |
| Books | `PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb())` with defaults: `KellyLogUtility`, `BakerMcHale` shrink, `ExecutionConfig(commission = PerBetCommission(0.02), max_selection_stake = 0.50, min_selection_stake = 1e-4, budget = 0.99, barrier_mu = 1e-6, require_complete_markets = true)`. Books are built with every canonical market, so `k_shrink` is unchanged. |
| Basket (legs entering the solve) | 1X2 Home, 1X2 Draw, 1X2 Away, Under 2.5. Every other leg is dropped **before** the joint solve. |
| Trust | `TieredTrust`: Home 0.35, U2.5 0.35, Draw 0.25, Away 0.25, default 0. |
| Allocator | `PF.ScenarioKelly()`, one joint solve per day over all basket legs of all fixtures that day: 1000 scenarios, seed 20261010, budget 0.99, per-leg cap 0.50. |
| Risk | `PF.SlateDrawdown(12.0)`, i.e. **λ 12** |
| Cap / filter / grouping | `PF.FixedCap(0.25)` / `PF.KeepAll()` / `PF.DailySlate()` |
| Stake pipeline | joint a → ×trust → ×k_shrink → ×k_risk(λ 12) → FixedCap 0.25 → KeepAll. Stakes are fractions of the current bankroll. |
| Commission | 2% on net winnings |

**Fallback**, ready to switch to: the same config at **λ 23** (`SlateDrawdown(23.0)`). Its 25/26 g was 5.57×10⁻³ with MDD −11.22%.

**Benchmark** to paper-trade alongside: m12 × `MD.canonical_scottish_lower_policy()` (per-match allocator, λ 23).

**Watch:** cumulative drawdown against −20.39% (1.5 × the m12 24/25 MDD target). The 25/26 backtest came within 1.13 pp of it.

## What the later src/MatchDay integration needs

Nothing here is implemented. The experiment code lives in `experiments/scotland/08_…` and `09_…`, and src was frozen.

1. **Weekly market-rate inversion feed.** qs_marketobs observes market-implied team rates. Live use needs a weekly job that inverts the latest Betfair prices into those rates (the TODO 039 KL inversion) and stores them where the fit can read them, with no look-ahead past the fit date.
2. **Live QS fits.** A scheduled refit of qs_marketobs on data up to each slate, at the same sampler settings as the backtest fit (4 chains × 1000 draws). Saved fits must be loadable by run UUID, the way MatchDay loads m12 today, and fold staleness must be handled (cf. the m12 live-fold card injection).
3. **Scenario adapter in src.** The joint daily problem builder is wave 4's `QSBacktestScenario.scenario_problem(...; native = true)`. It builds per-fixture posterior draw grids, uses the witness groups, gives one joint scenario matrix per day, and feeds `PF.allocate(ScenarioKelly(), …)`. This has to move into `BF.Portfolio`/`MatchDay` as a supported allocator path. Today MatchDay only has the per-match allocator.
4. **Basket-filtered legs.** Filter each book's legs to trust > 0 **before** the joint solve: a column subset of the day problem. Wave 5 did this with `QSWave5.solve_days` plus `filtered_book`, and showed it equals rebuilding the problem on filtered books. MatchDay's policy path currently applies trust after allocation.
5. **Daily bookkeeping.** Slate grouping by day, `SlateDrawdown` state carried across days with the current bankroll, and the ledger fields the backtest used: stake, P&L, CLV against the T−20m→kick-off TWA close. These are needed for the drawdown watch and the ruling-8-style review.
