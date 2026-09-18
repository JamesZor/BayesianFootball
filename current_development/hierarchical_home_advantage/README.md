# Task 008 Phase 1 — Hierarchical team home advantage, Scottish Lower

> Status: **Phase 1 complete (2026-09-12).** Converged on all three tiers; no proper-score
> gain on any; σ_γ identified only under MultiScaleGRW; turf grounds do not carry a larger
> home effect on the work package's surface list; the 2026-09-12 card would not have changed.
> Every number below is copied from a runner report in `results/`.

## The question

On 2026-09-12 the live account `live_scottish_m12_500` priced the nine-fixture Scottish
League One/Two card from Run 67 (`m12_joint_hybrid_synergy`, one global γ), took six away
legs against home sides the work package lists as playing on synthetic turf, lost all
six, and closed −£45.89. A single scalar γ ≈ 0.159 cannot express a ground that is harder
to visit than the league average.

Phase 1 asks whether the existing `HierarchicalTeamHomeAdvantage`

```
γ_i = γ_base + σ_γ · z_i,    z_i ~ N(0, 1),    γ_base ~ N(0.2, 0.2),    σ_γ ~ N⁺(0, 0.1)
```

(i = home club) is identified on this data, improves out-of-sample proper scores, and
would have changed that card. Phase 2 replaces the club effect with explicit
`is_synthetic_pitch` and travel covariates.

### Hypotheses, written so they can fail

| | claim | evidence that would refute it |
|---|---|---|
| H1 | σ_γ is identified | posterior P(σ_γ < 0.02) no lower than the prior's 0.159 |
| H2 | turf grounds carry a larger home effect | P(mean γ_turf > mean γ_grass) < 0.90 at fold 40 |
| H3 | team HA improves proper scores | paired ΔLogLoss(hier − flat) interval includes 0 |
| H4 | any gain sits on turf home fixtures | Δ on turf home fixtures no more negative than on grass |

## The ladder

Every candidate is its persisted flat-HA control with exactly one slot changed.

| candidate | dynamics | pillars | flat-HA control (UUID) |
|---|---|---|---|
| `m05_joint_production_wealth_hier_ha` | TimeDecay(180) | wealth | `m05_joint_td_raw` (`ed541a7c…`, Exp 06) |
| `m12_joint_hybrid_synergy_hier_ha` | TimeDecay(180) | wealth + shots-RAPM lineup | `m12_hybrid_td_raw` (`132df5c2…`, Exp 06) |
| `m12_joint_hybrid_synergy_grw_hier_ha` | MultiScaleGRW | wealth + shots-RAPM lineup | `m12_joint_hybrid_synergy_grw` (`3a9a4c7e…`, Task 013) |

All three use the two-arm `JointGammaPoissonObservation`. Non-HA components, the
`GroupedCVConfig(history_seasons = 2, dynamics_col = :match_biweek)` splitter and every
audit helper come from Task 013's `l01_loader.jl` by include, so they cannot drift from
what the controls were fitted with.

## Files

| file | role |
|---|---|
| `l01_loader.jl` | config, models, sampler, HA site report, ground effects, surface classification |
| `l02_evaluation.jl` | arms, fold team-map rebuild, turf − grass posterior contrast, split bootstrap |
| `l03_slate.jl` | pinned replay slots, live-ledger read, full-fill settlement, reproduction diff |
| `r01_smoke.jl` | gates G1–G5 on folds 1–2 |
| `r02_production_grid.jl` | 40-fold grid, audit, ground-effect side artefacts, persistence |
| `r03_extend_2627.jl` | extends the TD hybrid to Fold 43 (the fold Run 67 priced the card from) |
| `r04_evaluate.jl` | proper scores, paired bootstrap, H1–H4 |
| `r05_slate_repricing.jl` | 2026-09-12 T−25 counterfactual through the replay engine |

## Contracts

* **Comparability.** One 710-fixture panel (24/25 + 25/26), one book (de-vigged Betfair
  TWA(−20, 0] close), and `m12_hybrid_td_raw` must reproduce its published LogLoss
  0.64337 / ECE 0.0100 before any contrast is printed.
* **Sampler.** Candidates: 4 × (500 warmup + 1,000 retained), δ = 0.80, max depth 10 —
  Task 013's budget. The TD controls were fitted at Exp 06's 4 × (800 + 800), δ = 0.65.
  That changes Monte-Carlo noise, not the target, and is stated beside every score.
* **T003 extraction asymmetry.** A held-out fixture whose home club is absent from its
  fold's `team_map` is priced at γ = 0 by the hierarchical extraction but at γ_global by
  the flat one (`src/models/pregame/builder/engine.jl:629`). 3 of 710 grid fixtures are
  affected; every contrast is reported with and without them. `src` is not patched here —
  see `docs/tickets/T003-home-advantage-population-fallback.md`.
* **Re-pricing filtration.** `MD.price_slate` materialises only `MD.INJECTABLE_KEYS`,
  which omits `:player_lineup_ratings_map`, so a hybrid priced today through it would
  read the played XI. r05 prices through the replay engine's `slot_latents` instead,
  with the book and lineup scrapes sliced at `as_of`.

## Smoke gates (r01)

Folds 1–2, pooled 56/57, 24/25. Gates as defined in `r01_smoke.jl`: G1 gradient
correctness plus Δalloc(hier − flat) = 0; G2 no divergences; G3 R̂ ≤ 1.05 on every site
and every `ha.*` site, ESS ≥ 400; G4 latents; G5 save/load round-trip.

### Gradient audit (identical at both budgets)

| model | fold | clubs | θ (flat) | tape (flat) | grad ms (flat) | alloc B (flat) | Δalloc | RD vs FD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m05 TD | 1 | 23 | 77 (53) | 227 (207) | 0.066 (0.064) | 129,072 (128,560) | +512 | 6.7e-16 |
| m05 TD | 2 | 25 | 83 (57) | 227 (207) | 0.066 (0.064) | 133,344 (132,800) | +544 | 3.0e-16 |
| m12 TD | 1 | 23 | 79 (55) | 252 (232) | 0.073 (0.072) | 175,728 (175,216) | +512 | 4.6e-16 |
| m12 TD | 2 | 25 | 85 (59) | 252 (232) | 0.073 (0.072) | 181,536 (180,992) | +544 | 4.9e-16 |
| m12 GRW | 1 | 23 | 127 (103) | 836 (816) | 0.094 (0.092) | 175,920 (175,408) | +512 | 2.7e-16 |
| m12 GRW | 2 | 25 | 189 (163) | 1,410 (1,390) | 0.115 (0.117) | 181,728 (181,184) | +544 | 4.3e-16 |

The compiled tape is exact at three perturbed points on every row. The hierarchical slot
costs 20 tape instructions and 16 B per club per gradient; G1's Δalloc = 0 therefore
**fails**, and the work package's literal "zero heap allocations" is unreachable for the
flat controls too (129–182 KB). Not tuned away; recorded.

### Sampling at 4 × (400 + 400) — verdict FAIL 0/3

| model | R̂ | ha R̂ | ESS bulk | ESS tail | div | failures | run |
|---|---:|---:|---:|---:|---:|---|---|
| m05 TD | 1.0093 | 1.0084 | 564 | 497 | 0 | G1 | `3a3e1509…` |
| m12 TD | 1.0104 | 1.0104 | 528 | 251 | 0 | G1, G3 tail ESS (non-HA site) | `50bd541b…` |
| m12 GRW | 1.0146 | 1.0098 | 361 | 285 | 0 | G1, G3 ESS (`ha.σ_γ` tail 285) | `59f31574…` |

The flat Task 013 GRW twin failed G3 identically at this budget (bulk 355 / tail 343).

### Sampling at 4 × (500 + 1,000) — re-smoke chosen by the user; G1 only

| model | R̂ | ha R̂ | ESS bulk | ESS tail | div | failures | run |
|---|---:|---:|---:|---:|---:|---|---|
| m05 TD | 1.0056 | 1.0044 | 1,166 | 1,228 | 0 | G1 | `6ced7c3c…` |
| m12 TD | 1.0050 | 1.0050 | 1,299 | 734 | 0 | G1 | `3b748983…` |
| m12 GRW | 1.0087 | 1.0042 | 878 | 610 | 0 | G1 | `550a0d00…` |

All runs in `mcmc_experiments / smoke_hier_ha`. The grid was launched on this basis.

### Early read on σ_γ (smoke, 4 × 400, not evidence)

| model | fold | γ_base | σ_γ median [5%, 95%] | P(σ_γ < 0.02) |
|---|---:|---:|---|---:|
| m05 TD | 2 | 0.134 ± 0.045 | 0.039 [0.003, 0.115] | 0.28 |
| m12 TD | 2 | 0.126 ± 0.043 | 0.035 [0.004, 0.098] | 0.29 |
| m12 GRW | 2 | 0.126 ± 0.035 | 0.101 [0.033, 0.163] | 0.03 |

(prior P(σ_γ < 0.02) = 0.159). On one season of history the TimeDecay models push σ_γ
*toward* zero while the GRW model holds it near 0.10 — a divergence between dynamics that
the 40-fold grid either confirms or dissolves.

## 40-fold grid (r02) — PASS 3/3

`mcmc_experiments / scottish_lower_hierarchical_ha`, 4 × (500 + 1,000), δ = 0.80. Audit on
all 4,000 retained draws per fold; artefacts keep every 2nd draw.

| model | folds | OOS | max R̂ | max ha R̂ (site) | ESS bulk | ESS tail | div / 160k | BFMI | wall | run |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| `m05_joint_production_wealth_hier_ha` | 40 | 710 | 1.0081 | 1.0070 (`σ_γ`) | 921 | 652 | 4 | 0.643 | 6.7 min | `6117c711-a9f9-4539-a36c-40d6ddb6593c` |
| `m12_joint_hybrid_synergy_hier_ha` | 40 | 710 | 1.0097 | 1.0097 (`γ_team_raw[2]`) | 1,071 | 835 | 0 | 0.682 | 8.5 min | `87c1052d-a181-434c-a965-3c8c801f4142` |
| `m12_joint_hybrid_synergy_grw_hier_ha` | 40 | 710 | 1.0106 | 1.0106 (`σ_γ`) | 716 | 439 | 0 | 0.553 | 48.0 min | `744cf6bd-0441-462c-8bd1-2a5169d70b4b` |

Tree-depth saturation 0.00% everywhere. The GRW hybrid misses Task 007's advisory strict
R̂ ≤ 1.01 (1.0106, not gated), and `ha.σ_γ` is its slowest site — the scale of a
non-centred random effect is where the funnel shows. `m05` has 4 divergences (0.0025%,
gate < 0.1%); the smoke gate was zero, the production gate is a rate.

### σ_γ by fold (H1)

| fold | m05 TD: σ_γ median [5%, 95%] · P(σ_γ<0.02) | m12 TD | m12 GRW |
|---:|---|---|---|
| 1 | 0.051 [0.006, 0.130] · 0.20 | 0.043 [0.004, 0.116] · 0.26 | 0.117 [0.054, 0.183] · 0.006 |
| 10 | 0.038 [0.004, 0.111] · 0.28 | 0.034 [0.003, 0.098] · 0.31 | 0.112 [0.046, 0.172] · 0.016 |
| 20 | 0.048 [0.005, 0.119] · 0.20 | 0.049 [0.005, 0.115] · 0.21 | 0.098 [0.047, 0.153] · 0.008 |
| 21 | 0.046 [0.005, 0.116] · 0.21 | 0.043 [0.004, 0.107] · 0.24 | 0.083 [0.029, 0.136] · 0.032 |
| 30 | 0.060 [0.007, 0.134] · 0.15 | 0.048 [0.005, 0.119] · 0.21 | 0.103 [0.060, 0.155] · 0.002 |
| 40 | 0.037 [0.004, 0.097] · 0.28 | 0.035 [0.004, 0.094] · 0.30 | 0.087 [0.048, 0.131] · 0.006 |

Prior P(σ_γ < 0.02) = 0.159. γ_base sits at 0.11–0.15 in every model and fold (the flat
γ_global of Run 67 is 0.159 ± 0.040).

**H1 splits on the dynamics, and does so on every fold.** Under MultiScaleGRW the club
spread is identified (lower 5% bound 0.03–0.06, boundary mass ≤ 3%). Under TimeDecay(180)
the posterior piles *more* mass at the boundary than the prior does — the data pull σ_γ
toward zero. A club's persistent strength at home and away is absorbed differently by the
two dynamics: TimeDecay's static-per-fold α_i/β_i can soak up a club's whole-venue level,
while the GRW state moves, so a stable ground-specific residual has somewhere to live only
in γ_i. That is a hypothesis about *why*, not a measurement; r04's club tables are where it
can be checked.

### T003 fixtures (priced at γ = 0 by the hierarchical extraction)

| fold | match | note |
|---:|---|---|
| 1 | inverness-caledonian-thistle v dumbarton | relegated into League One |
| 1 | arbroath v montrose | relegated into League One |
| 21 | east-kilbride v the-spartans-fc | promoted; **a turf home ground** — so r04's "turf home" cut carries this bias, and the "excluding T003" rows are the clean read |

## Proper scores and ground effects (r04)

Panel: 710 walk-forward fixtures, of which 627 carry a de-vigged Betfair TWA(−20, 0]
close — 2,899 scored (fixture, selection) rows. Reproduction gate passed:
`m12_hybrid_td_raw` LogLoss 0.64337 / ECE 0.0100, exactly as published.

### Scores (scope = all)

| arm | role | LogLoss | Brier | RPS (1X2) | ECE | market LogLoss / ECE |
|---|---|---:|---:|---:|---:|---:|
| `m05_joint_production_wealth_hier_ha` | candidate | 0.64272 | 0.22574 | 0.22416 | 0.0114 | 0.64182 / 0.0139 |
| `m05_joint_td_raw` | flat control | 0.64299 | 0.22586 | 0.22415 | 0.0149 | |
| `m12_joint_hybrid_synergy_hier_ha` | candidate | 0.64333 | 0.22603 | 0.22454 | **0.0068** | |
| `m12_hybrid_td_raw` | flat control | 0.64337 | 0.22605 | 0.22447 | 0.0100 | |
| `m12_joint_hybrid_synergy_grw_hier_ha` | candidate | 0.64502 | 0.22689 | 0.22514 | 0.0111 | |
| `m12_grw_raw` | flat control | 0.64437 | 0.22659 | 0.22493 | 0.0086 | |

Per market ECE, candidate vs flat: m05 1X2 0.0184/0.0216, O/U 2.5 0.0087/0.0100, BTTS
0.0164/0.0040; m12 TD 1X2 0.0107/0.0155, O/U 2.5 0.0041/0.0100, BTTS 0.0207/0.0087; m12 GRW
1X2 0.0081/0.0140, O/U 2.5 **0.0306/0.0133**, BTTS **0.0610/0.0402**. ECE carries no
interval here; read these as descriptive.

### Paired ΔLogLoss (hierarchical − flat twin), fixture-clustered bootstrap, B = 10,000

| candidate | scope | cut | Δ | 95% interval | P(Δ<0) |
|---|---|---|---:|---|---:|
| m05 TD | all | all fixtures | −0.00027 | [−0.00092, +0.00038] | 0.79 |
| m05 TD | all | excluding T003 | −0.00021 | [−0.00086, +0.00042] | 0.74 |
| m05 TD | all | turf home | **+0.00095** | **[+0.00005, +0.00181]** | 0.02 |
| m05 TD | all | grass home | −0.00074 | [−0.00158, +0.00006] | 0.96 |
| m12 TD | all | all fixtures | −0.00004 | [−0.00055, +0.00046] | 0.56 |
| m12 TD | all | excluding T003 | +0.00002 | [−0.00050, +0.00052] | 0.47 |
| m12 TD | all | turf home | +0.00052 | [−0.00019, +0.00121] | 0.07 |
| m12 TD | all | grass home | −0.00026 | [−0.00092, +0.00038] | 0.79 |
| m12 GRW | all | all fixtures | +0.00065 | [−0.00084, +0.00215] | 0.20 |
| m12 GRW | all | excluding T003 | +0.00072 | [−0.00077, +0.00224] | 0.17 |
| m12 GRW | all | turf home | **+0.00271** | **[+0.00004, +0.00525]** | 0.02 |
| m12 GRW | all | grass home | −0.00015 | [−0.00193, +0.00163] | 0.58 |
| m05 TD | 1X2 | all fixtures | −0.00006 | [−0.00064, +0.00052] | 0.58 |
| m12 TD | 1X2 | all fixtures | +0.00005 | [−0.00046, +0.00056] | 0.42 |
| m12 GRW | 1X2 | all fixtures | +0.00039 | [−0.00120, +0.00194] | 0.32 |

No O/U 2.5 or BTTS contrast is significant in any cut. Full table:
`results/evaluation/r04_paired_bootstrap.csv`.

### Turf vs grass ground effect (fold 40; fold 20 agrees)

| model | turf / grass clubs | E[mean γ_turf − mean γ_grass] | 90% interval | P(turf > grass) |
|---|---|---:|---|---:|
| m05 TD | 7 / 16 | −0.002 | [−0.039, +0.031] | 0.48 |
| m12 TD | 7 / 16 | +0.002 | [−0.031, +0.036] | 0.52 |
| m12 GRW | 7 / 16 | −0.032 | [−0.084, +0.013] | **0.13** |

Largest and smallest club effects under the GRW hybrid (the only model with identified σ_γ):
Annan Athletic 0.229 [0.125, 0.337], Queen of the South 0.192, Peterhead 0.192, Bonnyrigg
Rose 0.183, Edinburgh City (turf) 0.176 … Alloa (turf) 0.074, Hamilton 0.046, Clyde (turf)
0.045, The Spartans (turf) 0.014, Kelty Hearts −0.023 [−0.139, 0.089]. Under both TimeDecay
models every club sits within 0.09–0.15 — γ_i is shrunk almost to γ_base.

### Verdicts

| | verdict | evidence |
|---|---|---|
| H1 σ_γ identified | **model-dependent** | yes under GRW on every fold; no under both TimeDecay models (boundary mass above the prior) |
| H2 turf grounds carry more HA | **refuted** (for the work package's turf list — caveat 1) | P(turf > grass) 0.48 / 0.52 / 0.13 |
| H3 team HA improves proper scores | **refuted** | every all-fixture and 1X2 interval includes 0; GRW point estimate is worse |
| H4 gain concentrated on turf home fixtures | **refuted, reversed** (same caveat) | on turf home fixtures m05 and GRW are *significantly worse*; the small point gains sit on grass |

The one thing the hierarchical slot buys on TimeDecay is calibration — m12 TD ECE 0.0100 →
0.0068 and m05 0.0149 → 0.0114, at unchanged LogLoss — which is the property Exp 06 found
converts into Kelly growth. On GRW it costs calibration on totals and BTTS.

### T003 does not touch the surface result

The r04 re-run added surface cuts excluding the three T003 fixtures. The turf-home cut is
**unchanged** — 813 rows, 178 fixtures, identical Δ — because the one turf-home T003
fixture (fold 21, East Kilbride v The Spartans) has no Betfair close and was never among
the 2,899 scored rows. The grass cut loses the two fold-1 fixtures (10 rows):

| candidate | grass home | grass home excluding T003 |
|---|---|---|
| m05 TD | −0.00074 [−0.00158, +0.00006] | −0.00066 [−0.00151, +0.00015] |
| m12 TD | −0.00026 [−0.00092, +0.00038] | −0.00017 [−0.00081, +0.00045] |
| m12 GRW | −0.00015 [−0.00193, +0.00163] | −0.00005 [−0.00184, +0.00176] |

So the significant turf-home deficits (m05 +0.00095, GRW +0.00271) are not an artefact of
the γ = 0 extraction, and removing T003 moves the grass-home point gains toward zero.

### Caveats that bound the verdicts

1. **The surface classification is the work package's list, not verified ground data** —
   see below. This, not T003, is what H2/H4 are conditional on.
2. **The surface classification is the work package's list, not verified ground data.**
   Several clubs it classes as grass — among them Hamilton Academical and Kelty Hearts, who
   sit at the bottom of every γ table — should be checked against their actual 2024–26
   pitch surfaces before H2/H4 are read as statements about turf. A mislabelled club moves
   rows between the two cuts. This is Phase 2's first deliverable (`is_synthetic_pitch`,
   dated per club-season), not something to patch into Phase 1.
3. **Seven turf clubs.** Even with a correct list the turf − grass contrast rests on 7 grounds.

## 2026-09-12 counterfactual (r05)

Card: 9 played fixtures (Ross County v Hamilton, postponed, removed). as_of 2026-09-12
13:35 UTC (T−25). Flat arm: Run 67. Hierarchical arm: `m12_joint_hybrid_synergy_hier_ha`
`87c1052d…` extended to 43 folds by r03 (new folds R̂ 1.0101, ESS 782, 0 div). Both on
Fold 43. Priced through the replay engine; nothing written to either ledger.

### Reproduction of the live orders

The live ledger reads 11 legs, Fold 43, realised net **−£45.89** (summed from
`paper_settlements`, matching the quoted figure), commission 2%. The `flat_optB` re-price
contains **all 11 live legs** plus 4 more, with max |Δrisk| £3.64 and max |Δp_model| 0.023 on
the shared legs. The residual is the lineup: **0 of 9 SofaScore provisional XIs existed at
T−25**, so the point-in-time source priced every fixture with a neutral lineup pillar, while
the live chain read BBC's confirmed XIs first (a network source with no point-in-time
archive). Both re-priced arms see the identical inputs, so the flat-vs-hierarchical
comparison below is unaffected; the absolute sheets are close to, not identical with, what
was staked.

### What moved: P(home), flat → hierarchical

| fixture | home ground | score | raw | Option B |
|---|---|---|---|---|
| Queen of the South v East Fife | grass | 1-1 | 0.420 → 0.419 | 0.401 → 0.401 |
| East Kilbride v Peterhead | turf | 3-1 | 0.422 → 0.425 | 0.494 → 0.496 |
| Montrose v Cove Rangers | turf | 2-1 | 0.418 → 0.421 | 0.465 → 0.466 |
| Airdrieonians v Alloa | turf | 1-0 | 0.403 → 0.402 | 0.416 → 0.416 |
| Edinburgh City v Stirling Albion | turf | 7-3 | 0.411 → 0.409 | 0.494 → 0.493 |
| Clyde v Kelty Hearts | turf | 4-1 | 0.426 → 0.426 | 0.478 → 0.478 |
| The Spartans v Forfar | turf | 5-1 | 0.430 → 0.427 | 0.512 → 0.509 |
| Annan Athletic v Elgin | grass | 2-0 | 0.429 → 0.432 | 0.396 → 0.397 |
| Stranraer v Dumbarton | grass | 3-1 | 0.432 → 0.429 | 0.504 → 0.503 |

**No fixture moves by more than 0.003.** That is what r02/r04 predict for this candidate:
under TimeDecay σ_γ is pulled toward zero and every club's γ_i sits within 0.09–0.15, a
few hundredths from Run 67's flat γ_global.

### Stake sheets, settled at full fill

| arm | legs | risk | net P&L | away legs (on turf grounds) | away net | home legs | Under 2.5 legs |
|---|---:|---:|---:|---|---:|---:|---:|
| flat_raw | 19 | £124.28 | −£84.27 | 7 (6) | −£54.77 | 2 | 6 |
| hier_raw | 19 | £124.26 | −£83.50 | 7 (6) | −£54.52 | 2 | 6 |
| flat_optB | 15 | £73.78 | −£54.52 | 7 (6) | −£28.17 | 1 | 6 |
| hier_optB | 14 | £73.51 | −£54.26 | 7 (6) | −£28.08 | 1 | 6 |
| live ledger (realised fills) | 11 | £55.73 | **−£45.89** | 6 | | 0 | 5 |

Full-fill settlement assumes every leg filled at its planned risk; the live account filled
some legs partially through `LadderSweep`, so the ledger row is not directly comparable
with the four re-priced rows.

**Did the hierarchical model eliminate or scale down the away bets?** No. `hier_optB` backs
the same seven away legs as `flat_optB`, at the same six turf grounds, with total away risk
£28.08 against £28.17. The only leg it drops is the East Kilbride v Peterhead draw (£1.04).
**Did it take home or Under bets on turf grounds instead?** No new ones; the home and Under
2.5 legs are the same fixtures in both arms. **Counterfactual P&L:** −£54.26 against
−£54.52 at full fill — a £0.26 difference on a card where the stake sheet itself did not
change.

### What the card actually shows

The away-underdog bias on this card is **not a home-advantage-sized problem**. Before
calibration the model prices every home side at P(home) 0.40–0.43 — a 3-point band across
nine very different fixtures — while the market's implied home probabilities on the turf
favourites are far higher (The Spartans: 1 − 0.190 − 0.243 ≈ 0.57). A 0.13–0.16 gap
against a slot that moved 0.003 cannot be closed by re-parameterising γ. The flatness of
the model's own 1X2 across the card, at the first Saturday of 26/27 with no lineup, is the
more promising lead; this runner does not measure its cause.

The GRW hybrid is the only candidate whose σ_γ is identified, and it was **not** re-priced
here (it has no Fold 43). Its fold-40 ground effects, though, rank two of the six turf home
sides — The Spartans (γ 0.014) and Clyde (0.045) — among the *weakest* grounds in the
league, so on its own evidence it would have added, not removed, away stake at those two.
That is an inference from r04's table, not a priced counterfactual.

## Conclusions

1. **`HierarchicalTeamHomeAdvantage` is mechanically sound on all three tiers.** Exact
   gradients, 0–4 divergences per 160,000 transitions, R̂ ≤ 1.011, persisted and
   round-tripped. It costs 20 tape instructions and 16 B per club per gradient.
2. **It does not improve out-of-sample proper scores on any tier**, overall or on 1X2,
   O/U 2.5 or BTTS; and on turf home grounds two of the three tiers are significantly worse.
3. **Whether club HA is even identified depends on the dynamics.** Under TimeDecay(180) the
   data shrink σ_γ toward zero; under MultiScaleGRW it is clearly positive — but the GRW
   club ranking does not line up with the work package's turf list, and its scores do not
   improve either.
4. **On TimeDecay it buys calibration** (m12 ECE 0.0100 → 0.0068, m05 0.0149 → 0.0114) at
   unchanged LogLoss. Whether that converts into bankroll the way Exp 06's calibration gains
   did is a portfolio question this phase did not run.
5. **It would not have changed the 2026-09-12 card.** Same away legs, same grounds, £0.26.
6. **Phase 2 should start from verified data, not the hypothesis.** A dated
   `is_synthetic_pitch` per club-season is the first deliverable, because H2/H4 here rest on
   an unverified seven-club list; and the flat 0.40–0.43 home pricing on an opening-weekend
   card deserves its own look before another home-advantage component is built.

### Recommendation

Do not promote any `_hier_ha` candidate to production on predictive grounds. If the TD
calibration gain is of interest, the next step is an Option B portfolio simulation of
`m12_joint_hybrid_synergy_hier_ha` against `m12_hybrid_td_raw` on the same panel — not a
live swap.

## Reproduction

```bash
# mcmc-beast
cd /root/BF_hier_ha
julia --project -t 16 current_development/hierarchical_home_advantage/r01_smoke.jl
R01_SAMPLES=1000 R01_WARMUP=500 R01_CHAINS=4 julia --project -t 16 current_development/hierarchical_home_advantage/r01_smoke.jl
julia --project -t 16 current_development/hierarchical_home_advantage/r02_production_grid.jl
julia --project -t 16 current_development/hierarchical_home_advantage/r04_evaluate.jl
cd /root/BF_hier_ha_slate   # any cache through 2026-09-05; the card itself is read from betdb
julia --project -t 16 current_development/hierarchical_home_advantage/r03_extend_2627.jl
julia --project -t 16 current_development/hierarchical_home_advantage/r05_slate_repricing.jl
```

---

# Task 008 Phase 2 — Contextual home advantage: pitch surface and match timing

Work package: [`WORK_PACKAGE_PHASE_2_TURF_TIMING.md`](../../WORK_PACKAGE_PHASE_2_TURF_TIMING.md).
Run 2026-09-18 on `mcmc-beast` (smoke too — the beast was idle, and the standing rule is
that no MCMC runs on archpc). Results under `results/phase2/`.

## The model

```
η_h = base + γ_base + u_i + β_asym·turf_i(1−turf_j) + β_gen·turf_i + β_mid·midweek
           + β_rest·(rest_i − rest_j) + β_pace·turf_i + att_h + def_a + wealth
η_a = base + β_pace·turf_i + att_a + def_h − wealth
u_i = σ_stadium·ũ_i,  ũ_i ~ N(0,1),  σ_stadium ~ N⁺(0, 0.05),  γ_base ~ N(0.15, 0.05)
β_asym ~ N(0.05, 0.05)  β_gen ~ N(0, 0.05)  β_pace ~ N(0, 0.05)  β_mid ~ N(0.05, 0.05)  β_rest ~ N(0.02, 0.02)
```

No engine change. `γ_base + u_i` is `HierarchicalTeamHomeAdvantage` with the work package's
priors; every per-fixture term is a scalar `ContextualCovariate{K}` through the builder's
covariate contract (`l04_contextual_loader.jl`). The HA terms use a new `HomeOnlyRole`
(`η_h += q`, away side is the engine's structural `nothing`); `turf_pace` uses the existing
`LevelRole`. Two engine accumulator methods (`_predictor_acc(::Nothing, y)`) are added from
the loader so a home-only term may sit anywhere in the covariate tuple.

## Data decisions

* **Surface registry.** `is_synthetic_pitch` in `src/features/data/scottish_stadium_geocodes.csv`,
  spot-checked 2026-09-18 against public sources (Queen of the South, Falkirk, Forfar, East
  Fife, Stirling Albion, Peterhead, Dumbarton, Stranraer, Bonnyrigg). One dated override:
  **Dumbarton is turf from 2026/27** (The Rock relaid; registry says grass). Falkirk's summer
  2023 install replaced an older artificial surface, so Falkirk is turf throughout. The
  registry keys the home club; every groundshare in the 2022/23+ panel is turf-to-turf.
  Two notes fields held unquoted commas (Elgin, Inverness) — quoted; parsing was unaffected
  because `is_synthetic_pitch` precedes them.
* **The older list in `src/features/extractors/time_extractors.jl` (`PLASTIC_TEAMS`) disagrees**
  with the registry: it lists Bonnyrigg (grass) and omits East Fife (turf since 2017). Not
  touched here; it is only read by `PlasticPitchFeature`.
* **Midweek** = Tuesday–Thursday, or Friday with kickoff ≥ 16:00 UTC. Every flagged kickoff in
  the store is 18:00–20:00 UTC (`r06_midweek_cells.csv`); no Saturday or daytime Friday is flagged.
* **Rest days are league-only.** betdb holds tournaments 56/57 and no cup fixtures, so a
  midweek cup tie is invisible. Each side is capped at 14 days; days since the last kickoff on
  an earlier calendar day. 125 of 710 held-out fixtures have a non-zero difference.
* **OOS design** (`r07_oos_design.csv`): 710 fixtures, 462 at turf grounds, 171 with a grass
  visitor at a turf ground, 53 midweek.

## Ladder and gates

| rung | model | terms | control |
|---|---|---|---|
| 1 | `m05_joint_td_raw` | flat γ ~ N(0.2, 0.2) | persisted, Exp 06 (`ed541a7c`) |
| 2 | `m05_joint_td_turf_asym` | u_i + asym | 1 |
| 3 | `m05_joint_td_turf_dual` | u_i + asym + gen + pace | 1 |
| 4 | `m05_joint_td_contextual` | u_i + asym + gen + pace + midweek + rest | 1 |
| 5 | `m12_joint_hybrid_contextual` | rung 4's terms on the Gen 4 TD hybrid, 43 folds | `m12_hybrid_td_raw` (`132df5c2`) |

No rung won, so rung 5 carries the full contextual set — the most turf- and timing-aware
specification — for the slate mechanism test (r09), not a promoted winner.

Each rung differs from its control in the HA slot and its contextual terms **and** the HA
priors (work package's N(0.15, 0.05) vs the control's N(0.2, 0.2)), and in sampler budget
(4 × 500+1000, δ 0.80 vs 4 × 800+800, δ 0.65).

**G1 (r06, folds 1–2).** Every tape compiles; ReverseDiff == ForwardDiff to ≤ 7.1e-16; compiled
tape exact under perturbation. +25–31 parameters, +33 / +60 / +86 tape instructions, gradient
0.069–0.078 ms vs 0.064 ms flat; +512–544 B allocation per gradient (the 16 B/club Phase 1
already measured).

**G2 (r06, 4 × 500+1000).** PASS 3/3: 0 divergences, R̂ ≤ 1.0047, ESS ≥ 1059. Smoke runs in
`smoke_contextual_ha`: `6c52b6fc`, `31503bc7`, `617dd6ab`.

**G3 (r07 / r09, `scottish_lower_contextual_ha`).**

| model | folds | OOS | R̂ | ESS bulk / tail | div | run |
|---|---|---|---|---|---|---|
| `m05_joint_td_turf_asym` | 40 | 710 | 1.0121 | 1031 / 766 | 0 | `d20ff61d-652c-44fe-b6c4-48777e38bcd7` |
| `m05_joint_td_turf_dual` | 40 | 710 | 1.0082 | 926 / 809 | 1 | `a50e1171-fb42-47bb-8269-3a86a91447a0` |
| `m05_joint_td_contextual` | 40 | 710 | 1.0102 | 1074 / 643 | 1 | `37feea2c-69be-4626-b486-103523de984d` |
| `m12_joint_hybrid_contextual` | 43 | 769 | 1.0074 | 966 / 558 | 0 | `991e4991-624f-4166-a551-3233cd17ecb4` |

`turf_dual`'s **first** attempt failed the gate — tail ESS 327 at fold 29 on a non-HA site (all
`ha.*` and contextual sites ≥ 1052), 0 divergences, R̂ 1.0075 — and was not persisted
(`r07_production_runs_attempt1.csv`; checkpoints kept on the beast as
`checkpoints_attempt1_tailess327`). The sampler is unseeded; all 40 folds were resampled at the
same budget and passed. ~7.5 min per m05 rung, ~10 min for rung 5.

## Proper scores (r08) — H4 fails

710 fixtures / 2,899 scored rows, de-vigged Betfair TWA(−20, 0] close, B = 10,000 fixture-
clustered paired resamples. Controls reproduced Phase 1's r04 exactly (m05 0.64299 / ECE 0.0149;
m12 0.64337 / 0.0100).

| model | LogLoss | Brier | RPS | ECE |
|---|---|---|---|---|
| `m05_joint_production_wealth_hier_ha` (Phase 1, RE only) | 0.64272 | 0.22574 | 0.22416 | 0.0114 |
| `m05_joint_td_turf_asym` | 0.64290 | 0.22583 | 0.22403 | 0.0118 |
| `m05_joint_td_raw` (control) | 0.64299 | 0.22586 | 0.22415 | 0.0149 |
| `m05_joint_td_contextual` | 0.64299 | 0.22587 | 0.22410 | 0.0114 |
| `m05_joint_td_turf_dual` | 0.64300 | 0.22587 | 0.22421 | 0.0120 |
| `m12_hybrid_td_raw` (control) | 0.64337 | 0.22605 | 0.22447 | 0.0100 |
| `m12_joint_hybrid_contextual` | 0.64341 | 0.22607 | 0.22446 | 0.0111 |
| Betfair close | 0.64182 | 0.22529 | | 0.0139 |

ΔLogLoss vs flat twin, scope = all (95% interval):

| model | all | turf home (399) | grass home (228) | grass visitor at turf (149) | midweek (43) |
|---|---|---|---|---|---|
| turf_asym | −0.00009 [−0.00082, +0.00065] | −0.00019 [−0.00115, +0.00079] | +0.00007 | −0.00015 [−0.00246, +0.00221] | −0.00146 [−0.00445, +0.00142] |
| turf_dual | +0.00001 [−0.00082, +0.00083] | −0.00007 | +0.00015 | −0.00010 | −0.00270 [−0.00649, +0.00098] |
| contextual | +0.00001 [−0.00094, +0.00096] | +0.00028 | −0.00047 | +0.00041 | −0.00193 [−0.00718, +0.00310] |
| m12 contextual | +0.00004 [−0.00094, +0.00103] | +0.00034 | −0.00045 | +0.00063 | −0.00212 [−0.00745, +0.00301] |

Every interval, on every scope (all / 1X2 / OU2.5 / BTTS) and cut, contains zero — with one
exception out of 105: `turf_dual`, 1X2, midweek (41 fixtures), −0.0041 [−0.0081, −0.00002].
`turf_dual` has **no midweek term**, so this is multiple-comparison noise, not an effect.
Calibration: every m05 rung lowers ECE (0.0149 → 0.0114–0.0120), as Phase 1's RE-only model
already did; rung 5 raises m12's (0.0100 → 0.0111) and BTTS ECE (0.0087 → 0.0219).

## Coefficients (end of 25/26, fold 40) — H1, H2, H3 fail

| term | contextual (m05) mean [5%, 95%] | P(>0) | prior P(>0) | contraction |
|---|---|---|---|---|
| β_turf_asym | +0.046 [−0.020, +0.115] | 0.864 | 0.841 | 0.18 |
| β_turf_gen | −0.034 [−0.095, +0.028] | 0.186 | 0.500 | 0.23 |
| β_turf_pace | −0.011 [−0.072, +0.050] | 0.374 | 0.500 | 0.27 |
| β_midweek | +0.013 [−0.056, +0.080] | 0.623 | 0.841 | 0.15 |
| β_rest | +0.006 [−0.009, +0.021] | 0.739 | 0.841 | 0.54 |
| γ_base | 0.136 [0.080, 0.191] | | | 0.32 |
| σ_stadium | median 0.027 [0.002, 0.072]; P(σ < 0.02) 0.386 vs prior 0.311 | | | |

`contraction = 1 − sd_post / sd_prior`; 0 means the data said nothing.

* **H1 (turf asymmetry) fails.** P(β_asym > 0) is 0.80 (rung 2), 0.86 (rungs 3–4), 0.88 (rung 5)
  at fold 40 against a prior that already gives 0.84; the posterior sd shrinks by under 20%.
  It reaches 0.90 once in eight rung × fold reads: rung 5 at fold 20 (0.9075; m05 rungs
  0.8965–0.8995 there). The data neither confirms nor
  refutes a grass-visitor penalty; the prior is doing the work.
* **H2 (turf pace) fails.** β_pace is centred on zero; raw panel goals agree — 2.71 per game at
  turf grounds vs 2.67 at grass (non-midweek), `r08_panel_goals.csv`. The EDA impression of more
  goals on turf does not replicate on 24/25–25/26.
* **H3 (midweek) fails.** P(β_midweek > 0) = 0.62, *below* its prior's 0.84 — the data pull it
  toward zero. Raw: midweek at turf grounds, home sides lost on average (goal diff −0.46, n = 28).
* **β_turf_gen leans negative** (P(>0) 0.19): if anything, home sides at turf grounds do
  slightly worse than the rest of the model expects — the opposite of the Phase 2 premise.
* **σ_stadium** stays at its prior under TimeDecay, as in Phase 1.

## 2026-09-12 counterfactual (r09) — H5 fails

Replay engine, T−25 (13:35 UTC), Fold 43 both arms. `flat_optB` reproduces all 11 live legs
(+4 extra), max |Δrisk| £3.64 — Phase 1's figure, same cause (lineup source).
8 of 9 home grounds were turf, but only three visitors came from grass (East Kilbride v
Peterhead, Edinburgh City v Stirling Albion, Annan v Elgin). Dumbarton, now turf, played away.

| fixture | turf h/a | score | P(home) flat raw | ctx raw | Δ | Δ after Option B |
|---|---|---|---|---|---|---|
| queen-of-the-south v east-fife | 1/1 | 1-1 | 0.420 | 0.414 | −0.006 | −0.001 |
| east-kilbride v peterhead | 1/0 | 3-1 | 0.422 | 0.433 | +0.011 | +0.010 |
| montrose v cove-rangers | 1/1 | 2-1 | 0.418 | 0.418 | +0.000 | −0.000 |
| airdrieonians v alloa-athletic | 1/1 | 1-0 | 0.403 | 0.397 | −0.005 | −0.001 |
| edinburgh-city-fc v stirling-albion | 1/0 | 7-3 | 0.411 | 0.417 | +0.006 | +0.004 |
| clyde-fc v kelty-hearts-fc | 1/1 | 4-1 | 0.426 | 0.421 | −0.005 | −0.005 |
| the-spartans-fc v forfar-athletic | 1/1 | 5-1 | 0.430 | 0.424 | −0.006 | −0.007 |
| annan-athletic v elgin-city | 1/0 | 2-0 | 0.429 | 0.440 | +0.011 | +0.003 |
| stranraer v dumbarton | 0/1 | 3-1 | 0.432 | 0.429 | −0.003 | −0.001 |

| arm | legs | risk | full-fill P&L | away legs | on turf | away risk |
|---|---|---|---|---|---|---|
| flat_raw | 19 | £124.28 | −£84.27 | 7 | 6 | £54.77 |
| flat_optB (live arm) | 15 | £73.78 | −£54.52 | 7 | 6 | £28.17 |
| ctx_raw | 18 | £122.94 | −£79.20 | 7 | 6 | £54.49 |
| ctx_optB | 14 | £72.08 | −£52.61 | 7 | 6 | £27.10 |

Live realised: 11 legs, −£45.89. The contextual arm keeps **every** away leg; it drops only the
£1.04 East Kilbride draw leg (Option B) — the same outcome Phase 1's hierarchical arm had. The
asymmetry term lifts P(home) by ≤ 0.011 where it applies; β_turf_gen < 0 lowers it on the six
turf-vs-turf fixtures. The 0.40–0.43 raw home pricing is a gap to the market far larger than
any HA slot this task can build.

## Phase 2 conclusions

1. **No contextual home-advantage specification improves proper scores** on 710 walk-forward
   fixtures — overall, on turf grounds, for grass visitors at turf, or midweek, for m05 or m12.
2. **The data cannot identify the turf and timing coefficients.** Posteriors contract by
   15–30% and sit on (turf asymmetry) or below (midweek, general turf) their priors. H1's 0.90
   bar is unreachable from this panel without a prior that asserts it.
3. **The Saturday-slate story does not generalise.** Turf grounds show no excess home edge and
   no excess goals over two seasons; the 2026-09-12 away losses are not explained by surface.
4. **The ECE gain on m05 is the stadium random effect, not context** — Phase 1's RE-only arm
   already had it (0.0114).

**Recommendation:** close Task 008 without promoting a contextual HA. If home pricing on
opening weekends is still a concern, the lead is the model's flat 0.40–0.43 P(home) against
the market, not a home-advantage slot. The `ContextualCovariate` / `HomeOnlyRole` machinery is
reusable for any per-fixture home-side covariate (e.g. a cup-aware congestion feature once cup
fixtures reach betdb).

## Phase 2 reproduction

```bash
# mcmc-beast
cd /root/BF_hier_ha
julia --project -t 16 current_development/hierarchical_home_advantage/r06_contextual_smoke.jl
julia --project -t 16 current_development/hierarchical_home_advantage/r07_contextual_production.jl
cd /root/BF_hier_ha_slate   # 2026-09-12 DataStore cache; BF_DB_URL exported (T009)
julia --project -t 16 current_development/hierarchical_home_advantage/r09_slate_repricing.jl
cd /root/BF_hier_ha
julia --project -t 16 current_development/hierarchical_home_advantage/r08_contextual_evaluate.jl
```

Any process that `load_fit`s a Phase 2 run must `include("l04_contextual_loader.jl")` first — the
artefacts carry `ContextualCovariate` and `HomeOnlyRole`, which are defined there.
