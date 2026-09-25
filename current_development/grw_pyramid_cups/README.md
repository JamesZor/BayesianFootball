# GRW on the pooled Scottish pyramid + cup bridges — overnight 2026-09-25

Follow-up to TODO 029. Does training the Scottish Lower MultiScaleGRW on the whole SPFL,
and on the cup ties that bridge its divisions, improve pricing of the League One/Two book —
and does it relieve compression?

Run on `mcmc-beast` from `/root/BF_grw_pyramid_cups` (rsync of this branch; final code
`3876ceed` + the unpinned evaluator). Namespace `scottish_pyramid_grw_cups`.

## Arms

| Arm | Training rows (added to 56/57) | Likelihood | Run | R̂ max | ESS bulk / tail | Div | Wall |
|---|---|---|---|---|---|---|---|
| `g1_grw_all_spfl` | leagues 54/55 | Poisson | `f00ec78a-28ca-464e-91d4-dd1af384415c` | 1.0135 | 633 / 461 | 0 | 92 min |
| `g2_grw_all_spfl_cups` | 54/55 + 363 SPFL-vs-SPFL cup ties at club grounds | Poisson | `a6f62436-ec8a-461d-8bd5-dc1861a2daaa` | 1.0105 | 827 / 697 | 0 | 82 min |
| `g3_grw_joint_all_spfl_cups` | as g2 | Joint Gamma-Poisson (pxG), no wealth | `9babf9e9-0a04-43af-855c-619a4b7dac8b` | 1.0123 | 462 / 577 | 0 | 110 min |

All pass the six-part audit on 4 × 1,000 retained draws. Every 4th draw is persisted, because
at stride 2 the pooled artifact passed PostgreSQL's 1 GB hex message cap (see Caveats).

**Contract.**
- Held-out fixtures and fold boundaries are the canonical 40-fold / 710-fixture 56/57 grid
  (asserted fold by fold against `gph_splitter`). Only the training rows widen.
- Every added row kicks off before its fold's first held-out fixture (asserted).
- Added target-season rows sit on the 56/57 biweek clock (cross-checked against
  `Data._effective_step_map`).
- Cup season labels are rewritten to football seasons.
- Model recipe = `m00_baseline_grw` / `m05`-without-wealth, unchanged. There is no league
  offset, because TODO 029 measured flat SPFL goal levels.

## Results (710 fixtures, 2,899 scored selections, Betfair TWA(−20,0] close)

| Model | LogLoss | 1X2 LogLoss | ECE | market-on-model slope | model-on-market |
|---|---:|---:|---:|---:|---:|
| Betfair close | 0.64182 | 0.61312 | 0.0139 | 1 | 1 |
| `m05_joint_grw` (56/57, + wealth) | **0.64316** | **0.61561** | 0.0117 | 1.24 | 0.55 |
| `m12_td` (live, TimeDecay) | 0.64337 | 0.61636 | 0.0100 | 1.57 | 0.31 |
| **`g3_grw_joint_all_spfl_cups`** | 0.64385 | 0.61718 | 0.0095 | **1.10** | **0.60** |
| `m12_grw` | 0.64437 | 0.61701 | **0.0086** | 1.33 | 0.48 |
| `m00_baseline_grw` (56/57) | 0.64460 | 0.61689 | 0.0162 | 1.19 | 0.37 |
| `g2_grw_all_spfl_cups` | 0.64617 | 0.61634 | 0.0181 | 1.25 | 0.41 |
| `g1_grw_all_spfl` | 0.64644 | 0.61939 | 0.0122 | 1.10 | 0.39 |

Paired ΔLogLoss (left − right; fixture-clustered bootstrap, B = 10,000, 95% CI):

| Contrast | all | 1X2 |
|---|---|---|
| g1 − m00 (pooling alone) | +0.0018 [−0.0019, +0.0056] | +0.0025 [−0.0024, +0.0072] |
| **g2 − g1 (the cups)** | −0.0003 [−0.0035, +0.0029] | **−0.0031 [−0.0055, −0.0006]** |
| g2 − m00 | +0.0016 [−0.0031, +0.0062] | −0.0006 [−0.0051, +0.0038] |
| g3 − g2 (joint arm) | −0.0023 [−0.0089, +0.0044] | +0.0008 [−0.0055, +0.0071] |
| g3 − m05_joint_grw | +0.0007 [−0.0037, +0.0051] | +0.0016 [−0.0025, +0.0057] |
| g3 − m12_grw | −0.0005 [−0.0053, +0.0044] | +0.0002 [−0.0045, +0.0048] |
| g3 − m12_td | +0.0005 [−0.0063, +0.0071] | +0.0008 [−0.0059, +0.0075] |

## Reading

1. **Pooling the four divisions alone does not help the League One/Two book** (g1 is worse,
   not significantly).
2. **The cup bridges are the only effect that clears zero:** on 1X2 they recover
   −0.0031 against pooling-without-cups. That brings the pooled model back to the 56/57
   baseline, not past it.
3. **g3 is the least compressed joint model measured:** its market-on-model slope is 1.10,
   against 1.24 for `m05_joint_grw`, 1.33 for `m12_grw` and 1.57 for the live `m12_td`. It
   carries the most market spread (model-on-market 0.60) and has the second-best ECE (0.0095),
   at a log-loss statistically tied with every production candidate.
4. **Nothing beats the live models on log-loss.** All g3 contrasts straddle zero. The case for
   g3 is decompression at equal accuracy, which is what should matter for Kelly edge selection.
   It needs a portfolio backtest (Option B contract, T−25) before any promotion.

The outcome-on-model slopes in `r04_compression_scorecard.csv` are much noisier: realised goal
difference against expected goal difference, with no CIs. They disagree with the market slopes
(e.g. m12_td 1.07 on outcomes, 1.57 against the market), so don't read them on their own. The
"transition" subset (347 fixtures with a club whose tier differs from last season, including
League One↔League Two moves) is too broad to isolate relegation cold starts.

## Also done overnight

- **Live chains refreshed.**
  - `m12_joint_hybrid_synergy` TD (`132df5c2`): 43 → 44 folds, R̂ 1.010, MatchDay audit PASSED.
  - `m12_joint_hybrid_synergy_grw` (`3a9a4c7e`): 43 → 44 folds, R̂ 1.015.
- **Still stale for live use.** Fold 44 holds out 2026-09-12 → 09-19, so it is trained only
  on data before 09-12. The walk-forward extension cannot make a fold whose held-out block is
  unplayed, so a Saturday slate still prices from a chain that has not seen the previous one to
  two match-weeks. That needs a "fit on everything to date" path, not `extend_fit`.

## Caveats

- The binary-bytea `save_fit` fix (`_db_exec_binary`, memory note 2026-09-20) exists **only
  uncommitted** in the `shrinkage-decompression` worktree. This branch still hex-encodes,
  hence the stride-4 persistence. Commit that fix.
- No B-team, guest or non-league cup rows, and no wealth or lineup terms in g3.
- Single run per arm. The evaluation uses the persisted 1,000 draws per fold.

## Files

`l01_loader.jl` (segment, `PyramidGRWCV` splitter, `pcx_align_time!`, arms) ·
`r01_smoke.jl` · `r02_overnight.jl` · `r03_extend_m12_grw.jl` · `r04_evaluate.jl` ·
`run_overnight.sh` · `data/cup_bridge_allowlist.csv` (409 ties; 363 are present in
`sofascore.matches`) · `results/` (CSV + logs; checkpoints stay on the beast).
