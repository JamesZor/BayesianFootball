# TODO 039 market model: status, 2026-10-08

This page replaces `MORNING_SUMMARY.md` (2026-10-06) as the current state of the stream. The
chronology and every answer given on the human's behalf are in `NIGHT_LOG.md`.

## Where everything is

- **Stream branch:** `feat/market-model-inversion-eda`, worktree
  `.worktrees/BayesianFootball-market-model`. Merged locally, not pushed.
- **pi branches** (pushed): `pi/market-model-phase-{a,b,b2,b3,c}`, `pi/market-model-qs-eda` and
  `pi/market-model-qs-forms`. The last one contains all the others and is merged into the stream.
- **Saved posterior draws** are on mcmc-beast only. **Don't delete these folders:**

  | Folder | Holds |
  |---|---|
  | `/root/BF_runs/market_model_qs/…/results/QS/fits/<league>/` | 66 fits plus C0 synthetic draws |
  | `/root/BF_runs/market_model_qsf/…/research/qs_forms/answers/` | draw-level tables, listed in each batch's `manifest.csv` |
  | `/root/BF_runs/market_model_c*/` | Phase C runs |

## What is done

| Phase | Result | Report |
|---|---|---|
| A: KL inversion | 595/710 Scottish L1/L2 closes inverted. The double-Poisson grid misprices draws by −1.34 pp. | `results/A/` |
| B: two-stage Kalman ladder | Weekly GRW essential; home advantage flat; heavy tails mostly bad books. | `results/B/PHASE_B_REPORT.md` |
| B2: covariance | Correlated attack/defence steps (R6, ρ ≈ −0.82); quality/style rotation r ≈ 0.3 on Scottish L1/L2. | `results/B2/PHASE_B2_REPORT.md` |
| B3: copula and grids | Gaussian dependence adequate; Dixon–Coles global ρ = −0.064 is the best score grid (**not adopted**). | `results/B3/PHASE_B3_REPORT.md` |
| C: pooled and home advantage | C0 ≈ R6; γ ≈ 0.18; HA not linked to quality (κ ≈ 0); home/away split of HA unidentified. **C1 parked** (synthetic recovery failed on σ_s/σ_u). | `results/C/` |
| Literature | No football model estimates an attack/defence step correlation; recommends Spec A. | `research/ad_correlation/REPORT.md` |
| Quality/style EDA, 11 Betfair leagues | 9 suitable, 2 inconclusive (EPL, Finland), 0 unsuitable. r runs from 0.27 (Scottish L2) to 0.68 (Finland). Style adds real-goal information (+0.0067 nats/fixture). | `results/QS/QUALITY_STYLE_REPORT.md` |
| Formulation research (Astra + Sol) | Keep 45° q/s; **league-hierarchical r** is the leading extension; levels ≠ steps; Scottish Premiership quality tier (Celtic/Rangers); EPL style partly reverts. | `research/qs_forms/REPORT.md` |

The model is written up as the private artifact "Quality–Style GRW"
(https://claude.ai/artifact/U5emjQVY3pKMWzGercSpzw).

## Verification state, 2026-10-08

- The full test suite passes on the beast in fresh REPLs at the research-branch code. The results
  are in `NIGHT_LOG.md`.
- **T03 needs the original Scottish snapshot timestamp.** The beast copy of `c786e2…` needs its
  mtime set to 2026-09-25 12:57:15.48 UTC (`touch -d`); the laptop original already has it.
- The 10 key result CSVs are byte-identical between laptop and beast.
- The QS EDA reproduction passed: Premier League and Scottish League Two refits, plus all
  CSVs regenerated from the saved draws.

## Recommended next step

**A goal-model TODO for Spec A**, which needs `src/` changes:
- `MultiScaleGRW` on centred quality/style axes;
- r pooled across leagues with a tier covariate;
- separate level and step priors.

Judge it against the current model on forward goal prediction (40 folds), with total and
allocation scores. Detail: `research/qs_forms/REPORT.md` §4.

## Parked decisions (the human's)

1. **Forecasting test:** does the market model (C0, one-week-ahead) beat the goal models on
   25/26? It's cheap, and the predictions already exist.
2. **Dixon–Coles global-ρ grid** (B3's winner) for any market-model rerun.
3. **C2 thin-book pooling:** moot while C1 is parked.
4. **English `DataTournemantSegment`s in `src/`:** only needed for production use; the EDA used a
   stream-local `MarketModelEnglish` segment.
5. **Further candidate fits** from the research:
   - dynamic league level μ_t;
   - Scottish Premiership quality-tier or heavy-tail prior;
   - EPL style mean reversion.

   Test each separately.
6. **pi tooling trial** (pi 0.99.2, pi-context, pi-blackhole): re-measure with
   `~/agent_research/pi_context_review/scripts/`. Blackhole's compact-all once dropped a live
   instruction, and pi then stopped.
