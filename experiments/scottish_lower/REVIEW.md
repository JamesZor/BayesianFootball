# Scottish Lower experiments — review (2026-09-26)

A read-through of every suite in this folder (01–13), the three top-level reports
(`UNIFIED_PARADIGM_REPORT.md`, `MARKET_LINE_EDA_REPORT.md`, `compare_scottish_experiments.jl`)
and the closely related work outside it (`experiments/scotland/01–02`,
`current_development/grw_pyramid_cups`, `grw_player_hybrid`).

It answers four questions:
1. What do we actually know?
2. Which numbers can't be trusted or compared?
3. How did we work, and what worked about that?
4. What's missing before we extend to more tournaments?

Code-level inventory: [`docs/architecture/experiment_harness_suite_catalogue.md`](../../docs/architecture/experiment_harness_suite_catalogue.md).
Workflow and abstraction plan: [`docs/architecture/experiment_workflow_review.md`](../../docs/architecture/experiment_workflow_review.md).

---

## 1. What we know — the knowledge ledger

"Strength" = how much weight the evidence bears: **strong** (40-fold grid, paired CI,
replicated), **moderate** (one grid, consistent direction), **weak** (one run, CI spans zero,
or selection-biased), **untested**.

| # | Finding | Evidence | Strength |
|---|---|---|---|
| 1 | **The two-arm joint likelihood (Gamma on proxy xG + Poisson goals) is the biggest single predictive gain.** Covariates move LogLoss by about 1/5 as much. | 03 (−0.0028 vs its Poisson control; seven-arm covariate spread 0.0006); unified re-score 0.6438 vs 0.6545 | strong |
| 2 | **Overdispersion is mild** (NegBin r̂ ≈ 26). | 02 | strong on r̂; the LogLoss effect is disputed, see §2 |
| 3 | **Covariates (wealth, travel, bench depth) are small on proper scores.** Wealth was the betting edge in 01–03. Production wealth *hurts* Poisson calibration once re-scored (ECE 0.053). | 01–03, unified §3 | moderate |
| 4 | **Lineup RAPM buys calibration, not sharpness.** H1 refused: every lineup arm is behind the control on LogLoss/Brier/CRPS/RPS, and ahead on ECE (0.0088–0.0104 vs 0.0149; close 0.0139). m12 is live for this reason. | 05, 06 | moderate |
| 5 | **…but the lineup signal is mostly club identity.** With `fit_on = :history` there are only 2 distinct RAPM vectors across 40 folds (one per season); 89% of ΔL is club identity. | feature-compression EDA (2026-09-21) | moderate |
| 6 | **GRW beats TimeDecay for the pure-Poisson model on every axis.** LogLoss 0.6445 vs 0.6468, 1X2 0.6168 vs 0.6203, +197% vs +140%, market-on-model slope ~1.07 vs 2.53. | 10 (m01 vs m02), 11/12 controls, 023 | strong |
| 7 | **With joint + lineup, TimeDecay and GRW are tied.** m12_td 0.64337 vs m12_grw 0.64437, CI spans 0. GRW and proxy xG are partial substitutes. | TODO 013, `grw_pyramid_cups` | moderate |
| 8 | **Compression lives in the team α/β latent.** The market would weight the team contrast 2.4×, the lineup pillar 1.0× and wealth 1.6×. Priors and ridge penalty are not binding. | feature-compression EDA | moderate |
| 9 | **Fixes aimed at the observation or covariate layer don't decompress usefully.** NegBin + pxG covariate: slope 1.72 → 1.30, but it loses LogLoss and money (11). Decoupled funnel: worse slope and worse LogLoss, rejected (12). Momentum GRW: marginal (10). Hierarchical κ: nothing, loses 4.5–6 pts (06). Fast/slow mixtures: no (021). | 06, 10, 11, 12, 021 | strong (as negatives) |
| 10 | **Portfolio: calibrated T−25 container + Over 1.5 at tier 2 + a re-pointed risk ladder, all three or none** (super-additive). Over 2.5 not rehabilitated OOS; Under 0.5 broken; **per-line pruning rules don't survive out of sample.** | 07, `MARKET_LINE_EDA_REPORT` §5.1 | moderate (one 99-slate period, IS/OOS split) |
| 11 | **Pooling the four SPFL leagues alone doesn't help League One/Two; cup bridges do** (1X2 −0.0031, CI excludes 0). The joint GRW on leagues + cups (g3) is the least compressed joint model (slope 1.10) at tied LogLoss. | `grw_pyramid_cups` | moderate (one run per arm) |
| 12 | **Tier steps ~0.47 goals per tier, roughly linear; a relegated-into-L1 prior is about +0.12, not +0.90.** | scotland/02 (TODO 029) | moderate |
| 13 | **Goal decomposition** (penalties, own goals, open play). | 08: smoke 64/64, **grid never run** | untested |
| 14 | **Full-time/part-time status effect.** | 13: 8 fixtures, unknown for 223/252 club-seasons | untested |

**Net picture.** The levers that have worked are **the likelihood** (joint pxG), **the
dynamics** (GRW), **the data** (cup bridges) and **the portfolio container** (T−25
calibration + ladder). Adding features to the linear predictor has consistently bought
little. That is the argument for making data scope and dynamics first-class experiment
dimensions, rather than covariates.

## 2. Numbers not to trust or compare

| Issue | Where | What to do |
|---|---|---|
| **The headlines in 01–03 use an older scorer.** 01 says `m05_production_wealth` wins (0.6597 vs 0.6603). The unified re-score reverses it (0.6582 vs **0.6545**). 02 says NegBin gives "no material gain" (0.6598); the re-score has NegBin well ahead of Poisson (0.6457 vs 0.6545). The close scores 0.6568 on the old scorer and 0.6418 on the current one. | 01, 02, 03 READMEs; **AGENTS.md §6** and `model_generations_guide.md` cite the old numbers | Cite only `UNIFIED_PARADIGM_REPORT` numbers for Gen 1–4; fix AGENTS.md §6 |
| **The same Poisson TimeDecay(180) control scores 0.6545 (unified, original 01 fits) and 0.6468 (10/11/12 re-runs)** on the same 2,899 observations. Unexplained: the 10–12 controls may be the TODO 005/006 ReverseDiff re-fits under a different recipe. | unified §3 vs 10/11/12 | Resolve before any cross-suite table; the leaderboard must key on run UUID, not model name |
| **Portfolio returns come from different books.** 03: 3 markets, 30% trust, Betfair close. Unified: 6 markets, `FlatTrust(1.0)`. 10–12: a 622-fixture common panel. 07: T−25. Persisted `portfolio_runs` for 01/03 are **bookmaker**-priced (negative returns). "+136%", "+162%" and "+197%" are not comparable. | all suites | Compare returns only within one suite, or re-simulate under one spec |
| **Three different "slopes"**: market-on-model (11, 12, 023, pyramid; ideal 1, above 1 = compressed), model-on-market (10; below 1 = compressed), Platt β (unified). | 10, 11, 12, unified | Always name the convention; standardise on market-on-model |
| **Different fold counts.** Unified G2 `m00_negbin_baseline` has 42 folds / 749 fixtures, not 40 / 710. | unified §1 | Exclude it from Δ tables or trim to 40 |
| **The "favourite P(win) 76% vs ~52%" figure is from 18 market-selected fixtures.** | 10, 11, 12 | Treat as illustrative, not an estimate |
| **One 710-fixture period.** Every portfolio number is one path through 24/25–25/26; only 06 (r67) bootstraps the portfolio. | all | Report portfolio CIs, or treat returns as a ranking only |

## 3. How we worked

**The process that emerged:**
1. Brain dump.
2. `/grill-me` (Antigravity), then a TODO plus `WORK_PACKAGE_PROMPT.md`.
3. A worktree and branch per package.
4. A pi or Claude agent executes: EDA/preflight → smoke gates → 40-fold grid on the beast
   (from an rsync copy) → scoring → portfolio → README with a decision.

The shape changed over time:
- **01–03:** globally numbered scripts (`r20`–`r48`).
- **05–06:** loader/runner pairs.
- **07–13:** agent-executed work packages.

**What worked — keep it:**
- **An identical-spine control in every suite.** This is what made each effect attributable
  (03's `m00_poisson_control`, the 11/12 double controls).
- **Cheap deterministic screening before MCMC.** 06's ridge bake-off (`r59`) refused two of
  five hypotheses before any sampling; TODO 029 used analytic-gradient Dixon–Coles MAP fits.
  This is the most under-used idea in the folder.
- **Pre-registration.** Suite 12 fixed folds 21–40 (full proxy-xG coverage) as the clean
  comparison block *before* the grid ran.
- **Re-scoring saved fits from the DB under one recipe** (`compare_scottish_experiments.jl`).
  That is what exposed §2; it is a leaderboard prototype.
- **`gph_*` reuse** (`current_development/grw_player_hybrid`): suites 10–12 and the pyramid
  work got a tested runner for free.
- **Honest negative results with decisions and run UUIDs** (10–13).

**What didn't work:**
- **The harness was rebuilt per suite.** 05/06/07/08 and scotland/01 each wrote their own
  gates, runners and scorers. The gates drifted: ESS ≥ 100 / 200 / 300, divergences
  hard-fail or warn, a zero-byte tape rule in scotland/01 only. Details in the catalogue §A, §F.
- **One folder does three jobs.** EDA, new model types and benchmarking are mixed, so a
  benchmark waits on its own bespoke EDA and components, and EDA hypotheses enter prompts
  as facts. 027's prompt carried a false "54% ceiling"; 028 built its prior on +0.90
  before 029 measured +0.12.
- **Headline numbers are frozen into READMEs and AGENTS.md at the time of each suite**,
  then silently go stale when the scorer changes (§2, row 1).
- **Work ends unmerged.** `main` has 01–08 only; 10–13 live on a chain of stacked branches.
- **Runs come from rsync copies without git.**

## 4. What's missing before extending to more tournaments

1. **One scorer, one leaderboard**, keyed on run UUID, re-scoring every persisted run under a
   versioned scorecard (per market, per tier, transition and cold-start cohorts). Start from
   `compare_scottish_experiments.jl` + `gph_scores`.
2. **Data scope as an experiment dimension**: training tournaments, cup policy, scored and
   monitored tournaments, with the canonical 40 / 710 folds asserted. It exists twice today,
   inconsistently (`ScottishAll` + `GroupedCVConfig` in scotland/01; `PyramidGRWCV` +
   allowlist CSV in `grw_pyramid_cups`).
3. **A standard runner with gate severities**, from `gph_*`: correctness gates hard,
   convergence flagged for review (08 already does this), performance and tail mass as warnings.
4. **A stage-0 screen**: MAP/ridge fits in seconds, before any NUTS, so many candidates can be
   screened cheaply. This is the piece that makes a FunSearch-style loop affordable.
5. **Separate lanes**: `eda/` (numbers → priors and hypotheses), components (PR to `src/`
   with AD tests), experiments (recipes on the runner).
6. **Resolve the two §2 discrepancies** (01/02 vs unified; 0.6545 vs 0.6468) and correct
   AGENTS.md §6.
7. **Finish or close the open ends**: 08's grid (TODO 002), TODO 027's missing evidence, and
   a decision on 008's unmerged branch.

## 5. What to run once the pieces exist

The factorial the ledger points at, scored on League One/Two with the upper tiers monitored:
dynamics {TimeDecay, GRW} × data scope {56/57, SPFL, SPFL + cups} × observation {Poisson,
joint}. Four of the twelve cells already have runs to cross-check against (`m00_baseline_grw`,
g1, g2, g3). Then add, in order: the lineup pillar and wealth on the best cell; tier offsets
and a +0.12 relegated prior; and the T−25 portfolio for finalists. This is TODO 031.
