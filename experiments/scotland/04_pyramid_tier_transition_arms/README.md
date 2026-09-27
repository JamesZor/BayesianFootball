# 04 — Pyramid tier and transition arms (TODO 031 Wave 2)

**Status: SPEC AGREED 2026-09-27, not yet built.** This README is the wave-2 contract agreed with
the human in the 2026-09-27 grill-me session. Results will be appended below it. Experiment name
`scottish_pyramid_tier_transition_w2`. Predecessor: [`03_dynamics_scope_matrix`](../03_dynamics_scope_matrix/README.md)
(W1). Tracking: [`todos/031`](../../../todos/031_scottish_pyramid_dynamics_x_scope_x_observation_matrix.md).
Priors: [`02_cross_tier_cups_and_pyramid_eda`](../02_cross_tier_cups_and_pyramid_eda/README.md) §7 (TODO 029).

## Question

Can explicit pyramid structure fix the two pain points W1 left open?
- **Transition mispricing:** relegated clubs are under-rated and promoted clubs over-rated, in every W1 cell.
- **Cross-tier compression:** pooling across the SPFL shrinks every club toward one mean, which compresses
  the gaps between tiers. As a result, W1's pooled cells make the transition bias *worse*.

The candidate fixes:
- tier steps as the shrinkage target;
- a learned carry of the step at a league change;
- a club-specific jump at a league change;
- league deltas on the observation;
- a macro step for `MultiScaleGRW` into the target season;
- the TODO 029 transition prior.

## 1. Target panel (changed from W1)

- **Target seasons 23/24, 24/25, 25/26** (frozen), tournaments 56/57, `history_seasons = 2`
  (23/24 folds train on 21/22–22/23). About 1.5× the W1 folds.
- **Why:** on W1's 24/25–25/26 panel the transition cohorts are 3 relegated clubs (Arbroath and
  Inverness 24/25, Hamilton 25/26), 4 promoted (Dumbarton, Stenhousemuir, East Fife, Peterhead) and
  1 entered club (East Kilbride; "n = 15" was its fixtures). Adding 23/24 gives:
  - **5 relegated into L1:** + Cove, Hamilton 23/24;
  - **6 promoted into L1:** + Annan, Stirling;
  - **2 entered SPFL:** + The Spartans.

  (Checked in betdb 2026-09-27.)
- Betfair coverage for 56/57 starts at 21/22. **pxG starts at 23/24**, so 23/24 folds have pxG in
  the target season only. Comparisons are paired, so this does not bias them.
- 26/27 to date (Airdrie and Ross County relegated, East Kilbride promoted) is *not* in the panel.
  It is the separate live check in W2.5.
- **W2 rows are not comparable with W1 rows** (different panel). Every base is re-run.

## 2. New components (opt-in, straight into `src/`)

Units: log goal rate. α = attack, β = concession, θ = α − β. The tier step split is Δα = +0.48·τ and
Δβ = −0.52·τ (TODO 029 §7). The tier scale is τ_T = −s·(T − 1), with T1 = Premiership = 0.

### 2.1 `MultiScaleGRW(target_season_step = true)`
Today the walk has `n_history − 1` macro (σₛ) steps, all *inside* the history. The target season
starts from the last history season's level plus one **micro** step (σₖ ≈ 0.03). So the summer
before the season being scored carries one biweek's worth of movement. The flag adds one σₛ macro
innovation between the last history season and the first target state. **Default off**, so W1 and
m00 stay reproducible.
**Season-opening folds** (no target-season data, `n_target = 0`) have no fitted state to carry the step.
There, the forecast integrates it: per draw, σₛ·z is added to each club's last-season level, with z
seeded by fold and centred across teams. The mean is unchanged and the uncertainty is honest.
Confirmed by the human, 2026-09-27.

### 2.2 Pyramid tier component
For club i in a fixture of season s(t):

```
α_i,t = α_i,t(dynamics) + 0.48·[ τ_anchor(i) + κ·(τ_s(t)(i) − τ_anchor(i)) + o·OF_i ] + Σ_e η^α_{i,e}·1[s(t) ≥ s_e]
β_i,t = β_i,t(dynamics) − 0.52·[ same bracket ]                                     + Σ_e η^β_{i,e}·1[s(t) ≥ s_e]
```

| Part | Spec |
|---|---|
| Anchor | Club's tier in its **first season within the fold's training window** (pooled scopes). Lower scope: `:full_pyramid`, the club's first-window-season tier from `club_season_tiers` over all SPFL seasons, so a relegated club with no lower history anchors at T2 |
| Step | One linear step **s ~ TruncatedNormal(0.47, 0.16; 0, ∞)**, estimated (linearity p = 0.92 post-2020; the window starts 21/22) |
| Split | Fixed 0.48 / 0.52 |
| Old Firm | Celtic and Rangers as a pseudo-tier "T0": **o ~ TruncatedNormal(1.15, 0.10; 0, ∞)**, same split |
| Carry κ | Mode `κ ≡ 0` (anchor = pure shrinkage target) or **κ ~ Beta(1.5, 4)** estimated (mean 0.27). κ = 1 would mean "club keeps its position within its league", which contradicts TODO 029's +0.28 |
| Jump η | One event e per tier change (club i, season s_e), persisting from s_e onward, on α and β separately, non-centred. Mode `EstimatedJump`: η ~ N(0, σ_j²), **σ_j ~ Gamma(2, 0.08)**. Entered-SPFL clubs get no η |
| B1-literal mode | Same η, but with **fixed** TODO 029 priors per direction and the tier term off: relegated into L1 α N(+0.12, 0.20²), β N(−0.16, 0.26²); relegated into L2 α N(+0.03, 0.20²), β N(−0.04, 0.24²); promoted into L1 α N(0.00, 0.21²), β N(+0.01, 0.31²). Other directions get no η |
| Cup ties | `:senior_spfl_ties` keeps only non-neutral ties where both clubs have an SPFL tier that season, so each side uses its own club's tier. No T5/B-team nodes |

### 2.3 League observation deltas (replaces TODO 028's A1; amended 2026-09-27 after step 0)
- **Intercept δ_int:** zero-sum per-league deltas, indexed by the **attacking side's club tier**
  that season. Non-centred, fixed prior **N(0, 0.10)**. This is the A1 null check (TODO 029: |δ| ≤ 0.07).
- **League κ, `δ_κ[c]`:** zero-sum per **match competition** c ∈ {54, 55, 56, 57, cups}, shared
  by **both sides** of the match, on the **goals arm only**: `y ~ Poisson(κ·e^{δ_κ[c]}·μ)`.
  Non-centred, fixed prior **N(0, 0.25)**, no learned scale. The pxG Gamma arm and ν are unchanged,
  and home advantage stays shared.
- **Step-0 evidence** (2026-09-27, betdb pxG, 23/24–25/26, logs in `/root/BF_runs/logs/7ff12250/`):

  | Competition | goals ÷ pxG | vs pooled | Attempts per match | Blocked per match |
  |---|---|---|---|---|
  | 54 Premiership | 0.849 | −21% | 25.4 | 7.7 |
  | 55 Championship | 1.149 | +6% | 18.8 | 3.9 |
  | 56 League One | 1.231 | +14% | 19.0 | 2.8 |
  | 57 League Two | 1.237 | +15% | 18.4 | 3.0 |

  95% CIs are about ±4%, so the ±3% gate is passed decisively.
  - The cause is a pxG measurement property: many Premiership attempts are blocked and rarely score,
    and below the Premiership BBC under-records blocked attempts (SofaScore has T55 at 21.3 shots
    with 5.9 blocked).
  - Every W1 pooled joint cell read Premiership pxG about 27% high relative to goals. Lower-scope
    models are unaffected (56 ≈ 57).
- **Why key δ_κ by match rather than attacking side:** an attacking-side κ leaves a Premiership club's
  α and β both inflated, which would over-predict a lower-tier side's goals against it in cup ties.
  Those ties are what identify the tier step s.

## 3. Cells (15)

| # | Cell | Base | Role | Est. time |
|---|---|---|---|---|
| 1 | `td_base` = `td_spfl_cups_joint` | TD pooled + cups, joint | control (TD) | 15 min |
| 2 | td + A2 (κ ≡ 0) | 1 | candidate | 15 |
| 3 | td + A2 + carry | 1 | candidate | 15 |
| 4 | td + A2 + carry + jump | 1 | candidate | 15 |
| 5 | td + league δ | 1 | candidate | 15 |
| 6 | td + A2 + carry + jump + league δ | 1 | candidate | 15 |
| 7 | `grw_base` = `grw_spfl_cups_joint` | GRW pooled + cups, joint | control (GRW) | 140 |
| 8 | grw + step | 7 | candidate | 140 |
| 9 | grw + step + A2 + carry + jump | 7 | candidate | 150 |
| 10 | grw + step + league δ | 7 | candidate | 140 |
| 11 | grw + step + A2 + carry + jump + league δ | 7 | candidate | 150 |
| 12 | `td_lower_joint` | TD lower, joint | control (lower) | 10 |
| 13 | td_lower + A2(full-pyramid anchor) + carry + jump | 12 | candidate | 10 |
| 14 | td_lower + B1 literal | 12 | candidate | 10 |
| 15 | `m12_td` re-run on the new panel | — | champion reference (not gating) | 20–30 |

Pooled GRW-joint cells will probably be saved at stride 2 (the 1 GB field limit). No MAP screen stage
(it inflates non-centred scales, and η is non-centred). Smokes only.

## 4. Decision rule

- **Paired control = the arm's own base** (1, 7 or 12). Cell 15 is reported, not gating.
- **Gate 1, non-inferiority:** Δ target LogLoss vs own base, paired fixture-clustered bootstrap
  (B = 10,000), **CI upper < +0.002**.
- **Gate 2, transition fix:** bias = mean(model − market) win probability of the transitioning club,
  in pp, first 20 league matches in the new tier.
  - CI from a **club-season-clustered** bootstrap.
  - Cohorts: relegated_into_L1 and promoted_into_L1.
  - Pass = |bias| falls vs base on **both** cohorts in point estimate, **and** the Δ|bias| CI
    excludes 0 on **at least one**.
- **Reported, not gating:**
  - first-10 cohorts, `l1_l2`, entered SPFL (2 clubs, diagnostic);
  - compression slope (overall, t56, t57), ECE, 1X2 LogLoss;
  - the 54/55 monitor rows.
- **Winner:** the best arm passing both gates goes on to W3/W4. If none passes gate 2, the best on
  gate 1 + compression goes on, and this README records "transition fix not demonstrated".

## 5. Harness work (Phase B, built while the grids run)

- **(g)** a club-season-clustered bootstrap CI for `transition_bias_pp` and for Δ|bias| vs control.
- **Scorecard v1.2** + panel labelling: the leaderboard must never rank rows from different target
  panels together, and the scorer must not assume 710 fixtures.
- **(i)** score the 54/55 monitor subsets for pooled scopes.
- **(h)** `run_candidates.jl --only` must merge `harness_experiments.run_ids`, not overwrite them.
- **Validation:** re-score the W1 runs. Point estimates must reproduce the W1 numbers exactly, with CIs added.
- Out of scope: (a), (c), (d) (its own TODO), (e), (f).

## 6. Build (Phase A, pi Sol, branch `feat/w2-tier-components` off `feat/experiment-harness`)

Components go straight into `src/` as opt-in builder components, deliberately skipping the
`current_development/` prototype step, because harness fits must deserialize wherever they are scored
(W1 follow-up c). Features are built outside `@model`, following AGENTS.md §7. Required tests, each
rerun by the manager on the beast:

1. **GRW accumulators:** pin today's state composition, and show the flag adds exactly one σₛ step
   into the target season.
2. **Regression:** with every new option off, the log density equals the W1 base's, to machine
   precision, on a real fold (TD and GRW).
3. **Reductions:** κ ≡ 0 equals anchor-only; κ = 1 moves a changer by exactly +0.48·Δτ / −0.52·Δτ.
4. **Tier features:**
   - anchor = first tier in the fold window;
   - `:full_pyramid` anchor;
   - change events;
   - per-side cup tiers;
   - Old Firm flag;
   - no leakage (fold features unchanged when later fixtures are removed).
5. **Default priors** reproduce every value in §2.
6. **AD:** ReverseDiff gradient vs finite differences for each component under TD and GRW. No
   `if`/`view`/mutation in `@model`.
7. **Existing suites stay green:** runner 55/55, scoring 29/29, splitter 48/48, portfolio 25/25, and
   `test/run_parallel_tests.jl`.

`candidates.jl` for the 15 cells is part of Phase A. Controls are passed as the re-run UUIDs, once
they exist, for scoring.

## 7. Sampler, runs and authority

- `QueuedNUTSConfig` 4 × (500 + 1000), acceptance 0.65, max depth 10, `-t 16`, one grid at a time.
- If a cell fails the convergence review (R̂ > 1.05, amended from 1.01 by the human on 2026-09-27;
  or divergences above 0.1% of draws), re-run it once at
  acceptance 0.80. Record both runs.
- **Pre-authorised:**
  - **TD batch** (1–6, 12–15): launch once every smoke passes 5/5 hard checks.
  - **GRW batch** (7–11): launch overnight after the manager has read the TD batch. **Cut 9 and 11 if
    A2 is inert on TD**: cell 4 vs 1 changes |bias| by < 1 pp on both cohorts *and* compression by < 0.05.
- Anything else needs the human's approval.

## 8. After W2

1. **W2.5:** re-price the 2026-09-19 slate (Cove v Ross County, Hamilton v QoS) with the W2 winner and
   `m12`, using card injection.
2. **W3:** σ shrinkage, conditional. It shrinks or is dropped if the W2 winner already reaches
   compression ≈ 1.0–1.2 on TD.
3. **W4:** `PlayerLineupPillar` + wealth. Revisit B2 (market-derived prior) here if the outliers
   persist.
4. **W5:** T−25 calibrated finalist portfolio vs `m12_td`.
