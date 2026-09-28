# 04 — Pyramid tier and transition arms (TODO 031 Wave 2)

**Status: COMPLETED 2026-09-28. Transition fix demonstrated by the tier target (A2); two
co-finalists go on: `td_lower_a2full_carry_jump` and `grw_step_a2_carry_jump`** (human decision;
see [Result](#result--wave-2-grid-2026-09-28-scorecard-v12)). Sections 1–8 are the wave-2 contract
agreed with the human in the 2026-09-27 grill-me session; results follow them. Experiment name
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

## Result — Wave 2 grid (2026-09-28, scorecard `v1.2`)

**Decision (human, 2026-09-28): the transition fix is demonstrated, by the tier target A2. Two
co-finalists go on to W3/W4, one per dynamics family: `td_lower_a2full_carry_jump` (TD, lower scope)
and `grw_step_a2_carry_jump` (GRW, pooled + cups).**
- Eight arms pass both gates, and every one of them carries A2 (or its B1-literal sibling).
- The gate-passers tie on target LogLoss (paired table below), so "best" had no statistical winner.
- The literal lowest-LogLoss arm, `td_lower_b1lit`, is set aside for prior leakage (caveats).

Reference: de-vigged Betfair TWA (−20, 0] close. Target = 1,070 held-out 56/57 fixtures in 60 folds
(panel `56+57|23/24,24/25,25/26|n=1070`); every cell is scored on identical observation counts.
- **Compression:** market-on-model 1X2 slope. 1 is ideal; above 1 means the model is compressed.
- **Bias:** mean model − market win probability of the transitioning club, in pp, over its first
  20 league matches. Positive means over-rated. CIs are club-season-clustered.
- **Deltas:** paired fixture-clustered bootstraps (B = 10,000) against the arm's **own base**.
  Gate 1 needs ΔLL CI upper < +0.002. Gate 2 needs Δ|bias| < 0 on both cohorts, with at least one
  CI below 0.
- **Cohort sizes (first 20):** relegated into L1 90 fixtures, promoted into L1 107, entered SPFL 15.

| Cell | Target LL | ECE | Compression (all / t56 / t57) | Relegated → L1 bias | Promoted → L1 bias | ΔLL vs base [95% CI] | Δ\|bias\| rel / pro | Gates | Run |
|---|---:|---:|---|---:|---:|---:|---|:-:|---|
| `td_base` (ctl) | 0.6392 | 0.0099 | 1.51 / 1.58 / 1.38 | −9.50 [−13.76, −4.83] | +5.39 [+3.38, +7.38] | 0 | — | — | `1cdff82a` |
| `td_a2` | 0.6374 | 0.0107 | 1.51 / 1.50 / 1.51 | −4.16 [−7.30, +0.29] | −0.02 [−1.55, +1.40] | −0.0018 [−0.0040, +0.0005] | −5.34* / −5.36* | ✓✓ | `87330b48` |
| `td_a2_carry` | 0.6375 | 0.0104 | 1.58 / 1.59 / 1.57 | −5.14 [−8.32, −0.81] | +0.86 [−0.74, +2.30] | −0.0017 [−0.0037, +0.0003] | −4.36* / −4.52* | ✓✓ | `f29455af` |
| `td_a2_carry_jump` | 0.6369 | 0.0163 | 1.56 / 1.56 / 1.53 | −4.50 [−6.79, −1.01] | +0.77 [−0.97, +2.57] | −0.0022 [−0.0043, −0.0002] | −5.00* / −4.61* | ✓✓ | `e812e223` |
| `td_ldelta` | 0.6386 | 0.0076 | 1.53 / 1.61 / 1.38 | −9.04 [−13.27, −4.32] | +6.16 [+4.20, +8.13] | −0.0006 [−0.0024, +0.0012] | −0.46* / **+0.78*** | ✓✗ | `0af24493` |
| `td_a2_carry_jump_ldelta` | 0.6368 | 0.0112 | 1.61 / 1.63 / 1.57 | −4.30 [−6.61, −0.75] | +1.31 [−0.30, +3.08] | −0.0024 [−0.0050, +0.0002] | −5.20* / −4.07* | ✓✓ | `bf877fb1` |
| `grw_base` (ctl) | 0.6404 | 0.0113 | 1.14 / 1.19 / 1.03 | −5.44 [−10.14, −1.07] | +3.41 [+0.76, +6.86] | 0 | — | — | `ce7ea22f` |
| `grw_step` | 0.6384 | 0.0152 | 1.15 / 1.20 / 1.06 | −4.51 [−7.80, −1.67] | +3.20 [+0.67, +6.62] | **−0.0020 [−0.0037, −0.0004]** | −0.92 / −0.21 | ✓✗ | `79247533` |
| **`grw_step_a2_carry_jump`** | 0.6377 | 0.0102 | 1.20 / 1.22 / 1.14 | −2.65 [−5.20, +0.13] | +1.19 [−0.91, +4.12] | **−0.0026 [−0.0046, −0.0007]** | −2.78* / −2.22 | ✓✓ | `c4a0fa94` |
| `grw_step_ldelta` | 0.6379 | 0.0149 | 1.16 / 1.21 / 1.05 | −4.38 [−7.56, −1.40] | +4.05 [+1.62, +7.27] | −0.0025 [−0.0051, +0.0000] | −1.06 / **+0.64*** | ✓✗ | `8987f71e` |
| `grw_step_a2_carry_jump_ldelta` | 0.6375 | 0.0188 | 1.21 / 1.24 / 1.14 | −2.51 [−4.98, +0.28] | +2.02 [−0.24, +5.07] | **−0.0029 [−0.0054, −0.0005]** | −2.92* / −1.39* | ✓✓ | `399a9190` |
| `td_lower_base` (ctl) | 0.6383 | 0.0102 | 1.60 / 1.69 / 1.42 | −4.24 [−6.28, −2.19] | +4.46 [+2.87, +6.00] | 0 | — | — | `0dd74a52` |
| **`td_lower_a2full_carry_jump`** | 0.6369 | 0.0111 | 1.46 / 1.46 / 1.44 | **−0.63 [−1.56, +0.58]** | **+0.88 [−0.93, +2.56]** | −0.0014 [−0.0029, +0.0002] | −3.60* / −3.57* | ✓✓ | `6c0aa3ea` |
| `td_lower_b1lit` ⚠ | 0.6362 | 0.0099 | 1.32 / 1.34 / 1.26 | +0.97 [−0.86, +2.87] | +2.59 [+0.45, +4.91] | −0.0021 [−0.0044, +0.0002] | −3.27 / −1.86* | ✓✓ | `5cf71ee6` |
| `m12_td_ref` (reference) | 0.6377 | 0.0151 | 1.45 / 1.48 / 1.36 | −2.52 [−6.45, +1.37] | +4.72 [+2.28, +7.28] | −0.0006 [−0.0024, +0.0012] | −1.71 / +0.26 | (✓✗) | `973baafa` |

\* Δ|bias| CI excludes 0. `m12_td_ref` is paired against `td_lower_base` and does not gate. Against
`td_base` it has ΔLL −0.0015 [−0.0048, +0.0019] and Δ|bias| −6.98* / −0.67.

**Leading arms paired against `td_lower_b1lit`** (the lowest target LogLoss). No arm is
distinguishable on LogLoss:

| Arm | ΔLL target [95% CI] | Δ\|bias\| relegated | Δ\|bias\| promoted |
|---|---:|---:|---:|
| `td_a2_carry_jump_ldelta` | +0.0005 [−0.0023, +0.0033] | +3.33 [+0.37, +6.17] | −1.28 [−2.26, −0.12] |
| `td_lower_a2full_carry_jump` | +0.0007 [−0.0011, +0.0024] | −0.34 [−2.43, +1.22] | −1.71 [−2.72, +0.18] |
| `td_a2` | +0.0012 [−0.0024, +0.0047] | +3.19 [−0.36, +6.87] | −2.57 [−3.91, +0.89] |
| `grw_step_a2_carry_jump` | +0.0015 [−0.0024, +0.0053] | +1.69 [−0.94, +4.76] | −1.40 [−2.74, +0.40] |
| `m12_td_ref` | +0.0015 [−0.0015, +0.0045] | +1.55 [−2.21, +5.90] | +2.12 [−0.76, +5.03] |

**54/55 monitor (pooled scopes; reported, not gating).** Monitor LogLoss (all markets) and
compression by cell. The market scores 0.6235 on T54 and 0.6324 on T55.

| Cell | T54 LL | T55 LL | T54 compression | T55 compression |
|---|---:|---:|---:|---:|
| `td_base` | 0.6396 | 0.6452 | 1.58 | 1.71 |
| `td_a2` | 0.6352 | 0.6421 | 1.29 | 1.67 |
| `td_a2_carry` | 0.6350 | 0.6423 | 1.30 | 1.74 |
| `td_a2_carry_jump` | 0.6348 | 0.6422 | 1.30 | 1.67 |
| `td_ldelta` | 0.6338 | 0.6457 | 1.56 | 1.74 |
| `td_a2_carry_jump_ldelta` | 0.6304 | 0.6426 | 1.28 | 1.73 |
| `grw_base` | 0.6403 | 0.6433 | 1.26 | 1.19 |
| `grw_step` | 0.6395 | 0.6440 | 1.29 | 1.23 |
| `grw_step_a2_carry_jump` | 0.6377 | 0.6419 | 1.28 | 1.25 |
| `grw_step_ldelta` | 0.6315 | 0.6455 | 1.25 | 1.23 |
| `grw_step_a2_carry_jump_ldelta` | 0.6302 | 0.6434 | 1.25 | 1.26 |

### Findings

1. **The tier target A2 is the transition fix.** Every arm with A2 passes both gates, in all three
   scopes. Against its own base, relegated |bias| falls by 2.8–5.3 pp and promoted by 1.4–5.4 pp.
   No arm without A2 passes gate 2.
2. **Carry κ and jump η add nothing on the pooled TD scope.** `td_a2` alone has the best promoted
   bias there. In the lower scope, the full-pyramid anchor with carry and jump gives the smallest
   residual bias of any cell: −0.63 / +0.88 at first 20 (+2.03 / +0.89 at first 10), with both CIs
   including 0.
3. **League δ does its designed job and nothing else.** It fixes the Premiership pxG scale on the
   monitor: T54 LogLoss 0.6396 → 0.6338 on TD, and 0.6395 → 0.6315 for `grw_step`. It does not
   improve the League One/Two target, and it makes promoted clubs **worse** (Δ|bias| +0.78 and
   +0.64, both CIs above 0). Keep it only where the Premiership is priced.
4. **The GRW summer macro step improves LogLoss on its own** (−0.0020 [−0.0037, −0.0004]). It does
   not move transition bias. All four GRW arms beat `grw_base` on LogLoss with CIs below 0.
5. **Compression is not fixed on TD.** A2 leaves pooled TD at 1.51–1.61 and brings lower-scope TD
   from 1.60 to 1.46. GRW stays the least compressed (1.14–1.21); A2 adds about +0.06 there. On the
   Premiership monitor A2 does decompress TD (T54 1.58 → 1.29). §8's condition for dropping W3 (TD
   winner at compression ≈ 1.0–1.2) is **not met**, so W3 stays in.
6. **Every gate-passer is within noise of the live `m12` on target LogLoss.** `m12_td_ref` over-rates
   promoted clubs by +4.72 pp [+2.28, +7.28]. Neither co-finalist does.
7. **Entered-SPFL clubs get worse under A2 in the pooled scopes** (2 clubs, 15 fixtures; diagnostic
   only). TD moves from −9.0 to between −12.2 and −13.3; GRW from −4.4/−5.1 to −9.0/−9.1. A2 anchors
   those clubs at their first, lowest tier with no η. Revisit with B2 or wealth in W4.
8. **ECE rises on some arms:** `td_a2_carry_jump` 0.0163, `grw_step` 0.0152,
   `grw_step_a2_carry_jump_ldelta` 0.0188 and `m12_td_ref` 0.0151, against 0.010–0.011 for the bases.
   Both co-finalists are at ≤ 0.0111.

### Caveats

- **B1-literal leaks the target.** Its fixed TODO 029 priors precision-weight structural ratings
  with the **closing market's view of each transitioning club over its first five league games**,
  over 2021/22–2026/27 ([02 README](../02_cross_tier_cups_and_pyramid_eda/README.md) §7, Option B).
  That window includes this panel's relegated and promoted cohorts, and gate 2 scores exactly those
  clubs against the same market. So `td_lower_b1lit`'s lead on LogLoss and compression is not
  clean evidence, and it is not a finalist.
- **A weaker overlap exists for A2.** The step prior TruncatedNormal(0.47, 0.16) comes from
  2021/22+ results (not prices), and s is re-estimated in every fold.
- **The finalists were selected on this panel.** Treat their margins as exploratory until the W2.5
  live re-price and the 26/27 fixtures accumulate.
- **Convergence.** Every cell is under the human's rerun rule: R̂ ≤ 1.018 against 1.05, and
  divergences ≤ 70 / 240,000 (0.029%) against 0.1%. So no 0.80 reruns were made.
  - Three GRW arms carry a `review` fail from the harness's own ESS > 400 gate:
    `grw_step_a2_carry_jump` (tail ESS 372, fold 40), `grw_step_ldelta` (bulk 344, fold 59) and
    `grw_step_a2_carry_jump_ldelta` (329). ESS is not part of the rerun rule, so this is accepted.
  - A failed review stores `{}` in `harness_checks.value`. The R̂ and divergence figures above come
    from the grid logs (harness follow-up (k), TODO 031).
- **Persistence.** `grw_base` was saved at stride 2 (2,000 draws). The four GRW arms were saved at
  stride 4 (1,000 draws), because their stride-2 fits exceed PostgreSQL's 1 GiB field limit.
  `grw_base` itself is 1,015 MB at stride 2. That is ample for scoring, but the harness ladder ends
  at stride 4 (harness follow-up (l), TODO 031).
- **Stale rows.** `harness_scores` still holds pre-rescore v1.2 rows that pair `grw_base` and the
  three lower-scope cells with `td_base` (written in-process at grid time). They do not feed the
  leaderboard. Deletion awaits the human.

### Reproduction

- **Smokes:** 15/15 at `a76a65df`, each 6/6 hard checks.
- **Grids:** `a76a65df` on mcmc-beast via `harness_queue.sh`. The TD batch (10 cells) took 84 min;
  the GRW batch took 97 / 112 / 156 / 114 / 152 min per cell. Logs are in
  `/root/BF_runs/logs/a76a65df/`.
- **Scoring:** from a clean `4fc4582a` checkout with the grids' own datastore cache (md5
  `db3ed548`), `--target-seasons 23/24,24/25,25/26 --expected-fixtures 1070`, in three passes:
  1. `W2_runs.csv`: each arm against its own base;
  2. `W2_m12_vs_td_base_runs.csv`;
  3. `W2_finalists_vs_b1lit_runs.csv`.

  Then `scripts/leaderboard.jl`. Logs are in `/root/BF_runs/logs/4fc4582a/w2_score/`.
