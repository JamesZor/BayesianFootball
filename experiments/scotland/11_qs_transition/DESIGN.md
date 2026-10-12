# 11 — QS transition experiment: why QualityStyleGRW misprices clubs that change tier, and how to fix it

TODO 046. Branch `exp/qs-transition` (worktree `.worktrees/BayesianFootball-qs-transition`), cut from
`exp/qs-goal-vs-market` at `94aef191`. Continues TODO 031 W2 (`../04_pyramid_tier_transition_arms/`).
Scope agreed with James in three AskUserQuestion rounds on 2026-10-11.

## 1. The problem

- **Live evidence.** In the 26/27 paper trade (TODO 045, `../10_qs_marketobs_2627_paper/`), qs_marketobs
  bet 37 times on fixtures of the four clubs that changed tier (Ross County, Airdrie, Kelty, East
  Kilbride). Those bets returned −39%, against +22% on the other 54 (descriptive).
  - The model bet against Ross County five times at an average 11.7 pp "edge", and lost all five.
  - The losses come in every fold, not only the opener. CLV on these bets was −0.7 to −1.4 pp,
    against −0.2 to −0.8 pp on the others.
  - 1X2 log loss vs the mid close: qs +0.030 on transition fixtures, −0.013 elsewhere.
- **Earlier evidence (W2, 28 Sep).** Every base model under-rates relegated clubs (−4 to −10 pp win
  probability, first 20 league games) and over-rates promoted clubs (+3 to +5 pp).
  - A2 (the `PyramidTiers` tier target) cut that to −0.6 / +0.9 pp on `td_lower_a2full_carry_jump`.
  - A2 was never applied to QS.
  - The W2 finalists were selected on the panel they were scored on. The live check (W2.5) never ran.
- **Why QS is exposed.**
  - Every club starts at the league average plus noise (zero-centred `z_init ~ N(0,1)·τ₀`).
  - QS has no tier information.
  - In lower scope, a club relegated from the Championship has no League One/Two history.
  - qs_marketobs sees the market's re-rating only through past folds, and folds are biweekly.

## 2. Candidate causes (Stage 1 must separate them)

| | Cause | Signature in the bias-by-game curve |
|---|---|---|
| a | **Cold start:** the club starts at the league mean | large at game 1, falls as the club's own data arrives |
| b | **Tier scale:** the model has no value for a tier | the same sign for every club in a direction, persisting |
| c | **No summer step:** the season boundary carries a biweek of movement | non-movers also lag at the season start |
| d | **Lag:** the market re-rating reaches QS one or two folds late | the bias drops at each refit boundary |

## 3. Candidate fixes (factors)

| # | Fix | Build | Notes |
|---|---|---|---|
| F1 | **Per-club market prior:** each club's starting q/s from its inverted closes in earlier seasons, all four tiers | src: QS prior means | James's idea. **Re-derived per fold from earlier seasons only** (B1 leaked) |
| F2 | **Transition shift from past market re-ratings:** per direction or per group of clubs | with F1 | Estimated from movers in seasons before the fold |
| F3 | **A2 on QS:** `PyramidTiers` (tier step, full-pyramid anchor, carry κ, jump η) | probably recipe only | A2 is a builder predictor term, separate from the dynamics (`src/models/pregame/builder/pyramid_tiers.jl`); verify that it composes with `QualityStyleGRW` |
| F4 | **Summer macro step for QS** | src: QS keyword | `target_season_step` exists only on `MultiScaleGRW` |
| F5 | **More data:** pool 56/57 with the Championship (55), the Premiership (54) and cups | recipe plus scope | Cup ties identify the tier step. QS needs a pooled scope |
| F6 | **Weekly refits** instead of biweekly | splitter | The cheapest fix for cause (d) |
| F7 | League δ (W2's pxG scale fix) | exists | Only if a pooled arm uses pxG |

src changes are allowed if they are opt-in and off by default. Waves 1–5 and the frozen wave-5 config
must reproduce bit-identically. Each new component needs exact AD (≤ 1e-10), zero-allocation tapes,
3-seed recovery and unit tests.

## 4. Decisions made (2026-10-11)

1. **Goal:** find the cause, then fix it. Stage 1 diagnoses; later stages test only the fixes that
   target the causes found.
2. **Target panel:** W2's, 23/24–25/26, tournaments 56/57, 60 folds, 1,070 fixtures.
   - Cohorts: 5 relegated into L1, 6 promoted into L1, 2 entered SPFL.
   - Directly comparable with W2's A2 numbers.
3. **Scopes:** all of lower (56/57), + Championship, full SPFL (54–57), and + cups. An EDA with all
   the data comes first ("play around with it").
4. **Control:** qs_marketobs as is (QS + goals + market-rate observation, lower scope). Each fix is
   paired against it.
5. **Gate (W2's):**
   - Transition bias = mean(model − market) win probability of the transitioning club over its
     first 20 league games. Club-season-clustered CIs. The relegated_into_L1 and promoted_into_L1
     cohorts gate.
   - Plus log-loss non-inferiority: paired fixture-clustered CI upper < +0.002.
   - Harness code: `src/harness/cohorts.jl` (`club_season_tiers`, `transition_cohort`).
6. **Prior form:** the EDA proposes it. James signs off in the morning before any fit.
7. **Final test:** the 79 played 26/27 matches (4 transition clubs, 28 fixtures) are held back. They
   are scored once, on the frozen winner, at the end. The EDA may describe 26/27 coverage only.
8. **Agents:** pi. Stage 0 is research on `gpt-6-astra` (library and web). Later stages use
   `gpt-6.1-sol`, with a fresh pi per stage. Claude (`%113`) manages. Overnight: research, then the
   manager merges its questions into the EDA brief and launches the EDA. No fits and no src changes
   overnight.

## 5. Stages

| Stage | What | Agent | Gate to the next |
|---|---|---|---|
| S0 | **Research:** literature and web on promotion/relegation effects, cold-start priors, market-informed priors and hierarchical league models. Produces EDA questions and candidate prior forms | pi astra | Manager merges the questions (logged) |
| S1 | **EDA** on all Scottish data, 21/22–25/26 (§6). No new fits | pi sol 6.1 | **James's morning review:** prior form, which causes are live, arm design |
| S2 | **Build and smoke** the components the EDA justifies (F1/F2 prior, F4 step, pooled QS scope). Verify that F3 composes with QS | pi sol 6.1 | Gates in §3 |
| S3 | **One factor at a time** vs control on the W2 panel | pi sol 6.1 | W2 gate |
| S4 | **Combinations** of the passing fixes, plus scope arms | pi sol 6.1 | W2 gate; pick one winner |
| S5 | **Backtest** of the winner (wave-4/5 pipeline) and the **one-time 26/27 test** | pi sol 6.1 | Report |

The arm design for S3/S4 (one factor then combine, a small factorial, or diagnosis-led only) is
decided after the EDA. The manager recommends one factor, then combine.

## 6. Stage 1 EDA questions (base set; S0 research adds to it)

1. **How the market re-rates movers.** Per direction (relegated into L1/L2, promoted into L1/L2,
   entered SPFL), from inverted closes:
   - the club's rating in its new tier against its old-tier rating the season before;
   - how fast the market settles: the first quote, then games 5, 10 and 20.

   This gives the empirical shift and its spread, which become the F2 prior.
2. **Bias decomposition of existing fits**, by game number 1–20 and by fold age:
   - qs_marketobs `81fdf817` (24/25–25/26 cohorts);
   - the W2 fits on 23/24–25/26: `td_lower_base` `0dd74a52`, `td_lower_a2full_carry_jump`
     `6c0aa3ea`, `grw_base` `ce7ea22f`, `grw_step_a2_carry_jump` `c4a0fa94`, `m12_td_ref` `973baafa`.

   Map the patterns onto causes a–d.
3. **The value of a tier:**
   - from cup ties (goals, and SofaScore bookmaker odds, since cups have no Betfair);
   - from movers (market);
   - compare with A2's 0.47; check linearity across tiers 1–4 and the Old Firm.
4. **Summer movement:** how much non-moving clubs' market ratings change between seasons, against
   QS's boundary step.
5. **Coverage:** market history per club and season across tiers, including the Championship 25/26
   gap (7 of 175 closes), and clubs with no market history.
6. **A proposed prior form** for F1/F2, written as a per-fold, earlier-seasons-only rule with a fallback
   for clubs without market history, plus an out-of-sample check of the prior on its own: for movers
   before the panel, how well the prior predicts the market's first-quote rating.

## 7. Data facts

- **Inverted closes** in `current_development/market_model/results/QS/rates_sco_*.csv`:
  - Premiership 21/22–25/26 (956 accepted);
  - Championship 21/22–24/25 (477; 25/26 excluded);
  - League One 835;
  - League Two 693.

  Betfair coverage for 56/57 starts at 21/22.
- **Cups:** Scottish Cup, League Cup (knockouts only) and Challenge Cup are in betdb. Cups have no
  Betfair, but SofaScore `match_odds` 'Full time' covers all three.
- **Tiers:** `club_season_tiers` (`src/harness/cohorts.jl`).
- **pxG** starts 23/24. Premiership pxG reads about 21% high relative to goals (W2 step 0).
- **Leakage rules:**
  - Any market-derived prior is re-derived per fold from seasons strictly before the fold's target
    season.
  - No 26/27 tuning.
  - The read-only DB audit wraps every runner.

## 8. Overnight run (James, 2026-10-11 evening)

James asked for **all stages in order, without waiting for his sign-off**. He reviews the results in
the morning with an agent.
- The manager (Claude, `%113`) takes each gate decision: the prior form after S1, the arm design
  (default: one factor, then combine), and the winner after S4. Each decision is logged with its
  reason in `MANAGER_DECISIONS.md` (this folder) and in the night log.
- **Every stage writes `REPORT_S<n>.md`** in this folder: a summary of at most 12 lines, then its
  numbers with sources. The morning review starts from those.
- Each stage gets a fresh pi (S0 gpt-6-astra; S1–S5 gpt-6.1-sol). If a 5-hour limit is hit, the
  manager resumes at the reset on the same model.
- The 26/27 test still runs once, at the end of S5, on the frozen winner.
- Stop rules: any failed reproduction or leakage gate stops the chain at that stage, with a report.
  No merges to other branches. Push `exp/qs-transition` only.

## 9. Metric amendment (James, 2026-10-12 ~17:45): absolute transition error

**Why.** The frozen gate 2 asks the signed mean bias to move toward zero in both directions. The
control's relegated mean (+0.07 pp) is near zero only because errors cancel, over time (+1.91 at
games 2–5, −0.47 at 11–20) and across clubs (Inverness −0.36, Hamilton +4.05). So the gate penalises
fixes that reduce per-club misses. The market is the reference (S1 Q10 found no evidence that it
misprices movers), so model − market in **either** direction is error.

**Metric** (for relegated_into_L1 and promoted_into_L1; also reported for entered-SPFL):
- **Primary, MAE_club:** for each transitioning club-season, the mean of |P_model(club win) −
  P_market(club win)| over its first 20 league games; then the mean across club-seasons.
  - Δ vs `qs_control` is paired, with a club-season-clustered bootstrap (B = 10,000, 95%).
  - **Pass:** neither direction's Δ is positive in point estimate, and at least one direction's Δ CI
    upper is < 0.
- **Secondary:**
  - worst club: the maximum over club-seasons of |mean signed error over its first 20 games|;
  - the primary metric over games 1–10 and 11–20 separately.
- **Gate 1** (log-loss non-inferiority) is unchanged and still required.

**Status.**
- **Post hoc** on the 23/24–25/26 panel (chosen after S3). It is reported next to the frozen gate,
  which stays in the record.
- **Pre-registered** for the one-time 26/27 test (S5). This commit predates any 26/27 scoring.
