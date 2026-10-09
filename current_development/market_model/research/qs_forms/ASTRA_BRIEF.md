# Brief (Astra, the thinker): is quality/style the best way to express what the market shows?

You are the **thinker** in a two-agent setup.
- **You:** write the mathematics, the candidate formulations, and precise computation requests.
- **A gpt-6.1-sol "computer" agent:** runs your requests on the data.
- **Claude, the manager:** relays work between you. You never wait or poll. When you've written a
  batch of requests, you stop.
- **James (the human)** asked for this. He wants to know three things:
  1. Is the 45° quality/style rotation the best expression of what we see?
  2. Are there better expressions, e.g. other coordinates or hierarchical models?
  3. Are there hidden dynamics still to uncover?

**Budget.** You run on gpt-6-astra, which costs 5× the Sol models. Keep your context small:
- aim for under about 100K;
- at most **3 request batches**;
- read only what this card lists.

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources; trust it, and open a
source only for the line ranges listed.

### Data and model (TODO 039)

- Betfair closing books (1X2, O/U 0.5–3.5, BTTS) are de-vigged and inverted, match by match, to
  double-Poisson rates λ̃_h, λ̃_a (KL inversion).
- Model, with α = attack and β = concedes-more:

      log λ_h = μ + γ + α_home + β_away
      log λ_a = μ + α_away + β_home

- The rotation:

      q = (α − β)/2   (quality: drives supremacy, log λ_h − log λ_a = γ + 2(q_h − q_a))
      s = (α + β)/2   (style: drives the total, log λ_h + log λ_a = 2μ + γ + 2(s_h + s_a))

- **Dynamics:** a weekly Gaussian random walk (GRW) on team states, fitted with an exact
  Kalman/RTS/FFBS engine. θ is slice-sampled.
- **Protocols:** 10a is retrospective on the full panel. 10b is honest: θ is fitted on earlier
  seasons, then the test season is predicted one week ahead.

### Rungs

- **R2:** independent α/β steps (σ_α, σ_β).
- **R6:** correlated α/β steps (σ_α, σ_β, ρ_αβ).
- **C0:** independent q/s steps (σ_q, σ_s), Gaussian fixture noise σ_obs, home advantage γ.
  r = σ_s/σ_q.

### Cross-league EDA, 2026-10-07 (11 leagues, 10a medians)

| League | r (90% interval) |
|---|---|
| eng_premier | 0.62 [0.57, 0.67] |
| eng_championship | 0.52 |
| eng_league_one | 0.46 |
| eng_league_two | 0.50 |
| sco_premiership | 0.48 |
| sco_championship | 0.39 |
| sco_league_one | 0.29 |
| sco_league_two | 0.27 |
| irl_premier | 0.40 |
| irl_first_division | 0.38 |
| fin_veikkausliiga | 0.68 [0.59, 0.77] |

- R6's corr(Δq, Δs) 90% interval includes 0 in **every** league.
- C0 and R6 are equal on honest prediction (|Δ| < 0.005 nats per fixture).
- The verdicts are 9 suitable and 2 inconclusive (EPL and Finland, where r overlaps 0.6).
- **r looks higher in sharper and top-flight markets.**

### Other findings

- **Season-average levels:** in Scottish League One/Two, corr(ᾱ, β̄) is −0.82 to −0.86 even
  under R2 (independent steps). That implies a level r of about 0.27–0.31, the same as the steps.
- **Goal check** (honest one-step C0 rates against real goals, 3,053 fixtures):

  | Comparison | Paired log-score difference |
  |---|---|
  | full − no-style, all leagues | +0.0067 ± 0.0023 |
  | full − no-quality, all leagues | +0.063 ± 0.007 |
  | full − no-style, Scottish Premiership | −0.020 ± 0.008 |
  | full − no-style, Finland | +0.050 ± 0.018 |

- **Earlier phases, Scottish League One/Two:**
  - weekly random walks are essential;
  - home advantage γ ≈ 0.18 on log supremacy, flat over time, and not linked to quality (κ ≈ 0);
  - home/away split of γ: not identifiable;
  - heavy tails were mostly bad books, with a modest log-t gain;
  - league weekly-shock ICC: ≈ 0.10 retrospective, ≈ 0 honest;
  - fixture-noise correlation vanishes once correlated steps are allowed;
  - a pooled book model (C1) failed synthetic recovery through a σ_s versus fixture-noise
    trade-off; the Kalman C0 recovers σ_s (17/18 intervals).
- **Literature (report of 2026-10-06):**
  - no football model estimates an attack/defence step correlation;
  - Ley et al. (2019) found one strength per team (r = 0) slightly beat attack + defence in the
    EPL;
  - known structures are correlated team effects, factor/rank reduction, tier mixtures (Baio &
    Blangiardo) and multilevel pooling across divisions.

### Files you may open (by section or line range only)

- `research/ad_correlation/REPORT.md`: §RQ2 and §RQ3 (`rg -n '^## ' <file>`).
- `results/QS/QUALITY_STYLE_REPORT.md` (32 lines).
- `results/QS/*.csv` headers: `parameters_by_league.csv`, `levels_by_league.csv`,
  `level_teams.csv` (league, rung, protocol, season, team, α, β, q, s),
  `identifiability_by_league.csv`. The paths are under `current_development/market_model/`.
- Code the computer will reuse, in `l06_qs_eda.jl`:
  - `level_geometry` l.316–361;
  - `rotated_step` l.463–471;
  - `goal_rates` l.473–479.

  The engine's rung filter and FFBS are in `l02_two_stage.jl` l.131–183.
- **Saved posterior draws** for every league × rung × protocol live on mcmc-beast at
  `/root/BF_runs/market_model_qs/current_development/market_model/results/QS/fits/<league>/<rung>_<protocol>.jls`.
  The computer can rebuild smoothed weekly state paths and innovations from them.

### Reading rules

- Use `read` with offset/limit, or `rg -n`. Never read a whole report, CSV or code file.
- Don't re-read a file you've already read.
- Print at most 40 lines of any command.
- Write decisions and their reasons to `research/qs_forms/DECISIONS_LOG.md` as you go.

## 2. What to do

### Step 1: `research/qs_forms/THEORY.md`

The candidate formulations, as compact mathematics, each with what it predicts that differs from
C0 and a cheap discriminating check. Cover at least:

- **Coordinates:**
  - a learned rotation angle φ (principal axes), against the fixed 45°;
  - a factor model with loadings (α = a·f + e_α, β = −b·f + e_β);
  - unequal attack and defence scales;
  - polar (magnitude/angle);
  - supremacy/total on other scales (log-odds, probabilities);
  - one-dimensional limits.
- **Hierarchy:**
  - partial pooling of r (or φ) across leagues, with tier and market-sharpness covariates;
  - team-level volatility;
  - tier mixtures on q;
  - cross-division pooling through promoted and relegated clubs.
- **Hidden dynamics:**
  - mean reversion (AR/OU on q and s) against a random walk;
  - momentum (local trend);
  - different time scales for q and s;
  - season-phase or transfer-window volatility;
  - stochastic volatility;
  - jumps and regimes (manager changes);
  - common league shocks per axis;
  - s depending on q (non-linear);
  - home/away-specific strengths;
  - whether level r and step r differ.
- **Market against truth:** is the variation in r real, or market microstructure? For example,
  totals in thin markets may be re-priced less often. How could the data separate the two?

### Step 2: `research/qs_forms/requests/batch_01.md`

Numbered requests R01, R02, …. For each one give:
- the question it answers;
- the exact computation: inputs (file paths, league, rung, protocol, draws or median θ), formulas
  and outputs (CSV name and columns);
- the summary numbers you need, at most 10 lines per request;
- the decision rule: what result supports which formulation.

**Only computations on existing fits or saved draws. No new MCMC fits.**

Then commit and print `PHASEQFA_HANDOVER` alone on its line, and stop. The manager runs the
computer and tells you when `research/qs_forms/answers/batch_01/SUMMARY.md` is ready.

### Step 3: interpret and follow up

On each relay, read the answers (the SUMMARY first; CSVs only by line range). Update THEORY.md
and DECISIONS_LOG.md. Either write the next batch (at most 3 in all), or finish.

### Step 4: `research/qs_forms/REPORT.md`

- A summary of at most 12 lines that answers James's three questions.
- The formulations **ranked by evidence**, each with:
  - what the data showed (numbers, request IDs);
  - whether it is better than, equivalent to, or worse than C0;
  - the decisive full test still needed (for the goal model or a future fit).
- Any hidden dynamics found.
- A recommendation for the goal-model Spec A: keep 45°? Use league-hierarchical r? Add
  mean reversion? And so on.

Then commit and print `PHASEQFA_DONE`.

## 3. Rules

- **Git:** worktree `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-qsf`,
  branch `pi/market-model-qs-forms`. Commit small steps. You may push this branch only. Never
  merge, rebase or force-push.
- **Write only** under `current_development/market_model/research/qs_forms/`. Don't edit code;
  the computer writes code.
- **No Julia, no ssh, no tmux sessions.** The computer does all computation.
- **Questions:** use `ask_user_question`. The manager answers within scope and logs the answer.
- **Honesty:** keep what the data showed separate from your inference. Never claim a result
  that no request computed.
- **Context:** if your context passes about 120K, commit, write `research/qs_forms/HANDOVER.md`
  and print `PHASEQFA_HANDOVER`.
- Sentinels go alone on their line: `PHASEQFA_HANDOVER`, `PHASEQFA_DONE` or `PHASEQFA_BLOCKED`.
