# Market model — inverting the close and decomposing it into team strength

**TODO:** [039](../../todos/039_market_model_inversion_and_team_strength_eda.md)
**Branch:** `feat/market-model-inversion-eda`, off `feat/market-inverse-grw-dynamics` (TODO 023)
**Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-market-model`
**Status:** design agreed with the human 2026-10-05; not started.

---

## 1. Why

The human's design note (Boox, `~/Downloads/market_model_2026-10-05.pdf`, handwriting only)
proposes a model fitted to the *market* instead of to goals:

```
market odds ──f──▶ latent rates (λ̃_h, λ̃_a) per match ──▶ GRW on α, β

log λ_h = μ + α[home] + β[away] + γ
log λ_a = μ + α[away] + β[home]

goals model:   y_h ~ Poisson(λ_h),      y_a ~ Poisson(λ_a)
market model:  λ̃_h ~ Gamma(λ_h, ν),     λ̃_a ~ Gamma(λ_a, ν)     ← Gamma is a placeholder
```

`f` inverts a match's odds (1X2, O/U 0.5–3.5, BTTS) to the rates of a double Poisson.
The aim is to explore the merits of such a model: the odds aggregate information beyond
the raw stats everyone has. **"Dixon–Coles" in this stream means the attack/defence/home
parameterisation above, not the ρ low-score correction.** The score grid is a plain double
Poisson throughout.

This stream answers three questions before any forecasting work:

1. **How to invert.** Each match in isolation, or as a group that borrows from other matches?
2. **What the latent rates look like** (EDA), including which distribution fits λ̃.
3. **How well log λ̃ decomposes into team strength**, rung by rung, and what α, β look like.

Scoring the market model as a forecaster of results is the follow-on, not this stream.

## 2. Decisions (grill, 2026-10-05)

| Question | Decision |
|---|---|
| Isolated or group | **Ladder:** isolated per-match inversion first (the measurement), then a pooled fit of the team structure straight to the books with a per-match deviation u_m |
| Score grid | Double Poisson (ρ = 0) |
| Inversion loss | KL(market ‖ model), summed over markets |
| Panel | Tournaments 56/57, seasons 24/25 + 25/26 (710 fixtures) |
| Price | Betfair TWA(−20, 0] close, de-vigged within (match, market, line) |
| EDA strands | Inversion quality · rate distributions · rates vs realised goals · isolated vs pooled · team-strength parameters |
| Structure rungs | Static per team-season · weekly GRW · per-team home advantage · fixture deviation u_m (Gaussian vs Student-t) |
| Measures | Smoothed variance explained · one-step-ahead error · split by supremacy and total axes · per team and week |
| Engines | Two-stage rungs on TODO 023's exact Kalman/FFBS engine; pooled step as a standalone Turing NUTS model, MAP as its smoke check |
| Book weight (pooled) | One learned concentration n |
| Team-parameter EDA | vs goal-model ratings · attack vs defence · paths over time · stability across fits |
| Gamma question | Fit Gamma / log-normal / log-t on the static rung; compare by held-out density on the λ̃ scale |
| Bad books | Isolated step requires a 1X2 book (local gate, T015); pooled step keeps totals-only books; T014's swapped book dropped |
| Phasing | 4 phases (A–D), pi Sol builds, Claude reviews each, human signs off before the next |
| Output | Findings README + CSVs under `results/` + PNG figures |

## 3. What already exists, and what is reused

| Piece | Where | Use here |
|---|---|---|
| Close book, de-vig | `market_inverse_dynamics/l01…:closing_book` | reused as-is |
| SSE inversion, double Poisson | `src/Calibration/rate_pool.jl:invert_market_rates` | **control only**: the KL inversion is new and lives in this stream's loader, built on the same `Features.build_probability_matrix` |
| Market panel, weekly grid | `l01…:MarketPanel`, `build_market_panel` | reused; a method that takes an external inversion frame is added here |
| Kalman / FFBS / RTS, GRW1, GRW1Break, Student-t scale mixture, static feature states | `l01…` (TODO 023 Phases 1–2) | the whole two-stage ladder |
| Engine gates (batch joint Gaussian) | `l01…:mid_gates` | rerun as Phase B's first gate |
| TODO 023 rates (SSE) | `market_inverse_dynamics/results/production/market_targets.csv` | regression reference for the KL rates |
| Ratings fitted to inverted closes | CLV napkin §14 (`proto/clv-napkin`, not merged) | cross-check only (KL + bivariate grid, ridge RW, one-step RMSE 0.095) |

Known facts to beat or confirm (TODO 023, 623 SSE rates): sd(log λ̃) ≈ 0.23 per side; supremacy
sd 0.42; log-total sd 0.10; corr(log λ̃_h, log λ̃_a) = −0.66. One rating per team over both seasons
leaves RMSE 0.149 (≈ 59% of variance explained); GRW smoothed 0.076 (≈ 89%); GRW one-step 0.127
(≈ 70%); Student-t σ 0.055 with ν ≈ 2.9.

## 4. Method

### 4.1 Isolated inversion (Phase A)

For match m and each quoted market k ∈ {1X2, BTTS, O/U 0.5, 1.5, 2.5, 3.5}, with de-vigged
probabilities p_{m,k} and double-Poisson grid probabilities q_{m,k}(λ_h, λ_a):

```
(λ̂_h, λ̂_a) = argmin Σ_k KL(p_{m,k} ‖ q_{m,k}(λ_h, λ_a)),     max_goals = 10
```

Gates: a 1X2 book present (T015), ≥ 3 quoted selections, optimiser converged, KL ≤ a threshold
read off the KL distribution and reported, λ ∈ [0.05, 6]. Match 14035501 (T014) is dropped.
About 595 of 710 fixtures should pass (623 − 27 totals-only − 1).

### 4.2 Two-stage ladder (Phase B): Kalman on the isolated log-rates

```
y_{m,h} = log λ̂_h = μ + γ_(h) + α_{h,t} + β_{a,t} + ε_{m,h}
y_{m,a} = log λ̂_a = μ +         α_{a,t} + β_{h,t} + ε_{m,a}
```

| Rung | Team states | Home advantage | ε |
|---|---|---|---|
| R0 | none | league γ | Gaussian |
| R1 static per team-season | constant within a season, one jump over the summer (GRW1Break, in-season step fixed at 0) | league γ | Gaussian |
| R2 weekly GRW | TODO 023 a1 | league γ | Gaussian |
| R3 + per-team HA | as R2 | γ_i ~ N(γ, σ_γ²), static states | Gaussian |
| R4 + heavy tails | as R3 | as R3 | Student-t (scale mixture) |

Each rung reports, on log λ_h/log λ_a **and** on the supremacy axis log(λ_h/λ_a) and the total
axis log(λ_h + λ_a):

* smoothed variance explained (in-sample R², RMSE);
* one-step-ahead error, pre-week filtration, with θ both plug-in (full panel) and honest
  (fitted on 24/25, scored on 25/26), as TODO 023 §5;
* residuals per team and per week.

### 4.3 The Gamma question (Phase B)

On rung R1 (few parameters, Turing), three observation families for λ̃ given the structured
mean exp(η):

* Gamma with mean exp(η) and shape ν (the note);
* log-normal: log λ̃ ~ N(η, σ²);
* Student-t on log λ̃.

Compare by PSIS-LOO **on the λ̃ scale** (Jacobian −log λ̃ for the two log-scale families), on all
books and on books with ≥ 5 selections, plus a binned mean–variance plot (constant CV points to
Gamma/log-normal, variance ∝ mean to quasi-Poisson). The winner carries into the later rungs.
Expected: with σ ≈ 0.06–0.09, a Gamma's shape is ~120–330 and the three bulks coincide, so the
tails decide.

### 4.4 Pooled fit (Phase C): the team structure fitted straight to the books

```
log λ_{h,m} = μ + γ_(h) + α_{h,s} + β_{a,s} + u_{m,h}
log λ_{a,m} = μ +         α_{a,s} + β_{h,s} + u_{m,a}
u ~ N(0, σ_u²)  (and a Student-t variant)

p_{m,1X2}  ~ Dirichlet(n · q_{m,1X2}(λ_{h,m}, λ_{a,m}))
p_{m,k}    ~ Beta(n · q_{m,k}, n · (1 − q_{m,k}))       two-way markets
```

* One shared concentration **n** is learned. A tempered cross-entropy n·Σ p log q is not a
  proper density in p, so n would be unidentified; the Dirichlet/Beta is the proper version and
  its large-n limit is the KL of §4.1. n is identified by how far a book's own lines disagree
  with the grid; σ_u by how far matches deviate from the team structure.
* u_m free reproduces the isolated inversion; u_m = 0 is pure structure; σ_u sits between, so
  thin books shrink toward the structure. A totals-only book informs only the total.
* Rungs: P1 static per team-season, P2 weekly GRW (non-centred).
* Fit: MAP smoke, then NUTS (4 chains, ReverseDiff compiled tape). One-step-ahead: MAP refits at
  each of the 40 fold cutoffs (`GroupedCVConfig`, `match_biweek`), scored on the next fold's books.
* Known approximation: nested O/U lines of one book are treated as independent Betas, which
  over-counts totals information. n absorbs part of it; it is reported, not hidden.

### 4.5 EDA catalogue

| Strand | Contents | Phase |
|---|---|---|
| Inversion quality | per-line residual (q − p) by line, book size and league; KL distribution; **held-out-line test**: invert on 1X2 + O/U 2.5 only, predict O/U 0.5/1.5/3.5 and BTTS; KL vs SSE rates against TODO 023 | A |
| Rate distributions | λ_h, λ_a, total, supremacy by league, season and month; home advantage; λ_h–λ_a correlation; mean–variance | B |
| Rates vs realised goals | Poisson regression of goals on log λ̃ per side and on the total; binned calibration of totals; the 0.81 slope check | B |
| Isolated vs pooled | per-match Δ log λ by book size; the 27 T015 books; 3-price books; shrinkage against n_lines | D |
| Team-strength parameters | distributions; attack–defence correlation; spread per tier and season; the L1/L2 offset (a gauge direction with no tier term); paths for transition clubs (Ross County, Airdrie, East Kilbride, Kelty), season starts and ends; stability across rungs and two-stage vs pooled; vs goal models m01 GRW `2b42d3bf-28d7-47ac-8706-88798c9031ac` and grw_spfl_joint `82e01a3b-d686-4e29-a269-3702b96fc5ba` (spread ratio, correlation, biggest disagreements; centred within 56/57) | D |

## 5. Phases and acceptance

Each phase ends with a sentinel (`PHASE<X>_DONE` / `PHASE<X>_BLOCKED`), a Claude review against
this list, and a human sign-off before the next brief is released.

### Phase A — KL inversion and inversion quality

- [ ] `l01_market_model.jl` (module `MarketModel`) includes TODO 023's loader and adds the KL
      inversion, the 1X2 gate, and a `build_market_panel` method that takes an inversion frame.
- [ ] Synthetic recovery: books generated from known (λ_h, λ_a) on a grid invert back to 1e-6.
- [ ] `r01_kl_inversion.jl` writes `results/A/rates.csv` (one row per fixture: λ̂, KL, lines,
      flags, reason) and a refusal table by reason.
- [ ] KL vs SSE: distribution of Δ log λ against `market_targets.csv`, the 27 T015 fixtures
      shown refused.
- [ ] Inversion-quality tables and figures of §4.5, including the held-out-line test.

### Phase B — two-stage ladder, distributions, rates vs goals

- [ ] TODO 023 engine gates rerun and pass.
- [ ] Rungs R0–R4 fitted on the KL rates; R̂ ≤ 1.05, bulk and tail ESS ≥ 200 for every parameter.
- [ ] Measures of §4.2 for every rung, both axes, both θ protocols.
- [ ] Gamma / log-normal / log-t comparison of §4.3 (Turing; R̂, ESS and divergences ≤ 0.1%).
- [ ] Rate-distribution and rates-vs-goals strands of §4.5.

### Phase C — pooled fit

- [ ] `l02_pooled_market_model.jl`: the Turing model of §4.4.
- [ ] Synthetic recovery: books simulated from known α, β, u, n are recovered within their
      90% intervals.
- [ ] P1 and P2 by MAP, then NUTS; R̂ ≤ 1.05, ESS ≥ 200, divergences ≤ 0.1%.
- [ ] n and σ_u reported with intervals; posterior predictive check of KL by book size.
- [ ] One-step-ahead MAP refits at the 40 fold cutoffs.

### Phase D — comparisons and findings

- [ ] Isolated vs pooled and team-parameter strands of §4.5.
- [ ] `README.md` findings: every number copied from a CSV under `results/`; figures under
      `results/figures/`.
- [ ] TODO 039 updated with verification and findings.

## 6. Files

| File | Is |
|---|---|
| `DESIGN.md` | this document |
| `l01_market_model.jl` | KL inversion, gates, panel bridge, two-stage rung definitions, R1 family models |
| `r01_kl_inversion.jl` | Phase A runner |
| `r02_two_stage_ladder.jl` | Phase B runner |
| `l02_pooled_market_model.jl`, `r03_pooled_fit.jl` | Phase C |
| `r04_comparisons.jl` | Phase D |
| `results/{A,B,C,D}/`, `results/figures/` | outputs |

Runners follow `docs/prototype_runner_style_guide.md`.

## 7. Environment

* Long fits run on mcmc-beast with `-t 16` (not 32). The kaimon REPL was not reachable on
  2026-10-05; fall back to ssh + tmux.
* `Distributions` stays pinned at 0.25.126.

## 8. Out of scope

Scoring the market model as a forecaster of match outcomes and bets; the Dixon–Coles ρ and
other score grids (bivariate, COM); a builder-native observation type; fixing T015 in `src`.
