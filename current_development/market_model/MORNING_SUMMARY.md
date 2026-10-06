# Morning summary, TODO 039 market model (night of 2026-10-05/06)

**Status: draft, written while Phase C was still running.** Phase C's section is filled in when it
completes. Full chronology: `NIGHT_LOG.md`.

## What finished and was verified overnight

| Phase | Result | Verified by the manager |
|---|---|---|
| B3 (a) copula/tail check on team steps | **Gaussian dependence is adequate.** No posterior predictive p-value is extreme (0.28–0.63) on kurtosis, Kendall's τ vs the Gaussian copula, joint 95% tails in both directions, or volatility clustering. **No copula needed for attack/defence.** | fresh tests; independent beast rerun, 35/35 hashes |
| B3 (b) score-grid comparison | **Dixon–Coles with one global ρ = −0.064 is the best grid** (beats bivariate Poisson λ₃ and Frank copula): see the table below. Outcome log loss is unchanged for every grid (all CIs include 0). | same |
| C (pooled model) | *in progress*; see below | |

**B3 score grids:**

| Measure | Double Poisson | Dixon–Coles global ρ |
|---|---:|---:|
| Draw mispricing | −1.34 pp | −0.14 pp |
| BTTS-yes held-out error | −1.2 pp | −0.17 pp |
| 1X2-only totals bias | −0.103 | −0.009 |

R6 refitted on the Dixon–Coles rates keeps its structure (ρ_αβ −0.83).

**This makes the Dixon–Coles grid the natural Phase C2**, a decision for you. As you decided,
Phase C ran on double Poisson.

## Phase C: what happened

Four revisions, each blocked or narrowed for a stated reason. **No threshold was loosened.**

1. **Revision 1 (manager's brief) was wrong in three places.** pi caught all three at preflight,
   before any sampling:
   - the Dirichlet large-n limit is −n·KL(q‖p), the reverse of Phase A's KL;
   - a supremacy-only projection can't pass a gate against the full likelihood;
   - "total only" is a curved contour, not a fixed direction.
2. **Revision 2:** the exact Dirichlet likelihood with a joint-mode Laplace update.
   - The marginal is very accurate.
   - The posterior **mean** missed a 5% gate (up to 9% of SD) at low n with wide predictions.
3. **Revision 3:** a skewness-corrected (second-order) mean.
   - **Full books now pass everything.**
   - **Thin books fail at wide predictions** (BTTS + O/U badly; O/U-only marginally), because
     curved likelihood ridges defeat Laplace.
4. **Revision 4:** C0, C1 (full books) plus your home-advantage rungs H1/H2. **C2 (thin-book
   pooling) is deferred to you.**

**Preliminary results (first fast-rung attempt, unreviewed; final runs pending):** your
home-advantage question. Source: beast
`market_model_c_fast_final/.../results/C/v8_scalar_plot_attempt/home_advantage_rungs.csv` and
`paired_vs_c0.csv`.

- **The market prices home advantage at about 0.18 on supremacy, tightly:** C0's γ = 0.179
  [0.171, 0.187] (10a), 0.183 [0.170, 0.195] (10b).
- **No sign it scales with team quality (H2):** κ = 0.008 [−0.036, 0.050] (10a), 0.034
  [−0.049, 0.126] (10b). Paired honest log density against C0 is +0.0001 ± 0.0005 (10a) and
  −0.002 ± 0.004 (10b).
- **H1, home scoring vs away suppression, is not identifiable, and that's my design error.**
  Every league match has exactly one home team, so the split is aliased with the league mean μ.
  Only the sum (supremacy lift 0.179 [0.171, 0.187]) is identified; the level part is prior-only
  (0.07 [−0.21, 0.34]). Telling them apart needs neutral-venue matches.
- **C1 (the pooled book model):** the synthetic recovery and both production fits are running on
  the beast at the prescribed budget; ETAs are in pi's progress report. Expect this evening
  (about 17 h each); the budget decision is yours.

## Decisions waiting for you

1. **C2 thin-book method** (`experiments/pi_market_model_phaseC_v4_scope.md`):
   - (i) exact moment matching by quadrature;
   - (ii) NUTS on the exact Dirichlet;
   - (iii) gate at realistic prediction spreads (a setting change);
   - (iv) restrict thin-book types (a data change).
2. **Phase C2 score grid:** adopt Dixon–Coles global ρ (B3's winner)?
3. **Review a decision I took for you:** totals/BTTS-only books use their **exact** likelihood,
   not a "total-only" projection. O/U-only books still carry no supremacy information; BTTS + O/U
   books add the weak supremacy information they contain.
4. **Answers I gave pi** (all logged in `NIGHT_LOG.md`):
   - the B2 levels definition (season-average);
   - H1 priors: γ_att ~ N(0.15, 0.25²), γ_def ~ N(0, 0.25²).
5. **Open a goal-model TODO?** Correlated attack/defence (quality/style) steps in the production
   `MultiScaleGRW`, with the market's ρ ≈ −0.8 and σ_s/σ_q ≈ 0.3 as prior centres. No stream has
   tested this.
6. **English leagues:** add `DataTournemantSegment`s with betdb IDs (a `src/` task) before the
   pipeline can run there.

## Housekeeping

- Codex 5h quota hit 100% at ~02:42; reset at 03:20. Waited as instructed; 2 banked resets were
  shown and not used.
- Stream branch `feat/market-model-inversion-eda` holds the reviewed merges of A, B, B2 and B3.
  It is not pushed. pi pushed only its own `pi/market-model-phase-*` branches.
