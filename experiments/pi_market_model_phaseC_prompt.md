# Work package — TODO 039 Phase C: the pooled model, with team structure fitted to the books themselves

You are the implementer. Claude is the manager and reviews your work against §5. **Do Phase C only,
then stop.** Phase D and Phase C2 are not authorised.

**Night mode.** The human is asleep. The manager answers `ask_user_question` prompts, and only
within this brief's scope. Prefer making an in-scope choice yourself and documenting it. Anything
outside scope (`src/`, thresholds, data, the score grid, other phases): write it under "Blocked"
and stop.

## 0. Rules (B3's rules, unchanged)

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`, branch
  `pi/market-model-phase-c`, already created from the reviewed B3 merge.
  - You may `git push origin pi/market-model-phase-c`, and no other branch.
  - Never force-push, merge, rebase or touch the stash.
- **Julia** runs only through a persistent tmux REPL, with **pane IDs only**:
  - laptop session `pi_julia_mm_c`;
  - beast checkout `/root/BF_runs/market_model_c` with session `pi_mm_c`, `-t 16`, BLAS = 1;
  - logs in `/root/BF_runs/logs/market_model_c/`;
  - anchored end markers (`R05_DONE`, `T05_DONE`, …).

  Never touch any session you didn't create: the human's are `pi_julia_mm_b`, `agents`, `0` and
  `library`; the manager's are `claude_*`.
- **Data and packages:** the pinned cache (SHA256 `c786e2fc…b423b4`), `max_age_hours = 10^6`. No
  new or updated packages. No `src/` changes, no database writes. Everything is driven by
  `MarketModelConfig`.
- **Context handover:** at ~75% context, commit, write `results/C/HANDOVER.md`, print
  `PHASEC_HANDOVER` and stop.
- **Honesty:** if a number doesn't reproduce or a gate fails, say so with the numbers. Don't loosen
  thresholds or pick subsets.

## 1. Read first

1. `README.md`, and `DESIGN.md` §2, §4.4, §5 (B2, **Phase C revisions**, C as run) and §8.
2. `results/B2/PHASE_B2_REPORT.md` (R6, the quality/style rotation, the Phase C implications) and
   `results/B3/PHASE_B3_REPORT.md` (the tail check; the grid comparison is evidence for a later
   C2 only).
3. `results/A/PHASE_A_REPORT.md`, `results/A/rates.csv` and the Phase A book code: the gated
   close, KL inversion, gates and coverage.
4. `l01`–`l04` loaders, especially the B2 covariance engine (`l03`) and its gates.

## 2. The model

### 2.1 Fixed decisions (human)

- **Double-Poisson score grid.** B3's grids are not used.
- **Team dynamics on quality and style:**
  - q = (α − β)/2 and s = (α + β)/2, so α = q + s and β = s − q (β = "concedes more");
  - independent weekly GRW steps σ_q and σ_s; zero-sum projection as before;
  - season gap as in R6 (ordinary weekly steps; no special jump);
  - league μ and γ static.
- **No** per-team HA, Student-t, RBPF or weekly μ shock.
- **Per-match deviation** u_m ~ N(0, σ_u² I₂) on (log λ_h, log λ_a): independent, equal on both
  sides.
- **What each kind of book may inform:**

| Book type | Phase A status | Informs |
|---|---|---|
| Full books (1X2 + ≥ 1 more market) | 517 accepted | both directions |
| 1X2-only books | 78 accepted | **supremacy only** (their totals carry the −0.10 grid bias) |
| Totals/BTTS-only books (no complete 1X2) | refused at gate 2 | **the total direction only** |
| No Betfair book; T014 swapped book | refused | nothing (still forecast) |

### 2.2 The Laplace–Kalman book likelihood

For fixture m, write θ_m = (log λ_h, log λ_a). The book likelihood, with one concentration n
shared by all books, is

```
L_m(θ; n) = Π_k Dirichlet(p_{m,k}; n·q_{m,k}(θ))     (a Beta for each two-way market)
```

Approximate it around Phase A's KL inversion θ̂_m. That is its large-n maximiser; use the exact
maximiser if you show it matters. Let H_m be the Hessian of the cross-entropy
−Σ_k Σ_s p_s log q_s(θ) at θ̂_m:

```
log L_m(θ; n) ≈ log L_m(θ̂_m; n) − (n/2)(θ − θ̂_m)ᵀ H_m (θ − θ̂_m)
```

The pooled model is then linear-Gaussian, so the exact Kalman engine applies:
- θ̂_m acts as a pseudo-observation of θ_m with covariance (n H_m)⁻¹;
- θ_m = structure + u_m.

**Projecting each book onto the directions it identifies:**
- Eigen-decompose H_m = V diag(e) Vᵀ.
- **Full books:** both eigen-directions v_i. Each scalar observation is v_iᵀ θ̂_m with noise
  1/(n·e_i) + σ_u². Because u is isotropic, the noises stay independent and scalar updates are
  exact.
- **Totals/BTTS-only books:**
  - keep only directions with e_i > 10⁻⁶·max(e);
  - v_iᵀ θ̂_m is well defined even though θ̂ drifted along the flat direction (T015);
  - check the kept direction is close to the total direction, and report it.
- **1X2-only books:** keep only the supremacy direction d = (1, −1)/√2, i.e. observation dᵀθ̂_m.
  Its noise is dᵀ(nH_m)⁻¹d + σ_u², the information about supremacy once the total is integrated
  out.

**The n-dependent terms must enter the collapsed likelihood.** Integrating the book against the
Gaussian state prediction gives, per book:

```
log L_m(θ̂_m; n) + (k/2)·log 2π − ½·log det(n H_m)|_kept + [Kalman Gaussian term]
```

- k is the number of kept directions.
- The 1X2-only book uses the matching one-direction marginal.
- Derive this carefully and write the derivation in the loader docstring.
- **This term is what identifies n:** how far a book's own lines disagree with the grid.

**Inference:** the global parameters are θ = (σ_q, σ_s, σ_u, n). Use exact collapsed Kalman plus
slice sampling, 4 × (2,000 + 3,000), thin 1.

**Priors:**
- σ_q, σ_s ~ HN(0.10);
- σ_u ~ HN(0.20);
- n ~ LogNormal(log 1000, 1.5);
- μ, γ and the initial states as B2.

Report the posterior of n and its implied book standard deviation per book type.

### 2.3 Rungs

| Rung | Data | Observation model | Purpose |
|---|---|---|---|
| C0 | 517 full books, Phase A isolated rates | Gaussian σ_obs (B2 style), quality/style GRW | control. Must reproduce R6's metrics closely: the diagonal quality/style GRW is R6 with σ_att = σ_def; B2 found corr(Δq, Δs) ≈ 0. Report the gap. |
| C1 | 517 full books | Laplace book likelihood (n, σ_u) | isolates the change of measurement model |
| C2 | all informative books (517 + 78 supremacy-only + totals-only) | Laplace book likelihood | **the pooled model**: does borrowing from thin books help? |

### 2.4 Side rungs: is home advantage tied to team strength? (human request, 2026-10-06)

These are two-stage rungs on the isolated rates, on the 517 full books, with C0 as their control.
They are exact Kalman, because each new coefficient enters linearly for fixed θ. They answer a
question; they don't change C1 or C2.

| Rung | Home-advantage structure | New θ | Question |
|---|---|---|---|
| H1 split HA | log λ_h += γ_att; log λ_a −= γ_def (both static states, replacing the single γ on the home side) | — (static states) | does home advantage act through the home team scoring more, the away team scoring less, or both? In quality/style terms: does it lift supremacy only, or also the level (are home games higher scoring)? |
| H2 quality-linked HA | C0 plus κ·q_{h,t} on the home team's own scoring rate, so its quality loads (1 + κ) at home | κ ~ N(0, 0.5²) | do stronger teams get a bigger (κ > 0) or smaller (κ < 0) home edge? |

- Phase B's R3 found independent per-team home advantage negligible (σ_γ = 0.013), but its prior
  ignored any link to strength. H2 pools that link across all teams through one parameter, so it
  has far more power.
- Report:
  - the posteriors of γ_att, γ_def and κ;
  - 10b paired log density against C0, with its SE;
  - coverage.
- Gates:
  - H1 with γ_def = 0 reproduces C0;
  - H2 with κ = 0 reproduces C0 (≤ 1e-9);
  - batch joint-Gaussian checks on the toy panel.
- Outputs: `home_advantage_rungs.csv`.
- Cite `ridall2024-bayesian-state-space-models-modelling` p. 1/3 (a common, dynamic home
  advantage) as the convention being tested.

## 3. Gates and measures

**Gates** (`engine_gates_c.csv`):
1. **The Laplace approximation is accurate.** On ≥ 30 books of each type (full, 1X2-only
   supremacy-only, totals-only), compare the Laplace marginal log-likelihood with brute-force
   2-D quadrature of the **exact** Dirichlet/Beta book likelihood times a Gaussian state
   prediction. Use 3 values of n (posterior median ×0.25, ×1, ×4) and 2 prediction spreads.
   - Report the error distribution.
   - Gate: median |error| ≤ 0.01 nats and 95th percentile ≤ 0.05 per book.
   - If a book type fails, print `PHASEC_BLOCKED` with the numbers. Don't loosen the gate.
2. **The projections are exact:** the eigen-direction scalar updates equal the full 2-D update
   to ≤ 1e-10 on the toy panel.
3. **C0 reduces to R6:** at matching parameters (σ_att = σ_def, ρ from σ_q and σ_s), C0's
   log-likelihood equals R6's to ≤ 1e-9.
4. The batch joint-Gaussian gate for C1/C2 at fixed θ, on the toy panel; plus the B2, B and
   TODO 023 gates.
5. Convergence: R̂ ≤ 1.05 and bulk/tail ESS ≥ 200 for every θ, in every rung and protocol.

**Measures.** Score everything **on the same set: the 517 full books' isolated rates**, so
C0/C1/C2 are comparable with B2.
1. **One-step 10a/10b**, the B2 schema: RMSE, MAE, mean log predictive density and 90% coverage on
   supremacy, level, log λ_h and log λ_a. Also paired per-fixture log density against C0, with
   its SE. **The headline question: does pooling (C2) improve honest forecasts of the full books
   over C1 and C0?**
2. **Shrinkage:** the smoothed posterior mean of θ_m (structure + u_m) against the isolated
   θ̂_m, by book type, giving the share of each book's deviation kept. Three special cases:
   - **totals-only books:** pooled supremacy against the T015 artefact (+0.36…0.63);
   - **1X2-only books:** pooled total against their own double-Poisson 1X2-only total. Phase A
     measured that bias at −0.103, so the pooled total should come out about 0.10 higher, if
     the structure is right;
   - full books: how much of u_m is kept.
3. **n and σ_u:** posteriors, the implied book noise by type, and their ratio to σ_u.
4. **Ratings:** quality/style paths of the transition clubs (Ross County, Airdrie, East Kilbride,
   Kelty) under C1 and C2, and the posteriors of σ_q, σ_s and σ_s/σ_q against B2's R6.

Outputs go to `results/C/` (`ladder_summary_c.csv`, `onestep_metrics_c.csv`,
`onestep_fixture_c.csv`, `paired_vs_c0.csv`, `shrinkage_by_type.csv`, `laplace_gate.csv`,
`convergence_c.csv`, `engine_gates_c.csv`, `team_paths_c.csv`), plus figures `C_shrinkage.png`
and `C_paths.png`.

## 4. Tests (`t05_pooled_tests.jl`, laptop)

1. The Hessian matches finite differences (≤ 1e-6 relative) on 20 books.
2. **Eigen-projection rules:**
   - a full book keeps 2 directions;
   - a totals-only book keeps 1, within 10° of the total direction;
   - the 1X2-only supremacy projection is exactly d.
3. **The Laplace marginal:** on synthetic books, against quadrature, within the gate.
4. **C0 equals R6** at matched parameters.
5. **Synthetic recovery:** simulate a panel from known (σ_q, σ_s, σ_u, n), with books drawn from
   Dirichlet(n·q) and a book-type mix like the real one. The fit recovers each parameter within
   its 90% interval.
6. `t02`–`t04` still pass (counts).

## 5. Acceptance (the manager reruns each item)

1. `t05` and `t02`–`t04` pass in a fresh laptop REPL.
2. All gates in §3 pass, the Laplace gate included.
3. Convergence passes everywhere.
4. Every §3 table and figure exists; every number in the report is copied from them.
5. `r05_pooled.jl` runs top to bottom to `R05_DONE` in a fresh beast REPL, and a second fresh run
   gives byte-identical CSVs (`SCIENTIFIC_SHA256SUMS.txt`, `REPRODUCIBILITY.md`).
6. The README gains a Phase C section, including the human's "double Poisson for C; C2 later".
   DESIGN §5 Phase C stays as written; the report says how the run maps onto it.
7. TODO 039 has a dated `@pi` line; `./scripts/todo.sh check` passes.

## 6. Report and stop

- Write `results/C/PHASE_C_REPORT.md`:
  - evidence per acceptance item;
  - your choices where §2 left room;
  - **in plain words:**
    - isolated vs pooled: does pooling change the rates, and for which books;
    - does it improve honest forecasts;
    - what n says about how much a book is worth;
    - whether the 1X2-only pooled totals land about 0.10 above their isolated values;
    - what the T015 books' pooled supremacy looks like;
  - anything wrong in DESIGN or this brief.
- Commit, push and close your sessions.
- Print `PHASEC_DONE` on its own line and stop, or `PHASEC_BLOCKED` / `PHASEC_HANDOVER`.
