# Phase C revision 3 — fresh-session handover (2026-10-06)

## 0. State and scope — read this first

Worktree: `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`.
Branch: `pi/market-model-phase-c`. Implementation commit **8591099b**;
blocker report/artifacts commit **ce9b727d**. This handover is documentation only.

**No C0/C1/C2/H1/H2 fit exists. No sampling, production smoothing, forecast
metrics, shrinkage, ratings, figures or recovery fit has been run for Phase C.**
The l05/r05/t05 production-named files currently implement only the likelihood
update, numerical diagnostics and necessary tests—not a production pipeline.
TODO 039 is BLOCKED. All owned Julia panes are closed; there is no beast session
or checkout to resume. Do not operate unrelated existing sessions.

Full-book local accuracy now passes. Thin-book accuracy and the integrated
1X2 derivative check remain red. The complete all-types Gate 1 did not finish.
The revision-3 addendum says stop on any failure. **Before a fresh session
launches fits, the manager must explicitly confirm a full-books-only C0/C1
continuation despite the blocked C2/all-types requirements, or issue a revised
method and clear those gates.** This request to write a handover did not launch
or certify fits. Do not silently delete the failing test or call partial work
Phase C completion.

Read, in order:

1. `AGENTS.md`, Julia coding guide and prototype runner style guide.
2. `experiments/pi_market_model_phaseC_prompt.md` (revision 1),
   `pi_market_model_phaseC_v2_prompt.md`, and
   `pi_market_model_phaseC_v3_addendum.md` (copied verbatim from manager).
3. This file, `results/C/PHASE_C_REPORT.md` and `REPRODUCIBILITY.md`.
4. Stream README/DESIGN and the A/B2/B3 reports as required by the original brief.

All paths below are relative to `current_development/market_model/` unless
explicitly repository-root or absolute.

## 1. File map

| File | What it contains / current status |
|---|---|
| `l05_pooled.jl`, module `PooledMarket` | Production-grid-compatible AD probability/density algebra; finite-difference and AD derivatives; third tensor and Wick contraction; Newton joint mode; revision-2 marginal with revision-3 corrected mean; exact quadrature; level integral; C0 parameter mapping and fixed-parameter gate centres; all-types gate and summary. **No state-space training engine or sampler yet.** |
| `r05_pooled.jl` | Numbered config/data/gate runner. Loads pinned cache, calls `laplace_gate`, prints gate decision if it completes. No training/evaluation. It overwrites current gate CSVs; preserve evidence before diagnostic reruns. |
| `t05_pooled_tests.jl` | **130 necessary assertions**, currently **129 pass / 1 fail**. Gaussian integrals, Beta identity, leading reversed-KL limit, synthetic total book, cubic contraction, first-20-real-full-book grid/Hessian/third-tensor checks, integrated-likelihood check, quality/style algebra and existing R6 batch parity. Not full pooled/recovery acceptance. |
| `l05_laplace_preflight.jl`, module `LaplaceBookPreflight` | Revision-1 diagnostic implementation. Reused `market_vectors`, production `book_logdensity`, normal quadrature. Also contains literal old brief/projection tests. **Do not promote its old approximation.** |
| `r05_laplace_preflight.jl` | Original revision-1 diagnostic runner; would overwrite original preflight CSVs. Preserve these as historical blocker evidence. |
| `t05_laplace_preflight_tests.jl` | Original 11-test mathematical diagnostic, not pooled acceptance. |
| `l01_market_model.jl`, module `MarketModel` | `MarketModelConfig`, reviewed Scottish config, gated close, unchanged production probability calls, Phase A KL inversion and MarketPanel bridge. |
| `l02_two_stage.jl`, module `TwoStageMarket` | `phase_b_panel` gives the same 517 full-book fixture panel; prior conventions, B Gaussian engine/slice patterns and TODO 023 reuse. |
| `l03_covariance.jl`, module `CovarianceMarket` | R6 covariance engine, exact scalar Gaussian filtering, original-side independent batch check, schedule and slice-chain patterns. Reuse mechanics, **not its R6 priors or chains** as a C0 fit. |
| `l03_covariance_diagnostics.jl` | `covariance_predictions`, `smoothing_moments`, `covariance_smoothed`, `evaluate_covariance`; useful schemas/RTS/prediction conventions. |
| `results/A/rates.csv` | Untouched isolated KL(p||q) targets. Books are not expanded at these rates in the revised likelihood. |
| `results/C/laplace_gate*.csv` | Current partial all-types evidence: 792 completed fixture-setting rows, four 2-D types. |
| `results/C/engine_gates_c.csv` | 192 scalar checks for completed settings, 167 pass / 25 fail; **Gate 1 only**. Re-exported from emitted summary after the runner abort; `GATE_EXPORT.sql` records that export. |
| `results/C/integrated_derivative_check_c3.csv` | Both repeated integrated third-derivative discrepancies. |
| `results/C/LAPTOP_REVISION3_DEVELOPMENT.txt` | Development log, including failures and an earlier 130/130 version. **That passing earlier version is not the current implementation.** |
| `results/C/PHASE_C_V1_REPORT.md`, `REPRODUCIBILITY_V1.md` | Original blocker record; original preflight CSVs/source remain unchanged. |
| `results/C/PHASE_C_V2_REPORT.md`, `REPRODUCIBILITY_V2.md`, `v2_gate/` | Archived revision-2 mean-gap evidence. Archived hash manifests retain historical paths; current root manifest checks archived CSVs. |
| `results/C/{SOURCE,SCIENTIFIC}_SHA256SUMS.txt` | Current code/evidence hashes. These are **not** two-run production reproduction evidence. |

Module aliases in `PooledMarket`: `PF` = preflight, `CM` = covariance,
`MM` = MarketModel, `MID` = TODO 023 engine, `DF`, `LA`, `ST`, `DS`.

## 2. What passes: full books, all 12 settings

The first **30 full books in match-ID order**, not all 517, were checked at:

- n in **250, 1000, 4000**;
- equal-side Gaussian prediction SD in **0.05, 0.20**;
- offsets **0 and +0.10 to both log-rate sides**, centres at Phase A isolated rates.

All 12 full-book settings pass unchanged marginal and posterior-moment gates.
Sources: `laplace_gate.csv` and `laplace_gate_fixture.csv`.

- Largest median absolute marginal error: **0.003980333011027959 nats**, limit .01.
- Largest p95: **0.0051705855019272395**, limit .05.
- Largest corrected mean error / exact posterior SD:
  **0.0009284497092163062** (0.093%), limit .05.
- Largest relative SD error: **0.010184310576972222**, limit .05.
- All **360** saved full-book log marginals match the revision-2 archive exactly:
  maximum difference **0.0**. Covariance treatment is unchanged.
- AD third tensors versus central differences pass on all completed 2-D rows;
  first-20-full-book grid/density/Hessian parity tests also pass.

Implementation detail that must not be lost:

```
f = log L + log N(theta; a,S)
mode = argmax f
Sigma = (-Hessian(f) at mode)^-1
T = third derivative of log L at mode
mean_i = mode_i + 0.5 sum_jkl Sigma_ij T_jkl Sigma_kl
```

The Gaussian contributes no third derivatives. Wick's fourth-moment identity
is derived in the loader docstring. **Keep the revision-2 peak/determinant
log-marginal, including all n-dependent Dirichlet normalisers.** The Gaussian
book information is `covariance\corrected_mean - S\a`; do not shift a likelihood
centre and accidentally shrink the correction a second time.
`raw_covariance` is inverse bracket precision for the contraction;
`covariance` retains the prescribed clipping treatment. Those differ when J
has negative eigenvalues.

## 3. What fails, where, and why

### Thin books: prediction SD 0.20

All 25 O/U-only, all 10 BTTS+O/U and the single BTTS-only book were tested at
all 12 settings. The BTTS-only book passes its completed settings. O/U-only
has three failed settings; BTTS+O/U has six. The failures are not confined to
an unresolved high-n integration case: the **n=250, no-offset** settings are
independently resolved by orders 32/64.

| Type, n=250 / SD=.20 / offset=0 | Observed | Limit |
|---|---:|---:|
| O/U-only, maximum SD relative error | **0.058978186176263714** | .05 |
| BTTS+O/U, median absolute marginal error | **0.02060874306987226 nats** | .01 |
| BTTS+O/U, maximum mean error / SD | **0.05836204173970596** | .05 |
| BTTS+O/U, maximum relative SD error | **0.14878198870219494** | .05 |

O/U witness **14035540**: minimum likelihood J eigenvalue **-1.2364093919265855**;
clipping is active. Raw SD error is **0.03898563692037171**, versus clipped
**0.058978186176263714**. Curved total contours can give nonzero/negative
observed tangent curvature at a prior-regularised joint mode. A mean-only
correction cannot fix that covariance change. Do not silently remove clipping.

BTTS+O/U witness **14035709**: minimum J eigenvalue **40.28420140644717**, so
clipping is inactive. Its marginal error is **-0.048158097212724016 nats** and
SD error **0.14878198870219494**. Sum/product information is symmetric under
swapping rates; weak/curved and potentially multimodal supremacy geometry is
not accurately represented by one local Gaussian plus one cubic mean shift.
Do not force these books rank one or change their likelihood.

Order-32/64 marginal differences at n=250, SD=.20, no offset are at most
**2.2578857006294584e-8** (O/U-only) and **1.7352243730783812e-7** (BTTS+O/U).
Moment-order differences are also far smaller than the failures; see report.
High-n thin-book quadrature is less stable, explicitly unresolved in some
settings (O/U n=4000 maximum marginal order difference **0.0029324648804998077**).
Do not cite high-n numbers as fully resolved or discard them to obtain a pass.

### 1X2 level-integrated likelihood: current t05 check 1/130 fails

Exact location: `t05_pooled_tests.jl:104`, inside
`"Production-grid derivative parity: first 20 full books"`, **after** the loop
of full-book checks. It uses only the 1X2 market from fixture **12473328**, at
n=1000 and its isolated-rate d. It is not a failing full-book tensor check.

```
f(x) = level_integral(x[1], markets, 1000; order=64).marginal
T = third_ad(f,[d])
norm(T-third_fd(f,[d])) / norm(T) <= 1e-6
```

Failures recorded twice:
**5.405790669565064e-5**, then **5.405838441375254e-5** after including all
nested dual components in quadrature error control. Both exceed 1e-6.
Thus current necessary tests are **129 pass / 1 fail / 130**, not complete
acceptance. `T05_C3_UPDATE_DONE` is not reached by the failing current suite.

The initial moving-Gaussian level integral passed a preliminary test version,
then stalled in the real all-types Newton line search (reported score
**-0.06441733688240525**). It was replaced once by adaptive Gauss-Kronrod,
using the already-installed `Distributions.quadgk`. Current bounds [-8,4],
rtol=2e-12, atol=1e-14, inner rules 7/15; labels 32/64 select those rules.
The endpoint-density and integrated-value convergence assertions pass, but
agreement of the **third derivative** does not. A converged integral value
alone does not validate its derivative jet. **The precise discrepancy cause
remains unresolved**: do not claim AD is correct, FD is wrong, or that an
inflection/near-zero tensor is the explanation without measuring it.
The discrepancy is not the old full-book mode/mean gap. Two identical failing
checks caused the mandated stop; no third variation or further gate run followed.

No complete 1X2 type batch was written. Inventory tested=0 means no completed
batch persisted, not no attempted evaluations. The full all-types gate remains
incomplete. A future numerical investigation should separately measure T_AD,
T_FD, absolute error, step sensitivity and integrated Hessian convergence,
under an explicit new manager direction—not resume the old retry loop or
weaken the tolerance.

## 4. REPL-only commands: tests and gate

From the worktree root, verify the session name is unused. Never send/capture
by session/window name; use the returned **pane ID only**. No one-shot Julia.
No package updates. Fresh Julia avoids stale module/constant bindings.

```bash
PANE=$(tmux new-session -d -P -F '#{pane_id}' -s pi_julia_mm_c -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 julia --project -t 8')
printf 'owned laptop pane: %s\n' "$PANE"
tmux send-keys -t "$PANE" -l -- \
  'using LinearAlgebra; BLAS.set_num_threads(1); include("current_development/market_model/t05_pooled_tests.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
# Wait for result, then capture:
tmux capture-pane -t "$PANE" -p -J -S -2000
```

Expect the failing integrated derivative assertion, **not T05_DONE**. Do not
put tests and fitting in a semicolon chain and assume the tests succeeded.
For prior regressions, send separate includes of t04/t03/t02 in an owned fresh
REPL and wait for T04_DONE/T03_DONE/T02_DONE. Revision-3 did not rerun them;
revision-2 counts were 92/131/131. The t03 A regression uses temporary outputs.

For a **separate diagnostic gate run**, first preserve current result CSVs,
then in an owned pane:

```bash
tmux send-keys -t "$PANE" -l -- \
  'include("current_development/market_model/r05_pooled.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
tmux capture-pane -t "$PANE" -p -J -S -2000
```

Current r05 attempts every type; no completed current-source all-types run has
been established. It may abort before its decision marker. If it completes,
its stage markers are C05_C3_LIKELIHOOD_BLOCKED / C05_C3_LIKELIHOOD_GATE_DONE,
**not R05_DONE**. It has no training stage. CSVs are replaced after each
completed type; the end-only engine export can remain stale after an abort.
GATE_EXPORT.sql was used to make the saved engine table match the emitted
partial summary. An earlier 130/130 message in the development log refers to
an earlier integrator, not passing current-source tests.

After evidence capture, close only your owned pane:
`tmux kill-pane -t "$PANE"`.
Pinned cache `.cache/datastore_ScottishLower.jls` SHA256:
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`;
load `max_age_hours=10^6`, no cache refresh. Verify current hashes from the
stream directory with the commands in REPRODUCIBILITY.md.

## 5. Exact next implementation steps for C0/C1 on full books

**Conditional on explicit full-books-only continuation approval.** This does
not clear C2, mixed-book synthetic recovery, H1/H2, or whole-Phase-C acceptance.
Do not alter full t05 to conceal its integrated-likelihood failure; if a scoped
entry point is authorised, label it full-book-only and retain the failing check.

### A. Freeze population, parameterisation and priors

1. Use `TwoStageMarket.phase_b_panel(ds; config)` for exactly **517 fixtures /
   1034 isolated observations**. Obtain the same fixtures' gated books with
   `MarketModel.gated_close`. Keep IDs/team/week ordering and evaluation target
   joins asserted. No thin books, swapped T014 book or new data repairs.
2. Zero-sum team effects as B2, static mu/gamma and ordinary weekly summer steps.
   q=(alpha-beta)/2, s=(alpha+beta)/2. Existing alpha/beta-state engine can be
   reused without changing the initial prior or observation design:
   `scale=sqrt(sigma_q^2+sigma_s^2)` and
   `rho=(sigma_s^2-sigma_q^2)/(sigma_s^2+sigma_q^2)`.
   `PooledMarket.c0_parameters` already maps this to an R6 parameter vector.
3. C0 sampled parameters: sigma_obs, sigma_q, sigma_s. sigma_obs HN(.20);
   sigma_q/sigma_s independently HN(.10). C1: sigma_q, sigma_s, sigma_u, n;
   sigma_u HN(.20), n LogNormal(log(1000),1.5), q/s HN(.10).
   Mu N(log(1.35),.5), gamma N(.15,.25), initial unconstrained alpha/beta
   N(0,.5) independently; integrate these Gaussian states as B2 does.
   Implement new log-coordinate priors and Jacobians. **Do not reuse R6's
   independent sigma_att/sigma_def and Uniform-rho prior**, or treat its
   saved draws/rotation medians as fitted C0. Do not blindly apply an existing
   linked-coordinate bound to n and silently truncate its specified prior.

### B. Build and prove C0 first

4. Define a C0 arm/fit wrapper with the correct three physical parameter names.
   Its Gaussian filter can call `CM.covariance_filter(CovarianceRung(6), panel,
   c0_parameters(...))`. Prepared H/y can be cached like B2. The likelihood is
   exact Gaussian; only the prior differs from unconstrained R6 fitting.
5. Add an explicit C0-to-R6 matched-parameter likelihood reduction <=1e-9,
   independent batch Gaussian likelihood/RTS gates, and retain the B2/B/TODO 023
   gates. Current t05 proves covariance rotation and one existing R6 batch
   equality, **not a new C0 prior/sampler/reduction pipeline**.
6. Add C0 slice-chain training and physical-draw/convergence tables. Reuse
   `CM.covariance_chain`/`fit_covariance` mechanics, not its hard-coded prior,
   parameter names or a saved R6 fit. Run deterministic gates on laptop;
   posterior sampling only on the beast after approved preflight.

### C. Implement the C1 collapsed approximate state-space filter

7. For each chronological full-book fixture, construct its original-side
   2-by-state observation design H, preserving zero-sum projection. Let prior
   state be m,P and set `a=H*m`, `S=H*P*H' + sigma_u^2*I`. u is independent,
   equal-side per-match Gaussian; no per-team HA, weekly mu shock or t noise.
8. Call `laplace_update(raw_book_density,a,S; third_likelihood=AD_book_density)`
   with full markets and n. Add **only its returned marginal** to the collapsed
   likelihood: it already integrates the book times the state prediction.
   Do not append another Kalman Gaussian term, lose the density peak/normalisers,
   or use the old isolated-KL Hessian. Preserve the passing revision-2 marginal.
9. Propagate returned theta moments back into state moments. With
   `K=P*H'/S`, `b=update.mean`, `V=update.covariance`, Gaussian conditional
   identities give
   `m_new=m+K*(b-a)` and `P_new=P+K*(V-S)*K'`.
   This incorporates sigma_u through S; do not treat theta as H*x without u
   or add sigma_u twice. Independent tests must verify moment/information-form
   equivalence and PSD/symmetry. Avoid inverting the full latent-state P when
   static/centred directions can be singular. Keep raw versus clipped book
   curvature explicit, never substitute a covariance to obtain a gate pass.
10. Fix deterministic within-week fixture order. **All scored predictions for a
    week precede every update in that week**, as B2. Sequential nonlinear book
    approximations can be order-dependent even though Gaussian scalar updates
    are exact. Store pre-week moments and the forward pseudo-factors/moments
    needed for approximate RTS and fixture-u shrinkage.
11. Test a toy Gaussian likelihood against independent batch calculations,
    scalar/eigen versus full-2-D update <=1e-10, and conditional-state identities.
    For nonlinear books, freeze the local Gaussian factors and their constants
    for a batch check; do not pretend the original Dirichlet state model is
    globally linear-Gaussian. Label C1 filter/collapsed likelihood/RTS approximate.
    Add full-book synthetic parameter recovery as approved; the original mixed-
    type recovery requirement remains unmet while thin-book code is blocked.

### D. Fit and evaluate only after these gates

12. Implement the prescribed four chains, **2000 warmup + 3000 retained, thin 1**,
    slice adaptation ending at warmup, deterministic new seed manifest. C0/C1
    each need 10a full-panel and 10b honest_train-only hyperparameter fits.
    Convergence: every parameter/protocol Rhat<=1.05, bulk/tail ESS>=200.
    No sampling on laptop; beast checkout `/root/BF_runs/market_model_c`, owned
    session `pi_mm_c`, `-t 16`, core pinning and BLAS=1; logs under
    `/root/BF_runs/logs/market_model_c/`. These do not exist from this session.
    Read the remote execution guide before creating them, verify load and copy
    the pinned cache preserving metadata. Never operate unrelated panes.
13. Freeze source SHA before sampling, serialize prototype fits with loaders
    included for reloading, refuse inference promotion after failed convergence.
    No database writes. All segment/split/population choices flow from config.
14. Reuse B2 schema and filtration for one-step metrics on the **same 517 full
    isolated targets**, with 10b scoring only honest_test. C0 predictive noise
    is sigma_obs; C1 theta prediction is structure+u (`H*P*H'+sigma_u^2*I`).
    Make this target/noise convention explicit; do not condition a forecast's
    mean on its own book or silently introduce an evaluation-book variance.
    Add paired per-fixture supremacy+level marginal logpd differences versus
    C0 and SE, coverage, RMSE/MAE; retain all requested subsets/axes. Report
    C0's measured gap to B2 R6, not forced agreement: the matched-parameter
    likelihood identity does not make their differently constrained priors and
    posterior fits identical.
15. Implement approximate RTS, full-book theta=structure+u shrinkage, n/sigma_u
    and implied book-noise summaries, q/s paths and scale ratios, including the
    named transition clubs Ross County, Airdrie, East Kilbride and Kelty.
    No smoothed
    result should be described as an honest forecast. Use B2 diagnostics as
    schema references, not results to copy into new output tables.
16. Runner needs real engine/training/convergence/evaluation/output sections,
    not just a gate include. A second fresh beast run must reproduce scientific
    CSVs byte-for-byte at the frozen source/seeds. Existing hashes certify
    diagnostic evidence only. The authorised output set/report must clearly
    say **C0/C1 full-books only** if that scope is approved; C2 and H1/H2 remain
    separate pending work, not silently satisfied or removed.

## 6. Guardrails and final checks for the next session

No `src/`, package/data/grid/threshold changes; no DB writes; no Phase D or
later score-grid Phase C2. Never force-push, merge/rebase or touch the stash.
Only push `origin pi/market-model-phase-c`. Preserve historical artifacts.
Do not retry the integrated derivative failure a third time without a new
manager-directed diagnostic plan. No numerical cause was proved for that check.

Before handoff/commit: `git diff --check`, `./scripts/todo.sh check`, current
source/scientific hashes. Record dated @pi work log and status/index together
when claiming/changing task state. Stop with the appropriate sentinel and
close only owned panes. This session hands off **blocked, incomplete work**,
not a fit or acceptance success.
