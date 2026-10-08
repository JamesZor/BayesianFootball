# Phase C revision 4 — sampler/recovery BLOCKED (2026-10-06)

## State and scope

Worktree `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`,
branch `pi/market-model-phase-c`. **Full-book C0/C1/H1/H2 remains authorised.
Do not ask again for scope approval.** C2 remains separately deferred.
Statistical implementation **9de98548**; subsequent docs/results commit does not
change it. All owned laptop %49/%50/%51/%52 and beast %261/%262/%263 are CLOSED.
No running recovery/session to resume. Do not operate unrelated panes.

**STOP: the C1 synthetic recovery joint-mode line search failed twice**, first
with FD then with synthetic-only AD. Per the working agreement, no third
numerical variation/retry was attempted. TODO039 is BLOCKED on solver robustness
at supported parameters. Do not silently return -Inf for mode/PSD failures,
change seeds/prior/data/grid/thresholds or select synthetic fixtures to pass.

No completed C0/C1/H1/H2 posterior fit, retained recovery draws/intervals,
convergence, forecast evaluation, learned n/sigma_u, shrinkage, HA/ratings
posterior figures or production reproduction exists. Definition of done remains
all authorised rungs/gates/recovery/converged fits/measures/figures and two fresh
byte-identical full beast runs; **not met**. r05_pooled remains a preflight.

Read AGENTS, Julia/runner guides, manager revisions1–4, stream README/DESIGN,
PHASE_C_REPORT.md and REPRODUCIBILITY.md before changes. Engine-only prior
handover/report/repro preserved as *_ENGINE_CHECKPOINT*. V1/V2/V3 evidence and
A/B/B2/B3 remain unchanged. Sentinels PHASEC4_DONE/BLOCKED/HANDOVER alone on a line.

## Added implementation

`l05_fullbook_sampling.jl`, included inside PooledMarket after the engine:

- Dedicated full-book slice target/wrapper, correct normalised rung priors,
  4 explicit Xoshiro seeds, widths adapt during warmup only, 2000+3000, thin1.
  Physical positive coordinates exp; H2 kappa unchanged. No R6 prior/chains or
  inherited whole-target ±12 guard. Explicit numerical representability support.
- Failed mode/PSD errors propagate; @sync + cooperative cancellation stops peers,
  logs seed/iteration/last supported theta, never promotes incomplete chains.
- `restrict_fullbook` restricts fixture pairs AND metadata and joins books by ID.
  MID.restrict_panel keeps its original full metadata; do not rely on that for
  fullbook_markets/order or reuse full-panel markets positionally for 10b.
- `synthetic_fullbook_latents` / `synthetic_fullbook`: inherited initial raw
  alpha/beta prior, zero-sum design, static mu/gamma, ordinary independent q/s,
  fixture u and exact Dirichlet(n*q) full books on unchanged cutoff10 grid.
  Real full fixture/week/market schedule is preserved. Synthetic obs_y stores
  latent theta TRUTH, not isolated scoring targets. No thin books or gate-based
  selection. Gaussian generation paths exist but no recovery fits were run.
- `FullBookLogMarket` stores finite logp for Dirichlet draws whose display p can
  underflow. Exact log-Gamma(a)=log-Gamma(a+1)+log(U)/a for a<1, normalise log
  draws. No clipping/redraw; density normalisers unchanged. PF.book_logdensity
  has methods on this owned type, primal production grid / dual existing AD
  algebra. `fullbook_market_logp` leaves real-book log(p) algebra unchanged.
- `fullbook_derivative_rule`: real books keep existing FD, synthetic log books
  use AD. Joint-mode tolerances, clipping, Laplace marginal/correction and
  quadrature remain unchanged. AD did NOT resolve the actual recovery blocker.
- `fullbook_seeds.toml` freezes all recovery/production/inference seeds.
  Recovery generation3962, chain4961/4962/4963/4964. No seed selection was made.
- `recover_fullbook` serialises generated data, fits prescribed chains, would
  write convergence/90%-interval gates BEFORE promotion; it never got that far.
  `r05_fullbook_recovery.jl` is the readable beast-only prerequisite runner;
  not a production workflow. Load l05 before binary deserialization.

## Verified and failed evidence

Final laptop %52: **t05 270/270**, T05_C4_DETERMINISTIC_DONE, not T05_DONE.
Separate C2-pending unchanged **29pass/10fail/39**, C2_PENDING_REPORTED.
Fresh regressions t04 **92/92**, t03 **131/131**, t02 **131/131**, all markers.
Final r05 preflight passes full Gate1 **12/12** (360 rows), state gates **53/53**,
all four fixed 517-fixture filters/smoothers. All **six** v4_preflight CSVs are
byte-identical to the engine-only checkpoint, verified with diff -rq after both
log-space and synthetic AD changes. This is not production reproduction.
Logs: LAPTOP_C4_AD_TESTS.txt and LAPTOP_C4_FINAL_SAMPLING_REGRESSIONS_PREFLIGHT.txt.
Earlier local indexing and test-population errors are retained in development
logs; no non-C2 check was moved into the pending boundary or loosened.

Beast `/root/BF_runs/market_model_c`, detached at9de98548, existing Manifest
symlink, pinned metadata-preserving cache, Julia1.12.4, -t16, core pinning/BLAS1,
no DB writes. Three historical attempts:

1. 19ad4b99, %261: generator p underflow, fixture15336943, before sampling.
   Generating log rates [-.2925837158254979,2.7903082673178803]; OU0.5 under
   probability5.881035960900128e-7 (shape.0005881035960900128). Fixed by exact
   log-space storage, NOT another seed, clipping, narrower prior or exclusion.
2. c2547742, %262: generated all books, FD recovery sampler stopped on
   joint-mode line search; gradient[-2.533547593941421e-5,2.8639610683001138e-5].
3. 9de98548, %263: synthetic-only AD still fails same line search. Root chain
   seed4964, warmup8, last theta[-3.549527585137839,-4.460929121755582,
   -2.821549571263347,7.9567722491577495], gradient
   [-1.5699131339808048e-5,3.249019587192592e-5]. Other three chains cancelled
   at warmup7/8. Expanded CompositeException confirms root, not merely peers.

The last two synthetic_panel.jls binaries are BYTE-IDENTICAL:
SHA256 de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1.
This is generated-data identity, NOT fits or two production runs. No fit binary
or recovered parameter interval exists; no retained iterations reached.
Local evidence BEAST_C4_RECOVERY_FIRST_ATTEMPT.txt,
BEAST_C4_RECOVERY_LOGSPACE_ATTEMPT.txt, BEAST_C4_RECOVERY_AD_ATTEMPT.txt;
CSV directories v4_recovery_first_attempt/, v4_recovery_logspace/, v4_recovery_ad/
contain only passing deterministic engine gates, NOT recovery results.

Remote logs `/root/BF_runs/logs/market_model_c/`:
19ad4b99_recovery1.log, c2547742_recovery2.log, 9de98548_recovery3.log.
The c254 original output directory was moved (preserving its binary) to
`/root/BF_runs/logs/market_model_c/c2547742_recovery_outputs/` to avoid an
untracked-file checkout collision. Last binary remains in the dedicated
checkout's results/C/v4_recovery_ad/. Nothing was deleted or reset to bypass
that collision. No session currently runs on this checkout.

## Next dependency / work after review

**First review the repeated solver failure, not rerun it a third way.** The
first30 real Gate1 settings do not certify robustness throughout the frozen
synthetic panel/parameter support. FD roundoff was plausible but AD still
fails; the sole cause is not established. Remedy/revised diagnostic work needs
explicit review at this stop boundary. Full-book scope itself already authorised.

After a reviewed remedy: finish full-book recovery and t05 acceptance before
production; implement actual C0/C1/H1/H2 training in both protocols, convergence
Rhat<=1.05 and bulk/tailESS>=200 every parameter. Implement pre-week B2-compatible
scoring, no own-book forecast mean or extra evaluation-book variance; paired sum
of supremacy+level marginal logpd against C0 with fixtureSE, report measuredR6gap.
Then conditional theta=structure+u smoothing (not just H*RTS), independent toy
checks, full-book shrinkage/noise/n and transition-club q/s paths/ratios.
H1 gamma_att/gamma_def are static states requiring state uncertainty plus theta
mixing, not sampled parameter columns; H2 kappa is sampled. Check Ridall library
source before citing p1/3 (not done in this session). Production r05 needs actual
training/diagnostic/inference/evaluation/output sections; regenerate Gate1 there,
then two fresh frozen-source/seed beast runs with byte-identical scientificCSVs.
Keep root production artifacts distinct from v3 diagnostics; C2 stays deferred.

## Guardrails

Pinned cache SHA256 c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4,
max_age_hours10^6. No one-shot Julia, package/data/grid/threshold/src/DB changes,
no PhaseD/later score-grid C2, no force-push/merge/rebase/stash. Only push origin
pi/market-model-phase-c. Own pane IDs only; all listed are closed. Before any
handoff/commit: diff --check, todo.sh check, source/scientific hashes; tracker
and README row together. Hashes identify blocked evidence, NOT completed fits.
