# Phase C revision6 — frozen prerequisites, recovery continuation

Branch pi/market-model-phase-c, same dedicated worktree. Source file identity:
SOURCE_SHA256SUMS.txt. Revision5 evidence archived with `_V5` names, all earlier
results unchanged. No completed posterior/production reproduction is claimed yet.

Pinned cache SHA256 c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4,
max_age_hours10^6; existing Manifest, no package changes. Seeds/truth/population
unchanged: generation3962, chains4961–4964, truth(.03,.01,.06,1000),517 fixtures.

Owned laptop %56, Julia1.12.1,8 threads/BLAS1:
1. include t05_pooled_tests.jl:289/289 and T05_C6_DETERMINISTIC_DONE; separate
   C2-pending29pass/10fail, explicitly excluded.
2. Write v6_newton/newton_termination.csv in a separate REPL command, avoiding
   the previous same-expression include/world-age issue.
3. include t04_copula_grid_tests.jl, t03_covariance_tests.jl, t02_two_stage_tests.jl:
   92/92,131/131,131/131 with markers; A regression uses temporary outputs.
4. include r05_pooled.jl: full real Gate1 12/12, state53/53, all four fixed
   filters/smoothers, C05_C6_ENGINE_PREFLIGHT_DONE. Output v6_preflight/.

Exact-case fixture12476625 uses the same frozen seed4964-coordinate and original
forward-prefix capture; exact handover gradient matches. One undamped polish step:
mode gap2.0039525594484076e-13<=1e-8; marginal gap1.1574741165532032e-11<=1e-9.
Tighter diagnostic method/tolerances unchanged. Inputs/results in v6_newton/.
Accounting records two actual polished regression solves,2 total steps, max1,
max acceptedδ1.50342686280365e-24, acceptance gate passes. Unit mocks reset before
this accounting. Real preflight writes its separate accounting and root latest-run
copy. These are not synthetic-recovery or production counters.

Logs: LAPTOP_C6_T05.txt, LAPTOP_C6_REGRESSIONS_PREFLIGHT.txt. Only terminal trailing
whitespace is normalised. Source/scientific manifests identify prerequisites,
not two fresh complete production runs.

```bash
cd current_development/market_model
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
(cd results/C && sha256sum -c SCIENTIFIC_SHA256SUMS.txt)
```

Next prescribed run is beast-only r05_fullbook_recovery.jl, persistent owned Julia
REPL,16 threads, core pinning/BLAS1. Existing Manifest/cache must remain pinned;
no DB writes. Fresh output v6_recovery/; run-local and canonical root termination
accounting is written even if chains abort. Do not promote any failed or incomplete
chains. Implementation source is committed/pushed before checkout/execution.
Original remote output directories/binaries must be preserved across untracked
checkout collisions, never deleted/reset. A final recovery-outcome update will
record the precise remote source/pane/log/output status before handoff.

Definition of done remains full authorised gates/recovery/convergence/measures,
figures and two byte-identical production runs. C2 is separately deferred.
