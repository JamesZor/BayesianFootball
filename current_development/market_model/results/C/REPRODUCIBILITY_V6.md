# Phase C revision6 — blocked recovery checkpoint, not production reproduction

Branch pi/market-model-phase-c, same dedicated worktree. Primary statistical
source **d9e6c0030d6c56e2325a2448b0e2525971b121a0**; observer-only diagnostic
**04e67fa85b50d4793c33d7c8e5c1a0782786d1db**. Later evidence/docs commits change
no solver/sampler decisions. SOURCE_SHA256SUMS.txt includes both implementation
and diagnostic files plus the verbatim manager brief. Revision5 evidence archived
with `_V5` names; all prior-phase scientific artifacts unchanged.

**STOP:** exact old-case regression passes, but prescribed recovery hits a new
100-iteration Newton limit on zero-motion rounded Armijo updates. Read current
PHASE_C_REPORT.md/HANDOVER.md before any next run. No solver remedy/recovery
retry or threshold/seed/prior/grid/data/iteration-budget change was attempted.

## Frozen inputs and laptop verification

Pinned cache SHA256 c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4,
max_age_hours10^6, existing Manifest/no package changes. Same generation3962,
chains4961–4964, truth(.03,.01,.06,1000),517-fixture/91-week/full-market schedule.

Owned laptop %56, Julia1.12.1,8 threads/BLAS1, now closed:
1. include t05_pooled_tests.jl:289/289,T05_C6_DETERMINISTIC_DONE. Separate
   C2-pending29pass/10fail explicitly excluded/reported, not newly investigated.
2. Separate post-include command writes v6_newton/newton_termination.csv,
   avoiding the previous same-expression include/world-age issue.
3. include t04_copula_grid_tests.jl,t03_covariance_tests.jl,t02_two_stage_tests.jl:
   92/92,131/131,131/131, their markers; A regression uses temporary outputs.
4. include r05_pooled.jl:full Gate1 12/12,state53/53,four fixed517-fixture
   filters/smoothers,C05_C6_ENGINE_PREFLIGHT_DONE. v6_preflight/ output and accounting.

Exact fixture12476625: same original forward-prefix capture/seed4964-coordinate,
exact gradient match, same tighter reference. One undamped polish step:
mode gap2.0039525594484076e-13<=1e-8,marginal1.1574741165532032e-11<=1e-9.
Regression accounting2polished/2steps/max1,maxδ1.50342686280365e-24.
Real preflight accounting1597decrement/11polished/25steps/max3,
maxδ9.827548787014079e-13. These are deterministic prerequisites, not fits.

Logs LAPTOP_C6_T05.txt,LAPTOP_C6_REGRESSIONS_PREFLIGHT.txt; terminal trailing
whitespace only normalised. No claim that v6 outputs equal older termination outputs.

## Beast recovery provenance / failure

Dedicated checkout /root/BF_runs/market_model_c, owned %264, Julia1.12.4,
16 threads, core pinning/BLAS1, existing Manifest symlink and pinned cache.
Source d9e6c003 was committed/pushed/fetched before checkout and execution.
Historical v4_ad output was moved intact to
/root/BF_runs/logs/market_model_c/9de98548_recovery_outputs/ before checkout,
preserving binary/hash and avoiding untracked CSV overwrite. Nothing deleted/reset.

Historical run command (NOT an instruction to repeat the failed recovery):

```julia
include("current_development/market_model/r05_fullbook_recovery.jl")
```

Output results/C/v6_recovery/. Initial target for root seed4961 fails:
`joint mode did not converge in 100 iterations`, coordinate
[-4.3667389598945885,-4.699137880221636,-3.0309405303003616,7.090705824646739].
Other chains cancel at warmup1; @sync waits. Expanded CompositeException confirms
one root and three cancellations. No retained draws, fit/interval/convergence CSV.
The generated binary and53 passing engine gates exist, not recovery acceptance.

Run-local and canonical root newton_termination.csv are identical:
1696decrement,1polished,1 total/max step,max acceptedδ9.743716301157053e-13.
Failed mode is excluded. Aborted-run counts can depend on peer cancellation
scheduling; they are not two-run production-reproducibility evidence.

New synthetic_panel.jls is BYTE-IDENTICAL to revision4's two panels:
SHA256 de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1.
This is generated-data identity only; no C1_recovery.jls exists.

## Observer-only diagnostic, no numerical remedy

Once tasks had stopped, checkout04e67fa8 added only diagnostic source/evidence.
Primary solver code stayed unchanged, already loaded from d9e6c003. Through SAME
owned persistent REPL, include r05_newton_diagnostic.jl loads l05_newton_diagnostics
and the existing serialized panel, then observes one unchanged full filter.
It returns exact original derivatives into joint_mode and records x/g/H/δ/f(x).
Failed filter is reported as failed; no fitting/sampler restart or accepted fallback.
Load l05_pooled.jl BEFORE deserializing any prototype binary.

v6_diagnostic/newton_mode_trace_summary.csv identifies fixture12476686:
100 calls, gradient norm4.9497130117024364e-5,δ1.035864027765691e-12,
Newton step4.189252610998754e-8, last actual movement0.0. Calls7–100 are94
zero-motion updates; δ remains just above1e-12 and Armijo never exhausts.
Thus the reviewed polishing branch never runs. Raw precision is positive,
min eigenvalue965.3501213272148. Full trace includes every preceding book/call,
not a selected subset. Diagnostic accounting146decrement, failed mode excluded.

The activity CSV is an auditable file-only aggregation, not a threshold change:

```bash
duckdb -csv -c "SELECT match_id,count(*) AS derivative_calls,
min(derivative_call) FILTER (WHERE actual_movement_inf=0) AS first_zero_movement_call,
count(*) FILTER (WHERE actual_movement_inf=0) AS zero_movement_calls,
min(decrement) AS min_decrement,max(decrement) AS max_decrement
FROM 'current_development/market_model/results/C/v6_diagnostic/newton_mode_trace.csv'
WHERE match_id=12476686 GROUP BY match_id;"
```

No zero-motion classification patch, iteration-limit increase, tolerance/seed change
or sampler retry. Explicit review of this new stop boundary is required first.

## Preservation / cleanup / manifests

Owned %56/%264 CLOSED. No running recovery/session or unrelated pane operation.
Remote remains detached04e67fa8 with preserved root accounting modification,
untracked v6_recovery/v6_diagnostic directories and ignored binaries.
Original log /root/BF_runs/logs/market_model_c/d9e6c003_recovery1.log contains
recovery/error expansion/diagnostic. Local BEAST_C6_RECOVERY_ATTEMPT.txt and
BEAST_C6_RECOVERY_AND_DIAGNOSTIC.txt preserve normalised captures. Only CSVs
and captures copied back; no cache/data/.env/Manifest sync or database writes.

Old v4_ad binary now in preserved log directory above; original v4_recovery/
first attempt remains in checkout. Earlier c254 log/output locations unchanged.

```bash
cd current_development/market_model
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
(cd results/C && sha256sum -c SCIENTIFIC_SHA256SUMS.txt)
```

Manifests identify prerequisite, historical and aborted-run/diagnostic evidence,
NOT two complete byte-identical production runs. Definition of done remains
all authorised gates/recovery/convergence/measures/figures and two complete
reproductions: not met. Remaining priority order unchanged; C2 stays deferred.
No src/package/data/grid/scientific-threshold changes, DB writes, PhaseD/later
score-grid C2, force-push/merge/rebase/stash or unrelated pane operations.
