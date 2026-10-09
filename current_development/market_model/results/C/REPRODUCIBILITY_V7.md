# Phase C revision7 — deterministic proof and ACTIVE recovery, not production reproduction

Primary solver/recovery source:
`6335316f4e93ec91495341fa2f6960a64ce772a9` on pi/market-model-phase-c.
Later workflow preparation does not change that solver, sampler, generator,
priors, thresholds or frozen seeds. Revision6 evidence is archived with `_V6`.
Current definition of done is NOT met; no two complete production runs exist.

## 1. Frozen inputs

Pinned `.cache/datastore_ScottishLower.jls`, SHA256
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`,
max_age_hours10^6. Existing Manifest, Distributions0.25.126; no package updates.
Generation3962, recovery chains4961–4964, truth(.03,.01,.06,1000),517 fixtures,
91 weeks/full book mix. Each chain2000 warmup+3000 retained,thin1.

## 2. Laptop deterministic verification

Own persistent tmux REPLs only, Julia1.12.1,8 threads/BLAS1:

```julia
include("current_development/market_model/t05_pooled_tests.jl")
```

Fresh %58 and final fresh %61:305/305,T05_C7_DETERMINISTIC_DONE.
C2-pending executes separately,29pass/10fail/39 excluded, not T05_DONE.
Fixture12476625 retains1e-8/1e-9 limits, passes2.0039525594484076e-13 /
1.1574741165532032e-11. Fixture12476686 from recorded stalled mode AND original
prediction passes0.0/0.0,1 zero-motion polish step,delta2.287082719929954e-26.
Exact frozen revision6 prefix and unchanged tighter reference are test-only.
Actual regression accounting isolated before any workflow mocks:
2polished+2zero-motion polished,4 total steps,max1.

In %58, separate post-include accounting command avoids Julia world-age issues;
then t04,t03,t02:92/92,131/131,131/131 and all markers. A uses temporary outputs.
Include r05_pooled.jl for deterministic preflight only:full Gate1 12/12,
360 rows,state53/53,four fixed517-fixture filters/smoothers,
C05_C7_ENGINE_PREFLIGHT_DONE. Versioned output v7_preflight/; no identity to
older termination outputs is asserted. Accounting1597decrement,11zero-motion
polished,27steps,max3,maxdelta9.96108898035961e-13.

```julia
include("current_development/market_model/t05_fullbook_workflow_tests.jl")
```

Fresh final %60 and %61:85/85,T05_WORKFLOW_PREPARATION_DONE. Independent
augmented joint-Gaussian theta/u moments, matched-R6 forecast parity, whole-week
perturbation filtration, paired-fixture guards, schema smoke and missing/wrong
recovery-data refusal. Mock IID hyperdraws are NOT recovery or production fits.
Core runner Meta.parseall succeeds; it is NOT executed or runtime-verified.
Local panes %58–%61 closed after verified output. Logs in PHASE_C_REPORT.md.

## 3. ACTIVE beast recovery — monitor, DO NOT relaunch

Dedicated checkout `/root/BF_runs/market_model_c`, detached6335316f.
**Owned pane %265 is handed over ACTIVE**, Julia1.12.4,16 threads, core
pinning/BLAS1. This is the one prescribed recovery attempt, not a retry.
All four chains reported warmup200 at the recorded checkpoint; no abort seen.
No retained fit/interval/convergence result yet. Progress log labels say `C6`
(the unchanged sampler's historical text), but source is6335316f/revision7.

Original log `/root/BF_runs/logs/market_model_c/6335316f_recovery1.log`.

```bash
ssh root@mcmc-beast 'tmux capture-pane -t %265 -p -J -S -100'
```

Historical launch was include(r05_fullbook_recovery.jl). DO NOT include it again,
kill the pane, alter seeds/budgets, or change checkout/code while it is running.
Wait for R05_FULLBOOK_RECOVERY_DONE and then recheck ALL retained convergence
and interval gates. If error: preserve expanded CompositeException and original
coordinate/accounting, wait for cancelled peers, STOP without another variation.
There is no numerical blocker at the current in-progress checkpoint.

Only recovery_engine_gates_c.csv is returned; newton accounting writes on run
completion/failure after @sync waits for all tasks. Do not invent interim counts.
Generated synthetic_panel.jls remains byte-identicalde6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1.
This is data identity, not successful recovery/fit reproduction.

## 4. Remote historical preservation

Before6335316f checkout, untracked v6_recovery/ and v6_diagnostic/ directories
moved INTACT to `/root/BF_runs/logs/market_model_c/04e67fa8_outputs_preserved/`.
The synthetic binary's hash was verified there. Modified root aborted-run CSV
copied/compared to `04e67fa8_accounting_preserved/newton_termination.csv` before
restoring tracked HEAD's baseline for checkout. No reset/delete/stash. Original
v4_ad preservation and v4_recovery/ locations remain as HANDOVER_V6 records.
No .env/Manifest/cache/data sync, SQL writes or unrelated pane operation.

## 5. Prepared next stage, only AFTER recovery passes

New l05_fullbook_workflow/l05_fullbook_reports modules and r05_fullbook_core
stage are local committed preparation, NOT loaded into the running recovery.
After its tasks have finished, fetch the permitted branch and checkout the
prepared source safely (preserve any newly untracked outputs before checkout).
Use a fresh owned16-thread persistent beast REPL, default core output v7_core_1/
(or fresh C05_CORE_OUT) and recovery input v7_recovery/ (C05_RECOVERY_INPUT).
The core runner rechecks frozen panel hash/budget/truth/all convergence/intervals
before sampling. It regenerates Gate1/state gates and fits C0/C1,10b first.

It does not implement full H1/H2/posterior figures/two-run acceptance; marker
R05_FULLBOOK_CORE_DONE only. r05_pooled remains deterministic preflight.
Do not declare R05_DONE or PHASEC7_DONE for a partial core stage. H1 static-gamma
uncertainty/mixing, nonlinear smoothed total, HA literature/figures, all required
figures, a complete workflow and second complete reproduction remain pending.
C2 remains deferred with unchanged failed thin-book gates.

```bash
cd current_development/market_model
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
(cd results/C && sha256sum -c SCIENTIFIC_SHA256SUMS.txt)
```

These manifests cover historical/deterministic/in-progress evidence only, NOT
completed production reproduction. Final diff/todo/hash checks recorded at handoff.
