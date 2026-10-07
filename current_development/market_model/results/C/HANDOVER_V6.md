# Phase C revision6 — exact-case PASS, recovery zero-motion BLOCKED (2026-10-06)

## State / definition of done

Same worktree `/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`,
branch `pi/market-model-phase-c`. **C0/C1/H1/H2 full books remains authorised;
do not ask for scope approval. C2 remains separately deferred.**

Primary statistical source **d9e6c0030d6c56e2325a2448b0e2525971b121a0**;
observer-only diagnostic **04e67fa85b50d4793c33d7c8e5c1a0782786d1db**.
Subsequent results/docs changes do not change solver/sampler equations.
Revision6 brief copied unchanged into experiments/. Previous revision5 evidence
preserved as PHASE_C_REPORT_V5.md, REPRODUCIBILITY_V5.md, HANDOVER_V5.md and
SOURCE_V5_SHA256SUMS.txt. Earlier checkpoints and A/B/B2/B3 remain unchanged.

Definition of done: all authorised gates/regression/recovery/converged fits,
measures/figures and two fresh byte-identical beast production runs. **Not met**.
No completed recovery/production fit, retained interval/convergence, forecast
measure, learned n/sigma_u, shrinkage/HA/ratings posterior figure or reproduction.
r05_pooled remains a deterministic preflight, not the complete production workflow.

**STOP:** prescribed recovery hits a NEW100-iteration Newton failure on a rounded
zero-motion Armijo step. The same algorithm's observer trace identifies it; no
solver remedy/sampler retry was attempted. Do not raise iterations/changeδ or
verification thresholds, return-Inf, change seeds/prior/grid/data or select fixtures.
Review zero-motion stall recognition before further numerical work.

Owned laptop `%56` and beast `%264` CLOSED, all previous owned panes closed.
No running recovery/session to resume. Do not operate unrelated panes.

Read AGENTS, Julia/runner guides, manager revisions1–6, stream README/DESIGN,
PHASE_C_REPORT.md and REPRODUCIBILITY.md before changes.

## Implemented and passing

Revision6 polish only on Armijo exhaustion atδ<=1e-9: <=3 undamped Newton steps
from derivatives, no density comparisons. Stop at step infinity norm<=1e-12 or
nondecreasing gradient norm. Final norm cannot exceed pre-polish; acceptedδ
must still pass1e-9. Fail loudly otherwise. Counts `polished`, step totals/max.
Existing search floor/raw precision/density/normalisers/derivative dispatch,
likelihood clipping, marginal/skewness/quadrature, priors and population unchanged.

Exact old fixture12476625 at frozen seed4964 warmup8 coordinate now passes:
mode difference **2.0039525594484076e-13** (limit1e-8), marginal difference
**1.1574741165532032e-11** nats (limit1e-9), one polish step,
δ **1.50342686280365e-24**. Original gradient matches exactly; tighter reference
method/tolerances unchanged. v6_newton/ inputs/results/accounting preserve proof.

Fresh owned %56: t05 **289/289**, T05_C6_DETERMINISTIC_DONE, NOT T05_DONE;
C2-pending unchanged **29pass/10fail/39** separately excluded/reported.
t04 **92/92**, t03/t02 **131/131**, all markers; A uses temporary outputs.
Real Gate1 **12/12**,360 rows, state gates **53/53**, four517-fixture fixed
filters/smoothers complete in v6_preflight/, not fits. Preflight accounting:
1597decrement,11polished,25 total steps/max3, maxδ9.827548787014079e-13.
Historical v4/v5 preflight evidence is preserved, not declared byte-identical.

## New recovery blocker: numbers and location

Beast at d9e6c003, Julia1.12.4,16 threads, core pinning/BLAS1, existing Manifest,
pinned cache/max_age_hours10^6, no DB writes. Four frozen chains2000+3000,
generation3962, chains4961–4964, truth(.03,.01,.06,1000),517 fixtures/91 weeks.

Root **seed4961 INITIAL target**, iteration0, supported coordinate:
`[-4.3667389598945885,-4.699137880221636,-3.0309405303003616,7.090705824646739]`.
Error `joint mode did not converge in 100 iterations`. Other three chains cancel
at warmup1. Expanded CompositeException confirms root plus3 cancellations;
@sync waits for all tasks. No retained draws, C1_recovery.jls or recovery intervals.

Observer-only l05_newton_diagnostics/r05_newton_diagnostic then traces ONE filter
on the exact same serialized panel/coordinate with unchanged joint_mode. It
returns exactly original derivatives and logs x/g/H/δ/f(x); no fitting restart,
rule/budget/tolerance/seed change or failed-mode promotion. Source04e67fa8.

Failure is full **fixture12476686**:
- **100** derivative calls; calls7–100 (**94**) have zero actual movement;
- last mode `[0.11192915227756556,2.14396416446582]`;
- gradient `[1.8404678883143788e-6,4.9462900923558095e-5]`, norm
  **4.9497130117024364e-5**;
- δ **1.035864027765691e-12**, just above1e-12 decrement stopping rule;
- unscaled Newton step infinity norm **4.189252610998754e-8**, above1e-10;
- actual accepted-step movement **0.0**; raw min precision eigenvalue
  **965.3501213272148**, not PSD/clipping failure.

Calls6–100 repeat x/g/H/value. Armijo accepts a rounded unchanged candidate, so
scale never reaches the failure condition that enters revision6 polishing.
Iteration limit errors instead. Do not patch zero-motion recognition or widen
any threshold without review. Exact old-case PASS does not certify all support.

Artifacts: v6_diagnostic/newton_mode_trace.csv (all prefix/failed-book calls),
newton_mode_trace_summary.csv, newton_trace_activity.csv (auditable aggregation),
newton_termination.csv. Diagnostic prefix146decrement accepted, failed mode excluded.
Recovery/root newton_termination.csv are identical:1696decrement,1polished,
maxδ **9.743716301157053e-13**,1 total/max polish step. They count evaluated
accepted BOOK modes in an aborted run, not accepted chains; partial cancellation
counts can depend on scheduling, not production byte-reproduction evidence.

Logs: LAPTOP_C6_T05.txt, LAPTOP_C6_REGRESSIONS_PREFLIGHT.txt,
BEAST_C6_RECOVERY_ATTEMPT.txt (expanded exceptions),
BEAST_C6_RECOVERY_AND_DIAGNOSTIC.txt. Source/scientific hashes identify blocked
evidence only; no source or scientific gate was changed to force a pass.

## Remote preservation / exact paths

Dedicated checkout `/root/BF_runs/market_model_c`, now detached04e67fa8.
Only diagnostic-only code loaded after recovery stopped; primary solver unchanged.
Root newton_termination.csv is a preserved tracked output modification; untracked
v6_recovery/ and v6_diagnostic/ outputs remain. No reset/delete/stash performed.

Before d9e6 checkout, old v4_ad directory moved INTACT to
`/root/BF_runs/logs/market_model_c/9de98548_recovery_outputs/`, retaining binary/hash,
so tracked CSV checkout could not overwrite untracked historical evidence.
Original v4_recovery/ first attempt remains in checkout. c254 preserved directory
and historical logs remain as earlier handovers record them.

New generated binary is results/C/v6_recovery/synthetic_panel.jls; SHA256
**de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1**, BYTE-IDENTICAL
with both revision4 panels. This is data identity, not successful recovery/fits.
Original log `/root/BF_runs/logs/market_model_c/d9e6c003_recovery1.log` includes
recovery/error expansion/diagnostic. Only CSVs/terminal captures copied to laptop;
no .env/Manifest/cache/data sync. Both new owned panes closed after verification.

## Next dependency / priorities and guardrails

**Review zero-motion Armijo acceptance as a stall first.** No new numerical
remedy or production retry is authorised by this blocked checkpoint. Tolerances
stay unchanged; supported failures must remain loud. Full-book scope is approved.

After reviewed remedy: finish C1 recovery/t05 acceptance; C0/C1 both protocols
and honest10b measures; H1/H2; measures/figures and second reproduction. Remaining
requirements unchanged: all-parameter Rhat<=1.05, bulk/tailESS>=200;
B2-compatible pre-week scoring without own-book mean or extra evaluation-book
variance; supremacy+level marginal logpd sum paired vsC0/fixtureSE/measuredR6gap;
conditional theta=structure+u smoothing and independent toy gates; shrinkage/noise/n
and transition-club q/s paths/ratios; H1 static gamma-state uncertainty and theta
mixing, H2 sampled kappa; source-check Ridall before citation; real production
workflow sections/full Gate1 regeneration/two byte-identical complete beast runs.
C2 remains deferred; no thin books enter these fits, no PhaseD/later score-grid C2.

Pinned cache c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4,
max_age_hours10^6. No one-shot Julia, package/data/grid/scientific-threshold/src/DB
changes, force-push/merge/rebase/stash or unrelated panes. Only push origin
pi/market-model-phase-c. Before handoff/commit: diff --check, todo.sh check,
source/scientific hash verification; tracker and README row together.

PHASEC6_BLOCKED
