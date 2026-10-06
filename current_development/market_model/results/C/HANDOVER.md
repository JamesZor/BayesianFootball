# Phase C revision6 — recovery continuation checkpoint (2026-10-06)

Full-book C0/C1/H1/H2 remains authorised; no scope reapproval needed. C2 deferred.
Worktree/branch unchanged: BayesianFootball-market-model-pi-c / pi/market-model-phase-c.
Definition of done remains all gates/recovery/converged fits/measures/figures and
two byte-identical full beast production runs. Not met; no completed posterior fit.

Revision6 brief copied unchanged into experiments/. It authorises <=3 undamped
Newton polish steps only on Armijo stalls withδ<=1e-9. Derivative-only steps stop
at infinity norm<=1e-12 or nondecreasing gradient norm. Final norm cannot exceed
pre-polish norm and acceptedδ remains<=1e-9; otherwise propagate errors. Count
as polished, with step totals/max. Every tolerance/seed/prior/data/grid unchanged.

Exact fixture12476625 replay at seed4964's recorded warmup8 coordinate reproduces
the original gradient and now passes unchanged limits: mode gap2.0039525594484076e-13,
marginal gap1.1574741165532032e-11 nats, one polish step, acceptedδ1.50342686280365e-24.
Data and tighter-reference method are unchanged. v6_newton/ stores inputs/results.

Owned laptop %56 fresh t05 289/289 (T05_C6_DETERMINISTIC_DONE), separate C2-pending
29pass/10fail; t04 92/92, t03/t02 131/131. Real full Gate1 12/12, engines53/53,
all four517-fixture fixed filters/smoothers complete in v6_preflight/. These are
prerequisites, NOT recovery/production fits or reproduction.

Scientific source frozen by SOURCE_SHA256SUMS.txt before beast recovery.
Revision5 evidence archived: PHASE_C_REPORT_V5.md, REPRODUCIBILITY_V5.md,
HANDOVER_V5.md, SOURCE_V5_SHA256SUMS.txt. All prior scientific artifacts unchanged.
Main report/repro will be updated with exact recovery outcome before final handoff.

Next priority is prescribed beast C1 recovery (2000+3000,4 frozen chains).
Then C0/C1 both protocols/honest measures, H1/H2, figures and second reproduction.
Any new failure stops with numbers, no tolerance/seed/prior/data/grid change.
Read AGENTS, Julia/runner guides, manager revisions1–6, README/DESIGN and current
report/repro before changes. Do not operate unrelated panes; source/scientific
hashes and todo/diff checks are required before handoff/commit.
