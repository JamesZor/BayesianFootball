# Stage 2 handover — PHASES2_DONE (F6 explicitly deferred)

## Final state

- Worktree/branch: /home/james/bet_project/.worktrees/BayesianFootball-qs-transition, exp/qs-transition.
- Owned beast checkout: /root/BF_runs/qs_transition only; final local records are synced there.
- Source implementation: e2e956cf, now GATED. No src changes were needed on resume.
- Gate runner/evidence commit: 19d3ad4d. Recovery/smoke/final report evidence commit: 928d80f7.
- Final handover follows those commits locally; no push/merge/other-worktree changes.
- Owned jl s2 STOPPED after audit094308-29566; tmux confirmed jl-s2 absent. No active Julia jobs.
- REPORT_S2.md is the experiment record, including exact S3 run commands and timing projections.

## DONE WHEN outcomes (do not repeat verified work)

1. Source gates PASS: new46 + existing373 unit assertions; synthetic4/real6 exact AD/FD cases,
   five AD points each, exactly0 B linked tapes throughout. Worst exact AD1.331e-15.
2. Three fixed recovery seeds per arm PASS: entrant50/51, summer36/39 90% intervals;
   pooled86/90, no parameter misses all3. Runner r09_recovery.jl and all CSVs committed.
3. Post-src original wave2 fold1 chains396000 and latents160000 bit-identical to
   81fdf817-f82d-4f66-9bfd-5f456c4fe29f, fresh post_src checkpoints, no tolerance.
4. All four approved recipes have normal two-fold hard smoke PASS (source folds1/60):
   - qs_control b58afafc-80ee-4ee6-920b-d5df860b4483 (pre-src verified, not repeated).
   - qs_a2 9374341b-66aa-4460-83fe-af45f8d5a01b (pre-src verified, not repeated).
   - qs_entrant 8ac3ccfa-3aa8-4721-9e17-7216e394a2f1.
   - qs_summer 55c33930-2e27-418f-ba7d-9e330fdd6095.
5. Entrant endpoint overlaps0; chronology/first-biweek assertions PASS on folds1/2/60;
   prior means/SDs/source IDs persist exactly through fold2. Hyper overlaps separately declared
   under D9 (1158/1158/509 sums of per-club counts, not unique source unions).
6. Expanded frozen market1790 rows/1361 full books, all1430 legacy rows retained exactly;
   W2 panel60 folds/1070 fixtures. No26/27 outcomes used.
7. REPORT_S2 summary≤12 lines, per-arm status, gate table, F6 note, timing and exact S3 commands.
8. Local commits, todo check PASS, owned REPL stopped. F6 explicitly deferred by James's resume
   instruction; no falsely labelled weekly arm or five-arm timing estimate.

## Important: S2 completion is NOT automatic S3 promotion

- All four smokes retain convergence/performance REVIEW failures; full diagnostics are recorded.
- Read-only history audit has_passing_smoke: control=false, other3=true.
- Sole stale control failed hard row: harness_checks.id832, run_id NULL, tape_allocation57568 B
  from the resolved original D11 incident. Latest control hard smoke passes at0 B.
- Full archived row/history: historical_failed_hard_qs_control.csv and harness_history_qs_control.csv.
  No DB deletion, cleanup, gate weakening or bypass was performed.
- Before control grid, an explicitly authorised operator must review/archive and narrowly clean
  only this stale failed row per experiment harness guide§3, then re-evaluate readiness.
  S2's no-DB-writes-outside-normal-smokes rule does NOT authorise that cleanup.
- James decides F6. Fold cadence and dynamics clock remain coupled; cheapest separable proposal
  in F6_DECISION.md. No weekly fold count or valid F6 command exists yet.
- Four60-fold grids project22.3715h sampling, excluding scoring/persistence overhead; five-arm
  budget undefined. No full grid was launched.

## Next owner

Review REPORT_S2.md, resolve review flags/stale-check authorisation, decide F6, then separately
launch S3 only if approved. Preserve the owned frozen Pyramid cache SHA256
 a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12.
Normal smoke budget is2×(200+200)/fold; the old handover's4-chain description was wrong.
S3 recipes remain4×(500+1000). gate_commands.csv gives all resumed commands/job IDs.
Do not redo D11/D12 investigations or verified gates/smokes. Stop here: PHASES2_DONE.
