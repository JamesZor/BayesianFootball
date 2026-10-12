# Stage 2 progress

Phase: RESUMED after beast connectivity recovery, 2026-10-12 09:10 BST.
Branch/worktree: exp/qs-transition, /home/james/bet_project/.worktrees/BayesianFootball-qs-transition.
Owned beast checkout: /root/BF_runs/qs_transition. Source e2e956cf; synced branch ba33e752.

## Verified (do not redo)
- Expanded frozen market input1790 rows/1361 full books, all1430 legacy rows unchanged; panel60 folds/1070 fixtures.
- Pre-src legacy identity: chains396000/latents160000 bit-identical to wave2 81fdf817-f82d-4f66-9bfd-5f456c4fe29f.
- Control normal2-fold smoke hard PASS: b58afafc-80ee-4ee6-920b-d5df860b4483; Rhat1.0625, ESS67.1/75.5, divergences0; convergence REVIEW fail.
- A2 normal2-fold smoke hard PASS: 9374341b-66aa-4460-83fe-af45f8d5a01b; Rhat1.0533, ESS50.9/72.0, divergences2; convergence REVIEW fail.
- D11 baseline allocation recovery: fresh old/current/loader REPLs0 B, no source/package/gate changes; process-state cause unproved (ALLOCATION_INVESTIGATION.md).
- Resume: stopped owned s2, fixed only own remote invalid .git pointer, synced ba33e752, fresh beast16 REPL. Root checkout ownership fixed after legacy git metadata warnings.
- Component job091418-14872: new46 + existing373 unit checks PASS; opening/observed-target exact AD ≤1.56e-16, FD convergence, exactly0 B for entrant/summer (component_gates.csv, fd_convergence.csv).
- Post-src legacy job091659-1391 in FRESH REPL: byte identity PASS, no pre-src checkpoints reused (legacy_reproduction_post_src.csv, logs/legacy_post_src.log).
- Real-fold job092037-32396 in FRESH REPL: entrant/summer folds1/2/60 exact AD≤1.34e-15, FD convergence,0 B (real_fold_gates.csv). Endpoint overlap0, all source chronology assertions PASS, entry first-biweek rule PASS, fold1→2 moments/source IDs persistent (provenance_gates.csv, prior_manifests/). Hyper overlap separately reported under D9.

## Running / next
- Owned REPL: jl s2, beast16, BLAS1, checkout above.
- Recovery job092346-28511, watch s2-recovery-three-seeds; r09_recovery.jl. First attempt. No recovery result yet.
- Next: retrieve recovery outputs; if pass, fresh REPL normal2-fold qs_entrant smoke, then fresh REPL qs_summer smoke via r02_smoke.jl with S2_ONLY.
- Finish timing/report/exact S3 commands, todo check, local commits, stop owned REPL, HANDOVER.md and PHASES2_DONE. No S3 grid in S2.
- F6 remains deferred for James: fold cadence tied to dynamics_col; no fake weekly-refit arm, no src separation authorised. See F6_DECISION.md.

## Stop rules
All Julia beast via jl; options off by default; no 26/27 outcomes; no grids; only normal smoke DB writes.
Reproduction/leakage failure stops stage. A gate fails twice => stop and ask manager. No source gate failed on resume to date.
Records live here (hub absent); local commits only, no push/merge/other-worktree changes.
