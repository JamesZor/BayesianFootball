# Stage 2 progress

Phase: S2 DONE (F6 explicitly deferred for James); final records committed locally after verification.
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

- Recovery job092346-28511 PASS first attempt: entrant50/51 (98.04%), summer36/39 (92.31%) 90% intervals; no parameter misses all3. CSVs and log retrieved. Runner committed19d3ad4d.

## Completion / next owner
- Owned REPL: STOPPED after audit; tmux confirms jl-s2 absent. No active watches or Julia jobs. Other REPLs untouched.
- Runs used synced19d3ad4d (rsync reassigns folder ownership; chown own checkout root before git operations).
- Entrant normal2-fold smoke job092926-16179 PASS hard checks first attempt: 8ac3ccfa-3aa8-4721-9e17-7216e394a2f1; Rhat1.08015, ESS19.40/62.59, divergences0, convergence/performance REVIEW fail. Summary/checks/manifests/perf report retrieved. Normal harness smoke is2 chains200+200 per fold (old handover's4-chain description was wrong); recipes' S3 budget is4 chains500+1000.
- Summer normal2-fold smoke job093506-27894 hard PASS first attempt: 55c33930-2e27-418f-ba7d-9e330fdd6095; Rhat1.13926, ESS15.10/37.00, divergences0. Convergence/performance REVIEW fail; records retrieved.
- Read-only audit job094308-29566 completed: control has_passing_smoke=false solely due stale D11 failed hard check832 (run_id NULL); other3 true. Full archived history in CSVs. NO DB cleanup authorised/performed.
- Next owner: review convergence/performance flags; explicit manager/operator go-prod needed for narrow stale-check832 cleanup before control grid. No bypass. James decides F6. REPORT_S2 has exact S3 commands and frozen cache hash.
- Timing: four60-fold grids projected22.3715h sampling, no fifth-arm estimate while F6 deferred. All source gates and four latest smoke hard checks complete. TODO check PASS; local commits only; HANDOVER.md final. No S3 grid in S2.
- F6 remains deferred for James: fold cadence tied to dynamics_col; no fake weekly-refit arm, no src separation authorised. See F6_DECISION.md.

## Stop rules
All Julia beast via jl; options off by default; no 26/27 outcomes; no grids; only normal smoke DB writes.
Reproduction/leakage failure stops stage. A gate fails twice => stop and ask manager. No source gate failed on resume to date.
Records live here (hub absent); local commits only, no push/merge/other-worktree changes.
