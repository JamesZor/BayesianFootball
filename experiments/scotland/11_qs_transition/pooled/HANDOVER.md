# Pooled S4 prep — PHASEPOOL_DONE

## Completed; do not repeat verified gates
- Branch `exp/qs-transition-pooled`, worktree `/home/james/bet_project/.worktrees/BayesianFootball-qs-transition-pooled`.
- Recipe commit7e76dbfc; recovery8915cb8d; smoke archive/operator script27894088; final report/evidencec01caac6. Handover follows locally. No pushes or merges.
- `qs_pooled_a2`: Championship+lower goals, frozen lower-only market likelihood, A2 `:first_in_window`, no entrant prior/cups/pxG. No `src/` changes.
- Target panel60 folds/1070 fixtures exactly matches control per fold. Championship goal rows360/370/529 on1/2/60 have zero market masks;25/26 goals169 rows on60.
- Three-seed two/three-tier recovery PASS:42/45 +41/45 intervals; step/carry6/6 each; max q/s centring1.173e-15. Broad step intervals: prior-assisted identification, not a precise/data-only identification claim.
- Fresh normal two-fold hard smoke PASS at exactly0B tapes, including23/24 opener and final25/26 fold. Run `5cb867cd-f6b8-43ef-a4d7-3d5da9c47ca2`, recipe `0db99f621d0e430127f62404c530f07bdbdb22a459f8a345856193e0e09df887`.
- Convergence remains REVIEW: Rhat1.09424, bulk ESS20.49/tail48.03, zero divergences. Performance review PASS. Do not call this production acceptance.
- Failed attempts retained/documented. Two diagnostic failures reached the stop rule; user expressly authorised fixing/continuing. Initial smoke allocation2464/2560B cleared in a fresh REPL with no source/package/threshold changes; process-state cause unproved.
- Initial null-run check890 was fully archived before cleanup. User separately authorised deletion of ONLY890; guarded transaction removed1row. Backup SHA `f094f0ef37ed50489f905a0888be7ee8e283928dddd7e74a1ded26cc4bb46ef4`. Post-cleanup historical gate `has_passing_smoke=true`; original/current histories retained. No other rows changed.
- Estimated16thread60fold grid3.52h ideal scaling; budget4–6h (not measured). Exact unexecuted grid command and assumptions in `REPORT_POOLED.md`.
- Report summary9lines; todo check/git whitespace check PASS. Only owned pool/4threads used, now STOPPED; no active jobs/panes/watches. No grids or26/27 outcomes.

## Owned beast checkout / reproducibility
- `/root/BF_runs/qs_transition_pooled` only. Final records/source synced there; branch metadata imported from local bundle (no push). Never touched S3/F6 checkouts or REPLs.
- Frozen W2/S2 cache copied read-only from `/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls`.
- Cache SHA `a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12`; frozen market SHA `1098f12aff1db116d3fc46454ccc00297402b61f5011502c0ffd86f7298c9add`.
- Manifest copied on beast, never via rsync. SHA `f86f49b72b987dd4c87ac80d16f132e5342e988fededbe27e54c6afe8133b6cc`. Runtime versions in logs/package_versions.log. Preserve it/cache for like-for-like comparisons.
- Rsync excludes `/.git`, `.env`, `Manifest.toml`, `/.cache/`, `/data/`; use `--no-owner --no-group` to preserve owned checkout ownership. Early copied worktree.git pointer/ownership issue repaired only here.
- `gate_commands.csv` gives every recorded job ID and outcome. `r05_authorised_cleanup.jl` is one-off, NOT automatic/reusable; do not rerun it.

## Next owner
Manager reviews convergence and the identification caveat, then decides S4 after S3/F6 compute is free.
No extra fit/recovery/smoke is required to re-establish already passed checks. Use the report's normal
harness command only after approval, with the same frozen panel/recipe; do not combine with entrant priors.
All experiment records are in this pooled directory because the hub directory does not exist.
