# S1 handover — complete

Read `../REPORT_S1.md`, then `DECISIONS_LOG.md`. Stage 1 ends here; do not run fits from this packet.

## Done

- Reproduced all six existing runs through harness read-side scoring, without persistence.
- Frozen W2 cohort gate passed 5/6/2; W2 lower panel 1070, QS panel 710.
- Q1–Q8/Q10 CSV diagnostics complete; Q9 source/competition checks descriptive, rotation unresolved;
  optional Q11 explicitly skipped. Q8 whole/endpoint sensitivity completed; older-results causal
  robustness remains an open question.
- Proposed A2-centred P2 mean transport with earlier-seasons-only parameter construction and
  explicit fallback. Forward first-quote quality RMSE 0.160 vs league mean 0.210 and A2 0.169;
  improvement over A2 is uncertain, not a fitted-arm pass.
- Chronology/value-hash manifest passes. Naive historical-init placement and repeated market
  likelihood evidence are blocked. Do not interpret `q12_status.csv` as an unconditional F1 approval.
- Eight ten-table database audits unchanged; `verification.csv` passes 27 checks.
- No production fits, src edits, DB writes, pushes, merges or 26/27 outcome diagnostics.
- The beast REPL `s1` was stopped and its tmux session absence checked. No open panes/jobs/watches.

## Exact next action (manager)

1. Review the cause verdict: W2 persistent tier/transport bias, some history-poor cold start;
   QS promotion entry/transport error but **positive**, not negative, aggregate relegation bias.
2. Approve the S2/S3 list. Start with QS+A2 composition; decide the F1/F2 evidence policy before
   implementing prior means. Prefer disjoint/sequential semantics, or explicitly label an EB arm.
3. Resolve common-pyramid gauge and state covariance, which the direct-fixture mean test does
   not certify. Repeat prior-only checks if source eligibility, window or evidence policy changes.
4. S3 needs the unchanged QS recipe on the full 23/24–25/26 panel; the existing QS run only
   covers 24/25–25/26. Freeze the shared market-input recipe before that comparison.
5. Do not add cups/pxG/squad effects as a bundled first arm; do not score the held-back season.

## Outputs and execution

- Report: `experiments/scotland/11_qs_transition/REPORT_S1.md`.
- Code, results, figures and logs: this `eda/` directory; see `README.md` for replay order.
- Exact run UUIDs/namespaces: `results/resolved_runs.csv`.
- Exact input/parameter/result identity: `results/artifact_sha256.csv`,
  `results/q12_source_manifest.csv`, `results/q12_fold_prior_hashes.csv`.
- Remote isolated copy: `/root/BF_runs/qs_transition`. W2 cache read-only source:
  `/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls`, md5
  `db3ed548191bcd5287a6c1a18e34a105`.
- Last Julia job: `jl s1 232833-21881` (close coverage), ok and audit unchanged.
- Branch: `exp/qs-transition`; all commits local. Manager owns pushing.

## Important traps already resolved

- `match_latents.fold_id` attaches aggregate draws to first fold; canonical held-out vintage
  reconstruction is in `l01_manifests.jl`, not that FK.
- `SplitBoundary.target_match_ids` are fitted current-season rows; include them in actual cutoff.
- Cup JSON has an `event` wrapper. Regulation scores are explicit, not knockout winner flags.
- Calendar-year League Cup labels initially admitted held-back rows to a raw cache export.
  They were removed by an actual-date embargo before any outcome diagnostic; development git
  history preserves the correction. Only coverage was inspected for those rows.
- Frozen W2 Championship 25/26 qualifying TWA closes are 5/175; supplied inversion coverage is
  zero. Do not substitute a different archive/cache/estimator count into reproduction outputs.
