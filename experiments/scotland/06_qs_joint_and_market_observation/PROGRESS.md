# Wave 2 — progress

Updated: 2026-10-09 UTC. Owner: pi. Manager: Claude.
Branch: `exp/qs-goal-vs-market`. TODO: 041.

## Current phase

**Phase 0: preparing fresh beast branch-health tests and proxy-xG coverage.** No phase is accepted yet. No component, smoke, grid or scoring has started.

## Done

- Local branch clean at 9a86f7b1; task created/claimed.
- Beast checkout `/root/BF_runs/qs_experiment` clean at d16a5800, dependency/environment symlinks intact.
- Pinned snapshot digest and original mtime verified unchanged.
- Read wave-1 summary, panel/fold code and Phase 0 launcher; manager C2 exclusion preserved.

## Next action

Commit/push Phase 0 setup; update isolated beast checkout. Run all prescribed test files sequentially in fresh REPLs, then coverage audit. Stop/report any failed gate. Record exact counts, wall times and evidence before Phase 1.

## Beast panes and artifacts

No owned wave-2 panes yet. Never touch other sessions.
Logs: `/root/BF_runs/logs/qs_experiment_w2/`.
Outputs: `/root/BF_runs/qs_experiment_w2_out/`.
Local evidence: `experiments/scotland/06_qs_joint_and_market_observation/logs/` and `results/`.

## Definition of done

All five phase boundaries evidenced and committed/pushed; exact work-package tests, four smoke hard passes plus manager grid approval, ordered grids, complete CSV-traceable paired report and saved reference parity, fresh frozen-seed reproduction, todo/diff checks. Stop on a failed gate; do not loosen thresholds.
