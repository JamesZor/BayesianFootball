# MAE deep research handover

**Status: complete; manager owns any commit. No git writes performed.**

Read `REPORT.md` first: 13-line summary, measured candidate effects, literature, five ranked QS-arm designs and reproduction details.

## Main findings
- The small recent-relative-GD/shots/SoT diagnostic reduces held-out mover MAE **5.334→4.527 pp** (−0.808), non-movers −0.917. It survives six-arm, training-scope, delay and ridge sensitivities.
- Gains are in games 6–20. Games 1–5 **do not improve**. The opening information/boundary question remains separate.
- Opponent information matters: own-only form improvement −0.236 versus relative form −0.808. This is not exclusively a mover problem.
- Current SofaScore values/manager fields fail historical-vintage tests. Values are excluded from final regressions. BBC archived manager strings recover real changes but generic manager flags do not improve forward MAE.
- Other simple squad/resource/last-season proxies show little or no forward mover-MAE gain. Actual announced-XI strength, dated values and financial regimes remain untested or underidentified.
- A coherent 1X2 projection improves MAE but has outcome ΔLL **+0.0030 [−0.0038,+0.0095]** overall. No real QS arm or production gate has passed.

## Deliverables and checks
- `queries.sql` includes the SQL pipeline; companion files keep extraction/features/audits readable.
- 31 CSVs, scripts, literature references, `extract_snapshot.tar.gz` (1.4 MB), and `input_manifest.csv`.
- Final directory ~5 MB. Large loose extracts are removed after offline reproduction; archive retains them.
- `verify.py`: **12 checks passed**. Exact canonical reproduction: **221 observations, zero difference**; control win panel **1,795 rows / 898 fixtures**.
- 12 immutable input hashes verified unchanged; `./scripts/todo.sh check`: **exit 0** (`todo_check.log`).
- Offline reproduction verified after final exclusion of valuation features:
  ```bash
  PYTHON=/tmp/qs-s1-venv/bin/python bash experiments/scotland/11_qs_transition/research/mae_deep/reproduce.sh
  ```
  Use another interpreter with `requirements.txt` dependencies if the temporary environment disappears.

## Next decision
Approve at most one small, coherent fast-quality/shot-information QS attribution arm, with explicit forecast information times, both teams' features, unchanged outcome gates and an early-game guard diagnostic. Do not treat residual regression gains as evidence to deploy or to unblind 26/27.

No open jobs/panes. No Julia, QS fitting, database writes, source changes outside this directory, git writes, or 26/27 outcome analysis.
