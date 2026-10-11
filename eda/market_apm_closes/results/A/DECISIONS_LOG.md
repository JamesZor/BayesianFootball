# Phase A decisions

- 2026-11-03: User approved local commits only; manager must push, overriding the brief's push requirement to comply with worker rules.
- Pinned datastore SHA256s verified on mcmc-beast against QS/DATA_SNAPSHOTS.md; all three match exactly. No reinversion or DB access.
- Use a new isolated clone at /root/BF_runs/market_apm, populated from this branch via a git bundle. Existing beast checkouts remain untouched.
- Coverage is the first gate: pause for user direction if any league loses >15% of accepted closes. No downstream modelling before that gate passes.
