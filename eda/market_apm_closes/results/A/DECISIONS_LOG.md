# Phase A decisions

- 2026-11-03: User approved local commits only; manager must push, overriding the brief's push requirement to comply with worker rules.
- Pinned datastore SHA256s verified on mcmc-beast against QS/DATA_SNAPSHOTS.md; all three match exactly. No reinversion or DB access.
- Use a new isolated clone at /root/BF_runs/market_apm, populated from this branch via a git bundle. Existing beast checkouts remain untouched.
- Coverage is the first gate: pause for user direction if any league loses >15% of accepted closes. No downstream modelling before that gate passes.

- Coverage passed all leagues: no >15% loss. English cache embeds QualityStyleEDA.MarketModelEnglish; restored that module identity before loading. Initial schema attempt hit the loader fallback but BF_DB_URL was unset, so no database access occurred.
- Use all fixture-side histories (not just accepted closes) for causal regulars; ties break by player ID. First five means each team’s first five recorded fixtures, not first five complete closes. Missing prior/next fixture lineups are not silently skipped.
- Player design has separate own-XI attack and opposing-XI defence columns, stacked over home/away log rates; identifiable dimension is rank([team FE, starter design]) − rank(team FE). This is in-sample algebraic rank, not a reliability claim.
