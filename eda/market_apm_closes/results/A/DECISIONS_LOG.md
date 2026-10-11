# Phase A decisions

- 2026-11-03: User approved local commits only; manager must push, overriding the brief's push requirement to comply with worker rules.
- Pinned datastore SHA256s verified on mcmc-beast against QS/DATA_SNAPSHOTS.md; all three match exactly. No reinversion or DB access.
- Use a new isolated clone at /root/BF_runs/market_apm, populated from this branch via a git bundle. Existing beast checkouts remain untouched.
- Coverage is the first gate: pause for user direction if any league loses >15% of accepted closes. No downstream modelling before that gate passes.
- Coverage passed all leagues and seasons. English cache embeds QualityStyleEDA.MarketModelEnglish; restored that module identity. Initial schema loading attempted the default loader fallback but BF_DB_URL was unset, so no DB access occurred.
- Causal regulars use all fixture-side histories, ties break by player ID, and unknown XIs add no starts. First five refers to recorded fixtures, not complete closes. Complete-history sensitivity excludes earlier invalid sides.
- Absorbed starter dimension uses separate own-XI attack and opposing-XI defence columns stacked over both rates: rank([team FE, starters]) minus rank(team FE), with tolerance 1e-8. Rank is not reliability.
- Absence regressions report both match-HC1 and season-block CR1 uncertainty; prioritize small-block t intervals. Only the response is residualized, as the brief requests; this is not a full FWL coefficient.
- Form control uses immediately previous fixture residuals in team perspective. Placebo uses immediately next fixture absences. Missing immediate fixtures are not skipped. Include current-on-placebo-sample and complete-history sensitivity rows.
- Home-probability translation holds log total fixed for supremacy: lambda_h times exp(beta/2), lambda_a times exp(-beta/2). Not a causal intervention estimate.
- Design player-start thresholds/movers use joined closes; top-11 share and rotation use all valid selected-season fixture-side histories.
- Analysis job 012840-27203 failed at parse time (ternary loop iterables needed parentheses); corrected and reran 012924-4465 successfully.
- Remote output retrieval initially overwrote local progress/decisions with stale clone copies. Restored decisions from the session and rewrote final progress; no CSVs were lost. Future retrieval should exclude PROGRESS.md and DECISIONS_LOG.md.

- Final validity refinement: a player ID shared across the two sides invalidates both fixture-side histories as well as the joined match. Recomputed analysis in 013213-31593 (ok); primary results/ranks unchanged.
- Added failure_summary.csv to distinguish missing inversion rows, explicit inversion refusals and accepted-close join failures. Final coverage job 013327-19065 ok; prior relative-include eval 013319-21964 failed without touching outputs.
- Results: current supremacy association is similar in Scotland/England; next-match placebo is non-null in six leagues under season-block intervals. Do not interpret absence slopes as causal player impacts. Conditional exploratory Phase B only; none implemented. Scottish League Two design saturates residual observation rank.
- User approved keeping the experiment card in REPORT_A.md/HANDOVER.md because ~/bet_project/hub/experiments/ is absent. No external hub files created.
- Final checks: 013549-7917 ok, 11 synthetic history tests plus 97 output-contract tests. Rank audit 013135-6144 ok (16 tests); all nine SVGs parse as XML. todo check and git diff --check pass. mapm REPL stopped after environment job 013613-1852 ok.
