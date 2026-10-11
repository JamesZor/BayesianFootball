# Stage 1 decisions

- 2026-10-11: Follow Stage 0 quote-identification contract: validate priors against fixture D/U, not future-smoothed isolated club first-quote ratings.
- Read-only export will use harness `_score_one`/observation frame without score persistence; audits mirror the existing ten-table pattern including timestamp maxima.
- Remote checkout `/root/BF_runs/qs_transition` is absent; create an isolated copy of this worktree, never modify another checkout. Use W2 grid cache only as a read-only input with checksum verification.
- Q9 source/rotation diagnostics and Q11 squad discontinuity are optional; no absence of these data will be treated as evidence against their mechanisms.
- W2 reproduced all six fits and cohorts (5/6/2); QS relegation bias is +1.50 pp, not the W2 negative bias. Diagnose panels separately and on common fixtures.
- Provenance exploration: SplitBoundary.target_match_ids are fitted current-season rows, and boundary IDs reset by season. Reconstruct held-out vintages through Data.get_next_matches and match metadata fieldwise.
- Provenance exploration: match_latents.fold_id attaches the aggregated latents to the first fold (db_storage.jl:795–796), not each fixture's prediction vintage. Replaced that false provenance with canonical splitter-heldout assignment; require disjoint/full latent-ID coverage.
- Calendar-year cup season labels accidentally admitted 12 held-back League Cup rows in the first raw cache export. No outcomes were inspected or scored. Corrected exports enforce match date <2026-07-01; only 26/27 coverage is queried. Record this in the provenance audit, not as a silent drop.
- Cup raw JSON has no regulation-time score fields in 888 queried fixtures; do not treat knockout full scores/winner flags as regulation outcomes without settlement evidence.
