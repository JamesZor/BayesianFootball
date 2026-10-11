# Stage 1 decisions

- 2026-10-11: Follow Stage 0 quote-identification contract: validate priors against fixture D/U, not future-smoothed isolated club first-quote ratings.
- Read-only export will use harness `_score_one`/observation frame without score persistence; audits mirror the existing ten-table pattern including timestamp maxima.
- Remote checkout `/root/BF_runs/qs_transition` is absent; create an isolated copy of this worktree, never modify another checkout. Use W2 grid cache only as a read-only input with checksum verification.
- Q9 source/rotation diagnostics and Q11 squad discontinuity are optional; no absence of these data will be treated as evidence against their mechanisms.
