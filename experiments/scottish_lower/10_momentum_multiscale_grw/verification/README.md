# TODO 022 evidence

These CSVs are copied unchanged from the completed beast runs. No posterior
chains, credentials, DataStore caches or binary database artifacts are committed.

- `stage1/`: accepted smoke diagnostics and UUID lineage, common tradeable panel,
  posterior summaries and descriptive small-cohort diagnostics.
- `production/`: all-fold filtration and production run addresses, full-draw
  convergence summaries, storage stride and full local artifact paths.
- `evaluation/`: headline/per-market scores, 4,000-replicate paired fixture
  bootstrap results (seed 22), momentum posterior summaries, all market inversion
  outcomes, common portfolio panel/refusals, match-level predictions and ledgers.

Model source fingerprint:
`6869de6297b580cdc7aaf3052ba31c81c1facc0bc826d495e770bfbbb4a73c6f`.
Sampling code: `a6cca1bc`; Stage 3 evaluation: `168d90c4`. Both runners exited 0.
The parent README contains immutable model and portfolio UUIDs and conventions.

Full-draw diagnostics describe 3,200 retained draws per fold. Evaluation and SQL
artifacts use the user-approved one-in-four subset (800 draws per fold). The
initial smoke strict-zero threshold mistake was corrected by re-auditing the
same chains; original failed-verdict artifacts remain in the database for audit.

Run the read-only sign-off verification from the repository root:

```bash
python experiments/scottish_lower/10_momentum_multiscale_grw/verification/verify_results.py
./scripts/todo.sh check
```

This checks artifact consistency, not a new MCMC run or an independent
reimplementation of the optimizer/bootstrap. The Julia deterministic tests,
compiled AD checks and database round-trips are recorded separately in the parent
README. Missing market coverage and nonzero truncated score-grid tails remain
limitations; a successful consistency audit does not waive either.
