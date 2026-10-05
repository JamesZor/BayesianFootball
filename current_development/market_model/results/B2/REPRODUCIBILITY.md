# Phase B2 reproduction record

## Full covariance ladder

Statistical SHA: `25a408e2eb031f53d5cac2478a48ed277f4a3fc9`.
Two fresh owned beast REPLs, Julia 1.12.4, 8 threads during manager review,
core pinning and BLAS=1; pinned cache SHA256
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.

Both commands:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
include("current_development/market_model/r03_covariance_ladder.jl")
```

- Fresh pane `%250`: `R03_DONE`, 566.8761510848999 seconds.
- Fresh pane `%251`: `R03_DONE`, 565.3235380649567 seconds.
- Every one of the 10 original runner-generated CSVs and both B2 PNGs is
  byte-identical by `cmp`, with no normalization or omitted provenance columns.
- Actual timestamps/fit wall times append to `RUN_PROVENANCE.md`, deliberately
  outside the scientific CSV byte comparison.
- External logs: `/root/BF_runs/logs/market_model_b2/25a408e2_run1.log` and
  `25a408e2_run2.log`. First-run CSVs/PNGs remain in `25a408e2_run1/` there.

## Mid-run human addendum: no new fits

Post-processing SHA: `fd8b23a9f9b82fc75ebc7056282588bbe834bca5`.
The original fits are retained as ignored `R<rung>_<protocol>.jls` files beside
these CSVs. Include the current prototype loader before deserializing them.
Two subsequent fresh owned beast REPLs at this SHA read those existing fits:

```julia
include("current_development/market_model/r03_hierarchy_postprocess.jl")
```

- Fresh panes `%252` and `%253` both reach `R03_HIERARCHY_DONE`.
- `rotation_quality_style.csv` and `league_shock_test.csv` are byte-identical.
- All 10 original scientific CSVs and both figures remain unchanged.
- External logs: `fd8b23a9_post1.log` and `fd8b23a9_post2.log` in the same log
  directory. First addendum CSVs remain in `fd8b23a9_post1/` there.
- No additional sampling was launched after the addendum; the full runner now
  calls the same deterministic post-processing helper for future clean runs.

## Fresh laptop regression including addendum

Own pane `%31`, Julia 1.12.1, 8 threads, BLAS=1, addendum source:
`t03_covariance_tests.jl` **131/131**, `t02_two_stage_tests.jl` **131/131**;
anchored `R01_DONE`, `T03_DONE`, `T02_DONE`. Full log: `LAPTOP_TESTS.log`.
Phase A's 16 non-runtime CSVs match byte-for-byte in a temporary output directory.
The human approved excluding only nondeterministic `runtime.csv`.

## Verify returned scientific artifacts

From this directory:

```bash
sha256sum -c SCIENTIFIC_SHA256SUMS.txt
```

This covers all 12 scientific CSVs and both PNGs. To reproduce exact core hashes,
use the statistical SHA above in your own detached beast worktree, then update
that owned checkout to the post-processing SHA and run only the addendum runner.
Later artifact/documentation commits change HEAD-based convergence provenance,
not the fitted implementation. A fresh full run at later HEAD consequently
requires comparing numerical columns separately or checking out the statistical
SHA for strict byte identity. Never change a provenance column to fake equality.

Create and target only owned tmux pane IDs, as documented in the stream README;
never one-shot Julia, session-name targeting, or operating another agent's pane.
The beast checkout and logs are retained for manager review; owned REPLs are closed.
