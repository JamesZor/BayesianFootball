# B3 reproduction

Statistical source: `2e02a7854a9ccc59fd8ba4c81aa6cf9ca6f5090a`.
Later result/documentation commits do not change the mathematics; convergence
CSVs embed HEAD, so check out this exact SHA to reproduce their bytes.

## Environment and safety

- Dedicated checkout: `/root/BF_runs/market_model_b3`, detached at the source SHA.
- `.env` and `Manifest.toml` symlink `/root/BayesianFootball/` (never print URLs).
- Pinned cache copied **with mtime preserved** from
  `/root/BF_runs/market_model_b/.cache/datastore_ScottishLower.jls`.
  SHA256: `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
- Julia 1.12.4, 16 threads, core pinning, BLAS=1; Distributions 0.25.126.
  No package addition/update, data refresh or database write.
- Check load before launch; above 20 blocks. Only own `pi_mm_b3` session.

## Full rerun (persistent REPL, never one-shot)

From the dedicated checkout, create an owned session and retain its returned ID:

```bash
PANE=$(tmux new -d -P -F '#{pane_id}' -s pi_mm_b3 -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 /root/.juliaup/bin/julia --project -t 16')
printf 'owned beast pane: %s\n' "$PANE"
tmux pipe-pane -t "$PANE" 'cat >> /root/BF_runs/logs/market_model_b3/review.log'
tmux send-keys -t "$PANE" -l -- \
  'using LinearAlgebra; BLAS.set_num_threads(1); include("current_development/market_model/r04_copula_grid.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
tmux capture-pane -t "$PANE" -p -J -S -1000
```

Wait for anchored `R04_DONE`. The default runner always refits; binaries are not
resume inputs. It rewrites every generated scientific CSV and all B3 PNGs.
At this source the PPC has no extreme p-value, so no Student-t alternative is
fitted. The runner fails closed if a future snapshot has an extreme PPC: do not
promote Gaussian adequacy or silently skip the authorised alternative.

From `current_development/market_model/`, verify:

```bash
sha256sum -c results/B3/SCIENTIFIC_SHA256SUMS.txt
```

Repeat in a **new Julia process**, using a new returned pane ID after closing
only the owned old pane. Do not operate manager/human sessions.

All CSVs (including per-grid profile progress files and raw fixture/selection/PPC
rows) and all three PNGs are scientific hash targets. `RUN_PROVENANCE.md`, reports,
methods, logs and binary fits are not: timestamps, wall time and `fit.seconds`
are nondeterministic and are never overwritten with fictitious timings.

## Laptop acceptance tests

Julia 1.12.1, own `pi_julia_mm_b3` pane `%36`, 8 threads, BLAS=1, statistical
source above. In one fresh persistent REPL:

```julia
include("current_development/market_model/t04_copula_grid_tests.jl")
include("current_development/market_model/t03_covariance_tests.jl")
include("current_development/market_model/t02_two_stage_tests.jl")
```

Recorded: t04 **92/92**, `T04_DONE`, 35.6 s; t03 **131/131**, `T03_DONE`, 41.2 s;
t02 **131/131**, `T02_DONE`, 59.3 s. t03 also reaches `R01_DONE`, reproducing all
non-runtime A CSVs in a temporary directory. Evidence: `LAPTOP_FINAL_TESTS.txt`.
The t04 normalisation/marginal tests document finite-support errors rather than
claiming exact untruncated means at cutoff 10.

## Recorded beast runs

First fresh pane `%258`: `R04_DONE`, 501.5812849998474 s (runner timer, excluding
package load/Julia startup). Logs and first-run CSV/PNG snapshots remain under
`/root/BF_runs/logs/market_model_b3/`; first log `2e02a785_run1.log`, plain capture
`2e02a785_run1.txt`, hashes `2e02a785_run1.sha256`, snapshots `run1_csv/` and
`run1_figures/`. Second fresh pane `%259`, same SHA/seeds/settings: `R04_DONE`
in **508.820631980896 s**. `sha256sum -c 2e02a785_run1.sha256` passes **35/35**:
all **32 CSVs and three PNGs byte-identical**. Second original log/capture:
`2e02a785_run2.log` / `2e02a785_run2.txt`; committed plain captures:
`BEAST_RUN1.txt` / `BEAST_RUN2.txt`. The laptop copy also passes all 35 hashes.
Owned REPL panes are closed; no other session was operated.
