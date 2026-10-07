# Phase C revision 2 — reproducible blocker, not production acceptance

Statistical implementation: `7b6c7659`. Branch `pi/market-model-phase-c`, worktree
`/home/james/bet_project/.worktrees/BayesianFootball-market-model-pi-c`.
No pooled sampling or beast run occurred. **No R05_DONE, T05_DONE, convergence,
synthetic parameter recovery or byte-identical production rerun is claimed.**

The revision-1 notes are preserved verbatim in `REPRODUCIBILITY_V1.md`; its
report is `PHASE_C_V1_REPORT.md`, and its preflight code/CSV hashes are unchanged.

## Inputs and numerical choices

- Existing Manifest, Julia 1.12.1, Distributions 0.25.126; no package changes.
- Pinned cache `.cache/datastore_ScottishLower.jls`, SHA256
  `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
  Load with `max_age_hours=10^6`, no refresh or database writes.
- Existing `MarketModelConfig.scottish_lower_2425_2526()` supplies panel, seasons,
  tournaments, close window, exclusions and population assertions.
- Exact density via the unchanged production DoublePoisson grid (cutoff 10).
- First 30 full books in match-ID order. Prediction centres are their Phase A
  isolated rates; additional offsets are +0.10 on both log-rate sides.
- n=250/1000/4000, equal-side prediction SD=0.05/0.20, all combinations retained.
- Mode starts at the prediction. Central derivatives use step 2e-4; Newton with
  line search ends on score or <=2e-7 Newton displacement. The displacement
  criterion resolves finite-difference/density roundoff, not the gate threshold.
- Exact quadrature orders 32/64 independently evaluate the integrand and posterior
  moments in Gaussian importance coordinates. All densities include normalisers.
- The likelihood eigenvalue clip is 1e-8 as prescribed; it is inactive on all
  tested full books. Marginal uses the actual bracket Hessian; update moments
  use clipped information. Raw SD errors are saved as a diagnostic.
- Stop after the first failed book type. Availability for untested types is
  retained in `book_inventory_c.csv`, with tested=0, not fabricated pass rows.

## Fresh owned laptop REPL

Verify the session name is unused; never operate unrelated human/manager panes.
From the task worktree root:

```bash
PANE=$(tmux new-session -d -P -F '#{pane_id}' -s pi_julia_mm_c -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 julia --project -t 8')
printf 'owned pane: %s\n' "$PANE"
tmux send-keys -t "$PANE" -l -- \
  'include("current_development/market_model/t05_pooled_tests.jl"); include("current_development/market_model/t04_copula_grid_tests.jl"); include("current_development/market_model/t03_covariance_tests.jl"); include("current_development/market_model/t02_two_stage_tests.jl"); include("current_development/market_model/r05_pooled.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
# Wait for the anchored markers before capture/cleanup.
tmux capture-pane -t "$PANE" -p -J -S -2000
```

Fresh pane **%42** produced `LAPTOP_REVISION2.txt`:

- 24/24 necessary update unit tests, `T05_UPDATE_DONE`;
- 92/92 t04, 131/131 t03, 131/131 t02, corresponding DONE markers;
- `R01_DONE` from t03's temporary Phase A byte regression;
- `C05_LIKELIHOOD_BLOCKED`: marginal checks pass, posterior-mean gate fails.

No `T05_DONE` or `R05_DONE` is printed. Do not interpret the unit-test count or
successful runner execution as a passing scientific gate. The source revision
contains only the likelihood gate stage, not a completed production pipeline.
The mean failure also appeared during development in owned pane %41; no further
model variations or sampling were tried after fresh confirmation. Both owned
panes were closed after capture. No beast sessions/checkouts were created.

## Hashes and checks

`SCIENTIFIC_SHA256SUMS.txt` covers the three unchanged revision-1 CSVs plus the
four new revision-2 gate/inventory CSVs. It identifies committed evidence; it
is **not evidence of two fresh production executions**. `SOURCE_SHA256SUMS.txt`
adds the three new production-named (but incomplete) source files to the old
preflight source hashes.

```bash
cd current_development/market_model
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
cd results/C
sha256sum -c SCIENTIFIC_SHA256SUMS.txt
```

At handoff `./scripts/todo.sh check` and `git diff --check` pass. Existing
A/B/B2/B3 outputs, original preflight CSVs and all `src/` files remain untouched.
TODO 039/index say BLOCKED, not complete. See `PHASE_C_REPORT.md` for the
failed setting distributions, concrete mean witness and missing acceptance.
