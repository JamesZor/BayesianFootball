# Phase C revision 4 — blocked recovery checkpoint, not production reproduction

Statistical source **9de98548**. Branch `pi/market-model-phase-c`.
C0/C1/H1/H2 full books remain authorised; C2 remains deferred. Recovery mode
solver failed twice; no third variation or production launch is authorised by
this checkpoint. Read PHASE_C_REPORT.md and HANDOVER.md before resuming.

## Inputs and manifests

- Existing Manifest, Distributions 0.25.126; Julia laptop 1.12.1 / beast 1.12.4.
- Pinned `.cache/datastore_ScottishLower.jls`, SHA256
  `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
  Load max_age_hours=10^6; never refresh or update packages.
- `fullbook_seeds.toml`: recovery generation 3962, chains 4961–4964;
  fresh seeds reserved explicitly for all four production rungs/protocols.
- `SOURCE_SHA256SUMS.txt`: current source/seed paths relative to stream directory.
  `SOURCE_ENGINE_CHECKPOINT_SHA256SUMS.txt`: previous engine-only source at
  622f9a55; `SOURCE_RECOVERY_LOGSPACE_SHA256SUMS.txt`: source c2547742.
- `SCIENTIFIC_SHA256SUMS.txt`: historical/preflight/aborted recovery engine CSVs,
  relative to results/C. **Not** evidence of two successful production runs.

```bash
cd current_development/market_model
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
(cd results/C && sha256sum -c SCIENTIFIC_SHA256SUMS.txt)
```

Original engine-only report/repro/handover are preserved as
`PHASE_C_ENGINE_CHECKPOINT_REPORT.md`, `REPRODUCIBILITY_ENGINE_CHECKPOINT.md`,
`HANDOVER_ENGINE_CHECKPOINT.md`. V1/V2/V3 archives remain unchanged. Historical
root C CSVs are v3 diagnostics, not production results.

## Fresh final-source laptop verification

Owned %52, now closed. Log `LAPTOP_C4_AD_TESTS.txt`: accepted **270/270**,
T05_C4_DETERMINISTIC_DONE, then separate C2-pending **29 pass / 10 fail**,
C2_PENDING_REPORTED. No T05_DONE: parameter recovery is blocked.
`LAPTOP_C4_FINAL_SAMPLING_REGRESSIONS_PREFLIGHT.txt`: t04 **92/92**, t03
**131/131**, t02 **131/131**, and C05_C4_ENGINE_PREFLIGHT_DONE. t03's A rerun
uses temporary outputs; no prior-phase artifact replacement.

Final-source r05 regenerated all six `v4_preflight/*.csv` byte-identically to
the previous engine checkpoint (`diff -rq` against a captured directory).
Full Gate 1 **12/12** and state gates **53/53** pass. These are deterministic
preflight reductions, not posterior recovery/convergence or production runs.

For deterministic tests only, verify the session name is unused, start from
worktree root, use the returned pane ID ONLY:

```bash
PANE=$(tmux new-session -d -P -F '#{pane_id}' -s pi_julia_mm_c -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 julia --project -t 8')
printf 'owned pane: %s\n' "$PANE"
tmux send-keys -t "$PANE" -l -- \
  'include("current_development/market_model/t05_pooled_tests.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
tmux capture-pane -t "$PANE" -p -J -S -2000
```

Wait for accepted summary and C2_PENDING_REPORTED separately. In successive
commands, waiting for each result:

```julia
include("current_development/market_model/t04_copula_grid_tests.jl")
include("current_development/market_model/t03_covariance_tests.jl")
include("current_development/market_model/t02_two_stage_tests.jl")
include("current_development/market_model/r05_pooled.jl")
```

r05 replaces six preflight CSVs only. It does not sample/evaluate and must not
be represented as R05_DONE. Capture evidence before `tmux kill-pane -t "$PANE"`.
All laptop %49/%50/%51/%52 are closed.

## Beast provenance — no session to resume

Dedicated detached checkout `/root/BF_runs/market_model_c`, at **9de98548**.
Manifest symlink to `/root/BayesianFootball/Manifest.toml`; no .env required by
this cache-only path. Cache copied `cp -p` from market_model_b and hash checked.
Initial load .02. Only owned %261/%262/%263 were operated; all closed.
No other human/manager panes, src/package/data changes or DB writes.

| Source/pane | Output | Outcome |
|---|---|---|
| 19ad4b99 / %261 | results/C/v4_recovery (original); v4_recovery_first_attempt (committed engine CSV) | probability-storage underflow before sampling |
| c2547742 / %262 | original directory moved to `/root/BF_runs/logs/market_model_c/c2547742_recovery_outputs/`; committed engine CSV in v4_recovery_logspace | generated books, FD joint-mode line-search failure |
| 9de98548 / %263 | results/C/v4_recovery_ad | generated books, same joint-mode line-search failure with AD; peers cancelled |

Logs under `/root/BF_runs/logs/market_model_c/`:
`19ad4b99_recovery1.log`, `c2547742_recovery2.log`,
`9de98548_recovery3.log`. Normalised terminal captures are committed as
`BEAST_C4_RECOVERY_FIRST_ATTEMPT.txt`, `BEAST_C4_RECOVERY_LOGSPACE_ATTEMPT.txt`,
`BEAST_C4_RECOVERY_AD_ATTEMPT.txt` (the last expands ALL exceptions).

No C1_recovery.jls fit exists. Both latter attempts retain synthetic_panel.jls;
include l05_pooled.jl before Serialization.deserialize. Both panel binaries have
SHA256 **de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1**.
This records identical generated data, not completed recovery or reproduction.

The last run was invoked through its own persistent REPL with:

```bash
# Historical command, NOT an instruction to repeat the failing experiment.
env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 \
  C05_RECOVERY_OUT=/root/BF_runs/market_model_c/current_development/market_model/results/C/v4_recovery_ad \
  /root/.juliaup/bin/julia --project -t 16
```

```julia
include("current_development/market_model/r05_fullbook_recovery.jl")
```

Runner pins cores, BLAS=1, four chains of 2000+3000; seeds/truth from source.
Fails before any retained draws/intervals/convergence tables. Not
R05_FULLBOOK_RECOVERY_DONE, R05_DONE or T05_DONE.

## Stop boundary and remaining acceptance

Supported theta for the root failing chain (seed4964, warmup8):
`[-3.549527585137839,-4.460929121755582,-2.821549571263347,7.9567722491577495]`.
Line-search gradient `[-1.5699131339808048e-5,3.249019587192592e-5]`.
The previous FD attempt failed the same mode check. **No third variation**;
review a solver robustness remedy before restarting. Never turn these errors
into posterior -Inf, change seed/priors/settings/population or loosen gates.

After that dependency is resolved, the remaining authorised work is still:
complete recovery/acceptance; actual 10a and ID-restricted 10b C0/C1/H1/H2 fits;
convergence; B2-compatible pre-week scoring/paired contrasts; theta+u smoothing,
noise/n and q/s summaries; uncertainty-integrated H1/H2 HA posteriors; figures;
read Ridall source before citing; a production r05 workflow regenerating Gate1;
and **two fresh complete beast runs** at frozen source/seeds with byte-identical
scientific CSVs. No such reproduction currently exists.
