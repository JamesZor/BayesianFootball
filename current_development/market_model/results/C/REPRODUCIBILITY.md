# Revision 8 reproduction and frozen live-run ownership

Fast statistical/reporting source:
`70336b07a81aac7bc3d7188e7ae2138fb3b59072`.
C1 production source: `41e28736cdd990d8d701dcb0b48df98196c54674`.
Live synthetic recovery source: `6335316f4e93ec91495341fa2f6960a64ce772a9`.
Revision7 notes are preserved in `REPRODUCIBILITY_V7.md`.

## 1. Fixed inputs and mathematical contract

Pinned cache SHA256
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`,
max_age_hours10^6; existing Manifest, Distributions0.25.126. Julia1.12.4 on
beast,16 threads/core pinning/BLAS1; no package/cache/database writes or updates.
Full books only,517 fixtures/91 weeks/22 teams. Honest train/test244/273.
Priors, score grid, thresholds, density/mean correction/Newton rules and all
`fullbook_seeds.toml` entries are unchanged. Every production fit has exactly
4 chains,2000 warmup+3000 retained,thin1; no selected chains or draws.

Manager-approved Gaussian scalar callback changes arithmetic/allocation only;
it refuses C1. The default native-sampler callback remains the dense/Laplace
engine.107 state/parity gates,20 convergence gates, and the report methods are
recorded in `PHASE_C_PROGRESS_REPORT.md` / `C8_METHODS.md`.

## 2. Clean fast runs and byte comparison

Dedicated `/root/BF_runs/market_model_c_fast_final`, detached fast SHA above.
Two fresh persistent REPLs run the same top-to-bottom `r05_fast_rungs.jl`:
- First `%270`, output `v8_fast_1/`, start05:26:50Z,
  `R05_FAST_RUNGS_DONE` at05:46:27Z.
- Second `%271`, output `v8_fast_2/`, start07:29:02Z,
  `R05_FAST_RUNGS_DONE` at07:45:36Z. **29/29 CSVs/PNGs byte-identical**,
  verified remotely and on both laptop copies; see `BYTE_COMPARISON.txt`.

Logs on beast: `/root/BF_runs/logs/market_model_c/c8/70336b07_fast1.log`
and `70336b07_fast2.log`. Real start/end/fit wall times are metadata, not
scientific tables. Fits remain as immutable `.jls` binaries (with real timing),
not claimed byte-identical. Include l05_pooled before deserializing them.

First scientific manifest lists **26 CSVs +3 PNGs**; compare exact file lists
and every byte, not rounded numbers or a selection. Commands after BOTH markers:

```bash
cd /root/BF_runs/market_model_c_fast_final/current_development/market_model/results/C
(cd v8_fast_2 && sha256sum -c ../v8_fast_1/SCIENTIFIC_SHA256SUMS.txt)
```

A further rerun must use a **new checkout/output and fresh owned REPL**, at the
exact fast SHA, never an existing result directory or a live C1/recovery pane.
Link the existing Manifest/.env without printing credentials; copy/check the
pinned cache with cp -p. Do not instantiate/update packages or refresh data.

```bash
# In a NEW dedicated beast checkout at the fast SHA:
PANE=$(tmux new-session -d -P -F '#{pane_id}' -s pi_mm_c8_rerun -c "$PWD" \
  "env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 C08_FAST_OUT=$PWD/current_development/market_model/results/C/new_fast_run /root/.juliaup/bin/julia --project -t 16")
echo "$PANE"
tmux send-keys -t "$PANE" -l -- 'include("current_development/market_model/r05_fast_rungs.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
tmux capture-pane -t "$PANE" -p -J -S -100
```

The absence annotations for Ross County/Airdrie are intentional: no such teams
are in the fixed panel. `path_club_inventory.csv` records that fact. Do not
change data to fill those panels. All22 teams' conditional q/s CSV paths remain.

## 3. Fresh laptop verification

Owned persistent laptop `%66`, Julia1.12.1,8 threads/BLAS1:
`t05_fast_reports_tests.jl`22/22, `t05_fast_gaussian_tests.jl`17/17;
`t05_pooled_tests.jl`305/305, workflow85/85, t04 92/92,
t03/t02 each131/131. C2-pending separately29pass/10fail, excluded by scope,
not T05_DONE. A regressions use temporary outputs. Anchored markers/captures
are in the LAPTOP_C8_* evidence files; IID mocks are never posterior fits.

## 4. Live recovery and C1 production — monitor ONLY

Do NOT kill/relaunch/include again, edit/check out/reload these frozen sources,
or change seeds/budgets. `%265` has only been read. C1 production was explicitly
allowed in parallel by revision8, without removing the core runner's recovery
prerequisite or promoting the new draws.

| Run | Pane | Checkout | Output directory under results/C |
|---|---|---|---|
| Synthetic C1 recovery |%265|/root/BF_runs/market_model_c|v7_recovery|
| C1 honest 10b |%267|/root/BF_runs/market_model_c_fast|v8_c1_10b|
| C1 retrospective 10a |%268|/root/BF_runs/market_model_c_fast|v8_c1_10a|

```bash
ssh root@mcmc-beast 'tmux capture-pane -t %265 -p -J -S -100'
ssh root@mcmc-beast 'tmux capture-pane -t %267 -p -J -S -100'
ssh root@mcmc-beast 'tmux capture-pane -t %268 -p -J -S -100'
```

Recovery log: `logs/market_model_c/6335316f_recovery1.log` on beast.
C1 original logs: `logs/market_model_c/c8/41e28736_c1_10a.log` / `_10b.log`;
correct subsequent UTC logs end `_utc.log`. Early awk prefixes were constant
and are NOT valid per-iteration timestamps. They are preserved, not silently
rewritten. Correct UTC logger started05:11:26Z and only reads terminal output.

Recovery expected marker: `R05_FULLBOOK_RECOVERY_DONE`. Recheck ALL retained
parameter Rhat<=1.05, bulk/tailESS>=200, every truth within its90% interval and
frozen panel SHA `de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1`.
C1 expected marker: `R05_C1_PENDING_DRAWS_DONE`, **not promotion**. Its output
contains PROMOTION_PENDING.txt. All-parameter production convergence must pass
AND synthetic recovery must be accepted before any interpretation/evaluation.
No newton counts/intervals are inferred from progress; accounting writes after
completion/error and all peer cancellations.

On failure preserve expanded Base.current_exceptions()/CompositeException,
root coordinate/log/generated panel/accounting; wait for cancellations and STOP
without another solver variation/retry/threshold/seed change.

## 5. Preserved attempts and remaining scope

Dense41e28736 attempt `%266` was stopped for measured speed, not a numerical
failure or posterior selection, before any saved fit. Its checkout/output/log
and `logs/.../c8/dense_attempt_preserved/` remain. Scalar5ab9980b `%269`
passed gates/fits, then the club-path reporting assertion failed. It remains
in `market_model_c_fast_scalar`, local `v8_scalar_plot_attempt/` and
BEAST_C8_PLOT_ATTEMPT.txt; it is NOT either clean full-run reproduction.
No reset/delete/stash or unrelated pane operation occurred. Older recovery
preservation paths remain exactly as HANDOVER_V7 records.

Clean fast proof is NOT two complete C1/full-Phase-C production runs. Recovery,
accepted C1 inference/shrinkage/n/u/posterior figures and C1 reproduction remain.
C2 thin pooling/later score-grid phase/PhaseD remain deferred.

```bash
# Laptop repository root:
(cd current_development/market_model && sha256sum -c results/C/SOURCE_SHA256SUMS.txt)
(cd current_development/market_model/results/C && sha256sum -c SCIENTIFIC_SHA256SUMS.txt)
./scripts/todo.sh check
git diff --check
```
