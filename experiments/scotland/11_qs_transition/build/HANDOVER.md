# Stage 2 handover — PHASES2_BLOCKED (D12 network deadline)

## State / stop reason

D12 required waiting until 2026-10-12 03:30 BST. Connectivity watch used a 600-second
cadence and no intervening model calls. Deadline wake/check: 03:38:59 BST;
`ssh -o ConnectTimeout=5 -o BatchMode=yes root@mcmc-beast 'true'` still timed out.
The manager diagnosed a live beast but broken Tailscale path. No attempt was made to guess or
bypass it with a public IP. **Stop now; no source gate or Stage 3 grid is authorised.**

- Branch/worktree: `exp/qs-transition`, `/home/james/bet_project/.worktrees/BayesianFootball-qs-transition`.
- Owned beast checkout: `/root/BF_runs/qs_transition` only.
- Last confirmed owned Julia session: `s2`, beast16, in that checkout. Job `001011-15246`
  completed `ok`; no Julia job was running when connectivity was lost. Cannot confirm shutdown
  while SSH is unavailable. Do not touch anyone else's REPL/worktree.
- `e2e956cf`: source/recipe WIP, **UNGATED**. `bf0fd56f`: D12 wait record and post-src legacy runner.
- Final blocked report: `../REPORT_S2.md`; final handover commit will follow these records.
- Local commits only; no push or merge.

## Verified outputs — do not redo

1. Unchanged wave-2 fold1 reproduced saved run `81fdf817-f82d-4f66-9bfd-5f456c4fe29f`:
   396,000 chain values and 160,000 latents bit-identical; `legacy_reproduction.csv`.
2. Expanded frozen `market_rates.csv`: 1,790 rows, 1,361 full books; all 1,430 legacy rows
   equal; SHA256 `1098f12aff1db116d3fc46454ccc00297402b61f5011502c0ffd86f7298c9add`.
3. W2 panel: 60 folds, 1,070 fixtures; `panel.csv`.
4. Control normal two-fold smoke: all hard checks PASS;
   `b58afafc-80ee-4ee6-920b-d5df860b4483`, hash
   `f75623499919dd9e2e045595bdbe3735c1b2e7748b0655602844ba68a0f46ba7`.
   R̂1.0625, ESS bulk67.1/tail75.5, divergences0; convergence REVIEW fail.
5. A2 normal two-fold smoke: all hard checks PASS;
   `9374341b-66aa-4460-83fe-af45f8d5a01b`, hash
   `ecb6eab8243b0b7e27020e24498844667cb386555618d8222c91d8c1173cbf32`.
   R̂1.0533, ESS bulk50.9/tail72.0, divergences2 (0.25%); convergence REVIEW fail.
6. D11 recovered within its time box: old/current sources, identical Manifest, both0 B at
   matched folds1/21/41 in fresh REPLs; old loader remains0; normal smokes0 B on both folds.
   No gate/package/source changes were used for this recovery. Root process-state cause unknown.
   `ALLOCATION_INVESTIGATION.md` and allocation CSVs contain the evidence.

The pre-src control/A2 successes do NOT validate later source modifications.

## Committed but unverified

- `src/models/pregame/components/dynamics/team_level/quality_style_transition.jl`:
  separate opt-in summer/entry wrappers and in-model entry q/s walks.
- `src/models/pregame/builder/quality_style_transition.jl`: matrix designs, reconstruction,
  deterministic OOS uncertainty; incumbent-only centring for entry arm.
- Legacy constructor defaults/type fields intended unchanged; includes and reconstruction
  keyword extended in the existing QS files. **Post-src bit identity still required.**
- `../l01_entrant_prior.jl`: S1 transport inputs, input SHA, strictly upper endpoint IDs,
  earlier-season hyper IDs, fallback/gauge/moment manifests. **No manifests executed yet.**
- `../candidates.jl`: `qs_control`, `qs_a2`, `qs_entrant`, `qs_summer`.
- `test/test_quality_style_transition.jl` and `r07_component_gates.jl`: prepared but **never run**.
- `r00_legacy.jl` now uses `S2_LEGACY_TAG` to avoid overwriting/reusing pre-src checkpoints.
- F6 absent intentionally: cadence is coupled to dynamics clock; see `F6_DECISION.md`.

## Exact next step after connectivity recovers

First stop/restart **only** the owned `s2` and reconcile/sync the checkout before running anything.
The earlier transfer/bootstrap timed out before any success marker; do not assume remote `.git`
was repaired or new source transferred. Its original `.git` was an invalid local-worktree pointer.
`/tmp/qs-transition-s2.bundle` is a valid 301 MB complete-history bundle at `e2e956cf`, **not** the
latest records commit. Regenerate a bundle from the current branch if restoring git metadata.

Example isolated resume sequence (manager/next S2 worker, NOT Stage 3):

```bash
jl stop s2
# Local: bundle current branch; this can take >1 min, use job/watch for the transfer.
git bundle create /tmp/qs-transition-s2-resume.bundle exp/qs-transition
rsync -a /tmp/qs-transition-s2-resume.bundle root@mcmc-beast:/root/BF_runs/qs_transition/allocation_sources/
rsync -a --exclude '/.git' --exclude '/.cache/' --exclude '/data/' --exclude '/Manifest.toml' --exclude '/.env' ./ root@mcmc-beast:/root/BF_runs/qs_transition/
# Inspect own checkout's .git; if still a bad pointer, rename only that file and git init there.
# Fetch the bundle into FETCH_HEAD, update own exp/qs-transition ref, reset --mixed;
# never merge, push, or touch another checkout. Verify SHA before sampling.
jl start s2 --beast --threads 16 --dir /root/BF_runs/qs_transition
jl run s2 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/build/r07_component_gates.jl --bg
```

Watch the printed job ID. No gate was attempted for these components, so failure count is zero.
This runner covers new unit tests + existing QS tests, exact AD≤1e-10 at five points, directional
FD convergence, and **exactly 0 B** linked tapes for opening and observed-target synthetic cases.
Do not weaken any assertion. If a gate fails twice, stop/ask manager; leakage/reproduction
failures stop the stage immediately.

Then, in order:
1. Write/run the required three-seed synthetic recovery gate (NOT YET IMPLEMENTED): ≥80%
   pooled interval coverage, no parameter misses all three seeds. Record CSVs and commands.
2. Post-src legacy identity in a fresh REPL:
   `jl eval s2 'ENV["S2_LEGACY_TAG"]="post_src"'`, then
   `jl run s2 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/build/r00_legacy.jl --bg`.
   It must not reuse pre-src checkpoints. Restore a fresh source-build REPL afterwards.
3. Validate actual entrant features on source folds1/60 (and persistence through fold2),
   including the first-biweekly-period assertion, endpoint overlap0 and pre-node chronology.
   Hyper-source overlap is reported separately under D9, never claimed zero.
4. Run real-fold exact AD/FD/0 B probes for the opt-in arms before normal smokes.
5. Normal source-smoke commands, one selected arm per fresh REPL as needed:
   `jl eval s2 'ENV["S2_ONLY"]="qs_entrant"'` then `jl run s2 /root/BF_runs/qs_transition/experiments/scotland/11_qs_transition/build/r02_smoke.jl --bg`;
   repeat with `S2_ONLY="qs_summer"`. These use two folds only, four chains200+200,
   normal harness hard checks/storage, and a largest-fold full-concurrency timing probe.
6. Manager decides F6; do not relabel weekly dynamics as weekly-refit-only.
7. Update REPORT_S2 gate/status/timing tables, clear all DONE WHEN checks, stop owned REPL,
   commit locally and print PHASES2_DONE. Until then, no 60-fold grids.

## Remaining gates / limitations

New units, AD, FD, allocation, recovery, real-fold provenance, entrant/summer smokes and post-src
legacy identity are all outstanding. The source drafts have no correctness certificate.
Existing timings project control5.62 h + A2 5.71 h (11.34 h combined); three remaining arms are
unmeasured, so no defensible five-arm estimate exists. Grid templates in REPORT_S2 are explicitly
blocked and omit F6. No DB writes outside normal harness smokes occurred.
