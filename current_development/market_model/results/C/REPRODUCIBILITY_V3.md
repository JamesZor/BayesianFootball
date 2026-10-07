# Phase C revision 3 — blocker evidence, not complete gate or production reproduction

Implementation commit: `8591099b`, permitted branch `pi/market-model-phase-c`.
The manager's addendum is copied unchanged in experiments/. This continuation
stops on thin-book scientific failures and a repeated integrated-derivative
unit failure. **No fresh complete Gate 1, T05_DONE, R05_DONE, pooled sampling,
convergence, synthetic recovery or production byte-reproducibility is claimed.**

## Preserved evidence

- Revision-1 preflight source and three CSVs remain unchanged.
- `PHASE_C_V2_REPORT.md` / `REPRODUCIBILITY_V2.md` preserve revision-2 notes.
- `v2_gate/` contains its original four gate/inventory CSVs and historical hash
  manifests. Those manifests retain their original relative paths; use the
  current root scientific manifest to check archived CSVs, and git source
  `7b6c7659` for original source reproduction.
- Current `laplace_gate_fixture.csv` has 792 completed fixture-setting rows
  from four 2-D types. The attempted 1X2 batch aborted before completion.
- `engine_gates_c.csv` was regenerated from the emitted summary after that
  abort, to avoid leaving the stale revision-2 engine table in place.
  `GATE_EXPORT.sql` records the exact DuckDB export (run from repository root).
- `integrated_derivative_check_c3.csv` copies both repeated discrepancies from
  `LAPTOP_REVISION3_DEVELOPMENT.txt`. The second failure triggers the stop rule.

## Input and methods

Existing Manifest/packages; Julia 1.12.1, 8 threads, BLAS=1. No package addition
or update: the adaptive level integral calls `Distributions.quadgk`, already
imported by the installed Distributions package. No database writes or `src/`
changes. Pinned cache SHA256:

```
c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4
```

Load `MarketModelConfig.scottish_lower_2425_2526()` with `max_age_hours=10^6`.
The unchanged production renormalised cutoff-10 double-Poisson grid supplies
2-D exact quadrature and the revision-2 marginal. Generic recurrence of the
same probabilities supplies AD third tensors, with first-20-full-book parity
checks. Gaussian prediction centres and all n/spread/offset settings remain
those specified in revision 2. No selection on errors.

Revision 3 uses Sigma=(-joint Hessian)^-1 to contract the likelihood third
tensor, returns mode+shift for the mean, and retains revision-2 J/covariance
clipping and the log-marginal. Full-book saved log marginals join identically
with all 360 revision-2 rows. AD third tensors agree with central differences
of the AD Hessian on all completed 2-D rows (see saved relative discrepancies).

The current 1X2 integral uses [-8,4] in log-level, with tails diagnosed; the
upper endpoint was widened from 2.5 after its test was not negligible enough,
not after changing a scientific gate. Adaptive level tolerances are rtol=2e-12,
atol=1e-14. Order labels 32/64 select Gauss-Kronrod Gauss rules 7/15 inside the
level integral; the outer expectation uses normal Gauss-Hermite orders 32/64.
Nested dual components are included in the adaptive error norm. This current
integrated implementation **fails its third-derivative consistency test** and
must not be promoted.

## Owned laptop development run

Only pane **%43** (`pi_julia_mm_c`) was operated; it is closed. No beast session
or checkout was created. The log retains development failures and the passing
preliminary version instead of presenting it as current acceptance evidence.

Sequence:

1. Initial mean-correction unit tests: 129 pass/1 fail; endpoint-density check
   detected the old level upper endpoint 2.5 was insufficient.
2. Widen only that integration range to 4.0: preliminary Gaussian-level version
   passes 130/130 necessary checks, `T05_C3_UPDATE_DONE`.
3. All-types gate attempt completes full, O/U-only, BTTS+O/U and BTTS-only;
   the first three types' saved measurements include thin-book failures.
   Level-integrated 1X2 Newton line search stalls. This attempt does not reach
   either final gate marker or R05_DONE.
4. Replace the moving Gaussian level mesh once with adaptive Gauss-Kronrod.
   Integrated AD-vs-FD third-tensor test fails at 5.405790669565064e-5.
5. Include all nested derivative components in quadrature error control.
   The same test fails at 5.405838441375254e-5, unchanged test limit 1e-6.
   **Stop; no third variation, further gate attempt or sampling.**

The saved 2-D gate calculations were completed before the adaptive level
changes; those changes affect only 1X2 and do not alter the completed 2-D
arithmetic. Their thin-book failures are independent of the integrated solver
failure. Low-n quadrature stability is documented; high-n thin-book resolution
is explicitly incomplete. No claim of a fresh full current-source runner is
made. The final source's necessary suite has 129 passes / 1 failure, not 130
passing acceptance checks. t02–t04 were not rerun for this continuation.

## Manager reproduction commands

Create only an unused owned persistent tmux session; target its returned pane ID:

```bash
PANE=$(tmux new-session -d -P -F '#{pane_id}' -s pi_julia_mm_c -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 julia --project -t 8')
printf 'owned pane: %s\n' "$PANE"
tmux send-keys -t "$PANE" -l -- \
  'include("current_development/market_model/t05_pooled_tests.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
# Capture after the test result; current source has an expected failing test.
tmux capture-pane -t "$PANE" -p -J -S -2000
```

Do not interpret an earlier `T05_C3_UPDATE_DONE` in the development log as a
passing current-source test. Do not start sampling after that failure. A
separate manual gate diagnostic can include r05_pooled.jl, but its current
all-types completion has not been established by this handoff.

## Hash and scope checks

From the stream directory:

```bash
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
(cd results/C && sha256sum -c SCIENTIFIC_SHA256SUMS.txt)
```

These hashes identify evidence/code, not two successful production executions.
The addendum copy compares byte-identically with the manager's source file.
`todo.sh check` and `git diff --check` pass at handoff. TODO 039 stays BLOCKED;
all owned sessions are closed. See PHASE_C_REPORT.md for numbers, diagnosis and
all missing acceptance items.
