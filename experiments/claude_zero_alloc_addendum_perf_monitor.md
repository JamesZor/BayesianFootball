# Addendum §2.6 to `claude_zero_alloc_engine_prompt.md` — smoke sampling-performance monitor

Added 2026-09-27 at the human's request. **Part of this package**: commit this file with your work.
Same rules as §0.

**Why.** The zero-allocation fix and the hard `tape_allocation` check catch one cause. The human
wants the smoke to catch the *class* of problem: a model that samples inefficiently for any reason
(GC pressure, poor CPU utilisation, a slow in-situ gradient). The smoke should report it so the
human or an agent can review or improve the model.

## 2.6 What to build (in `src/harness/checks.jl` / `stages.jl` and a helper file, not the scoring files)

1. **Load-realistic throughput probe inside `smoke()`**, after the existing hard checks.
   - Run **16 chains × (50 warm-up + 50 draws)** (make it configurable; default = `Threads.nthreads()`
     chains) on the scope's **largest** fold, with the candidate's own NUTS settings (acceptance,
     max depth, the production `nuts_algorithm`), through the same queued/threaded execution the
     grid uses. That puts every sampler thread under the same concurrency the grid will see.
   - Today's smoke runs 2 folds × 2 chains = 4 chains on 16 threads, which cannot reproduce GC
     contention. Target wall time: 1–3 min per candidate. Report it.
2. **Metrics**, measured over the probe window:
   - **Static** (reuse `scripts/tape_allocation_audit.jl`'s code as a library function): bytes per
     compiled gradient; warmed-minimum gradient latency; tape instruction count; the list of
     allocating instructions (function, argument shapes, bytes).
   - **In situ:**
     - wall time;
     - total leapfrog / gradient evaluations (`n_steps`, warm-up included);
     - achieved ms per leapfrog and the **efficiency ratio** (in-situ ms per leapfrog ÷ bare-tape
       latency);
     - bytes allocated per leapfrog (`Base.gc_bytes` deltas);
     - **GC time share** (`Base.gc_time_ns` / `Base.gc_num` deltas ÷ wall);
     - **sampler CPU utilisation**: process CPU time from `/proc/self/stat` ÷ (wall × nthreads),
       plus the per-thread split (`/proc/self/task/*/stat`) into sampler threads vs other/GC
       threads. Don't rely on thread names; identify sampler threads by the thread ids Julia's
       worker threads report, and document how.
     - Degrade gracefully (record `unavailable`) where `/proc` is missing, e.g. on macOS.
3. **Severity:**
   - `tape_allocation` stays the **hard** check (> 1,024 bytes per gradient).
   - Add a **`sampling_performance` review check**, never hard. It flags `review` when **any** of
     these hold: GC share > 15%; sampler utilisation < 75%; efficiency ratio > 2.0.
   - Keep the thresholds as named constants in one place, with a comment linking this addendum.
   - Persist every metric as `harness_checks` rows (a diagnostic row carrying the JSON detail, plus
     the review row).
4. **Report.** Write a markdown performance report per smoke to
   `<log dir or a configurable path>/smoke_perf/<experiment>/<candidate>.md` (outside the git
   checkout; follow how the harness decides paths, and make it an argument). It contains:
   - the metrics table;
   - the allocating-instruction table;
   - a plain-language **diagnosis** mapping each flag to its likely cause and the AD-guide section
     with the fix: tracked scalar in a fused broadcast → lift it or sum first; `clamp` with scalar
     bounds → array bounds; high GC with zero tape allocation → look at the non-tape path
     (DynamicPPL `logdensity`, the sampler); low utilisation with low GC → too few chains, or
     serial work.
5. **Surfacing:**
   - Extend `scripts/run_candidates.jl`'s `[SUMMARY]` line with
     `gc=<%> util=<%> ms_leapfrog=<x> eff=<ratio> perf=<pass|review>`, so the queue summary shows
     it.
   - Surface a `perf` flag in `scripts/leaderboard.jl`'s REGISTER/LEADERBOARD output only if it
     doesn't touch the scoring logic. The Phase B agent owns `scripts/leaderboard.jl`; if you'd
     have to edit it, list the change in your report instead.
6. **Tests** (in the harness runner tests or the new tape test):
   - the metric collector runs on a synthetic league;
   - the review fires when a deliberately allocating model is used (e.g. the pre-fix
     scalar-in-broadcast form, kept as a test-only fixture) and passes for the fixed engine;
   - the report file is written with the expected sections;
   - the `/proc` fallback works.
7. **Evidence:**
   - Run the new smoke on the beast for W2 `td_base` and `grw_step_a2_carry_jump_ldelta` on the
     **fixed** engine.
   - Run the probe alone on the **pre-fix** engine (`8effca72`) for `td_base`, to show the monitor
     flags the original problem (it should show high GC share and low utilisation).
   - Include both reports in `docs/architecture/zero_alloc_engine_report.md`.

## Acceptance additions (the manager reruns them)

9. The smoke's performance probe, metrics, review check and report exist and are tested.
10. **Pre-fix `td_base` → `review` flagged** (GC share and/or utilisation), and fixed
    `td_base` → pass, with both reports attached.
11. The `[SUMMARY]` line carries the performance fields.
