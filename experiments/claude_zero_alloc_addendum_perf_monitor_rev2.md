# Revision 2 to addendum §2.6 (smoke sampling-performance monitor) — measurement tools

Agreed with the human, 2026-09-27. **Supersedes §2.6 item 2's measurement method.** Everything else
in the addendum stands. Commit this file with your work.

The manager trialled Julia's built-in tools on the pre-fix `td_base` tape (Julia 1.12.4, beast,
4 threads). The script is `/root/BF_runs/logs/w2_profile_tools.jl` and the output is
`/root/BF_runs/logs/1a8f4a03/profile_tools.log`. Use these tools in the monitor:

1. **GC counters: `Base.gc_num()` diffs via `Base.GC_Diff`** around the probe window. They give the
   GC time share (`total_time`), `allocd`, `pause`, `full_sweep`, `max_pause` and
   **`total_time_to_safepoint` / `max_time_to_safepoint`**. Trial result: 13% GC, 13.4 GiB,
   136 pauses in 5.0 s. This **replaces** the `/proc`-based GC estimate.
2. **The statistical profiler during the 16-chain probe:**
   - `Profile.init(n = 10^7, delay = 0.002)`, then `@profile` around the probe's `sample`.
   - `Profile.print(format = :flat, groupby = :thread, C = true)` reports **utilisation per
     thread** and labels GC threads `foreign: gc`. Parse it, or read the same numbers from
     `Profile.fetch(include_meta = true)` if cleaner.
   - Report: mean and min utilisation of the sampler threads; the share of sampler-thread samples
     in GC-stall frames (`jl_safepoint_wait_gc`, `jl_gc_collect`, `ijl_gc_collect`,
     `jl_gc_small_alloc*`, `ijl_gc_managed_malloc`); the share in JIT/LLVM compile frames; and the
     **top 15 frames** by self count, to go in the report.
   - Trial result: sampler threads were 68–72% utilised, and the hottest frames were
     `jl_safepoint_wait_gc` / `jl_gc_collect`.
   - The review threshold "sampler utilisation < 75%" now uses this per-thread utilisation.
   - `/proc/self/task` becomes a fallback only (for when the profiler is unavailable) and is
     recorded as such.
3. **Per-instruction tape audit** (your `scripts/tape_allocation_audit.jl`) remains **the**
   attribution tool for tape allocations. `Profile.Allocs` **cannot** attribute them: in the trial
   every tape allocation (422,627 bytes per gradient, matching the manual audit) stopped at
   ReverseDiff's `macro expansion`, with no project frame, because the compiled tape runs closures
   detached from model source.
4. **`Profile.Allocs`** (`sample_rate` ≈ 0.01–0.1, to keep overhead low) on a few full **sampler
   steps** (`LogDensityProblems.logdensity_and_gradient` via Turing's `LogDensityFunction` plus a
   short NUTS run). This catches allocation **outside** the tape (DynamicPPL wrapper, NUTS tree),
   where stack traces reach real source. Report the top 10 allocation sites by bytes, with their
   first project/package frame.
5. Report the **efficiency ratio** as before (in-situ ms per leapfrog ÷ bare-tape latency).
6. **Overhead:** measure it. The probe's wall time with and without the profiler must stay within
   ~10%. If it doesn't, profile only a sub-window of the probe.
7. **Deferred** (don't build; mention in your report as follow-ups): an HTML flame graph
   (ProfileCanvas/PProf) as a review artifact; `JET.@report_opt` for dynamic dispatch in model
   code. Both would need new dependencies, in a separate tools environment.

Acceptance item 10 (pre-fix `td_base` flagged, fixed `td_base` passing) must be shown with these
tools, with both reports attached.
