# Zero-allocation compiled gradients in the builder engine — report

Work package: `experiments/claude_zero_alloc_engine_prompt.md`, with the smoke performance-monitor
addendum `experiments/claude_zero_alloc_addendum_perf_monitor.md` (§2.6) and its measurement
revision `…_rev2.md`. TODO 034. Branch `fix/zero-alloc-engine`, cut from `8effca72`.

**Result.** Every one of the 27 W1/W2 recipes, on its largest and its season-opening fold, replays
its compiled ReverseDiff tape at **0 bytes per gradient** (was 35 KB–923 KB). Log-density and
gradient equal the pre-fix code to ≤ 3.7e-15 relative at a prior draw and at ±0.8 per coordinate.
All 27 `recipe_hash` values are unchanged. A matrix test, a hard smoke check and a review-level
sampling-performance probe now guard it.

| | `td_base` pre-fix | fixed | `grw_step_a2_carry_jump_ldelta` pre-fix | fixed |
|---|---|---|---|---|
| bytes per compiled gradient | 422,992 | **0** | 923,456 | **0** |
| 16 threads × 16 chains (100 + 100) wall | 34.3 s | **11.6 s** | 1,181 s | **656 s** |
| throughput at 16 threads | | **2.97×** | | **1.80×** |
| GC share at 16 threads | 40.9 % | 24.1 % | 15.3 % | 7.0 % |

**One acceptance item needs a human decision** (see *Blocked*): with a 0-byte tape, the fixed
`td_base` still spends ~24–32 % of wall time in GC at 16 threads. The cause is ~28 KB per leapfrog
allocated by AdvancedHMC and the DynamicPPL/DI wrappers, not by this repository. So the new
`sampling_performance` monitor flags it `review` instead of the `pass` acceptance 10 expected.

## Commits

| SHA | what |
|---|---|
| `dd0472c6` | `scripts/tape_allocation_audit.jl` + `src/harness/tape_profile.jl` (per-instruction profiler) |
| `edc48474` | the engine fix (`src/models/pregame/tape_scalars.jl` and every offending site) |
| `cfd71867` | hard `tape_allocation` smoke check, `SMOKE_REQUIRED_CHECKS`, `test/tape_allocation_tests.jl`, pyramid regrouping |
| `08f2f309` | smoke sampling-performance probe, review check, report, `[SUMMARY]` fields |
| `4fef9697` | AD guide §10.5 / Rule 7, `AGENTS.md` digest, loader notes, rev2 addendum committed |
| `9472300c` | rev2 measurement tools (GC_Diff, profiler per thread, Profile.Allocs), `n_steps` fix |
| `6f146ae2` | TODO 034 — **the SHA the audit, benchmarks, smokes and suites below ran on** |
| `49eba057` | profiler sub-window cap (5 s) and `sampler_work_share` — harness-monitor only; final probes, overhead and harness tests re-run here |

All evidence runs used clean beast checkouts (`scripts/beast_checkout.sh <sha>`), logs under
`/root/BF_runs/logs/<sha>/`. Pre-fix measurements run the NEW scripts against the OLD package:
`cd /root/BF_runs/8effca72 && julia --project=/root/BF_runs/8effca72 /root/BF_runs/<new>/scripts/…`
— possible because the audit/probe helpers (`src/harness/tape_profile.jl`, `perf_monitor.jl`) are
included by path and depend on nothing the fix added. Julia 1.12.4 on the beast (juliaup default),
ReverseDiff 1.17.0, Turing 0.41.4.

## 1. The defect and the audit (acceptance 1)

ReverseDiff 1.17 (`derivatives/broadcast.jl`, `get_implementation`) sends a fused broadcast to
`tracker_∇broadcast` whenever **any** argument is a `Real` — tracked or not — and that adjoint
allocates O(rows × args) on every reverse pass. With only arrays (or `Ref`s) it takes `∇broadcast`,
whose buffers are allocated once at recording. Unfused binary `+ - * /` of a tracked scalar and an
array has its own preallocated kernel. Two ReverseDiff rules also allocate: `fill(tracked, n)` and
`mean(tracked_matrix; dims)` (no rule: it tapes element by element and returns an
`Array{TrackedReal}`).

Command (before; after is the same with `--compare-parity`):

```
cd /root/BF_runs/8effca72 && julia --project=/root/BF_runs/8effca72 -t 1 \
  /root/BF_runs/dd0472c6/scripts/tape_allocation_audit.jl \
  experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl \
  experiments/scotland/03_dynamics_scope_matrix/candidates.jl \
  --save-parity /root/BF_runs/logs/dd0472c6/parity_before_8effca72.jls \
  --out /root/BF_runs/logs/dd0472c6/audit_before_8effca72.tsv
cd /root/BF_runs/6f146ae2 && julia --project=. -t 1 scripts/tape_allocation_audit.jl <same files> \
  --compare-parity /root/BF_runs/logs/dd0472c6/parity_before_8effca72.jls \
  --out /root/BF_runs/logs/6f146ae2/audit_after.tsv
```

Folds: `largest` = the scope's last fold; `opening` = the first fold of the scope's last target
season (a GRW there has no observed target steps). Bytes = minimum over 20 warmed replays behind a
function barrier, at a seeded prior draw in the **linked** space NUTS uses. Latency = warmed
minimum of 100, single thread (so it does not show the GC cost at 16 threads — §5 does).

### Distinct allocating instructions before the fix (per-instruction replay)

| instruction (T = tracked array, TReal = tracked scalar, Real = plain scalar) | source | bytes per replay, 2,390 rows |
|---|---|---|
| `tracker_∇broadcast[clamp](T, Real, Real)` ×2 | `apply_guard(::ClampGuard)` | 57,624 each |
| `tracker_∇broadcast[fused](TReal, Arr, TReal, Arr, T, TReal, T, TReal)` ×2 | joint Gamma pxG arm | 153,664 each |
| `tracker_∇broadcast[fused](TReal, Arr, Real/TReal, Arr, TReal, Arr)` ×2 | `PyramidTiers` `B_home/B_away` | 115,248 each |
| `tracker_∇broadcast[fused](Real, T, Real, T)` ×2 | `PyramidTiers` attack/defence shares | 76,832 each |
| `tracker_∇broadcast[fused](T, TReal, T)` ×2 | `CompetitionKappa` `ξ .+ log_κ .+ δ` | per rows |
| `tracker_∇broadcast[fused](Real, T, TReal)` ×2 | `CompetitionKappa` `scale .* (raw .- mean(raw))` | small |
| `tracker_∇broadcast[fused](TReal, Arr, TReal, Arr)` ×2 | lineup pillar `w_att .* h .- w_def .* a` (m12) | 34,336 each (1,060 rows) |
| `fill(TReal, Real)` ×2 | `GlobalInterception`, `GlobalHomeAdvantage` | 48 + 368 |
| `broadcast,-(T43×21, Arr1×21)` ×2 | GRW `raw .- mean(raw, dims = 1)` (Array{TrackedReal}) | 240 each |

The GRW `mean(dims = 1)` also put ~1,800 scalar instructions per side on the tape (17,619 → 13,972
instructions on the W2 largest fold after the fix).

Rows: `before` = `/root/BF_runs/logs/dd0472c6/audit_before_8effca72.tsv`; `after` = `/root/BF_runs/logs/6f146ae2/audit_after.tsv` (final engine code; later commits touch only the harness monitor). Fold number in brackets.

### Audit: bytes per compiled gradient and latency (before `8effca72` → after)

| wave | recipe | fold | rows | params | instr. before → after | bytes before | bytes after | ms before → after |
|---|---|---|---|---|---|---|---|---|
| W2 | `td_base` | largest (60) | 2390 | 92 | 137 → 144 | 422,992 | 0 | 0.175 → 0.132 |
| W2 | `td_base` | opening (41) | 1605 | 90 | 137 → 144 | 285,008 | 0 | 0.119 → 0.090 |
| W2 | `td_a2` | largest (60) | 2390 | 94 | 176 → 188 | 807,152 | 0 | 0.222 → 0.165 |
| W2 | `td_a2` | opening (41) | 1605 | 92 | 176 → 188 | 543,728 | 0 | 0.153 → 0.113 |
| W2 | `td_a2_carry` | largest (60) | 2390 | 95 | 197 → 209 | 807,152 | 0 | 0.225 → 0.167 |
| W2 | `td_a2_carry` | opening (41) | 1605 | 93 | 197 → 209 | 543,728 | 0 | 0.156 → 0.114 |
| W2 | `td_a2_carry_jump` | largest (60) | 2390 | 136 | 237 → 249 | 807,152 | 0 | 0.277 → 0.215 |
| W2 | `td_a2_carry_jump` | opening (41) | 1605 | 134 | 237 → 249 | 543,728 | 0 | 0.193 → 0.149 |
| W2 | `td_ldelta` | largest (60) | 2390 | 101 | 165 → 176 | 538,816 | 0 | 0.219 → 0.177 |
| W2 | `td_ldelta` | opening (41) | 1605 | 99 | 165 → 176 | 363,200 | 0 | 0.150 → 0.121 |
| W2 | `td_a2_carry_jump_ldelta` | largest (60) | 2390 | 145 | 265 → 281 | 922,976 | 0 | 0.318 → 0.263 |
| W2 | `td_a2_carry_jump_ldelta` | opening (41) | 1605 | 143 | 265 → 281 | 621,920 | 0 | 0.226 → 0.180 |
| W2 | `grw_base` | largest (60) | 2390 | 1816 | 17619 → 13972 | 423,472 | 0 | 0.737 → 0.587 |
| W2 | `grw_base` | opening (41) | 1605 | 176 | 1185 → 852 | 285,200 | 0 | 0.158 → 0.117 |
| W2 | `grw_step` | largest (60) | 2390 | 1902 | 18307 → 14660 | 423,472 | 0 | 0.752 → 0.615 |
| W2 | `grw_step` | opening (41) | 1605 | 176 | 1185 → 852 | 285,200 | 0 | 0.153 → 0.117 |
| W2 | `grw_step_a2_carry_jump` | largest (60) | 2390 | 1946 | 18407 → 14765 | 807,632 | 0 | 0.853 → 0.786 |
| W2 | `grw_step_a2_carry_jump` | opening (41) | 1605 | 220 | 1285 → 957 | 543,920 | 0 | 0.235 → 0.180 |
| W2 | `grw_step_ldelta` | largest (60) | 2390 | 1911 | 18335 → 14692 | 539,296 | 0 | 0.778 → 0.664 |
| W2 | `grw_step_ldelta` | opening (41) | 1605 | 185 | 1213 → 884 | 363,392 | 0 | 0.186 → 0.147 |
| W2 | `grw_step_a2_carry_jump_ldelta` | largest (60) | 2390 | 1955 | 18435 → 14797 | 923,456 | 0 | 0.893 → 0.750 |
| W2 | `grw_step_a2_carry_jump_ldelta` | opening (41) | 1605 | 229 | 1313 → 989 | 622,112 | 0 | 0.261 → 0.207 |
| W2 | `td_lower_base` | largest (60) | 1060 | 52 | 137 → 144 | 189,104 | 0 | 0.082 → 0.061 |
| W2 | `td_lower_base` | opening (41) | 720 | 50 | 137 → 144 | 128,560 | 0 | 0.060 → 0.043 |
| W2 | `td_lower_a2full_carry_jump` | largest (60) | 1060 | 96 | 237 → 249 | 360,784 | 0 | 0.135 → 0.101 |
| W2 | `td_lower_a2full_carry_jump` | opening (41) | 720 | 94 | 237 → 249 | 245,200 | 0 | 0.095 → 0.073 |
| W2 | `td_lower_b1lit` | largest (60) | 1060 | 74 | 168 → 175 | 189,104 | 0 | 0.105 → 0.083 |
| W2 | `td_lower_b1lit` | opening (41) | 720 | 72 | 168 → 175 | 128,560 | 0 | 0.075 → 0.058 |
| W2 | `m12_td_ref` | largest (60) | 1060 | 55 | 183 → 192 | 257,776 | 0 | 0.101 → 0.086 |
| W2 | `m12_td_ref` | opening (41) | 720 | 53 | 183 → 192 | 175,216 | 0 | 0.072 → 0.060 |
| W1 | `td_lower_poisson` | largest (40) | 1060 | 50 | 92 → 92 | 51,760 | 0 | 0.044 → 0.042 |
| W1 | `td_lower_poisson` | opening (21) | 720 | 48 | 92 → 92 | 35,248 | 0 | 0.032 → 0.029 |
| W1 | `td_lower_joint` | largest (40) | 1060 | 52 | 137 → 144 | 189,104 | 0 | 0.084 → 0.062 |
| W1 | `td_lower_joint` | opening (21) | 720 | 50 | 137 → 144 | 128,560 | 0 | 0.060 → 0.043 |
| W1 | `td_spfl_poisson` | largest (40) | 2183 | 90 | 92 → 92 | 105,680 | 0 | 0.087 → 0.083 |
| W1 | `td_spfl_poisson` | opening (21) | 1476 | 88 | 92 → 92 | 71,888 | 0 | 0.060 → 0.058 |
| W1 | `td_spfl_joint` | largest (40) | 2183 | 92 | 137 → 144 | 386,384 | 0 | 0.160 → 0.120 |
| W1 | `td_spfl_joint` | opening (21) | 1476 | 90 | 137 → 144 | 262,480 | 0 | 0.112 → 0.083 |
| W1 | `td_spfl_cups_poisson` | largest (40) | 2390 | 90 | 92 → 92 | 115,664 | 0 | 0.096 → 0.090 |
| W1 | `td_spfl_cups_poisson` | opening (21) | 1605 | 88 | 92 → 92 | 78,032 | 0 | 0.067 → 0.062 |
| W1 | `td_spfl_cups_joint` | largest (40) | 2390 | 92 | 137 → 144 | 422,992 | 0 | 0.172 → 0.130 |
| W1 | `td_spfl_cups_joint` | opening (21) | 1605 | 90 | 137 → 144 | 285,008 | 0 | 0.122 → 0.090 |
| W1 | `grw_lower_poisson` | largest (40) | 1060 | 974 | 9494 → 7520 | 52,240 | 0 | 0.334 → 0.294 |
| W1 | `grw_lower_poisson` | opening (21) | 720 | 94 | 660 → 480 | 35,440 | 0 | 0.051 → 0.045 |
| W1 | `grw_lower_joint` | largest (40) | 1060 | 976 | 9539 → 7572 | 189,584 | 0 | 0.381 → 0.308 |
| W1 | `grw_lower_joint` | opening (21) | 720 | 96 | 705 → 532 | 128,752 | 0 | 0.079 → 0.059 |
| W1 | `grw_spfl_poisson` | largest (40) | 2183 | 1814 | 17574 → 13920 | 106,160 | 0 | 0.628 → 0.535 |
| W1 | `grw_spfl_poisson` | opening (21) | 1476 | 174 | 1140 → 800 | 72,080 | 0 | 0.095 → 0.083 |
| W1 | `grw_spfl_joint` | largest (40) | 2183 | 1816 | 17619 → 13972 | 386,864 | 0 | 0.718 → 0.571 |
| W1 | `grw_spfl_joint` | opening (21) | 1476 | 176 | 1185 → 852 | 262,672 | 0 | 0.147 → 0.109 |
| W1 | `grw_spfl_cups_poisson` | largest (40) | 2390 | 1814 | 17574 → 13920 | 116,144 | 0 | 0.646 → 0.532 |
| W1 | `grw_spfl_cups_poisson` | opening (21) | 1605 | 174 | 1140 → 800 | 78,224 | 0 | 0.099 → 0.094 |
| W1 | `grw_spfl_cups_joint` | largest (40) | 2390 | 1816 | 17619 → 13972 | 423,472 | 0 | 0.771 → 0.576 |
| W1 | `grw_spfl_cups_joint` | opening (21) | 1605 | 176 | 1185 → 852 | 285,200 | 0 | 0.154 → 0.115 |

### Parity (after vs before, same θ) and recipe hashes

| recipe | fold | worst rel. log-density | worst rel. compiled gradient | worst rel. fresh gradient | recipe_hash before | after | same |
|---|---|---|---|---|---|---|---|
| `td_base` | largest | 4.17e-16 | 1.04e-15 | 1.04e-15 | `84a3541715dc09d9` | `84a3541715dc09d9` | yes |
| `td_base` | opening | 1.45e-16 | 1.61e-16 | 1.61e-16 | `84a3541715dc09d9` | `84a3541715dc09d9` | yes |
| `td_a2` | largest | 3.33e-16 | 1.35e-15 | 1.35e-15 | `c0aaee9d75f8f374` | `c0aaee9d75f8f374` | yes |
| `td_a2` | opening | 3.41e-16 | 1.11e-15 | 1.11e-15 | `c0aaee9d75f8f374` | `c0aaee9d75f8f374` | yes |
| `td_a2_carry` | largest | 4.09e-16 | 2.49e-16 | 2.49e-16 | `83e6081188f423ad` | `83e6081188f423ad` | yes |
| `td_a2_carry` | opening | 2.48e-16 | 2.08e-16 | 2.08e-16 | `83e6081188f423ad` | `83e6081188f423ad` | yes |
| `td_a2_carry_jump` | largest | 3.48e-16 | 2.88e-16 | 2.88e-16 | `c2f590664a649f7b` | `c2f590664a649f7b` | yes |
| `td_a2_carry_jump` | opening | 2.72e-16 | 1.49e-16 | 1.49e-16 | `c2f590664a649f7b` | `c2f590664a649f7b` | yes |
| `td_ldelta` | largest | 3.71e-16 | 1.08e-15 | 1.08e-15 | `344b526558c71e67` | `344b526558c71e67` | yes |
| `td_ldelta` | opening | 1.45e-16 | 1.50e-15 | 1.50e-15 | `344b526558c71e67` | `344b526558c71e67` | yes |
| `td_a2_carry_jump_ldelta` | largest | 2.35e-16 | 1.03e-15 | 1.03e-15 | `57b2845c169b3556` | `57b2845c169b3556` | yes |
| `td_a2_carry_jump_ldelta` | opening | 2.32e-16 | 1.17e-15 | 1.17e-15 | `57b2845c169b3556` | `57b2845c169b3556` | yes |
| `grw_base` | largest | 3.56e-16 | 1.13e-15 | 1.13e-15 | `ea340e06d96c6c4d` | `ea340e06d96c6c4d` | yes |
| `grw_base` | opening | 1.51e-16 | 1.20e-15 | 1.20e-15 | `ea340e06d96c6c4d` | `ea340e06d96c6c4d` | yes |
| `grw_step` | largest | 4.65e-16 | 3.67e-15 | 3.67e-15 | `6fd0f5b5a00a8e6f` | `6fd0f5b5a00a8e6f` | yes |
| `grw_step` | opening | 1.51e-16 | 1.20e-15 | 1.20e-15 | `6fd0f5b5a00a8e6f` | `6fd0f5b5a00a8e6f` | yes |
| `grw_step_a2_carry_jump` | largest | 1.23e-16 | 9.89e-16 | 9.89e-16 | `819df2c97b7e6ef1` | `819df2c97b7e6ef1` | yes |
| `grw_step_a2_carry_jump` | opening | 3.64e-16 | 2.10e-16 | 2.10e-16 | `819df2c97b7e6ef1` | `819df2c97b7e6ef1` | yes |
| `grw_step_ldelta` | largest | 3.09e-16 | 3.14e-16 | 3.14e-16 | `144b5649ec7debe6` | `144b5649ec7debe6` | yes |
| `grw_step_ldelta` | opening | 1.94e-16 | 1.34e-15 | 1.34e-15 | `144b5649ec7debe6` | `144b5649ec7debe6` | yes |
| `grw_step_a2_carry_jump_ldelta` | largest | 3.61e-16 | 1.04e-15 | 1.04e-15 | `ea2e2e545d649ff1` | `ea2e2e545d649ff1` | yes |
| `grw_step_a2_carry_jump_ldelta` | opening | 2.56e-16 | 1.88e-15 | 1.88e-15 | `ea2e2e545d649ff1` | `ea2e2e545d649ff1` | yes |
| `td_lower_base` | largest | 2.41e-16 | 1.17e-16 | 1.17e-16 | `8324c5989c295902` | `8324c5989c295902` | yes |
| `td_lower_base` | opening | 1.34e-16 | 1.81e-16 | 1.81e-16 | `8324c5989c295902` | `8324c5989c295902` | yes |
| `td_lower_a2full_carry_jump` | largest | 2.39e-16 | 1.20e-15 | 1.20e-15 | `4f93564bad26c4df` | `4f93564bad26c4df` | yes |
| `td_lower_a2full_carry_jump` | opening | 3.78e-16 | 1.26e-15 | 1.26e-15 | `4f93564bad26c4df` | `4f93564bad26c4df` | yes |
| `td_lower_b1lit` | largest | 4.33e-16 | 1.16e-15 | 1.16e-15 | `24e6ca9dce988ddd` | `24e6ca9dce988ddd` | yes |
| `td_lower_b1lit` | opening | 2.59e-16 | 1.14e-16 | 1.14e-16 | `24e6ca9dce988ddd` | `24e6ca9dce988ddd` | yes |
| `m12_td_ref` | largest | 2.54e-16 | 1.05e-15 | 1.05e-15 | `e93f64658512c2a2` | `e93f64658512c2a2` | yes |
| `m12_td_ref` | opening | 8.51e-16 | 1.90e-15 | 1.90e-15 | `e93f64658512c2a2` | `e93f64658512c2a2` | yes |
| `td_lower_poisson` | largest | 0.00e+00 | 0.00e+00 | 0.00e+00 | `bf880214cbb367f4` | `bf880214cbb367f4` | yes |
| `td_lower_poisson` | opening | 0.00e+00 | 0.00e+00 | 0.00e+00 | `bf880214cbb367f4` | `bf880214cbb367f4` | yes |
| `td_lower_joint` | largest | 2.41e-16 | 1.17e-16 | 1.17e-16 | `722ed88a7efe8c82` | `722ed88a7efe8c82` | yes |
| `td_lower_joint` | opening | 1.34e-16 | 1.81e-16 | 1.81e-16 | `722ed88a7efe8c82` | `722ed88a7efe8c82` | yes |
| `td_spfl_poisson` | largest | 0.00e+00 | 0.00e+00 | 0.00e+00 | `86c53147e9b0f430` | `86c53147e9b0f430` | yes |
| `td_spfl_poisson` | opening | 0.00e+00 | 0.00e+00 | 0.00e+00 | `86c53147e9b0f430` | `86c53147e9b0f430` | yes |
| `td_spfl_joint` | largest | 2.33e-16 | 4.79e-16 | 4.79e-16 | `588f1b29d9eead0a` | `588f1b29d9eead0a` | yes |
| `td_spfl_joint` | opening | 0.00e+00 | 1.04e-16 | 1.04e-16 | `588f1b29d9eead0a` | `588f1b29d9eead0a` | yes |
| `td_spfl_cups_poisson` | largest | 0.00e+00 | 0.00e+00 | 0.00e+00 | `7f437551e05c9df6` | `7f437551e05c9df6` | yes |
| `td_spfl_cups_poisson` | opening | 0.00e+00 | 0.00e+00 | 0.00e+00 | `7f437551e05c9df6` | `7f437551e05c9df6` | yes |
| `td_spfl_cups_joint` | largest | 4.17e-16 | 1.04e-15 | 1.04e-15 | `20be74da5946bfa6` | `20be74da5946bfa6` | yes |
| `td_spfl_cups_joint` | opening | 1.45e-16 | 1.61e-16 | 1.61e-16 | `20be74da5946bfa6` | `20be74da5946bfa6` | yes |
| `grw_lower_poisson` | largest | 0.00e+00 | 1.27e-16 | 1.27e-16 | `7b7e98bf312dfeef` | `7b7e98bf312dfeef` | yes |
| `grw_lower_poisson` | opening | 1.91e-16 | 1.20e-17 | 1.20e-17 | `7b7e98bf312dfeef` | `7b7e98bf312dfeef` | yes |
| `grw_lower_joint` | largest | 1.31e-16 | 1.32e-16 | 1.32e-16 | `d373b3393d507912` | `d373b3393d507912` | yes |
| `grw_lower_joint` | opening | 1.12e-16 | 1.98e-15 | 1.98e-15 | `d373b3393d507912` | `d373b3393d507912` | yes |
| `grw_spfl_poisson` | largest | 1.90e-16 | 1.87e-16 | 1.87e-16 | `1d315f06d785436f` | `1d315f06d785436f` | yes |
| `grw_spfl_poisson` | opening | 0.00e+00 | 1.52e-17 | 1.52e-17 | `1d315f06d785436f` | `1d315f06d785436f` | yes |
| `grw_spfl_joint` | largest | 4.05e-16 | 1.35e-16 | 1.35e-16 | `c820801aa35a6352` | `c820801aa35a6352` | yes |
| `grw_spfl_joint` | opening | 0.00e+00 | 3.39e-16 | 3.39e-16 | `c820801aa35a6352` | `c820801aa35a6352` | yes |
| `grw_spfl_cups_poisson` | largest | 0.00e+00 | 1.16e-16 | 1.16e-16 | `ae66db96d20906f0` | `ae66db96d20906f0` | yes |
| `grw_spfl_cups_poisson` | opening | 0.00e+00 | 3.38e-17 | 3.38e-17 | `ae66db96d20906f0` | `ae66db96d20906f0` | yes |
| `grw_spfl_cups_joint` | largest | 3.56e-16 | 1.13e-15 | 1.13e-15 | `432636a56677dbbd` | `432636a56677dbbd` | yes |
| `grw_spfl_cups_joint` | opening | 1.51e-16 | 1.20e-15 | 1.20e-15 | `432636a56677dbbd` | `432636a56677dbbd` | yes |

Worst over all 54 rows: log-density 8.51e-16, compiled gradient 3.67e-15, fresh gradient 3.67e-15.


## 2. The fix, per site (design choice and why)

Helpers: `src/models/pregame/tape_scalars.jl` — `tape_scalar(x)` (a tracked scalar → one-element
tracked vector recorded by our own `SpecialInstruction`; anything else → `Ref(x)`) and
`tape_fill(x, n)` (a `fill` whose instruction owns its buffer). Outside ReverseDiff both spell the
bare scalar exactly, so Float64/ForwardDiff/extraction arithmetic is unchanged.

| site | file | choice | why |
|---|---|---|---|
| `ClampGuard` | `builder/components.jl` | `clamp.(η, Ref(lo), Ref(hi))` | constants: a `Ref` is untracked to ReverseDiff, a scalar to every broadcast. No struct change (the `l10` `ArrayClampGuard` needed a new type; this does not). |
| joint Gamma pxG arm (shared, hierarchical, competition κ, joint NegBin) | `builder/engine.jl` `_gamma_proxy_ll`, `league_deltas.jl` | **reduce first**: `(ν−1)·Σw log x − ν·Σ(w x)e^{−ξ} − ν·Σwξ + (ν log ν − log Γ(ν))·Σw` | every term is linear in a function of ν; ν now appears only in scalar instructions, and nothing per match is differentiated w.r.t. ν — faster as well as allocation-free. One function for all four observations. |
| NegBin goals arm (single-arm and joint) | `engine.jl` `_negbin_goals_ll` | **lift** `r` with `tape_scalar` | `r` sits inside `loggamma.(y .+ r)` and `log.(r .+ λ)` — non-linear, cannot be reduced first. Same per-element operations, bit-identical values. |
| `PyramidTiers` | `builder/pyramid_tiers.jl` `_tier_side` | **regroup**: `step·A + (step·carry)·D + old_firm·O` with `A, D, O` the data combined across sides by the shares | first tried lifting (allocation-free but a six-input dual kernel: 0.222 → 0.291 ms on `td_a2`). Regrouped, each scalar meets one data vector in an unfused binary broadcast. |
| league deltas `scale .* (raw .- mean(raw))`, `ξ .+ log_κ .+ δ` | `league_deltas.jl` | **split** into unfused binary steps | same operations in the same order — bit-identical. |
| hierarchical κ `σ_κ .* (raw .- mean(raw))`; hierarchical HA `γ .+ raw .* σ` | `engine.jl`, `components/home_advantage.jl` | **split** | as above. |
| lineup pillar weights and bench weight | `builder/player_dynamics.jl` `_player_side_effects` | **lift** with `tape_scalar` | four-to-six-term linear combination of design vectors; lifting keeps it one kernel (a fixed bench weight becomes a `Ref` and the term stays plain data). |
| `fill(μ, n_seasons)`, `fill(γ, n_teams)` | `components/interception.jl`, `home_advantage.jl` | `tape_fill` | ReverseDiff's `fill` rule allocates on replay. |
| GRW centring `raw .- mean(raw, dims = 1)` | `components/dynamics/team_level/multiscale.jl` `_grw_centre` | `fill(1/n, 1, n) * raw` then unfused subtract | no ReverseDiff rule for `mean(; dims)`; one preallocated matrix-product instruction. |

Constraints held: no struct field layout changed; `string(model)` and all 27 recipe hashes
unchanged; sampled sites (names, shapes, order) unchanged — the parity run evaluates the SAME θ
vectors saved from the pre-fix code, which would fail on any layout change.

Parity (acceptance 2) is in the second table of §1: the SAME θ points saved from `8effca72` — the
linked prior draw and θ + δ·sin(i) for δ ∈ {0.003, −0.8, +0.8} — evaluated under the fixed code.
Worst over 54 rows × 4 points: log-density **8.5e-16**, compiled gradient **3.7e-15**, fresh
ReverseDiff gradient **3.7e-15**, all ≤ 1e-12. Where a site was only split or lifted the values are
bit-identical (W1 Poisson rows: 0.0); the reduce-first Gamma arm, the regrouped pyramid term and the
matrix-product GRW centring differ by summation order only.

## 3. Recipe hashes and persisted fit (acceptance 3)

- **Recipe hashes**: the table in §1 prints `recipe_hash` before (from `8effca72`) and after
  (`6f146ae2`) for every row: 27 distinct hashes, all unchanged (the audit script compares the full
  64-hex strings and prints `same=true`; e.g. `td_base` `84a3541715dc09d9…`, `td_lower_joint`
  `722ed88a7efe8c82…`, which `pyramid_tier_tests.jl` also pins).
- **Persisted fit**: `Training.load_fit(PostgresStorage("scottish_dynamics_scope_matrix_w1"),
  UUID("decb01be-f214-47bb-bc7b-bcc309026bda"))` on `6f146ae2` →
  `LOADED PoissonCountModel folds=40 fixtures=710 model=grw_lower_joint`
  (`/root/BF_runs/logs/6f146ae2/fit_load.log`). No struct changed, so older artefacts deserialize.

## 4. Regression guards (acceptance 4, 5)

**(a) `test/tape_allocation_tests.jl`** (in `runtests.jl` and `run_parallel_tests.jl`). A synthetic
pooled pyramid (13 matches, four SPFL tiers, a tier changer, a promoted club, the Old Firm, a senior
cup tie) crossed as {TimeDecay, MultiScaleGRW, GRW + `target_season_step`} × {Poisson, NegBin, joint
Shared / Hierarchical / Competition κ, joint NegBin} × {ClampGuard, NoGuard} × {none, PyramidTiers
NoCarry/NoJump, EstimatedCarry/EstimatedJump, FixedTransitionPriors (no tier term), a production
wealth covariate, a bench-weighted lineup pillar with sampled bench weight, a positional lineup
pillar} = 252 combinations; the builder rejects NoGuard for every NegBin/joint observation (105),
and the test asserts exactly that pruning, so a combination that silently stopped building would
fail. 147 recipes, each asserting **0 bytes** per compiled gradient (min over 20 warmed replays,
linked space), naming the offending instructions otherwise.

| run | result | log |
|---|---|---|
| fix (`cfd71867`, and again `6f146ae2`) | **150 / 150 pass** — 147 recipes at 0 B, 105 pruned as predicted | `/root/BF_runs/logs/cfd71867/tests_tape_harness.log`, `6f146ae2/suite_tape_allocation_tests.jl.log` |
| pre-fix `8effca72` (`cd /root/BF_runs/8effca72 && julia --project=. -t 16 -e 'using Test, BayesianFootball; include("/root/BF_runs/6f146ae2/test/tape_allocation_tests.jl")'`) | **fails: 3 pass, 147 fail** — every recipe allocates (e.g. `Evaluated: 9152 == 0`, first offender `td__poisson__clamp__none … tracker_∇broadcast[clamp](T13, Real, Real)`) | `/root/BF_runs/logs/6f146ae2/tape_test_on_8effca72.log` |

**(b) harness smoke gate.** `tape_allocation` is a **hard** check in `smoke()` (after `gradient`):
bytes per compiled gradient on every smoke fold, fails above 1,024 B, the error naming each
allocating instruction (`#110 115248 B tracker_∇broadcast[fused](…)`) and pointing at the guide.
`SMOKE_REQUIRED_CHECKS` (checks.jl) now includes it, so `has_passing_smoke` refuses a grid for a
recipe whose only smoke predates the gate. Tests (`harness_runner_tests.jl` §5/§5b): the synthetic
smoke records a passing `tape_allocation` row with 0 B; a pre-fix-style scalar clamp guard (a
test-only `AbstractRateGuard`) fails it with `tracker_∇broadcast[clamp]` in the message and a
failing hard row; a store lacking the row does not unlock the grid.

> **Deviation, flagged.** §0 asked me to keep out of `src/harness/store.jl`, but §2.3(b) requires the
> check in the `has_passing_smoke` required set, which is a literal inside `store.jl`. I changed
> exactly those two literals to `Set(SMOKE_REQUIRED_CHECKS)` (constant defined in `checks.jl`). I
> checked the other worktree's uncommitted `store.jl` diff (read-only): it touches the
> `harness_experiments` schema and `write_experiment!`, not `has_passing_smoke`, so the two should
> merge without conflict. Likewise my `stages.jl` changes are in `smoke()`; theirs are in `grid()`.

**(c) documentation.** AD guide: header note, new **Rule 7** (no bare scalars in fused broadcasts),
**§10.5 "Bytes per gradient is a metric"** (failure mode, cases, fixes table, how to measure,
the td_base story — latency alone hid it), §10.1 bytes snippet and table row, §9 `Ref` clamp,
checklist "Allocation" block. `AGENTS.md` §7 one-line digest (20,283 B < 22,000). Julia coding
guide: `Ref` clamp and an anti-pattern row. `l10`/`l11` loaders: adapters marked superseded.

## 5. Performance (acceptance 7, §2.4)

Each candidate on its scope's largest fold (fold 60, 2,390 rows), W2 sampler settings (target
acceptance 0.65, max depth 10, uniform init). Pre-fix = the new `scripts/sampling_perf_probe.jl` run
against `8effca72`; fixed = `6f146ae2`. `-t 16`, `pinthreads(:cores)`, BLAS 1 thread, nothing else
running. Logs: `/root/BF_runs/logs/6f146ae2/bench_{single,16x}_<candidate>_{PRE,NEW}.log`.

```
julia --project=<checkout> -t 16 <new>/scripts/sampling_perf_probe.jl <W2 candidates.jl> --only <name> --single-chain 150,150
julia --project=<checkout> -t 16 <new>/scripts/sampling_perf_probe.jl <W2 candidates.jl> --only <name> --chains 16 --warmup 100 --samples 100 --no-profile --no-allocs
```

| metric | `td_base` pre-fix | `td_base` fixed | `grw_step_a2_carry_jump_ldelta` pre-fix | … fixed |
|---|---|---|---|---|
| bytes per compiled gradient | 422,992 | **0** | 923,456 | **0** |
| bare gradient (warmed min, 1 thread) | 0.178 ms | 0.136 ms | 0.905 ms | 0.781 ms |
| **single chain 150 + 150, `@timed`, `discard_adapt = false`** | | | | |
| wall | 3.92 s | 2.96 s | 62.5 s | 52.2 s |
| GC % | 15.3 % | 3.1 % | 4.2 % | 1.4 % |
| GiB allocated | 7.03 | 0.58 | 69.1 | 15.8 |
| leapfrogs (incl. warm-up) | 16,110 | 18,311 | 62,082 | 63,476 |
| KB per leapfrog | 457.4 | 33.2 | 1,167.1 | 260.3 |
| ms per leapfrog | 0.244 | 0.162 | 1.007 | 0.822 |
| **16 threads, 16 chains × (100 + 100), queued** | | | | |
| wall | 34.3 s | **11.6 s** | 1,181 s | **656 s** |
| leapfrogs | 422,285 | 424,049 | 1,635,026 | 1,636,662 |
| **throughput (leapfrogs / s)** | 12,330 | **36,670 (2.97×)** | 1,384 | **2,493 (1.80×)** |
| in-situ ms per leapfrog (per thread) | 1.27 | 0.42 | 11.27 | 5.74 |
| efficiency ratio (in-situ ÷ bare) | 7.14 | 3.11 | 12.45 | 7.34 |
| bytes per leapfrog | 455.6 KB | 28.0 KB | 1,183.6 KB | 257.9 KB |
| GC share (`GC_Diff`, full-concurrency window) | 40.9 % | 24.1 % | 15.3 % | 7.0 % |
| GC pauses | 1,739 | 116 | 27,505 | 5,298 |
| sampler-thread CPU ÷ (wall × 16) (`/proc`) | 59.9 % | 76.4 % | 84.7 % | 92.5 % |
| non-sampler CPU (GC etc., cores) | 0.87 | 0.40 | 0.63 | 0.31 |

**Throughput gain at 16 threads: 2.97× for `td_base`, 1.80× for `grw_step_a2_carry_jump_ldelta`**,
on the same number of leapfrogs. Single-threaded the gain is 1.5× and 1.2× — latency alone
understated the defect, as the work package said.

What is left (both are reported by the monitor, §6):

- `td_base` fixed still spends 24 % of wall time in GC at 16 threads, from **28 KB per leapfrog
  allocated outside the tape**. `Profile.Allocs` on NUTS steps attributes it to AdvancedHMC's
  integrator and tree (`∂H∂r` hamiltonian.jl:58, `step` integrator.jl:237–247, `build_tree`
  trajectory.jl:661/668) and ~2 KB per gradient to the DynamicPPL/DifferentiationInterface
  `logdensity_and_gradient` wrappers. None of it is in this repository.
- The GRW candidate allocates 260 KB per leapfrog outside its 0-byte tape — AdvancedHMC's per-leapfrog
  vectors scale with its 1,955 parameters — and its in-situ leapfrog is 7.3× the bare gradient at 16
  threads although it is 1.05× on one thread (0.822 vs 0.781 ms) and GC is only 7 %. That is
  contention rather than GC: most plausibly memory bandwidth on a ~14,800-instruction tape per chain
  (§8). It is the next thing to fix for GRW throughput.

## 6. Smoke sampling-performance monitor (addendum §2.6 + rev2; acceptance 9–11)

Built per the addendum, with rev2's measurement tools (`src/harness/perf_monitor.jl`, wired in
`checks.jl` `_smoke_sampling_performance!` and `stages.jl` `smoke()`):

- **Probe** — after the hard checks, `nthreads()` chains (configurable: `perf_chains`) × (50 warm-up
  + 50 draws) on the smoke's largest fold, one task per chain behind a semaphore of `nthreads()`
  (the `QueuedExecution` schedule), each chain exactly `Samplers.run_sampler(::QueuedNUTSConfig)`
  except `discard_adapt = false`, so every leapfrog is counted. A 1-chain, 4-iteration run first
  JIT-compiles outside the window. Wall: 5.9 s for `td_base`, ~5 min for the GRW candidate (its
  16 chains take that long; see §5).
- **Metrics** — static via the tape audit (bytes, warmed-min latency, instructions, allocating
  instructions, warm record + compile time); in situ: wall, leapfrogs, in-situ ms per leapfrog =
  (Σ chain busy − chains × tape setup) ÷ leapfrogs, efficiency ratio, bytes per leapfrog
  (`Base.gc_bytes`); **GC via `Base.gc_num()`/`GC_Diff`** (time share, allocd, pauses, full sweeps,
  summed time-to-safepoint, process max pause / max time-to-safepoint) over the full-concurrency
  window (start → first chain done), whole-run beside it; **the statistical profiler**
  (`Profile.init(n = 10^7, delay = 0.002)`, started at the window start, stopped at the first chain
  finishing, 90 % buffer or 5 s) parsed from `Profile.fetch(include_meta = true)` into per-thread
  utilisation — sampler threads = Julia's `default` pool, the pool the chain tasks run in; GC
  threads report as `foreign: gc`; thread 1 is the interactive pool Julia 1.12 adds for `-t N` —
  plus the share of awake sampler samples in GC-stall frames (`jl_safepoint_wait_gc`,
  `[i]jl_gc_collect`, `jl_gc_small_alloc*`, `ijl_gc_managed_malloc`) and in JIT/LLVM frames, and the
  top 15 self frames; **`Profile.Allocs`** (`sample_rate = 0.05`) on 50
  `logdensity_and_gradient` calls through Turing's `LogDensityFunction` and on 20 NUTS transitions
  from `AbstractMCMC.steps` after the setup step, top 10 sites each with the first non-Base frame.
  `/proc/self/task` is kept only as the recorded fallback (`utilisation_source = "proc"`, or
  `"unavailable"` without `/proc`); sampler OS thread ids come from `gettid()` inside
  `Threads.@threads :static` (iteration i runs on the pool's i-th thread).
- **Severity** — `tape_allocation` stays hard. `sampling_performance` is **review** only: GC share
  > 15 %, sampler utilisation < 75 %, efficiency ratio > 2.0 (`SAMPLING_PERF_THRESHOLDS`, one place)
  — plus `tape_bytes > 1,024` so a standalone probe (which skips the hard check) still names that
  cause. Rows: `sampling_performance_metrics` (info, every metric as JSON) and
  `sampling_performance` (review; the failing detail carries each flag's diagnosis and the report
  path). `has_passing_smoke` does not require it.
- **Report** — `<log_dir>/smoke_perf/<experiment>/<candidate>.md`, `log_dir` = `BF_LOG_DIR` or
  `../logs/<short sha>` beside the checkout (the beast's `/root/BF_runs/logs/<sha>`), outside git;
  `run_candidates.jl --log-dir DIR` sets it. Sections: verdict, metrics, threads, top frames, the
  tape-audit allocating instructions, allocation outside the tape, diagnosis (tracked scalar in a
  fused broadcast / clamp bounds → §10.5 fixes; high GC with a zero tape → the non-tape path, with
  its `Profile.Allocs` sites; low utilisation with low GC → too few chains / serial work / JIT;
  efficiency → contention or per-leapfrog overhead).
- **Surfacing** — `run_candidates.jl` smoke `[SUMMARY]` now ends
  `gc=<%> util=<%> ms_leapfrog=<x> eff=<ratio> perf=<pass|review> run_id=…` and prints a `[PERF]`
  line with the report path (acceptance 11). `scripts/leaderboard.jl` not edited (§9).
- **Tests** (`harness_runner_tests.jl`, 136 pass on `49eba057`): the collector on a synthetic league
  (GC, profiler and `/proc` fields, `Profile.Allocs` sites); the review firing for the deliberately
  allocating fixture (a test-only scalar-clamp guard) with `tracker_∇broadcast[clamp]` named, and
  not for the fixed engine's tape; each threshold alone and the diagnosis branches; the report's
  sections; the profile-buffer parser on a hand-built buffer; the `/proc` fallback and the
  no-`/proc` degradation; the smoke integration (info + review rows, report path, summary).
- **Overhead** (rev2 item 6), `--overhead` = the same probe with and without the profiler and
  allocation pass: pre-fix `td_base` 18.91 vs 18.45 s (**+2.5 %**), fixed 5.94 vs 5.98 s (−0.5 %).
  Profiling the whole 16.7 s window had cost +13.7 % on the pre-fix run, hence the 5 s cap
  (`49eba057`).

### Acceptance 10 — the four reports

`docs/architecture/zero_alloc_engine_perf_reports/`:

| report | what | verdict | key numbers |
|---|---|---|---|
| `probe_td_base_prefix_8effca72.md` | probe on the **pre-fix** engine (`scripts/sampling_perf_probe.jl --overhead` against `8effca72`) | **review**: tape_allocation, gc_share, efficiency_ratio | GC 44.5 %, 1,018 pauses; **59 % of sampler samples in GC stalls**; work share 41 %; 458 KB / leapfrog; efficiency 7.9 |
| `probe_td_base_fixed_49eba057.md` | same probe, fixed engine | **review**: gc_share, efficiency_ratio | GC 31.6 %, 51 pauses; 28.5 % GC-stall samples; work share 71.5 %; 28 KB / leapfrog, all AdvancedHMC/DynamicPPL; efficiency 3.1 |
| `smoke_td_base_6f146ae2.md` | the real smoke (`run_candidates.jl --stage smoke`), run `61da30d7-…` | review: gc_share | GC 32.7 %, util 99.9 %, 0.383 ms/leapfrog, eff 2.76; **all 6 required hard checks pass** |
| `smoke_grw_step_a2_carry_jump_ldelta_6f146ae2.md` | smoke, run `0cb9ed3e-…` | review: efficiency_ratio | GC 6.1 %, 5.8 ms/leapfrog vs 0.77 ms bare (7.6×); **all 6 hard checks pass** |

Both W2 smokes are recorded in `scottish_pyramid_tier_transition_w2.harness_checks`;
`has_passing_smoke` is `true` for both under the new required set. Their `convergence` review rows
fail as short smokes usually do (2 × 200 draws).

**Two measurement findings.** (1) At 16 threads the profiler shows every sampler thread 100 %
"awake" even when more than half its samples sit in `jl_safepoint_wait_gc`/`jl_gc_collect`: a
thread waiting at a safepoint spins rather than sleeps. The manager's 4-thread trial saw 68–72 %.
So rev2's utilisation threshold does not catch GC stalls at this thread count. GC share does, and
`sampler_work_share` = utilisation × (1 − GC-stall share) is now reported beside it: 41 % pre-fix,
71.5 % fixed. I left the threshold where rev2 put it. (2) Instruction counts in the probe and smoke
(255 for fixed `td_base`) exceed the audit's (144) by a constant 111 before and after the fix:
DynamicPPL evaluates through a `ThreadSafeVarInfo` when `nthreads() > 1`. Bytes are 0 in both.

## 7. Test suites (acceptance 6)

From a clean checkout of `6f146ae2`, `julia --project=. -t 16`, one process per file
(`/root/BF_runs/logs/6f146ae2/suite_*.log`):

| suite | result |
|---|---|
| `test/tape_allocation_tests.jl` | pass |
| `test/pyramid_tier_tests.jl` | pass |
| `test/test_multiscale_grw.jl` | pass |
| `test/builder_tests.jl` | pass |
| `test/harness_runner_tests.jl` | pass (136 tests; rerun on `49eba057`: 136 / 136) |
| `test/harness_scoring_tests.jl` | pass |
| `test/harness_portfolio_tests.jl` | pass |
| `test/scoped_splitter_tests.jl` | pass |
| `test/run_parallel_tests.jl` | **24 / 25**; the one failure is the known T007 (`features_tests.jl`: ``UndefVarError: `SplitClockProbe` ``) — same as the TODO 030 baseline |

## 8. Residual allocation and other findings

- **None on the compiled tape**: 0 B for all 54 recipe × fold rows and all 147 test recipes.
- **Turing's `logdensity_and_gradient` wrapper** still allocates a little per gradient outside the
  tape (DifferentiationInterface/DynamicPPL); see the `Profile.Allocs` sites in the monitor reports.
- **GRW tapes remain long**: 13,972 instructions on the W2 largest fold (0 B, 0.59 ms). The
  `mean(dims)` scalarisation was ~3,600 of the 17,619; the rest is another scalarising construct in
  the GRW path (not investigated — not an allocation). Worth a follow-up: GRW gradients are 4–5× TD's.
- **Plain `logdensity`** (not on the NUTS path) allocates tens of MB per call through Distributions'
  `filldist` logpdf; reported by the manager, not changed.

**Legacy engines** (`src/models/pregame/engines/`, out of scope, not fixed, not audited
dynamically). A static scan finds the same offenders in **all 28** files: `clamp.(…, -10.0, 10.0)`
with scalar bounds (→ `tracker_∇broadcast[clamp]`), plus `view(A, idx)` selection (T002, which
scalarises the whole likelihood). Anyone reviving one should run
`scripts/tape_allocation_audit.jl` on it and apply §10.5.

## 9. Deferred (rev2 item 7) and proposed follow-ups

- HTML flame graph (ProfileCanvas/PProf) as a review artifact, and `JET.@report_opt` for dynamic
  dispatch in model code — both need new dependencies in a separate tools environment.
- `scripts/leaderboard.jl` (owned by the Phase B agent) — **not edited**. Proposed change: read the
  latest `harness_checks` row with `check = 'sampling_performance'` per `recipe_hash` and
  `stage = 'smoke'`, and print `perf=pass|review` (row status) beside each REGISTER/LEADERBOARD
  entry; no scoring logic involved.
- The GRW scalar-instruction finding above.

## Blocked

**Acceptance 10, second half — "fixed `td_base` → pass" — is not met, and meeting it needs a human
decision, not more engine work in this package.**

- The fixed `td_base` tape allocates 0 B. The monitor still flags it `review`: GC share 31.6 % in the
  probe and 32.7 % in the real smoke (threshold 15 %), and efficiency 3.1 (threshold 2.0).
- `Profile.Allocs` attributes the remaining ~28 KB per leapfrog to AdvancedHMC's leapfrog and tree
  code (`∂H∂r`, `step`, `build_tree`: fresh parameter-length vectors per step) and ~2 KB per
  gradient to the DynamicPPL/DifferentiationInterface `logdensity_and_gradient` wrappers. At 16
  threads that is still 51 collections in 5 s, each stopping all 16 sampler threads.
- Changing Turing, AdvancedHMC or DynamicPPL is out of scope (§3), and lowering a threshold until
  the check goes green is exactly what the repo's escalation rules forbid.

Options for the human:
1. Accept the `review` as a true finding. The monitor is doing its job: the next throughput limit
   after the tape is the sampler's own allocation. Read acceptance 10 as met by pre-fix
   44.5 % GC / 7.9× vs fixed 31.6 % / 3.1×, with the 2.97× throughput gain in §5.
2. Re-set the thresholds deliberately — for instance GC share relative to a measured sampler-side
   floor — as a stated policy.
3. Open a follow-up on sampler-side allocation: in-place AdvancedHMC integrator state, a custom
   `LogDensityProblems` wrapper around the compiled tape, or GC tuning (`--gcthreads`, heap
   targets). The GRW candidate's 7.3× in-situ/bare ratio at 7 % GC is a related, separate item
   (contention, likely memory bandwidth on a ~14,800-instruction tape).

Everything else in the package, including every other acceptance item, is done and evidenced
above.
