# Array-backed MarketRatesFeature — revalidation PASS

Manager-authorised component-only parity fix at **b006743f**, 2026-10-09.
`src/harness` unchanged. Original dictionary failure and all older evidence retained.

## Fresh beast entry points

16 Julia threads, pinned cores, BLAS 1; distinct REPLs, sequential execution.

| Entry point | Pane | Result | Include window (UTC) |
|---|---:|---:|---|
| test/test_market_rate_observation.jl | %381 | 3831/3831 | 17:59:33–18:07:42 |
| test/tape_allocation_tests.jl | %382 | 150+2+8 = 160/160, 0 B | 18:07:42–18:13:39 |

Likelihood still uses the production accumulation seam and independent Normal
constants at absolute 1e-12. Whole FitConfig serialization clone passes the
UNCHANGED structural comparator; mutations to each of the four feature vectors
fail parity. Exact frozen-table equality covers all 1430 rows / 1107 full books.
Sorted input permutations, empty/missing IDs and binary masks are covered.
Component-local canonical display/digest responds to IDs, rates, flags and prior;
serialization and table permutation preserve it. No cached mutable digest.

## Exact AD and finite-difference convergence

Both 975-parameter dynamics checked at all three frozen perturbed points.

| Arm | max compiled/fresh RD error | max RD/ForwardDiff error | Richardson checks | max Richardson error |
|---|---:|---:|---:|---:|
| GRW | 0 | 4.0510180339559594e-16 | 249 | 4.1923948189319555e-16 |
| QS | 0 | 4.1098944827644414e-16 | 486 | 4.264873620725574e-18 |

Exact AD gate remains relative 1e-10, Richardson 1e-8. The historical failed
coordinate totals 65/305 are reproduced; the 33/112-coordinate unions plus
50 fixed random remaining coordinates, all three points and all four h steps
remain unchanged. Float64 and BigFloat128 curves/floor estimates are retained.

## All three declared recovery seeds freshly rerun

Same truths, data/sampler seeds and 2×(200+200) budget; no tuning.

| Arm | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|
| GRW | 6/7 | 6/7 | 7/7 |
| QS | 7/7 | 6/7 | 7/7 |

**39/42 = 92.86%** coverage (gate >=80%); **0/14** parameters miss all three
seeds. All 42 newly computed truth/interval/coverage values exactly equal the
previously committed three-seed evidence, including the original 14 seed-1
intervals. Original CSVs are untouched; new runtime measurements are retained.

## Traceability and next gate

- `results/phase1_arrays/`: exact AD, coordinate manifests, both-precision FD
  curves, Richardson and all 42 recovery intervals.
- `results/gate_logs/phase1_arrays_{market,tape}.console.txt`, matching status
  files and `phase1_arrays_launcher.log`: full console evidence and PASS receipts.
  Exact raw terminal bytes retained as corresponding `.log.gz`; text removes only
  terminal control codes/carriage returns/trailing whitespace.
- Frozen table SHA256 remains
  `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.
- Original inversion manifest SHA256 rechecked unchanged:
  `57941606a195010fe86cbfda0564d61f456379d05d0e8988dce3bb2edd600fb1`.
- Joint smoke passes stand. Next ONLY market smokes, with new phase2_arrays
  outputs/logs so the old failed attempt is not overwritten. Grid still requires
  manager approval with all convergence flags and the full four-arm ETA.
- AbstractDict comparator gap remains an out-of-package TODO 041 follow-up.
