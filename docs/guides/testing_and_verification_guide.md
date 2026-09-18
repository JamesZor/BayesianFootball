# Testing and verification — pick the fastest tier that covers your change

> Extracted from `AGENTS.md` (formerly §10). For the verification ladder that applies
> *before* a test run (tape checks, log-density re-derivation), see
> [`julia_coding_context_for_agents.md`](julia_coding_context_for_agents.md) and
> [`../turing_ad_performance_guide.md` §10](../turing_ad_performance_guide.md#10-benchmarking-and-verifying-your-model).

## 1. The five tiers

```bash
# 1. Single module (FASTEST, 15-20s)
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/unified_portfolio_tests.jl")'

# 2. Concurrent full suite (4 workers, ~40-45s)
julia --project -t 8 test/run_parallel_tests.jl

# 3. Standard sequential suite (full baseline, ~3.5 min)
julia --project -t 8 test/runtests.jl

# 4. MatchDay replay console (1,015 assertions; NOT in the parallel runner)
julia --project -t 8 test/test_matchday_replay.jl

# 5. Layer-2 calibration v2 alone (in runtests.jl; T10 needs mcmc_experiments)
julia --project -t 8 -e 'using Test, BayesianFootball; include("test/test_calibration_v2.jl")'
```

Suites live in `test/` (`data_tests.jl`, `features_tests.jl`,
`pregame_tests.jl`, …). `run_parallel_tests.jl` dispatches the seventeen
module-level suites across four worker processes; the MatchDay live-pipeline and
replay suites are run on their own because their upper tiers need `betdb`, the
experiment database, and a warm DataStore cache.


## 2. Last verified state and the known T007 failure

Verified 2026-09-03: `runtests.jl` **3,195 / 3,195** in 5m42s;
`test_matchday_replay.jl` **1,015 / 1,015** in 3m03s with no tier skipped.
`run_parallel_tests.jl` reports **16 / 17** — `features_tests.jl` fails in
isolation with `UndefVarError: SplitClockProbe`, because that probe type is
defined in `splitting_tests.jl` and the two share a `Main` only in the sequential
runner. This is the known open [T007](../tickets/T007-parallel-feature-test-hidden-dependency.md),
not a regression; confirm against `runtests.jl` before chasing it.


## 3. The four-tier replay suite

The replay suite runs in four tiers — pure (clock and filtration contract, no
database), the ladder desk, the ledger (`paper_replay` execution and settlement
plus the `paper_runbook` isolation assertion), and models (a real Saturday, real
canonical fits, hot-swapping, the lineup shock). The ledger and model tiers skip
**with a message** when the database or cache is out of reach, never silently. A
"passed" line from a tier that skipped is not evidence.
