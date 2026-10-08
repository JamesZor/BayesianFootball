# Batch 01 analysis artifacts

Start with [SUMMARY.md](SUMMARY.md), then [SCHEMA.md](SCHEMA.md) and [VERIFICATION.md](VERIFICATION.md). [manifest.csv](manifest.csv) records inputs, selected draw indices, seeds and hashed output paths. R01–R08 are analysis requests, not fitted-model competitions; PARTIAL/NOT_AVAILABLE/MC_UNSTABLE remain explicit.

## Reproduction — beast only

Owned session: `pi_mm_qsf`, pane `%304`. Worktree: `/root/BF_runs/market_model_qsf`; 16 Julia threads, pinned cores, BLAS1. Pinned datastore caches alias the original `/root/BF_runs/market_model_qs/.cache/`; saved original fits resolve there through the existing fit-root alias. No new MCMC, database writes, package changes or `src/` edits.

In the existing owned Julia REPL:

```julia
include("current_development/market_model/research/qs_forms/compute/q07_batch_01.jl")
ENV["QSF_REQUEST"] = "R01"  # then R02 ... R08 in order
include("current_development/market_model/research/qs_forms/compute/r07_batch_01.jl")
```

Do not run Julia on the laptop. Retain the compute caches: R03 pre-week states feed R07; R04 conditional panels feed R05/R06. R01 retains all physical draws; R02/R03 use 128 chain-balanced indices; R07 adds two/four joint latent samples without replacing those indices.

Bulky draw/null/path-mean CSVs and binary caches remain at:

`/root/BF_runs/market_model_qsf/current_development/market_model/research/qs_forms/answers/batch_01/`

Their exact paths, row counts, columns and SHA256s are in the manifest. Raw book rows for selected movements are linked by exact source-cache row ordinal, not fabricated timestamps. Per-request logs and the owned REPL capture live under `/root/BF_runs/logs/market_model_qsf/`; compact check output is retained in `logs/checks.log`.

The final report amendment, metadata-repair and audit helpers operate on existing results only. They are one-shot report operations; rerunning a request itself replaces that request's summary section and updates output manifest hashes. Historical verification rows retain earlier runtimes; final artifact checks audit the current tables, all 99 raw null-statistic records per cell, score identities and size/hash bounds.
