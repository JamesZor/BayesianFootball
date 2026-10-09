# Decisions — quality/style goal vs market

- 2026-10-09, Phase 0: use a detached checkout in `/root/BF_runs/qs_experiment` rather than the shared beast repository, because the latter has another branch and untracked files; retain all other sessions untouched. Run only the specified base test gate before implementation.
- 2026-10-09, Phase 0 stop: the market t05 pooled entry point returns normally despite 10 failed assertions in its intentionally excluded C2-pending child testset. Treat those assertions as failures under this package's literal "if anything fails" gate; stop before Phase 1. Its C4 deterministic full-book checks passed 305/305. No threshold, numerical code, or acceptance rule changed. Manager must resolve gate scope explicitly.
