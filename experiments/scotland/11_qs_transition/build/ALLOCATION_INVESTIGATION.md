# D11 allocation investigation — recovered without weakening gates

Time box started ~2026-10-12 00:02+01. Resolved by ~00:20+01 (fresh normal control/A2 smoke complete).

1. Initial shared REPL (after legacy sampling) failed the unchanged control's normal
   tape-allocation gate at 57,568 B, before smoke sampling. A subsequent diagnostic found
   86,544 B on the original wave-2 fold and 80,752 B on the expanded control; individual
   instruction allocations varied. No source edits had been shipped.
2. `PHASE0.md:3–16` records Julia 1.12.4 and 0 B at source `ec96c079`. The saved wave-2
   grid fit reports `metadata.git_commit = 081ba5a6`.
3. `/root/BF_runs/qs_transition/Manifest.toml` and `/root/BF_runs/qs_experiment/Manifest.toml`
   are byte-identical: SHA256 `f86f49b72b987dd4c87ac80d16f132e5342e988fededbe27e54c6afe8133b6cc`.
   Julia 1.12.4, Turing 0.41.4, DynamicPPL 0.38.10, ReverseDiff 1.17.0, Distributions 0.25.126.
4. Archived `081ba5a6` source/Project with that current Manifest: current self-contained
   `scripts/tape_allocation_audit.jl`, expanded control 25/26 opening fold, **0 B**.
   `r05_matched_allocation.jl` then measured opening folds 1/21/41: **0/0/0 B**.
5. Fresh current-source REPL, same data/Manifest and matched diagnostic: **0/0/0 B**.
   Including the original wave-2 recipe/prototype loader retained **0/0/0 B**.
   Thus a source bisect cannot distinguish a good/bad commit: current source itself is good.
   The only later src commits were `01f42219` (fusion observation) and `13677357` (portfolio).
6. Fresh normal harness smokes for control and A2 both passed every hard check, including
   0 B linked tapes on BOTH folds and persisted save/load parity. See `smoke_checks_*.csv`.

Repair: isolate legacy reproduction from subsequent build/smoke work with a fresh owned `s2`
REPL. Source, packages and gate code/limits were not changed. This resolves the observed process
state, but does NOT establish which background activity/runtime state contaminated the first
measurement; no unproved package-drift or source-regression attribution is claimed.
The fresh matched-source comparison and normal-smoke evidence are the acceptance evidence.

Commands are committed in `r04_old_source_audit.jl`, `r05_matched_allocation.jl`, and
`r06_loader_effect.jl`. Archived source lives only inside our beast checkout at
`allocation_sources/081ba5a6`; no other checkout or session was modified.
