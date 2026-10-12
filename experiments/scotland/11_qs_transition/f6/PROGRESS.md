# F6 progress
Phase: legacy reproduction, then normal smoke.
Done: opt-in fold_col + positional compatibility; separate held-out clock; same-state QS forecasts without innovation draws; qs_weekly identical control model/sampler. Units PASS 497 assertions; baseline vs post-source splits PASS 1368 folds across 4 existing policies × 2 panels × 3 clocks. Weekly panel PASS 115 folds/1070 fixtures, 55 same-state folds, strict day filtration all folds.
Next: commit source/gate runners; wave-2 raw-byte reproduction; normal two-fold qs_weekly smoke; timing/report.
Open REPLs/jobs: beast jl f6, four threads, /root/BF_runs/qs_transition_f6; no active job. Unit100427-24646, split100902-14374, panel100959-25700 all ok.
Constraints: only f6 REPL, four threads, no grids; stop/ask if any gate fails twice. Records here (hub directory absent).
