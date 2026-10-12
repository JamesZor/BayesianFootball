# F6 progress
Phase: legacy reproduction, then normal smoke.
Done: opt-in fold_col + positional compatibility; separate held-out clock; same-state QS forecasts without innovation draws; qs_weekly identical control model/sampler. Units PASS 497 assertions; baseline vs post-source splits PASS 1368 folds across 4 existing policies × 2 panels × 3 clocks. Weekly panel PASS 115 folds/1070 fixtures, 55 same-state folds, strict day filtration all folds.
Next: re-gate wrapper compatibility fix (approved by James), restart f6 because scope struct layout changed; rerun units/split/panel then second wave-2 reproduction attempt and smoke. First legacy attempt failed BEFORE sampling at old-fit deserialize; no checkpoints exist.
Open REPLs/jobs: beast jl f6, four threads, /root/BF_runs/qs_transition_f6; no active job. Unit100427-24646, split100902-14374, panel100959-25700 all ok on first source. Legacy101337-6824 err during load_fit, recorded legacy_attempt1.log; stop/ask if legacy fails again.
Constraints: only f6 REPL, four threads, no grids; stop/ask if any gate fails twice. Records here (hub directory absent).
