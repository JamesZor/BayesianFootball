# Corrections retained for audit

- R04 was rerun using the same cached 99 panels after eligibility was made strictly conservative: both endpoints must fall after the conditioned prefix. Earlier runtimes remain in VERIFICATION; current tables use the stricter rule.
- R04/R05 populated null envelopes incorrectly inherited a non-sentinel fallback label. The label was corrected to SIMULATED; all numerical statistics, raw replicate values and ranks were unchanged.
- First final-artifact validation failed because its lookup used `1` instead of the saved stratum `lag_1`. The exact stored keys were inspected and the validator was corrected.
- The next validation reached output hashing and failed because it called nonexistent `filehash` instead of the loader's existing `sha` helper. The validator was corrected after reading that definition. No analysis was refitted or resampled for either validator correction.

Failure traces are retained on the beast as `/root/BF_runs/logs/market_model_qsf/final_checks_initial_failure.log`, `final_checks_hash_helper_failure.log` and in the owned REPL log. Only an actual completed final-check pass is evidence of current artifact verification.
