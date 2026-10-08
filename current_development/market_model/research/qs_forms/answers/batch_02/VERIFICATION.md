
## R09
Command: `ENV["QSF2_REQUEST"]="R09"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS energy decomposition observed/all99 nulls absolute tolerance1e−10 (finite squared log-rates); full-centered product/covariance identities ≤1e−10. ALL11 common loading/prior-mean checks ≤1e−10. Canceled honest partition NOT_IDENTIFIABLE; no forecast scoring or future-cohort selection performed; no new draws.

## R10
Command: `ENV["QSF2_REQUEST"]="R10"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS old continuous count/G/normalized shape absolute errors≤1e−10; masks constructed once from actual schedule and reused identically for observed/R6/all99 nulls; matched baseline=numerator rows; h1 identity≤1e−10. Low-support common horizon labeled, no Q-time replacement or gap interpolation.

## R11
Command: `ENV["QSF2_REQUEST"]="R11"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS unique exact paired fixture intersections; original points and999 ordinary percentiles≤1e−10;1999 circular draws/L, exactly T resampled weeks/stratum; joint=total+allocation per fixture and replicate≤1e−10; multiplicities identical within week. Derived seeds recorded. Circular sensitivity assumes stationarity; never used to create lag pairs. R09 comparisons explicitly unavailable; existing integrated MC sensitivity not recomputed or dismissed.
