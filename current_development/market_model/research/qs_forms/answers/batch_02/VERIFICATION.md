
## R09
Command: `ENV["QSF2_REQUEST"]="R09"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS energy decomposition observed/all99 nulls absolute tolerance1e−10 (finite squared log-rates); full-centered product/covariance identities ≤1e−10. ALL11 common loading/prior-mean checks ≤1e−10. Canceled honest partition NOT_IDENTIFIABLE; no forecast scoring or future-cohort selection performed; no new draws.

## R10
Command: `ENV["QSF2_REQUEST"]="R10"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS old continuous count/G/normalized shape absolute errors≤1e−10; masks constructed once from actual schedule and reused identically for observed/R6/all99 nulls; matched baseline=numerator rows; h1 identity≤1e−10. Low-support common horizon labeled, no Q-time replacement or gap interpolation.

## R11
Command: `ENV["QSF2_REQUEST"]="R11"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS unique exact paired fixture intersections; original points and999 ordinary percentiles≤1e−10;1999 circular draws/L, exactly T resampled weeks/stratum; joint=total+allocation per fixture and replicate≤1e−10; multiplicities identical within week. Derived seeds recorded. Circular sensitivity assumes stationarity; never used to create lag pairs. R09 comparisons explicitly unavailable; existing integrated MC sensitivity not recomputed or dismissed.

## R12
Command: `ENV["QSF2_REQUEST"]="R12"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS128 full-season means/draw IDs per C0/R6 season, active-cohort team sets match; original largest gap/IQR and aligned same-side persistence reproduce≤1e−10; low/high orientation fixed, Jaccard/ARI calculated only nontrivial shared partitions; same point/null full and suffix windows;99 C0 panels reused. Posterior suffix and complete-window mask unavailable, explicitly reported rather than regenerated.

## final_checks
Command: `ENV["QSF2_REQUEST"]="final_checks"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS13 output hashes/row counts; 49 immutable input hashes;11 common-mode zero-loading/prior checks; energy/product/covariance tolerance≤1e−10; old horizon reproduction; original score point/interval reproduction≤1e−10;1999 circular B per L4/8/12; paired goal point/bootstrap-mean additivity≤1e−10;128 full-season tier means, suffix posterior explicitly unavailable; all4 summaries≤10 lines. No MCMC/new draws/SQL/src/package changes. Julia only on beast, pane%304,threads16/BLAS1.

## R12
Command: `ENV["QSF2_REQUEST"]="R12"; include("current_development/market_model/research/qs_forms/compute/r07_batch_02.jl")` in owned beast pane %304, threads16/BLAS1.
PASS128 full-season means/draw IDs per C0/R6 season, active-cohort team sets match; original largest gap/IQR and aligned same-side persistence reproduce≤1e−10; low/high orientation fixed, Jaccard/ARI calculated only nontrivial shared partitions; same point/null full and suffix windows;99 C0 panels reused. Later-season suffix posterior reuses identical full-season masks/means; only first-season truncated posterior suffix and complete-window mask unavailable, explicitly reported rather than regenerated.
