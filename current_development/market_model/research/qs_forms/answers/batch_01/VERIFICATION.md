
## R01
PASS covariance round-trip relative error ≤1e−10; R2 r=1 ≤1e−12; C0 c=0; 792000 draws. Runtime=10.04s. Invocation: Q07.r01() in owned beast REPL; seed=610801 (algebra deterministic).

## R02
PASS all published 10a point summary/team quantities within 1e−6; max error=3.091971123581061e-14. 33 fits ×128 exact joint FFBS paths; static mu/gamma preserved. Runtime=91.46s. Invocation: Q07.r02().

## R03
PASS pre-week last_observed<t in every loop; theta training dates and week labels strictly precede test; all predictive covariances PD, finite scores, unique paired fixture keys; n=3053. Median marginal score reproduction max=1.2323475573339238e-13. Bootstrap seed deterministic, 999 samples, blocks4/8/12 for comparisons. Runtime=127.8s. Invocation: Q07.r03().

## R04
PASS exactly99 shared conditional nulls for each sentinel, unchanged prefix, original missingness/design/Q/R; all comparisons suffix-only. Runtime=54.70492911338806s; invocation Q07.r04().

## R04
PASS exactly99 shared conditional nulls for each sentinel, unchanged prefix, original missingness/design/Q/R; all comparisons suffix-only. Runtime=51.11627411842346s; invocation Q07.r04().

## R05
PASS reuses R04 exact99 paths; actual sigma_q/s² per calendar step; fixed-cohort centering; steps/gaps separated. Runtime=51.85799193382263s; invocation Q07.r05().

## R05
PASS reuses R04 exact99 paths; actual sigma_q/s² per calendar step; fixed-cohort centering; steps/gaps separated. Runtime=52.45690202713013s; invocation Q07.r05().

## R06
PASS C0 suffix statistics reuse99 shared conditional panels; full-season FFBS kept separate; same teams/window in observed/null; largest gap fixed once per season, side labels may align across seasons without searching gaps. Runtime=50.89862394332886s; invocation Q07.r06().

## R07
PASS published fixture intersection3053 and every plug-in joint score ≤1e−8; joint=total+allocation ≤1e−10; same whole-week joint current-state samples across variants, same128 indices as R03; first2/all4 MC check retained. Runtime=63.224801778793335s; invocation Q07.r07().

## R08
PASS all12000 theta draws retained per fit; fixed independent league permutations preserve matched r/σ_obs; available field units/limits documented, no network or SQL. Movers explicitly NOT_AVAILABLE. Runtime=9.50099492073059s; invocation Q07.r08().

## metadata
PASS corrected populated null envelopes to SIMULATED; original fallback label incorrectly said non-sentinel even when all99 replicates existed. Numerical values unchanged.

## R05_audit
PASS selected fixture IDs linked to panel observation rows and full saved-rate rows, with exact cached raw table row ordinals; no SQL/network/raw book recreation. No quote timestamps were fabricated.
