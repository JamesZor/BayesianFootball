# Batch 02 results

Sources: batch 01 relay a95ca9ca; code commits, cache/input/output SHA256s and exact beast paths in [manifest.csv](manifest.csv). Seed 610802; rates natural log, energies squared log-rates, scores nats/fixture. Large tables stay on beast. Conditional-null ranks=(1+#null≤observed)/100, not p-values.

## R09
PARTIAL (manager amendment). Raw RTS suffix energies and matched99-null ranks: axis_projection.csv; identities: projection_identity.csv. Fixed active roster, actual calendar transitions; old continuous-appearance selection shown separately.
eng_premier style common-energy fraction/rank: 22/23 0.2523830180531469/1.0; 23/24 0.23400469201427254/1.0; 24/25 0.05764772199856976/1.0; 25/26 0.05243664835014145/1.0. Active-cohort mean motion is relative to the full structural roster, not a directly observed league intensity; no 1/n process-variance claim.
sco_league_two style common-energy fraction/rank: 21/22 0.2525303638740728/1.0; 22/23 0.25606316765644566/1.0; 23/24 0.16160977238484056/0.99; 24/25 0.09865395982009112/0.81; 25/26 0.10827993438676613/0.91. Active-cohort mean motion is relative to the full structural roster, not a directly observed league intensity; no 1/n process-variance claim.
sco_premiership style common-energy fraction/rank: 21/22 0.03835091512373541/1.0; 22/23 0.11053930715190201/1.0; 23/24 0.06172947423273247/1.0; 24/25 0.013431240546539639/0.9; 25/26 0.04833410143349079/1.0. Active-cohort mean motion is relative to the full structural roster, not a directly observed league intensity; no 1/n process-variance claim.
fin_veikkausliiga style common-energy fraction/rank: 2021 0.012669778181976737/0.46; 2022 0.3069205024198031/1.0; 2023 0.093657911391177/1.0; 2024 0.11781776759694945/1.0; 2025 0.2550954752734638/1.0. Active-cohort mean motion is relative to the full structural roster, not a directly observed league intensity; no 1/n process-variance claim.
Complete centered off-diagonal products = −sum(v²)/(n(n−1)); aligned temporal covariances = −trace(C)/(n(n−1)), max error 5.421010862427522e-20. R05 used appearance subsets and pair-specific temporal covariance rows: not common-shock evidence.
ALL11 C0 10b: design common-mode max loading 1.096345236817342e-15; filtered style common mean minus prior max 5.703770789011742e-15. Check passes: the full-structural-roster common mode is prior-only. The smaller season-cohort mean can move relative to teams outside that cohort; its retrospective energy is not the engine's unobserved common mode.
Honest partition, b0, component score variances and full−partition scores NOT_IDENTIFIABLE under C0; no literal uncentered formula computed, no gauge shift or dynamic mu silently introduced. Gauge/allocation/leakage assertions for these canceled scores are NOT_RUN, not passes.
A league scoring state needs dynamic mu_t in a future fit (out of scope). Existing related evidence per manager: earlier B2 weekly-shock ICC ≈0.10 retrospective, ≈0 honest; not recomputed here. These energy diagnostics do not estimate OU or goal benefit.

## R10
COMPLETE. horizon_screen.csv beast-only (manifest): primary relative q/s screens plus common/raw sensitivity, season and pool; energy and normalized shape reported separately. Old continuous counts/G/ratios reproduced ≤1e−10, including R6. All masks fixed by design;99 reused nulls/sentinel; calendar Q time=h.
eng_premier matched style shape/rank/pairs: h2 1.68895658289586/0.67/1365; h4 2.723636452359212/0.19/1216; h8 3.9072040720487626/0.14/1075; h13 3.8617639152600374/0.01/978.
sco_league_two matched style shape/rank/pairs: h2 1.7729307760157877/0.21/432; h4 3.372236219849261/0.87/398; h8 5.714430164741221/0.61/366; h13 7.397094262421799/0.26/268.
sco_premiership matched style shape/rank/pairs: h2 1.7345279309712953/0.42/674; h4 3.000703394405636/0.3/645; h8 4.585187169107571/0.26/640; h13 5.784872248810725/0.23/523.
fin_veikkausliiga matched style shape/rank/pairs: h2 1.6886755059460994/0.73/577; h4 2.587661174216756/0.17/524; h8 3.4870704481819845/0.17/411; h13 3.254791727417929/0.04/268.
eng_premier endpoint OLS axis/slope/rank q 0.0008161853231838937/0.19; s -0.00250440922026755/0.01; R6 style shapes h2=1.6842157580222632; h4=2.6634325654154085; h8=3.728914232038434; h13=3.7387407181195504.
sco_league_two endpoint OLS axis/slope/rank q -0.003059358533504876/0.03; s -0.004509383208728955/0.01; R6 style shapes h2=1.7606423384633028; h4=3.3858621292479594; h8=5.551152645045167; h13=7.119470952465711.
Common-horizon LOW_SUPPORT cells (quantity rows)=1980; raw/common modes kept separate. Season-wise slopes/shapes/counts permit inspection of pooled cancellations. h1 normalized rank suppressed; no half-life or prospective-fit superiority claim.
