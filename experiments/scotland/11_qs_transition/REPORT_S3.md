# Stage 3 — four grids complete; no transition factor clears both gates

## Summary (10 lines)
- Four frozen 60-fold grids cover W2's 1,070 fixtures in 23/24–25/26; no 26/27 outcomes used.
- All three fixes pass target LogLoss non-inferiority against `qs_control` (95% CI upper <0.002).
- None passes the two-direction transition-bias gate: relegated-club absolute bias rises in every arm.
- A2 and summer significantly reduce promoted-club absolute bias, but this is insufficient for Gate 2.
- Entrant transport worsens relegated first-20 bias from +0.067 to +0.755 pp; no significant promotion gain.
- Summer reduces promoted bias from +1.290 to +0.684 pp, but relegated bias becomes −0.464 pp.
- A2 received exactly one seeded rerun for each of nine offending folds; fold53 remains a review blocker.
- Control, entrant and summer require no reruns; all four original grids pass harness hard checks.
- Stale check832 was archived/committed before deletion; recipe checks14→13, database checks684→683.
- S4 should combine **no factors** under the frozen gate; retain control and plan pooled scope separately.

## 1. Run addresses and reproduction

| Arm | Final scored UUID | Original grid wall |
|---|---|---:|
| qs_control | `866a947c-be65-467d-8dfd-c58aed0ad1f9` | 4,559.26s /75.99min |
| qs_a2 | `2e13b786-7d47-47c0-8e04-1529e705c663` | 5,744.25s /95.74min |
| qs_entrant | `c6118c34-a2a5-46fd-b4d6-21ef7b795b49` | 4,380.72s /73.01min |
| qs_summer | `13b5d0fb-0f6a-43f0-a9bf-9d5b6efb856c` | 4,819.54s /80.33min |

A2's immutable initial UUID is `21ab856f-7c72-46be-8732-dc6122a58dc4`.
Its final fit preserves 51 original folds and replaces the nine offending folds with their
single seeded reruns, then re-audits/extracts all 60 folds. Model, sampler and splitter are
unchanged; added tags identify original run and rerun folds. Full latent/target coverage
and save/load parity were checked. `grid/reruns_qs_a2.csv` records seeds and diagnostics;
`*_qs_a2_initial.csv` preserves initial scores/diagnostics. `runs.csv` selects final fits.

Sampling uses4 chains×(500 warmup+1,000 retained), accept0.65, depth10, beast16,
pinned cores/BLAS1, one grid at a time, fresh owned `s3` per arm. The exact launch shape is
REPORT_S2 §7, with an **absolute** `scripts/run_candidates.jl` include in `jl eval`
(relative includes resolve under jl's job-code directory). Entrant prototype
`l01_entrant_prior.jl` must be preloaded as a separate `jl run` **before** calling `main`.
Two pre-sampling entrant dispatch failures were stopped and reviewed; James explicitly
allowed diagnosis/retry. Preload resolved dispatch; no source/recipe modification was needed.

The frozen datastore SHA256 was checked before each grid:
`a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12`.
`grid/r02_score_export.jl` uses `scripts/score_runs.jl`'s scoring functions with
`--target-seasons 23/24,24/25,25/26 --expected-fixtures 1070` semantics and B=10,000.
Every final run is compared against qs_control, scorecard **v1.2**, panel
`56+57|23/24,24/25,25/26|n=1070`. CSVs include control UUID and paired intervals.
No src, priors, thresholds or sampler changes; no push/merge of another worktree.

## 2. Frozen gate table

Gate1 uses paired fixture-clustered target **all-market** LogLoss differences;
Gate2 uses transitioning-club model-minus-market win probability over the first20 league
games, with paired club-season-clustered absolute-bias differences. Bias units are pp.
Both direction point estimates must improve, with at least one Δ|bias| CI excluding0.
CIs below are95%, B=10,000. Exact values: `grid/gate_table.csv`.

| Arm | ΔLL [CI] | Gate1 | Relegated bias; Δ\|bias\| [CI], pp | Promoted bias; Δ\|bias\| [CI], pp | Gate2 | Both |
|---|---|---|---|---|---|---|
| Control (self-reference) | 0 [0,0] | reference | +0.067;0 [0,0] | +1.290;0 [0,0] | reference | — |
| A2, post-rerun | +0.000313 [−0.000429,+0.001104] | PASS | +1.256; +1.189 [−1.855,+1.530] | +0.543; −0.747 [−1.164,−0.373] | FAIL | FAIL |
| Entrant | +0.000212 [−0.000491,+0.001064] | PASS | +0.755; +0.687 [−1.503,+1.862] | +1.170; −0.120 [−0.385,+0.021] | FAIL | FAIL |
| Summer | +0.000267 [−0.000504,+0.001027] | PASS | −0.464; +0.396 [−0.736,+0.841] | +0.684; −0.607 [−1.069,−0.059] | FAIL | FAIL |

The near-zero signed **aggregate** control relegation bias makes this a stringent two-direction
absolute-mean gate. Positive early-game bias and negative later-game bias cancel; this is not
proof that control has no transition problem. Do not substitute a bin-wise or absolute-individual
error criterion after seeing results. Cohorts are small (5 relegated/6 promoted club-seasons).

## 3. First-10 and entered-SPFL diagnostics

| Arm | Relegated first10 bias, pp | Promoted first10 bias, pp | Entered-SPFL first10 /first20 bias, pp |
|---|---:|---:|---:|
| Control | +0.626 | +1.151 | −0.650 /+0.949 |
| A2 | +3.021 | −0.353 | −1.470 /+0.615 |
| Entrant | +2.039 | +0.928 | −0.570 /+1.193 |
| Summer | +0.283 | +0.476 | −0.817 /+1.092 |

First10 priced observations: relegated46 (44 unique fixtures), promoted55 (53 fixtures).
Entered-SPFL has only7 first10/15 first20 priced observations from **one** club-season;
its cluster-bootstrap first20 intervals collapse to a point and must not be interpreted as
strong evidence. This is closing-price availability, not a change to the1,070-fixture target panel.
All first10/20 cohort CIs and counts are in `grid/cohort_bias.csv` and `scores_qs_*.csv`.

## 4. Bias-by-game-bin curves

Mean model-minus-market transitioning-club win probability, pp. Bin membership counts
**all league games before intersecting the priced panel**, as in S1; no renumbering of
missing prices. `grid/bias_game_bins.csv` has B=10,000 clustered CIs, observation/fixture
and club-season counts for both gate cohorts and entered-SPFL. Raw scoring observations
are committed for each arm; `r04_bias_bins.jl` reproduces the curves.

| Direction /arm | Game1 | Games2–5 | Games6–10 | Games11–20 |
|---|---:|---:|---:|---:|
| Relegated /control | +0.163 | +1.905 | −0.373 | −0.468 |
| Relegated /A2 | +7.380 | +5.522 | −0.129 | −0.435 |
| Relegated /entrant | +4.775 | +3.804 | −0.108 | −0.476 |
| Relegated /summer | +0.408 | +1.633 | −0.912 | −1.179 |
| Promoted /control | +2.824 | +1.829 | +0.204 | +1.424 |
| Promoted /A2 | −4.686 | +0.196 | −0.027 | +1.408 |
| Promoted /entrant | −0.769 | +2.043 | +0.226 | +1.403 |
| Promoted /summer | +3.114 | +1.572 | −1.043 | +0.884 |

Promotion game1 has5 priced club-seasons, subsequent bins6; relegation bins each5.
A2/entrant move the opening relegated probability upward, not toward the already-small
control aggregate bias. Summer's promotion improvement occurs mostly after the opening game.
These descriptive patterns do not identify causal mechanisms.

## 5. Calibration, compression and derivative markets

| Arm | Target all LL | All ECE | 1X2 ECE | Compression slope | O/U2.5 LL | BTTS LL |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0.633975 | 0.007273 | 0.007982 | 1.040113 | 0.680642 | 0.687344 |
| A2 | 0.634288 | 0.009851 | 0.010230 | 1.036019 | 0.680845 | 0.687862 |
| Entrant | 0.634187 | 0.006046 | 0.008006 | 1.035510 | 0.680185 | 0.687762 |
| Summer | 0.634242 | 0.006885 | 0.007920 | 1.039568 | 0.681656 | 0.687409 |

Compression is harness's **market-on-model log-rate supremacy** slope; do not confuse it
with the reciprocal model-on-market diagnostic. `grid/target_metrics.csv` carries counts.
Small ECE/derivative differences are secondary, not substitutes for the preregistered gate.

## 6. Convergence and timing

| Final arm | Max R̂ | Min bulk/tail ESS | Divergences /240,000 | Fold-level review |
|---|---:|---:|---:|---|
| Control | 1.0228 | 277 /414 | 0 | PASS |
| A2 post-rerun | 1.0255 | 290 /624 | 66 | **FAIL: fold53** |
| Entrant | 1.0358 | 262 /590 | 1 | PASS |
| Summer | 1.0301 | 304 /582 | 0 | PASS |

Initial A2 had293 divergences and min bulkESS32. Offending folds6,15,25,35,38,42,44,53,57
were each rerun once with seed2026101200+fold. Fold53 still has25/4,000 divergences (0.625%);
there is **no second rerun**. Other retained folds can have up to4 divergences, permitted by
the strict >0.1% trigger. The final aggregate66/240,000 rate passes an aggregate review,
but must not conceal the unresolved per-fold blocker. All maxR̂ values remain below1.05.
ESS<400 is recorded, not James's rerun criterion; the stricter fit audit prints FAIL despite
control/entrant/summer passing harness review. Initial hard-check CSVs and final fold CSVs
are committed; the A2 combined fit also passed explicit latent/coverage/parity checks.

Four original grid fit walls total **325.06min /5.418h**, versus S2's22.37h sampling projection.
A2 seeded rerun job added about19min plus scoring/export overhead. `timing.csv` records
control/entrant/summer fit walls and A2's72.61s **replay-only** fit wall; use
`timing_a2_initial.csv` for A2's actual original95.74min grid, not the replay figure.
Successful grid job IDs:100145-31050,112704-30834,142543-15880,154435-670;
rerun130952-11594. Fit git telemetry has `-dirty` from owned checkout research/checkpoint
artifacts. Final receipt confirms no tracked diff in src, candidate/entrant loader or
Project/Manifest; the recorded checkout commits are in timing CSVs.

## 7. S4 handoff and scope

**No F1F2/F3/F4 factor passes both frozen gates, so combine none.** Retain qs_control.
A2 also has an unresolved convergence blocker. Summer's significant promotion improvement
is a direction-specific finding, not authorisation to relax Gate2 or launch a combination.
Entrant remains labelled **exact entrant endpoint, EB transport hyperparameters** (D9):
endpoint overlap zero was gated in S2; earlier-mover hyperparameter overlaps are separate.
Full grid-generated entrant manifests are preserved under `grid/entrant_prior_manifests/`.

D14's planned S4 pooled arm is LeagueOne/Two+Championship, **goals only**, tested separately
on the same held-out lower-tier panel. It is not fitted here and is not an approved winner.
F6 is a separate build/arm, not one of these four and not touched here. No26/27 test was run.

Check832 archive and before/after counts: `archived_harness_check_832.csv`,
`stale_check_counts.csv`, readiness all-four true in `s3_readiness.csv`. Only that authorised
stale row was removed; ordinary run/score persistence followed. All evidence and report are
local commits only. Owned s3 is stopped; `grid/HANDOVER.md` is the next-owner record.
