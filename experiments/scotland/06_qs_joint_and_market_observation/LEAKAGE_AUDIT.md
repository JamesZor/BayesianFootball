# Market-observation leakage audit — PASS (2026-10-10)

Read-only `r10_leakage_audit.jl`, beast pane `%407`, Julia 1.12.4, 16 threads, cores pinned, BLAS=1. No fitting, inversion optimisation, sampling, database writes, or changes to frozen inputs. Pane closed after verification. Run UUIDs are recorded in the CSV.

## Every fold's likelihood inputs

`results/leakage_audit.csv` contains **80 rows: both arms × all 40 folds**. The saved models equal the declared models structurally; reconstructed held-out IDs equal those derived from each saved fold's metadata. Counts use the actual feature builder's `ordered_match_ids` aligned to `flat_market_available`, and verify the observation design uses that binary mask.

- Active market observations whose match is in the held-out target set: **0 in every fold**.
- Active market observations with kickoff ≥ first held-out target kickoff: **0 in every fold**.
- Any likelihood row belonging to the held-out target set, masked or otherwise: **0 in every fold**.

Kickoffs use the harness's frozen match date/hour convention. `SplitBoundary.target_match_ids` names already-fitted past-season rows, **not** held-out fixtures; the audit correctly obtains held-out targets via `get_next_matches`. Earlier held-out fixtures may legitimately become training history in later folds; their own closes never enter the likelihood that produces their held-out prediction.

## Own-close / future-match provenance

The pinned datastore digest and frozen table digest are checked, and the original inversion manifest must match. `results/leakage_provenance.csv` checks **all 1,430 frozen rows**:

- Frozen full-book flags and home/away log rates exactly match the saved inversion CSV; uncovered rows are zero dummies. Fixture dates agree.
- Reconstructing each fixture's close after deleting **every other fixture's raw odds** gives exactly the same gated prices/probabilities. All cross-match source-row and post-kickoff-used-tick counts are zero.
- Evaluating the stored rates against only that fixture's own book reproduces every finite saved KL objective exactly (maximum gap **0**). No optimiser is called.

Read-only source inspection confirms the dependency chain: `src/Data/betfair_util.jl:summarize_odds` windows ticks at TWA(−20,0] and groups by match ID/market/selection; `current_development/market_model/l01_market_model.jl:gate_book` de-vigs within a match/market; `invert_panel` selects `bymatch[id]`; `solve_kl` uses only that book and fixed dispersed starts, with no pooled/future-match estimation. `l01_market_table.jl` injects each fixture's own rates, and `src/features/market_rates.jl` aligns by match ID. The table can contain held-out closes, but those rows are absent from each prediction's fitted likelihood; close probabilities enter only subsequent evaluation.

## Saved-prediction LogLoss parity

`results/leakage_logloss.csv`: saved posterior latents are priced without fitting; resulting probabilities match committed prediction rows exactly. On the identical **595 fixtures / 1,785 1X2 selections**, direct recomputation of the report's selection-level binary LogLoss gives:

| Arm | Recomputed / reported LogLoss | Absolute gap |
|---|---:|---:|
| grw_marketobs | 0.6117542636747713 | 0 |
| qs_marketobs | 0.6119173567756958 | 0 |

This is the existing score law, not categorical match LogLoss. Passing this audit rules out the specified leakage paths; it does not establish a statistically significant advantage over the close or change the no-promotion decision.

The initial audit stopped at prediction-key comparison because runtime selections were Symbols versus CSV strings; all provenance checks and the first arm's 40 zero-count rows had passed. Normalising only key types produced the final PASS. Failed and final execution evidence is preserved in `logs/leakage_audit_v1.log` and `logs/leakage_audit_final.log`.
