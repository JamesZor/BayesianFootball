# D10 entry-node implementation plan (not a passed component)

- Keep the existing `QualityStyleGRW` concrete layout/default unchanged. New opt-in
  concrete configurations wrap it; never alter serialized legacy fits.
- Feature construction determines entrants from pre-target-season in-scope matches,
  not target-season fitted matches. This eligibility persists through the target season.
- Dense, constant projections centre the incumbent trajectories over incumbents only.
  Entrant rows do not contribute to that projection, so moving an entrant prior mean
  cannot change any incumbent fitted state (required exact unit check).
- An entrant's separate standard-normal q/s entry sites are transformed by fixed
  per-club means and SDs at the first target state. They do not appear at historical
  states. Target innovations accrue after that entry state (no duplicate first step).
- In a zero-target opening fold the entry site has its declared prior in-model,
  integrates to one in the training likelihood, and extraction carries that same
  posterior draw into the first held-out target node. This is NOT a fresh OOS override.
- Prior endpoint sources are 54/55 only and strictly before target-season start;
  SDs combine endpoint uncertainty, S1 predictive residual scale and hypermean
  uncertainty diagonally. S1 parameters are retained under D9; their overlapping
  parameter-learning/normalisation IDs must be separately declared.
- The q gauge adds the destination tier's A2 reference relative to the lower scope
  once; the transport already contains division rebase. No A2 predictor in this arm.
  Style needs an explicitly recorded earlier-season lower-scope reference, not an
  unexplained zero or a second count of the historical market likelihood.
- F4 independently uses the existing QS pair on MultiScaleGRW's opt-in season map;
  opening-fold macro q/s draws must be centred and deterministic per fold/draw.

All items require the brief's gates before becoming executable S3 recipes.
