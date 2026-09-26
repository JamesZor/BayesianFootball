# Scottish Lower grouped walk-forward fold inventory

**Run date:** 2026-09-24  
**Scope:** TODO028 pre-production boundary inventory only; no model loader, SQL collection, or sampling.

## Configuration

```julia
Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
)
```

The inventory was created from `Data.create_id_boundaries` and retrieved each next observed calendar bin through `Data.get_next_matches`. This tests the standard pooled-calendar `GroupedCVConfig` route, rather than the deprecated per-tournament clock behavior.

## Command and runtime

Executed on `mcmc-beast`, isolated worktree `/root/BayesianFootball-scotland-cross-tier`:

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 \
  /root/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project \
  -e 'include("experiments/scotland/01_time_decay_cross_tier_and_priors/fold_inventory.jl")'
```

- **Julia:** 1.12.6
- **Data source:** existing isolated cache only, `.cache/datastore_ScottishAll.jls` (no SQL call).
- **Cache age at run:** 4.0 hours; this is the successful fresh-SQL cache documented in `results/data_preflight.md`.

## Results

| Target season | Predictive boundaries | Held-out match observations | Unique held-out match IDs |
|---|---:|---:|---:|
| 24/25 | 20 | 360 | 360 |
| 25/26 | 20 | 350 | 350 |
| **Total** | **40** | **710** | **710** |

| Integrity check | Result | Verdict |
|---|---:|---|
| All ID boundaries emitted | 40 | Pass |
| Predictive boundaries with non-empty next observed bin | 40 | Pass |
| Repeated held-out IDs across boundaries | 0 | Pass |
| Train/held-out ID overlap (sum over folds) | 0 | Pass |
| Strict cutoff (`max(train kickoff) < min(held-out kickoff)`) | true for all 40 folds | Pass |

## Conclusion

The standard grouped lower-tier configuration produces the work-package inventory exactly: **40 predictive boundaries and 710 unique held-out fixtures**, split 360/350 across 24/25 and 25/26. Its ID boundaries are disjoint from their corresponding OOS bins and pass the strict kickoff-time cutoff check for every fold. The cached `ScottishAll` datastore is therefore suitable for the planned lower-tier smoke/production split construction; this result does not validate any likelihood, feature builder, or sampler.
