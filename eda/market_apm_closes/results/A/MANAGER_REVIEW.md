# Manager review of Phase A (2026-10-11)

## The placebo failure is absence persistence, not a confound

REPORT_A §summary says the next-match placebo is "not generally null". That placebo regresses the
current residual supremacy on **next-match absences alone**. But absences persist: an injured
regular is usually still out next week. Current and next `absent_home − absent_away` correlate at
0.55–0.66. So next-match absences stand in for current ones.

Regress on both together. The next-match coefficient goes to about zero in every league, and the
current-match coefficient is unchanged:

| League | n | r(current, next) | Current alone | Next alone | **Next, joint** | **Current, joint** |
|---|---:|---:|---:|---:|---:|---:|
| eng_championship | 1905 | 0.565 | −0.0281 | −0.0191 | −0.0048 | −0.0254 |
| eng_league_one | 1901 | 0.574 | −0.0257 | −0.0178 | −0.0045 | −0.0232 |
| eng_league_two | 1893 | 0.652 | −0.0217 | −0.0141 | −0.0008 | −0.0211 |
| eng_premier | 1267 | 0.564 | −0.0270 | −0.0144 | +0.0010 | −0.0276 |
| sco_championship | 414 | 0.650 | −0.0277 | −0.0213 | −0.0054 | −0.0242 |
| **sco_league_one** | 661 | 0.653 | −0.0225 | −0.0144 | **+0.0012** | **−0.0233** |
| **sco_league_two** | 523 | 0.656 | −0.0277 | −0.0164 | **+0.0020** | **−0.0291** |
| sco_premiership | 774 | 0.550 | −0.0327 | −0.0208 | −0.0045 | −0.0302 |

These are point estimates only, from OLS without an intercept adjustment beyond centring (`covar_pop`).
No standard errors were computed here. Phase B should report the joint model with season-block
intervals.

**Reading.** The close prices **who starts in this match**, at about −0.02 to −0.03 log-supremacy
per missing regular, in every league. Next week's absences add nothing once this week's are known.
That is the pattern a real lineup effect produces. It also answers the Phase B "persistence first"
caveat in HANDOVER.md.

Query (DuckDB over `joined_matches.csv`, eligible rows, finite values):

```sql
with d as (select league, try_cast(residual_supremacy as double) y,
             try_cast(absent_home as double) - try_cast(absent_away as double) c,
             try_cast(next_absent_difference as double) n
           from read_csv_auto('joined_matches.csv', all_varchar=true) where lower(eligible)='true'),
e as (select * from d where isfinite(y) and isfinite(n) and isfinite(c))
select league, count(*), corr(c,n), regr_slope(y,c), regr_slope(y,n),
  (covar_pop(y,n)*var_pop(c) - covar_pop(y,c)*covar_pop(c,n)) / (var_pop(c)*var_pop(n) - covar_pop(c,n)^2) as b_next_joint,
  (covar_pop(y,c)*var_pop(n) - covar_pop(y,n)*covar_pop(c,n)) / (var_pop(c)*var_pop(n) - covar_pop(c,n)^2) as b_cur_joint
from e group by league order by league;
```

## Still open for Phase B
- The Scottish designs are nearly saturated (stacked-rate df 66 / 0 for L1 / L2). An unrestricted
  player model can't be estimated there. Use pooling across leagues through the 576 cross-tier
  movers, strong shrinkage, and a single quality coefficient per player (not separate attack and
  defence).
- Held-out stability (split-half reliability) is the gate before any rating is interpreted.
