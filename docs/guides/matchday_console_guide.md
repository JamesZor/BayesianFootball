# MatchDay — the live and replay consoles

> Extracted from `AGENTS.md` (formerly §7, with the L5 architecture note from §2).
> Section numbers here map one-to-one onto the old ones: §1 = old §7.1 … §5 = old §7.5.
> Operator loop: [`QUICKSTART_LIVE.md`](../../current_development/match_day_inference/QUICKSTART_LIVE.md).
> Full operator detail, keyboard map and the replayable-day table:
> [`current_development/match_day_inference/README.md`](../../current_development/match_day_inference/README.md).
> The paper-ledger schemas (`paper_runbook`, `paper_replay`) and their unique constraints are in
> [`experiment_database_and_config_truth_guide.md` §2.1](experiment_database_and_config_truth_guide.md#21-betdb--the-operational-database).

## L5 — the operational layer (`src/MatchDay/`)

The operational layer. It prices a whole simultaneous fixture slate at a stated
instant, records the planned stake vector in a paper ledger, and serves it to an
operator. **The slate is the execution atom**: `Portfolio` solves one joint
problem for every fixture that settles together, so the stake vector is only
valid *as a vector*, and reservation is one transaction for the whole of it.

```
fixtures → identity → lineups → book → features → inference → gate → stake_sheet
```

Every stage is a seam with an abstract type (`AbstractFixtureSource`,
`AbstractIdentityResolver`, `AbstractLineupSource`, `AbstractBookSource`,
`AbstractGate`, …), which is what lets the replay console swap only the sources
that read a clock or a network while keeping the gates, the instrument rule, the
stake rounding, the market set and the portfolio policy identical to the live
path. Posteriors are never sampled here: `MD.canonical_fit` loads a completed run
out of `mcmc_experiments`.

## The two consoles

Two long-running processes, side by side, and **neither can reach the other's
rows**.

| | runner | port | tmux | schema | clock | for |
|---|---|---|---|---|---|---|
| **live** | `r07_serve_console.jl` | **8085** | `matchday_console` | `betdb.paper_runbook` | `now()` | committing a slate on a Saturday |
| **replay** | `r08_replay_console.jl` | **8086** | `replay_run` | `betdb.paper_replay` | a scrubber | backtesting a past Saturday |

Isolation is **structural, not conventional**: `assert_replay_schema` refuses
`paper_runbook` at every ledger call site and `serve_replay` refuses to bind
8085. Both refusals are asserted directly in the suite (R1, R2, R18), the last of
them by counting `paper_runbook` rows either side of a full execute-and-settle.

Verify a console is up before assuming it is:

```bash
tmux ls                                  # session names drift; check, do not trust
curl -s localhost:8086/api/health        # {"ok":true,...,"port":8086,"schema":"paper_replay",...}
```

## 1. The replay console (8086)

It answers: *"what would this model have said, at this minute, against the book
that actually existed then — and what would it have won?"* It drives the **same**
pipeline as the live console and replaces only the sources that read a clock or a
network:

| replacement | closes |
|---|---|
| `PreloadedBook` | one query for the whole day's ladders, read with `searchsortedlast(stamps, as_of)` — a tick from after the replayed instant is *unreachable*, not merely unqueried |
| `PreloadedLineups` | `scraped_at <= as_of`, with **no historical fallback behind it** — before the scrape lands a player model prices with no lineup and contributes exactly zero |
| `PointInTimeLineupRatings` | `:player_lineup_ratings_map` is emitted over *every* match in the store, so for a finished fixture it already holds the teamsheet that took the field; this materialiser overwrites it each tick from the visible XI |
| `FrozenIdentity` | resolved once at load |

A replay that relaxed a gate, the instrument rule, the stake rounding, the market
set or the portfolio policy would prove nothing about a Saturday, so none of them
are relaxed.

The clock is **minutes relative to kick-off**, T−60 to T+105; the absolute
`as_of` is derived from it and never stored separately. Latents are memoised on a
hash of the point-in-time lineups, so `Features.create_features` runs once per
*model* (~10 s team-level, ~80 s hybrid) rather than once per tick; a tick then
costs ~0.5 s, and 60× means one simulated minute per wall second.

`2026-08-08` is the default and the only replayable day carrying both an archived
book and a provisional-XI scrape (nine fixtures, published T−13 to T−40). On that
day, fixture 16362410 (1X2 away), `m12`'s `p_model` steps 0.3934 → 0.3637 across
the XI drop while `m00` and `m05` are bit-identical across the same transition.
That control is what makes the move attributable to the lineup rather than to the
book.

## 2. The Gödel-terminal workspace

The replay page is a modular quant workspace: six draggable, resizable,
stackable windows — **Staking Ticket, Slate Radar, Multi-Ladder Desk, Trajectory
Chart, Team Form, Model Scorecard** — with top-dock toggles, a bottom dock for
minimised panels, tile/cascade, and a layout persisted per browser.

* **Multi-Ladder Desk** — a Bet Angel exchange screen: three bid and three ask
  levels per runner, spread in currency **and in Betfair ticks**, three-level
  weight of money (WOM), the de-vigged market probability beside `p_model`, fair
  odds, EV, and the simulated order marked on the runner it would actually touch
  with the £ consumed per level.
* **Trajectory Chart** — market best back/lay against stepped model fair odds,
  the T−25…T−12 execution band, the XI drop, a needle synced to the replay clock,
  and the matched-volume S-curve beneath.
* **Team Form & Lineup Delta** (`fixture_stats`) — last five results with BBC
  shots/SoT, plus the announced XI against the regular one. Form reads matches
  strictly *before* the replayed day and the XI through the pipeline's own
  point-in-time source, so the panel cannot see a teamsheet the model could not.
* **Model Scorecard** (`model_scorecard`) — three sources kept **apart** because
  they are three different claims: `fold_results` (what the run scored),
  `match_latents` scored against the de-vigged close on one match set (the only
  figure that earns "vs market", CRPS via `Evaluation.compute_crps`), and
  `paper_replay.clv_audit` (what this account's bets did). A run that wrote no
  proper scores reports `nothing` **with the reason**, never an average of the
  folds that did.

## 3. The dynamic slate re-solver

A human places bets one at a time, and the vector the account ends up holding is
not the vector `Portfolio` solved. `StakingOverride` records the three facts the
console cannot derive — which legs filled, at what price, and which were skipped —
and `resolve_slate_with_overrides` re-optimises around them:

```
max k   s.t.  Σ_t log Σ_i p_t,i (1 + [R_t a_frozen]_i + k [R_t a_free]_i)^(-λ) ≤ 0
              Σ committed_frac + k · Σ free_frac ≤ exposure_cap
```

Frozen legs enter the wealth relative as **constants**, which is what a placed bet
is; one factor scales what is left. This is a constrained form of the same
`SlateDrawdown` solve and **not a rescale**, so `Portfolio._bisect_k`'s `[0,1]`
search becomes `[0, k_cap]` — a skipped leg can genuinely entitle the survivors to
more than the full-slate solution gave them (measured: skipping one leg of a
25-leg card re-solved the other 24 at `k = 1.0167`).

A placed leg is never reduced (the money is at the venue), its payoff column is
repriced at the price it actually got, and a commitment that fills the cap sends
the uncommitted legs to zero rather than treating a negative residual as room.
`execute!` commits `active_slate`, which is the re-solved vector when one is valid
for the priced minute; a reprice retires the re-solve and keeps the overrides.

Test R36 pins the counter-intuitive finding: under a **joint** log-utility budget
the surviving legs do not automatically grow when one is skipped. At the optimum
the per-fixture penalty terms sum to zero with some negative, so removing a
subsidising leg *tightens* the constraint. Growth is unambiguous only when the
exposure cap is what binds; both regimes are asserted.

## 4. What the consoles refuse to pretend

* **No traded VWAP.** `betfair_live.order_book_1m` archives resting depth and a
  running matched total, never a traded price series. The desk shows a **book
  VWAP** and labels it as such.
* **No levels beyond the third.** The archive carries at most three, verified over
  635,765 rows; every depth and WOM figure says `(3 lvls)`.
* **No model opinion on a gated fixture.** A refused fixture shows its book and an
  empty model column, never a number derived from inputs the pipeline declined to
  use. A fold that cannot represent a fixture refuses it **by name** in the
  `NOT COVERED BY …` panel rather than pricing it at the league mean.
* **No xG in the form panel.** `sofascore.match_statistics` holds zero rows for
  tournaments 56/57, so the panel carries shots and says why.
* **`LadderSweep` is the optimistic fill model.** It crosses up to three archived
  levels instantly; the live system rests at the touch. A replay P&L built on it
  is an **upper bound**. `fill_model` is recorded per fill row so a
  `ladder_sweep_v1` track and a `touch_only` one are never pooled by accident.
* **After kick-off the posterior is pre-game and the book is in-play.** Post-T−0
  "edges" measure that gap, not a signal. Execute is disabled and the API refuses
  unless `{"allow_in_play": true}` is passed deliberately.

## 5. API surface

The live console (8085) serves `/api/snapshot`, `/api/health`, `/api/execute`,
`/api/kill`. The replay console (8086) adds the VCR, the desk, the ticket and the
intelligence widgets:

```
GET  /api/snapshot | /api/health | /api/replay/matchdays
GET  /api/replay/ladder ?match_id=&market=MATCH_ODDS|OVER_UNDER_25|BOTH_TEAMS_TO_SCORE
GET  /api/replay/history ?match_id=&symbol=&market=[&from=&to=]
GET  /api/replay/stats | /api/replay/model_scorecard
POST /api/replay/play | pause | speed | step | jump | seek
POST /api/replay/set_model {"model":"m00|m05|m12"} | set_matchday {"day":"2026-08-08"}
POST /api/replay/stake/override | stake/resolve | stake/reset
POST /api/replay/execute [{"allow_in_play":true}] | settle | reset
```

Every control also accepts a query string (`POST /api/replay/seek?t=-15`), so the
whole console is drivable from `curl`. Full operator detail, keyboard map and the
replayable-day table are in
[`current_development/match_day_inference/README.md`](../../current_development/match_day_inference/README.md).
