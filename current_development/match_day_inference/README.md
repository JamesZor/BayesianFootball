# MatchDay live execution

This directory is the active operational suite for MatchDay live execution. It prices an
entire simultaneous fixture slate, records the planned portfolio in the paper ledger, and
presents the slate through the operator console. The slate is the execution atom: reservation
is performed for the whole stake vector in one transaction before individual orders are
submitted.

## Two consoles, two ports, two schemas

They run side by side and neither can reach the other's rows.

| | port | schema | clock | what it is for |
|---|---|---|---|---|
| **live** — `r07_serve_console.jl` / `r09_live_calibrated_slate.jl` | **8085** | `paper_runbook` | `now()` | committing a slate on a Saturday |
| **replay** — `r08_replay_console.jl` | **8086** | `paper_replay` | a scrubber | backtesting a past Saturday |

The replay process refuses `paper_runbook` at every ledger call site (`assert_replay_schema`) and
refuses to bind 8085 (`serve_replay`). Both refusals are asserted in
[`test/test_matchday_replay.jl`](../../test/test_matchday_replay.jl) (R1, R2, R18) rather than
left to convention.

## Active files

- [`QUICKSTART_LIVE.md`](QUICKSTART_LIVE.md) — operator-oriented quickstart for pricing,
  reserving, submitting, settling, and serving a paper slate.
- [`r06_slate_ledger_console.jl`](r06_slate_ledger_console.jl) — replayable end-to-end Scottish
  slate runner: canonical fit, slate pricing, paper ledger, settlement, and console snapshot.
- [`r07_serve_console.jl`](r07_serve_console.jl) — the **live** operator console on port 8085.
- [`r08_replay_console.jl`](r08_replay_console.jl) — the **replay** console on port 8086 (below).
- [`r09_live_calibrated_slate.jl`](r09_live_calibrated_slate.jl) — the 2026-09-05
  **Option B** live runner: read-only pre-flight/dry-run, exact T−25 calibration, and an explicit
  `--commit` path restricted to `paper_runbook` and port 8085.
- [`replay_state.jl`](replay_state.jl) — replay loader: the clock, the point-in-time sources, the
  model registry, execution and settlement.
- [`replay_server.jl`](replay_server.jl) — replay read model and HTTP/WebSocket surface.
- [`replay_console.html`](replay_console.html) — the replay single-page dashboard.
- [`test_replay_workspace.jl`](test_replay_workspace.jl) — verification for the workspace layer:
  the tick algebra, the tall ladder, the policy configurator, the λ inversion, and (with
  `R08_FULL=1`) every HTTP route against a live console on 8086.
- [`RESEARCH_MATCHDAY_ARCHITECTURE.md`](RESEARCH_MATCHDAY_ARCHITECTURE.md) — live execution
  design, dataflow, controls, and validation rationale.
- [`AI_AGENT_HANDOVER.md`](AI_AGENT_HANDOVER.md) — system state and operational context for
  follow-on work.

Run the worked example from the repository root:

```bash
julia --project -t 8 current_development/match_day_inference/r06_slate_ledger_console.jl
```

The historical exploratory inference and single-fixture runbook prototypes are retained under
[`current_development/archived/matchday/`](../archived/matchday/):
`legacy_inference/` and `legacy_runbook/`, respectively. They are archival material, not the
active live execution path.

---

# The replay console (`r08_replay_console.jl`, port 8086)

## What it answers

> *"What would this model have said, at this minute, against the book that actually existed
> then — and what would it have won?"*

It drives the **same** pipeline the live console drives —

```
fixtures → identity → lineups → BOOK → features → inference → gate → stake_sheet
```

— and replaces only the sources that read a clock or a network. The gates, the instrument rule,
the stake rounding, the market set and the portfolio policy are the live ones, unchanged. A
replay that relaxed any of them would prove nothing about a Saturday.

Nothing here samples: every posterior is loaded from a completed run in `mcmc_experiments` via
`MD.canonical_fit`.

## Launching it

```bash
# from the repository root
julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl

# then open
#   http://localhost:8086          (LAN: http://192.168.1.88:8086)
```

Environment:

| variable | for |
|---|---|
| `BF_DB_URL` | `betdb` — fixtures, the order book, lineups, and the `paper_replay` ledger |
| `BF_EXPERIMENTS_DB_URL` *or* `~/.pgpass` | `mcmc_experiments` — the canonical fits |

Overrides, so the file does not have to be edited:

```bash
R08_DAY=2026-08-15  julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl
R08_MODEL=m12       julia --project -t 8 current_development/match_day_inference/r08_replay_console.jl
```

Boot takes roughly 60–90 s: DataStore (cached), the match day read into memory (~3 s), then one
canonical fit plus its feature collection. The other two models load lazily on first selection.

## Which Saturdays can be replayed

A day is replayable when it has fixtures **and** an archived order book. The console's
`matchday` dropdown reports both counts and disables the days that have no book; the same table
is printed at boot and is available at `GET /api/replay/matchdays`. Measured on this database:

| day | fixtures | book rows | book span | scraped XI |
|---|---|---|---|---|
| 2026-08-01 | 10 | 123,796 | 00:02 → 23:57 | 0 |
| **2026-08-08** | 10 | 43,298 | 12:00 → 16:03 | **9** |
| 2026-08-15 | 10 | 21,113 | 12:04 → 16:04 | 0 |
| 2026-08-22 | 10 | 44,687 | 14:06 → 20:26 | 0 |

**2026-08-08 is the default and the only one that shows the lineup shock.** It is the single
Saturday carrying both a book and a provisional-XI scrape — nine fixtures, published T−13 to
T−40 with a median near T−29. On the other three the player pillar contributes zero throughout,
which is correct and is also not very interesting.

## Using it

### The VCR

| control | route | keyboard |
|---|---|---|
| ▶ / ⏸ | `POST /api/replay/{play,pause}` | `space` |
| ⏮ −1m / +1m ⏭ | `POST /api/replay/step` `{"minutes": ±1}` | `←` `→` |
| speed 1x / 5x / 30x / 60x | `POST /api/replay/speed` `{"speed": 60}` | |
| scrubber | `POST /api/replay/seek` `{"t": -15}` | |
| jump | `POST /api/replay/jump` `{"target": "lineups"｜"exec"｜"kickoff"｜"settlement"}` | `x` `e` `k` `f` |

The clock is **minutes relative to kick-off**, spanning T−60m to T+105m; the absolute `as_of`
handed to `MatchDay` is derived from it and never stored separately. 60x means one simulated
minute per wall second *including* the re-pricing that minute costs — the tick time is measured
and subtracted, and the header shows `ms/tick`.

### The suggested pass

1. **T−60m** — the opening book. Thin, wide, no XI. Player models contribute zero.
2. **`x` → T−30m** — watch the XI land. On m12 the green (model) bars move and the slate
   (market) bars do not: that difference *is* the lineup shock. On m00 and m05 nothing moves,
   which is the control.
3. **`e` → T−15m** — the entry window. Press **EXECUTE**: the whole stake vector is reserved in
   one transaction and filled against the ladder that minute actually had, with `LadderSweep`.
4. **`f` → settlement** — grade every filled leg against the real score, book the P&L and the
   2% commission, and measure CLV against the de-vigged close at T−0.
5. Press **Results** for the leg-by-leg table, the aggregate ROI and beat-close, and the equity
   before/after bars.

### The workspace

Eight windows on one desktop, each draggable, resizable, minimisable to the dock and
maximisable, with the arrangement persisted in `localStorage`. Nothing about a window touches
the payload, so a layout survives every push, every reprice and every model swap.

| # | window | key | what it is |
|---|---|---|---|
| 1 | 📋 Staking Ticket | `1` | every recommended leg, what the operator did with it, and what the re-solver would now stake |
| 2 | ▦ Slate Radar | `2` | the card grid, segmented by division |
| 3 | ☱ Multi-Ladder Desk | `3` | the primary exchange screen |
| 4 | 📈 Trajectory Chart | `4` | one runner's price against the model's fair odds |
| 5 | 📊 Team Form | `5` | recent form, pxG, and the lineup delta |
| 6 | 🎯 Model Scorecard | `6` | out-of-sample proper scores, the paired test, and CLV |
| 7 | ⚖ Policy & Trust | `7` | the staking policy, as controls |
| 8 | λ Lambda Radar | `8` | market-implied goal rates against the posterior |

Plus any number of **pop-out ladder desks** (`↗` on a card): each is an independent window with
its own fixture, market and ladder mode, so two or three matches can be watched at once.
`⊞ Tile` (`t`) arranges every open window; `☰ Tile desks` (`d`) tiles only the pop-outs, side by
side across the desktop. `?popout=1,2,3` in the URL is the whole arrangement in one link.

**Ergonomics.** `[ S / M / L ]` in the header scales the console — written onto the root font
size, so Tailwind's rem-based type *and* spacing scale with it — and is persisted. Trading
colours are high-contrast throughout: blue `#3b82f6` for the bid side, magenta `#ec4899` for the
ask side, green for positive EV, amber for orders and drawdown, red for gated and in-play.

### Division segmentation

The card is ten fixtures across two divisions and is no longer one list.

- **Tabs** — `[ All Leagues (10) ] [ Scottish League One (5) ] [ Scottish League Two (5) ]`.
  The counts are the *card's*, not the filtered view's, so a tab still says how many League Two
  fixtures exist while showing none of them.
- **Split view** (`⇹ Split`) — League One in the left pane and League Two in the right, at once.
  A layout rather than a filter: watching both is the thing a tab cannot do.
- **Badges** — every card carries `L1` (cyan) or `L2` (violet).
- **Subtotals** — per-division risk, risk-weighted EV, leg count and bankroll exposure, from the
  `leagues` block on the payload.

### Lineup pricing status and the XI shock

The first question about any price is whether it was made against *estimates* or against eleven
names. Every card answers it at the top:

- `⏳ Pre-Lineup (Est)` — the scrape exists and the clock has not reached it.
- `✓ Confirmed XI (Priced) T−29m · sofascore` — priced on the teamsheet, with the drop minute
  and the source.
- `⌀ No XI scrape` — no archived lineup on this match day at all; the player pillar contributes
  zero throughout and there is nothing to measure.

Once the clock passes the drop, the card also carries the **shock delta**: the model's
probability move across the teamsheet, in points, on the selection that moved most —
`+2.9pp XI Shock · Home`. Moves of |Δp| ≥ 2.0pp are flagged `⚡ HIGH IMPACT` and pulse, and the
radar header counts them.

**The delta is absent before the clock reaches the drop, and that is structural.** Computing
`p_post` at T−45 would be asking the model a question about the future; `lineup_shock` refuses
rather than answering it. `GET /api/replay/lineup_shock` is the same data without the browser.

### The two views

**▦ Slate Radar** is the card grid the live console shows, plus two depth facts per leg:

- a **WOM pill** (`████████░░ 68% WOM`) — the share of the three archived levels of resting size
  that sits on the BACK side. Green above 60% (money queued to back, price shortening), pink
  below 40% (queued to lay, drifting), slate in between.
- `depth £X (3 lvls)` — the whole archived ladder on the side the order would consume, against
  `depth_touch`, which is only the first level.
- `↗ Ladder Desk` — opens that fixture on the desk.

**☱ Ladder desks** are the exchange screen: one fixture, one market, every runner side by side.
Each desk offers two genuinely different instruments, not two skins of one — `⇅ Tall` answers
*where has this price come from?* and `≡ Touch` answers *can this order fill?*

| part | what it shows |
|---|---|
| runner header | fair odds `1/p_model`, the market mid, EV%, and the model/market overhang bars |
| WOM bar | back share against lay share, three levels deep |
| ask levels | the three lay prices, worst-first, so the touch sits against the spread row |
| spread row | `SPREAD 0.15 (3 ticks) 4.26%` — currency, **ticks**, and relative |
| bid levels | the three back prices, touch-first |
| order marker | amber, on the runner the order actually **touches**, with the £ consumed per level |
| footer | matched volume and the **book** VWAP (see below) |
| `📈 chart` | opens that runner's trajectory window |

Markets: `[ Match Odds ] [ Over/Under 2.5 ] [ BTTS ]`. Fixtures come from a dropdown, and
`?desk=<match_id>[&market=…][&mode=tall|compact][&chart=home]` is a deep link to any of it.

#### The tall centred ladder

`⇅ Tall` draws one row per Betfair **tick** across a window centred on a **reference price** —
the first archived mid at or after T−60m — with resting size on the side it rests on and the
price column as the spine.

Fixing the frame on the anchor and letting the shelves move inside it is what makes drift
visible: a bid stack walking down the column *is* a price lengthening, and it reads the same on
every runner because the axis is ticks. `▲ +6 ticks` beside each runner is that move as a
number, signed toward a longer price, with the reference price and the minute it was taken.
The window is `±12` ticks by default (`±6 … ±30` on the slider) and **widens** rather than
scrolls when the touch drifts outside it — an operator comparing three runners needs the
reference row at the same height in each.

Marked on it: the best back and best lay rows, the dead band between them (nothing can rest
inside its own spread), the reference rule, and `▸ £x` wherever our own order would be consumed.

There are no per-tick traded volumes, because there are none to have — see (1) below.

Three things the desk does **not** pretend to know:

1. **A traded VWAP.** `betfair_live.order_book_1m` archives resting depth and a running
   `market_matched` total, never a traded price series. What is shown is `book vwap` — the
   probability-space volume-weighted average of the visible ladder — and it is labelled as such.
2. **Levels beyond the third.** The archive carries at most three, verified over 635,765 rows, so
   every depth and WOM figure says `(3 lvls)`.
3. **A model opinion on a gated fixture.** The model column is priced by the same pipeline the
   card grid uses, on the same gate-passed set; a refused fixture shows its book and an empty
   model column rather than a number derived from inputs the pipeline declined to use.

### The trajectory chart

`📈 chart` opens a draggable, minimisable window over the desk:

- **top pane** — market best back (blue) and best lay (pink) against the model's fair odds
  (green, dashed, **stepped**). On m12 the green line steps at the minute the XI became visible;
  on m00 and m05 it is flat, which is the control. A shaded band marks the T−25…T−12 execution
  window, a dashed amber vertical marks the lineup drop, and a solid blue needle tracks the
  replay clock.
- **bottom pane** — matched volume, i.e. the liquidity S-curve into kick-off.

`GET /api/replay/history` is the same data without the browser. The model is evaluated on a
coarse grid with the drop minute pinned, not every minute: the posterior is memoised on the
lineup signature and moves only when an XI lands, so a 165-minute chart costs two extractions
rather than 165.

### Switching models mid-replay

`POST /api/replay/set_model
{"model": "m00"|"m05"|"m12"|"m05_optB"|"m12_optB"}` swaps the posterior **in the running
process** and re-prices at the current instant. The clock does not move — the question is what
*this* model says at *this* minute.

Two registries are available, because the runs live in different experiments and neither is a
superset of the other. `R08_REGISTRY` picks one at launch; `r08_replay_console.jl` defaults to
`joint_player` so the Option B controls are present.

**`player_grid`** (legacy override)

| key | run | experiment |
|---|---|---|
| `m00` | `m00_poisson_control` | `scottish_lower_joint_2426` |
| `m05` | `m05_joint_production_wealth` | `scottish_lower_joint_2426` |
| `m12` | `m12_hybrid_production_wealth_player_rapm` | `scottish_lower_player_grid_2426` |

**`joint_player`** — `R08_REGISTRY=joint_player julia --project -t 8 …/r08_replay_console.jl`

| key | run | experiment |
|---|---|---|
| `m00` | `m00_poisson_control` | `scottish_lower_joint_2426` |
| `m05` | `m05_joint_production_wealth` | `scottish_lower_joint_player_2426` |
| `m12` | `m12_joint_hybrid_synergy` | `scottish_lower_joint_player_2426` |
| `m05_optB` | `m05_joint_production_wealth` + Option B | `scottish_lower_joint_player_2426` |
| `m12_optB` | `m12_joint_hybrid_synergy` + Option B | `scottish_lower_joint_player_2426` |

The `_optB` slots apply `scot_lower_t25_inv` to the exact de-vigged MatchDay quote before both
probability rendering and staking. The transform is authorised only at **T−25**: selecting one
at another replay minute refuses rather than transferring a T−25 calibration law to a different
price instant. Raw and calibrated slots use the same Option B book/policy in `r08`, so switching
`m12` → `m12_optB` isolates the rate transform. The validated system is available as
`MD.option_b_system()`; Portfolio's `min_selection_stake = 0.001` is a bankroll fraction, while
the £1 exchange minimum is applied later by MatchDay's `FloorOrDrop`.

The second pair was extended to **43 folds** for 26/27 while the others stop at 42, so on a card
whose next observed round is fold 43 the default registry conditions on an earlier fold and
`select_split` says so in a warning that is easy to miss at 60x. The fold each model actually
selected is on its info card either way — the flag changes which chains are offered, never which
fold is claimed.

Hovering a model button opens its info card: the run and experiment, the architecture read off
`typeof(config.model)`, the fold index, coverage (`8/10`), and the **convergence audit** the fit
was loaded with — max R̂, divergences, minimum bulk ESS, and the verdict. Two glyphs sit on the
button itself and mean two different refusals: `▲` is *this fold cannot represent every fixture*
and `✕` is *this chain failed its convergence gates*. The second matters most and is the one that
used to be invisible: an unconverged posterior is too **narrow**, so every `p_model − p_market`
edge reads larger than the evidence supports and Kelly stake is monotone in the edge. A fit
carrying no audit at all is reported as *unknown* and is never reported as passing.

The first selection of a model costs one `Features.create_features` (≈10 s for the team-level
pillars, ≈80 s for the hybrid player pillar) and is then held for the life of the process;
switching back is instant. Switching the *match day* rebinds the loaded models — a fold
re-selection, about a second — rather than rebuilding their features.

A fold that cannot represent a fixture refuses it **by name** in the `NOT COVERED BY …` panel
rather than pricing it at the league mean. On 2026-08-08 two teams (`ross-county`,
`airdrieonians`) are absent from these folds' `team_map`, so 8 of 10 fixtures are priced.

### The policy configurator (⚖)

Every number on the batch header is a function of three settings that used to be literals in
`r08_replay_console.jl`. The drawer makes them controls and
`POST /api/replay/set_policy` rebuilds the `PortfolioSystem` **in the running process**, then
re-prices the visible minute against the same posterior and the same book.

| control | what it is | range |
|---|---|---|
| τ Home / Draw / Away | `TieredTrust` on the 1X2 triple | 0 … 1 |
| τ BTTS | both Yes and No | 0 … 1 |
| **τ Under / Over, per strike** | **one independent pair for each of 0.5, 1.5, 2.5, 3.5** | 0 … 1 |
| τ uniform | `FlatTrust` — one weight everywhere | 0 … 1 |
| λ drawdown | `SlateDrawdown(λ)` — bigger is **tighter** | 5 … 40 (default 23) |
| cap % | `FixedCap` — the hard simultaneous-exposure ceiling | 5 … 50 (default 25) |
| shrinkage | `BakerMcHale()` / `FractionalKelly(k)` / `NoShrinkage()` | |
| risk | `SlateDrawdown` / `NoRisk` (cap only — there is deliberately no `NoCap`) | |

#### Twelve tiers, one per strike

The drawer used to collapse every total onto a single `τ Under` / `τ Over` pair written across a
fixed line set. **The edge is not a property of "unders" — it is a property of a line**, and
that collapse made the finding inexpressible:

| strike | audited | why |
|---|---|---|
| Under 2.5 | **+18.7% Kelly ROI** | the retail public's Over bias; the one total the study endorses |
| Over 2.5 | −11.4% | the other side of the same bias |
| Under 0.5 | −30.7% | at a deep total Jensen's inequality (`E[e^−Λ] ≥ e^−E[Λ]`) inflates the tail and **manufactures a Kelly edge that is not there** |
| Under 1.5 / 3.5 | dilutive | they compete for the same drawdown budget as Under 2.5 without carrying its edge |

One shared slider either gated the alpha away with the distortion or staked both. Each of the
four strikes now has its own Under and Over control, reaching its own `TieredTrust` row —
`("over_under", 2.5, :under)` and `("over_under", 0.5, :under)` were always different keys in
that table; the drawer simply was not using them.

The line set is **derived from `canonical_markets()`**, not declared: adding a total to the
market set adds its two sliders, and dropping one removes them. A tier key *is* the canonical
selection symbol (`under_25`, `over_05`), so the drawer, the request body and the odds feed all
name a total with the same string. There is deliberately no bare `under` or `over` key — it
would have to pick a line, and picking one silently is what this revision exists to stop.

Presets: **CanonicalScottishLowerTrust** (`P1_conservative_tilt` — Home 0.35, **Under 2.5 0.35**,
Draw 0.25, Away 0.25, and **zero on every other strike**, including Over 2.5 and both sides of
0.5, 1.5 and 3.5; λ 23; cap 25%), **FlatTrust**, **NoShrinkage**, and an unshrunk no-budget upper
bound. `canonical_tiers()` builds it by zeroing every tier and then naming the four the audit
endorsed, so a total added to the market set arrives **gated**, not staked. The preset is
asserted equal to `MD.canonical_scottish_lower_policy()` row for row by
`test_replay_workspace.jl` — a preset that drifted from the production policy would be a console
showing one thing and pricing another.

```bash
curl -XPOST localhost:8086/api/replay/set_policy -d '{"lambda": 12, "cap_pct": 30}'
curl -XPOST localhost:8086/api/replay/set_policy -d '{"preset": "flat", "flat": 0.5}'

# one strike at a time — each lands on its own TieredTrust row
curl -XPOST localhost:8086/api/replay/set_policy \
  -d '{"preset":"canonical","trust":{"under_05":0.08,"over_35":0.12}}'
curl -XPOST 'localhost:8086/api/replay/set_policy?trust_under_15=0.2'
```

Every field is optional and absent means *leave it*, so `{"lambda": 12}` is a one-slider request
rather than a policy that silently reset the other five controls. Refusals are **reported, not
thrown**, and leave the system where it was: `FixedCap` will not take a cap outside `(0,1)`,
a trust weight outside `[0,1]` is refused, an unknown tier name is refused **by name** (so a
stale `{"trust":{"under":0.3}}` fails loudly rather than landing on one line or on all four),
and `SlateDrawdown` will not take λ ≤ 0. Both trust weights are validated **on the way in**, not
only in the builder that happens to read them — in `flat` mode the tier table is never built, so
a check that lived only in `build_trust` would accept a bad tier, store it, and let it take
effect the moment the operator switched back to tiered. (That was a real bug;
`test_replay_workspace.jl` asserts both modes.)

Two structural notes:

- `PolicySpec` is "everything that is a pure post-multiplier on an already-built book", so a
  policy swap invalidates **nothing** — not the fold FeatureSet, not the chain, not the latents
  cache, not the model-probability cache. It is exactly one `reprice!`.
- **Shrinkage is different.** It lives in `BookSpec`, which *is* the pricing cache key, so
  changing it rebuilds the book. The response says `book_rebuilt: true` when it did.

The live console on 8085 has no equivalent of this drawer and must not acquire one: a slider that
can raise the exposure ceiling between the moment a vector is priced and the moment it is
committed is a slider that can commit an over-cap vector.

### The Lambda Radar (λ)

Every other panel compares model and market in **probability** space, one selection at a time.
That is the right space to stake in and the wrong one to think in: a 3pp gap on Home and a 2pp
gap on Under 2.5 are either *one* disagreement about how many goals are in this match or two
unrelated ones, and in probability space those look identical.

`GET /api/replay/lambda?match_id=…` inverts the book's de-vigged mids for 1X2, O/U 2.5 and BTTS
back through a double-Poisson — via `Features.fit_market_implied_parameters`, the same inversion
the market-feature pillar uses — and puts the resulting (λ_home, λ_away) beside the posterior
the model actually holds.

- **Two gauges.** The posterior is drawn as its 10–90% band with the mean as a rule inside it;
  the market is one amber needle. That asymmetry is honest — the model has an interval and the
  book has a point — and it is what tells a real disagreement from noise: the same 0.2-goal gap
  means different things against a tight fit and a wide one.
- **Two derived readings.** Total (`λ_h + λ_a`) is the O/U opinion and supremacy (`λ_h − λ_a`) is
  the handicap opinion, stated separately because a model can hold one without the other.
- **A trajectory.** The book's rate moves every minute; the posterior steps once, when the XI
  lands. On one axis, that is the claim a replay can make and nothing else can.

Two refusals: the inversion needs a **complete** 1X2 book (two rates cannot be identified from a
totals line alone, and an answer from one would be the optimiser's starting guess wearing a
number), and an incomplete market is dropped rather than contributed un-normalised.

The caveat is carried on the payload rather than left to be rediscovered: λ_market is *the
double-Poisson rate pair that reproduces this book*, and m05 and m12 do not have double-Poisson
score grids. The two are comparable as mean goal rates, not as generating processes.

### Execution and settlement

Everything lands in `betdb.paper_replay`, under account `replay_scottish`:

```
paper_slates · paper_orders · paper_fills · paper_settlements · clv_audit · account_ledger
```

`↺ Reset ledger` (`POST /api/replay/reset`) deletes that one account's rows and restores the
bankroll, so a replay can be re-run without dropping a schema.

## What is honest about it, and what is not

Three leaks are possible in a replay and all three are closed **structurally**, not by care:

1. **The book.** `PreloadedBook` holds each runner's ladder sorted by `ts` and reads it with
   `searchsortedlast(stamps, as_of)`. A tick from after the replayed instant is unreachable.
2. **The XI.** `PreloadedLineups` filters `scraped_at <= as_of` and has **no historical fallback
   behind it**, so before the scrape lands a player model prices with no lineup and contributes
   exactly zero. The live spec chains `LastHistorical` there; here that would hide the event the
   console exists to show.
3. **The player ratings.** `:player_lineup_ratings_map` is emitted by the feature extractor over
   *every* match in the store, so for a finished fixture it already holds the teamsheet that took
   the field. `PointInTimeLineupRatings` overwrites it each tick from the visible XI. Without
   that materialiser a T−60m decision would be priced off the teamsheet.

The fold is chosen by `MD.select_split`, which identifies it positively — the fold whose next
observed round *is* this card — and steps back from any fold whose target window contains the
fixtures being priced.

Two things it does **not** claim:

- **`LadderSweep` is the optimistic fill model.** It assumes we cross up to three archived
  levels instantly, which is what a market order does and not what the live system does (it rests
  at the touch). A replay P&L built on it is an **upper bound** on the resting-order path.
  `fill_model` is recorded per fill row, so a `ladder_sweep_v1` track and a `touch_only` one are
  never pooled by accident.
- **After kick-off the model is pre-game and the book is in-play.** The book has seen goals the
  posterior has not, so post-T−0 "edges" reach four figures and are a measurement of that gap
  rather than a signal. The console says so in red and disables Execute; the API refuses too
  unless `{"allow_in_play": true}` is passed deliberately.

## Verification

```bash
julia --project -t 8 test/test_matchday_replay.jl
julia --project -t 4 current_development/match_day_inference/test_replay_workspace.jl
R08_FULL=1 julia --project -t 4 current_development/match_day_inference/test_replay_workspace.jl
```

803 assertions in four tiers — pure (clock, filtration contract; no database), the ladder desk
(ticks, weight of money, the three-level book, the order marker, one runner's history), ledger (`paper_replay` execution and settlement, plus a direct assertion that `paper_runbook`
row counts are unchanged), and models (a real Saturday, real canonical fits, hot-swapping and the
lineup shock). The ledger and model tiers skip **with a message** when the database or the
DataStore cache is out of reach, never silently.

`test_replay_workspace.jl` covers the workspace layer in two tiers. **Part A is pure** — 827
assertions over the isolation constants, the Betfair tick algebra (every position on the ladder,
round-tripped through `tick_index`/`tick_price`), the tall ladder's geometry, the policy builders
and their refusals, and the λ inversion checked against books generated from *known* rates
(recovered to ±0.05 goals). It needs no database, no chain and no network. **Part B**
(`R08_FULL=1`, 75 assertions) boots a console on 8086 — falling back to 18086, or to
`R08_TEST_PORT`, when a console is already serving it, because a developer running the thing
while testing it is the normal case — and drives every route. It asserts that `set_policy`
actually re-solves rather than merely re-labelling, that each of its refusals leaves the system
exactly where it was, that no XI shock is reported for a minute the clock has not reached, and
that `serve_replay` still refuses to bind 8085.

## API reference

```
GET  /                          the page
GET  /api/snapshot              the whole payload (replay · account · batch · cards · settlement)
GET  /api/health                liveness, client count, port, schema, current minute
GET  /api/replay/matchdays      which days are replayable, and how well
GET  /api/replay/ladder         ?match_id=…&market=MATCH_ODDS|OVER_UNDER_25|BOTH_TEAMS_TO_SCORE[&ticks=12]
GET  /api/replay/history        ?match_id=…&symbol=home&market=…[&from=-60&to=105]
GET  /api/replay/stats          ?match_id=…[&n=5&threshold=0.7]
GET  /api/replay/model_scorecard ?model=m12[&baseline=m00]
GET  /api/replay/policy         the loaded policy, its presets and its bounds
GET  /api/replay/lineup_shock   [?model=m12]  per-fixture XI status and the Δp across the drop
GET  /api/replay/lambda         ?match_id=…[&trajectory=0&from=-60]
POST /api/replay/set_policy     {"preset"|"kind"|"trust"|"flat"|"lambda"|"cap_pct"|
                                 "shrink"|"shrink_k"|"risk"}  — all optional
                                trust keys: home draw away btts
                                            under_05 over_05 under_15 over_15
                                            under_25 over_25 under_35 over_35
POST /api/replay/play
POST /api/replay/pause
POST /api/replay/speed          {"speed": 1|5|30|60}
POST /api/replay/step           {"minutes": 1|-1}
POST /api/replay/jump           {"target": "start|lineups|exec|kickoff|settlement"}
POST /api/replay/seek           {"t": -60 … 105}
POST /api/replay/set_model      {"model": "m00|m05|m12|m05_optB|m12_optB"}
POST /api/replay/set_matchday   {"day": "2026-08-08"}
POST /api/replay/execute        [{"allow_in_play": true}]
POST /api/replay/settle
POST /api/replay/reset
```

Every control also accepts a query string (`POST /api/replay/seek?t=-15`), so the whole console
is drivable from `curl` without a browser.
