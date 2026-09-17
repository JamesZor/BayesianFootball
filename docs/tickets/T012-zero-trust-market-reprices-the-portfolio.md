# T012 — Declaring a market in the `BookSpec` reprices the whole portfolio even at trust 0

| Field | Value |
|---|---|
| Severity | medium |
| Area | `src/Portfolio/pricing.jl`, `src/Portfolio/implementations/shrinkage.jl` |
| Status | open |
| Raised | 2026-09-17, while building Task 016 (`current_development/grw_smile_spine/r08`) |

## Evidence

A trust-pruning sweep needs one reference: the same book with the new market present but at
trust 0, so a sweep row differs from its reference in the trust table alone. Task 016's r08
measured whether that reference is in fact inert, over three posteriors × two market
environments, by adding **only** O/U 4.5 to Option B's book and staking the identical policy
(`TieredTrust` with 4.5 absent from the table, so its weight is the default 0):

| environment | arm | canonical book | extended book | Δ return | bets |
|---|---|---:|---:|---:|---|
| close | `m05_joint_grw_baseline` | +385.7776% | +368.8441% | **−16.93 pp** | 1247 → 1246 |
| close | `m05_joint_grw_smile_supremacy_w040` | +553.4909% | +536.6523% | **−16.84 pp** | 1250 → 1253 |
| close | `m05_joint_grw_smile_spine_w040` | +469.3535% | +449.8734% | **−19.48 pp** | 1232 → 1247 |
| t25 | `m05_joint_grw_baseline` | +531.7811% | +509.5218% | **−22.26 pp** | 1124 → 1123 |
| t25 | `m05_joint_grw_smile_supremacy_w040` | +381.2263% | +382.3781% | **+1.15 pp** | 1112 → 1117 |
| t25 | `m05_joint_grw_smile_spine_w040` | +457.6981% | +467.9181% | **+10.22 pp** | 1104 → 1111 |

Six for six: the ledger is never identical. **The sign is not stable** — −22.3 pp for one arm and
+10.2 pp for another in the same environment — so this is an arbitrary reprice, not a bias that
could be corrected or bounded. Bet counts move in both directions too (−1 to +15).

Task 015's `r08_trust_sweep.jl:166-178` already recorded the same observation for one arm at the
close and worked around it (it compares every sweep row against P0 on the extended book). This
ticket is the generalised measurement: it is not arm-specific, not environment-specific, and not
an artefact of NaN-unsafe `==` (the comparison here is `isequal`).

## Root cause

A market with trust 0 contributes no stake, but it does contribute **columns to the payoff
matrix**, and the payoff matrix is an input to two things that set every other stake in the
fixture:

1. `payoff_matrix(sels, max_h, max_a, commission)` (`src/Portfolio/payoff.jl:50`) is built from
   ALL admitted selections, and `_finish_book` (`pricing.jl:360-375`) passes it to
   `allocate(spec.allocator, p_grid, R, spec.exec)`. The Kelly solve therefore runs in a
   different coordinate space, and `pricing.jl`'s own RULE 2 records that this matters:
   "LBFGS on a permuted problem does not return the exactly-permuted answer in floating point."
   Here the problem is not merely permuted, it is widened.
2. `grid_shrink_factor` → `shrink_factor(::BakerMcHale, …)`
   (`src/Portfolio/implementations/shrinkage.jl:53-85`) re-solves the allocator on 128 posterior
   draws with the SAME widened `R` and returns **one scalar `k` per fixture**. That `k`
   multiplies every stake in the fixture. So a changed `k` rescales the whole slate, which is
   the mechanism that turns "one extra unstaked column" into a double-digit swing in terminal
   return.

The trust weight is applied downstream of both (`PolicySpec`/`TieredTrust`), so it cannot undo
their effect. Zero trust zeroes the new bet; it does not remove the new bet from the geometry
the other bets were solved in.

## Reproduction

```julia
# any converged fit `f`, the Option B contract, and a panel of quoted fixtures
book, policy = MatchDay.option_b_system().book, MatchDay.option_b_system().policy
ext = Portfolio.BookSpec(markets = Data.MarketConfig(vcat(
          Data.AbstractMarket[Data.Market1X2(), Data.MarketBTTS()],
          [Data.MarketOverUnder(i + 0.5) for i in 0:4])),
      price = book.price, allocator = book.allocator, shrink = book.shrink, exec = book.exec)

a = Portfolio.simulate_portfolio(policy, Portfolio.build_books(book, f, odds, ds)...)
b = Portfolio.simulate_portfolio(policy, Portfolio.build_books(ext,  f, odds, ds)...)
isequal(a.trajectory.bets, b.trajectory.bets)   # false, though 4.5 has trust 0 in `policy`
a.summary.total_return_pct - b.summary.total_return_pct   # ~17-22 pp on ScottishLower
```

`current_development/grw_smile_spine/results/trust_sweep/r08_gates.csv` holds the six measured
rows (`gate == "S0 extended book inert at trust 0"`).

## Blast radius

* **Any comparison across two `BookSpec`s is confounded**, including "what does adding this market
  do?" — the intended question of every trust sweep. Task 015's r08 and Task 016's r08 both
  work around it by fixing the book and varying only the trust table; a study that did not would
  attribute a shrinkage artefact to the new market.
* **A market set is not a free parameter of a persisted portfolio.** `portfolio_artifacts` stores
  the `BookSpec` beside the result, so two runs differing only in an unstaked market are not
  comparable even though their policies are identical.
* **MatchDay live vs replay**: both build from `option_b_book_spec()`, so they agree with each
  other. But adding a market to the console's book — a plausible operational change — would
  silently move the stake on every existing leg, not just add a new one.
* Not a live-money error today: nothing in `src/` adds a zero-trust market. The exposure is to
  research conclusions and to any future book-spec edit.

## Proposed fix (options, with trade-offs)

1. **Drop zero-trust selections before `_finish_book`.** Filter `sels` by the policy's trust
   weight so `R` contains only stakeable columns. Cheap and it makes trust 0 exactly equivalent
   to absence — but it couples the book to the policy, which `pricing.jl` deliberately keeps
   apart ("a `MatchBook` is a pure function of the data and the `BookSpec`"), and it would make
   the cache boundary policy-dependent.
2. **Make `k` per-selection rather than per-fixture.** Solve the shrinkage on the core basket and
   apply it only to those columns. Removes the rescaling channel, but changes `BakerMcHale`'s
   published semantics and every existing portfolio number.
3. **Refuse the comparison.** Have `simulate_portfolio` record a hash of the admitted selection
   set, and make the reporting layer refuse to diff two results whose hashes differ. Fixes
   nothing but makes the confound unrepresentable rather than merely documented.
4. **Document and standardise the workaround.** Add a `reference_book` concept to the sweep
   helpers so every trust study is forced to hold the book fixed. This is what both tasks already
   do by hand.

Option 1 is the smallest change that makes the gate pass; option 3 is the one that prevents the
error class. They are not mutually exclusive.

## Acceptance criteria

* With a market present at trust 0, `simulate_portfolio` returns a bet ledger `isequal` to the
  one from a book that omits that market, for a `CountLatents` and a `SmileLatents` arm, in both
  the close and T−25 environments (the six rows above).
* `BakerMcHale`'s `k` is unchanged for a fixture whose stakeable selection set is unchanged.
* Existing Option B portfolios reproduce bit-identically (`r06`'s P1 row: +385.78% / ROI 11.68% /
  1,247 bets on the 632-fixture close panel).
* A test in `test/unified_portfolio_tests.jl` covering the zero-trust-market case.

## Scope guard

Do not change `TieredTrust`, the allocator, or the Option B contract's market set. Do not
re-tune `BakerMcHale.n_draws` or its grid. This is about whether an unstaked selection may
influence a staked one, nothing else.
