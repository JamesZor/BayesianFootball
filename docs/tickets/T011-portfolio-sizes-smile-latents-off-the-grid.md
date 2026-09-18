# T011 — Portfolio prices a smile container's totals through φ but sizes every stake off the plain grid

| Field | Value |
|---|---|
| Severity | medium |
| Area | `src/Portfolio/pricing.jl` |
| Status | open |
| Raised | 2026-09-13, Task 015 (`current_development/grw_market_smile/r07`, `r08`) |

## Evidence

A `SmileLatents` container reaches `Portfolio.build_books_reported` and its O/U market books are
priced through `λ_tot·φ(K)` (`BookWorkspace{SmileScoreGrid}`, `_fill_extra!`, `pricing.jl:134`).
That price becomes each `Selection`'s `p_model` (`_collect_market!`, `pricing.jl:240–257`).

It never reaches a stake. `_finish_book` (`pricing.jl:360–375`) sizes the whole fixture from the
mean score grid:

```julia
p_grid = vec(mean(w.S, dims = 3)[:, :, 1])          # the (λ_h, λ_a) grid
R      = payoff_matrix(sels, max_h, max_a, commission)
res    = allocate(spec.allocator, p_grid, R, spec.exec)   # Kelly on scorelines
k      = grid_shrink_factor(spec.shrink, w.S, R, ...)     # BakerMcHale on grid draws
```

Measured, Task 015 r07: for three smile arms, a calibrated container whose totals are priced through
`λ_tot·φ(K)` (`t25_inv_pooltot`, smile routing verified to 1e-15 on every staked totals bet) and the
same container with φ removed (`t25_inv_grid`) stake **identical ledgers** — 955 / 947 / 940 shared
bets, zero exclusive, ROI equal to 1e-14 — while their recorded mean edge differs (e.g. 1.869 vs
1.905 pp). The smile changes the reported price and nothing else.

## Root cause

The allocator's scenario space is the scoreline grid, and a per-strike smile is not a scoreline
distribution: `P(total ≤ K) = cdf(Poisson(λ_tot·φ(K)), K)` is a separate marginal per strike, not a
partition of one 12×12 tensor. `SmileScoreGrid` was built for PRICING (evaluation, the replay desk),
and the portfolio path reuses its pricer for `p_model` but not its geometry for the solve.

## Blast radius

* Every portfolio built from a smile container (Task 015 smile rungs; the TimeDecay smile engines in
  `src/models/pregame/engines/`): bankroll, ROI and drawdown are those of the model's (λ_h, λ_a) grid.
  Differences between a smile arm and a non-smile arm come from λ, not from φ.
* The ledger is internally inconsistent for totals: `p_model − p_market` (the edge column) is a smile
  edge, the stake is a grid stake. Edge-based attribution (capture ratio, edge by family) mixes the two.
* Any trust study premised on the smile correcting deep-totals prices (e.g. re-enabling Under 0.5
  because φ₀ < 1) cannot be run through `build_books_reported`: the Under 0.5 stake is still sized
  from the grid the smile was meant to correct.

## Reproduction

```julia
# a restricted smile fit `f` (SmileLatents) and its grid twin
twin = Fit(f.config, f.folds, CountLatents(f.latents.match_ids, f.latents.λ_home, f.latents.λ_away, nothing),
           f.diagnostics, f.metadata, f.save_path)
a, _ = build_books_reported(option_b_book_spec(), f, odds, ds)
b, _ = build_books_reported(option_b_book_spec(), twin, odds, ds)
all(x.a == y.a for (x, y) in zip(a, b))   # true: identical allocations
any(s.p_model != t.p_model for (x, y) in zip(a, b) for (s, t) in zip(x.sels, y.sels))  # true
```

## Proposed fix (options, with trade-offs)

1. **Totals-aware scenario matrix.** For a smile container, build a joint scenario over (1X2 outcome,
   total goals) whose total-goals marginal is the smile's `Λ(K)` curve and whose conditional 1X2 split
   comes from the grid; `payoff_matrix` then indexes that space. Coherent and exact, but a new scenario
   type through `allocate` and `BakerMcHale`.
2. **Smile-reweighted grid.** Reweight the scoreline grid so its total-goals marginal matches the
   smile at K = 0…Kmax (iterative proportional fitting on the anti-diagonals), keep everything
   downstream unchanged. Cheap and uses the existing allocator; the reweighted grid is an approximation
   beyond Kmax.
3. **Refuse.** `build_books_reported` errors on `SmileLatents` unless the caller opts into grid sizing,
   so the inconsistency is at least explicit.

## Acceptance criteria

* For a smile container, the allocator's implied `P(total ≤ K)` equals `mean cdf(Poisson(λ_tot·φ(K)), K)`
  for every staked strike to ≤ 1e-9, or the builder refuses the container.
* A container with φ ≡ 1 stakes the same ledger as its grid twin (bit-identical).
* `CountLatents` portfolios are unchanged (bit-identical ledgers on the r06 panel).

## Scope guard

Do not change `SmileLatents`, the evaluation pricer, or any engine. Do not change `CountLatents`
sizing.
