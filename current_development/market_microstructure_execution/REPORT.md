# Market microstructure and staged execution — research report

## 1. Decision and scope

This is a **read-only archive experiment and pure execution prototype**, not a live
execution upgrade or a demonstration that alpha is proven. A profitable Saturday,
a model-positive edge, and a statistically demonstrated tradable advantage are
three different claims. No model is fitted, no paper orders are submitted, and no
`src/Portfolio/` or production MatchDay code is modified.

The work-package arithmetic needs reconciliation: £451.09 − £165.65 is **£285.44
(63.28%)**, not £261.43. £261.43 is 57.96% of £451.09. The quoted P&Ls also mix
actual and confirmed-lineup counterfactuals. They must not share one baseline.

Measured evidence, input identifiers and hashes of replaceable artifacts are recorded in
[EMPIRICAL.md](EMPIRICAL.md) and the `results/` artifacts. The protocol below governs
what those measurements mean; no archive-derived fill is a real exchange fill.

### Audited live ledger (not a confirmed-XI reconstruction)

The extracted slate is `0fd23606-80bf-4fcd-be44-66d708562fe6`, account
`live_scottish`, run name `m12_joint_hybrid_synergy`, bankroll £2,400. Direct
summation of `results/2026-09-05_archive/frozen_orders.csv` gives:

| Persisted fact | Value |
|---|---:|
| Orders / fixtures with orders | 17 / 9 |
| Sum of rounded parent risks | £427.07 |
| Slate header total risk | £427.08 |
| Recorded filled liability | £144.77 |
| Recorded filled venue stake | £146.53 |
| Recorded settlement net P&L | **+£19.14** |

The £0.01 parent/header difference is retained, not silently reconciled away.
These rows do **not** reproduce the prompt's £165.65 filled-risk number or its
18-leg £451.09 confirmed-lineup scenario. All fill timestamps are
`2026-09-05 14:50:21.555 UTC`, after the 14:00 kickoff. The slate's `as_of` is
13:35; quote stamps are sixteen at 13:35 and one at 13:32. Thus this is evidence
of recorded paper-account P&L, **not proof of actual T−25 exchange submission**;
conversely, a late ledger fill stamp does not prove the reference quote was in-play.
All persisted `bf_market_id` values are blank: replay markets must be reconstructed
through metadata, with ambiguity refused and that provenance limitation visible.
This study freezes persisted `p_model`; it does not independently certify Fold 43,
reconstruct the source lineups, or prove when those probabilities first became
available. A counterfactual replay conditional on these parents is not an audited
end-to-end point-in-time inference backtest.

## 2. Units, identity and filtration

- `p` is the probability of the **position's favorable event**. For a back it is
  the venue runner winning; for a lay it is the runner losing. A synthetic back
  Over by laying Under retains the model's Over probability. A direct lay of a
  1X2 runner needs `1 − p_runner`, not the probability of another single runner.
- Decimal odds are `d`; displayed sizes `x` are **backer stake on both sides**.
  Back risk is `x`. Lay risk is `(d−1)x`, at the **actual child fill price**.
- The archive calls available-to-back prices `back`, and available-to-lay prices
  `lay`. These are not generic equity bid/ask inventory labels.
- Freeze model probabilities, original venue identities, parent risks, bankroll,
  commission and decision time across execution policies. Historical confirmed
  lineups are not automatically known at T−25. A confirmed-XI hindsight variant
  must be separately labelled, never substituted for the live ledger probability.
- At decision time `t`, select the last quote with `ts <= t`, require age ≤90 s,
  and never reuse the same quote stamp for the same position. No post-T−5 execution.
- Only three levels exist. `market_matched` is a cumulative **market-wide** total,
  not a runner-specific traded-price or order-flow series.

## 3. Reservation prices and slippage

Let `c` be commission and `α` the required net expected return per pound of
liability. The kernel applies a standalone positive-win haircut; exact exchange
commission on a portfolio must instead be netted at market scope.

Net favorable payoff per unit risk:

\[
b_B(d)=(d-1)(1-c),\qquad b_L(d)=\frac{1-c}{d-1}.
\]

Expected return is `e(d)=p b(d) − (1−p)`. Requiring `e(d) ≥ α` gives

\[
\boxed{d_B^*=1+\frac{1-p+\alpha}{p(1-c)}}\quad\text{(minimum back odds)},
\qquad
\boxed{d_L^*=1+\frac{p(1-c)}{1-p+\alpha}}\quad\text{(maximum lay odds)}.
\]

The hurdle is deliberately comparable in **liability units**, the portfolio's
budget denominator. It is not the same return per pound of venue stake: for a lay
its equivalent expected-profit hurdle per venue pound is `α(d−1)`. Report sides
and venue stakes separately; do not average venue-stake ROI with liability ROI.

Every child independently clears this price. A favorable level cannot subsidise a
negative-edge deeper child through a good average. Equality clears the hurdle;
any worse price is refused. An operational adapter must round a back minimum UP
and a lay maximum DOWN on Betfair's odds tick ladder, not to arbitrary decimals.
The prototype consumes existing archived prices and does not submit limit prices.

Relative adverse distance from reference `d₀` is `(d₀−d)/d₀` for backs and
`(d−d₀)/d₀` for lays. The initial reservation headroom in decimal-odds units is
`max(0,d₀−d*_B)` or `max(0,d*_L−d₀)` respectively. It is not an extra permission to
spend edge: the strict reservation and slippage constraints both apply.

The policy's `max_slip = δ` bounds cumulative **venue-stake arithmetic VWAP**:

\[
\bar d=\frac{\sum_j x_j d_j}{\sum_j x_j},\qquad
\bar d\ge d_0(1-\delta)\;(B),\quad
\bar d\le d_0(1+\delta)\;(L).
\]

All three prototype policies use the **same 1% tolerance** by default; TouchOnly
changes only the level count, not the acceptable price budget. The anchor stays
at the frozen order's reference odds for the whole working window. This is a
parent-reference implementation-shortfall budget, **not distance from each tick's
contemporaneous touch**. Favorable drift may finance a deeper late sweep; adverse
drift may block even level 1. That is a price-budget refusal, not a depth shortage.
Reference quote timestamps remain part of order provenance.

A back needs a **minimum**, not maximum, VWAP. Higher back odds are better. For a
fixed runner, arithmetic VWAP exactly reproduces back win proceeds and lay loss
liability. The harmonic mean `Σx/Σ(x/d)` does not have that property for stakes
weighted in the archive's denomination. This is **book-fill VWAP**, never traded VWAP.

## 4. Three-level log-utility optimum

### 4.1 Single-position problem implemented

Let `B` be original bankroll; frozen fills have risk `Q₀` and net favorable payoff
`G₀`. Available level `j` has stake `s_j` and odds `d_j`. Let `h_j=1` for backs or
`d_j−1` for lays; risk `q_j=h_j x_j`, favorable payoff `b_j q_j`.

\[
\max_{0\le x_j\le s_j}
 p\log\left(B+G_0+\sum_j b_j h_j x_j\right)
 +(1-p)\log\left(B-Q_0-\sum_j h_j x_j\right)
\]

subject to remaining parent liability, the current cumulative TWAP allowance,
strict per-child reservation prices and the linear VWAP inequality above. Wealth
must remain positive. `target_risk < B` is validated at construction.

Prices are sorted best to worst. For any total new risk, shifting risk from a worse
price to spare better-price capacity increases favorable payoff without changing
loss payoff; it also improves the VWAP. Thus the best-first frontier is optimal
for this one-position problem. It is **not** independent Kelly on three unrelated
bets: all three fills have the same outcome.

After previous levels, write current wealth as `W+=B+G`, `W−=B−Q`. The maximum
additional risk at a level before its marginal log utility becomes zero is

\[
q_K=pW_- - (1-p)W_+/b_j.
\]

Take the minimum of positive `q_K`, available level risk and remaining scheduled
parent risk, then clip by VWAP. With existing venue size `S`, odds notional `N`,
and bound `v`, the additional permissible stake at an adverse level is

\[
x_{VWAP}\le\frac{N-vS}{v-d_j}\;(B,d_j<v),\qquad
x_{VWAP}\le\frac{vS-N}{d_j-v}\;(L,d_j>v).
\]

This admits a **partial** final level, unlike an all-or-nothing level check. It
also permits one child beyond the VWAP boundary when accumulated price improvement
finances it, but never beyond the separate reservation price. No minimum Betfair
stake or penny rounding is applied in this continuous capacity study; production
rounding must round down and revalidate actual liability and minimum-order rules.

### 4.2 Full-slate problem required for deployment

For score outcome `ω`, frozen net cashflows `F_ω`, child payoff columns
`A_{ω,j}`, and score probabilities `π_ω`, the proper problem is

\[
\max_x\sum_\omega \pi_\omega\log(B+F_\omega+\sum_j A_{\omega,j}x_j),
\]

with the original exposure cap, conditional drawdown budget, depth, price and
working-window constraints. Correlated 1X2, totals and BTTS on one fixture share
its score tensor. Commission netting couples positions in the same market too.

Our scalar kernel does **not** solve this problem. Capping each leg at its approved
original risk bounds total worst-case liability, but **does not preserve the
original slate's drawdown certificate**: deleting a subsidising leg can tighten
the remaining portfolio's risk budget. More matched volume is not automatically
more expected log growth of the actual partial-fill vector.

## 5. Working window and maker/taker choice

### 5.1 Implemented, deliberately untuned control

`StagedTWAP(start_minutes=25,end_minutes=5)` offers 21 cumulative tranches. At minute
`m=0…20`, due risk is `Q_target (m+1)/21`. Already acknowledged/simulated fills are
subtracted; missed tranches carry forward. If 40% is filled at T−15, the schedule
allows `11/21−0.40 = 12.38%` of the original target as catch-up, subject to current
prices, remaining depth and Kelly room. At T−5, unfillable residual is cancelled;
the deadline never overrides the reservation price.

This is a **staged taker capacity experiment**, not a queue-aware maker algorithm,
not a time-optimal stochastic controller, and not a new Portfolio allocator.
Probabilities and instruments stay frozen; no best-future-minute selection occurs.

### 5.2 Maker/taker design, not fitted execution alpha

A future execution policy should compare conditional incremental log utility, not
raw fill percentage. If `u_T` is immediate taker utility and a passive order has
conditional fill probability `q_f`, filled utility `u_M` including adverse
selection, and continuation value `V_next`, a maker is preferred only when

\[
q_f E[u_M\mid\text{fill},z]+(1-q_f)V_{next}(z)>u_T(z).
\]

The state `z` includes remaining liability, time, spread in **ticks**, side-specific
price headroom, stale/suspended status, displayed imbalance, lagged market-volume
velocity and acceleration, current slate exposure, and verified lineup information.
Near deadline the continuation value falls, but it can never justify a
negative-edge or risk-infeasible take. Pending maker liability must remain reserved
until cancellation is acknowledged before a replacement taker is sent.

One-minute snapshots cannot identify `q_f`, queue priority, cancellation flow or
fill-conditioned adverse selection. Therefore the honest passive lower scenario
is **zero credited passive fills**. A crossed-price snapshot alone is not a fill
proof. Live shadow orders and higher-frequency exchange updates are prerequisites
for calibrating that policy; the prototype emits only descriptive imbalance labels.

## 6. WOM and traded-volume hypotheses

\[
w_t=\frac{\sum_{j=1}^3 s^{available\ to\ back}_{j,t}}
 {\sum_{j=1}^3(s^{available\ to\ back}_{j,t}+s^{available\ to\ lay}_{j,t})}.
\]

Zero depth produces `nothing`, not neutral confidence. Because available-to-back
liquidity is supplied by resting layers, `w>0.65 ⇒ shortening` is not a mathematical
identity. The sign depends on the quote convention and flow/cancellation behavior;
it must be measured separately for each side/market. Sizes also represent different
liabilities at different odds. Test stake-WOM against liability-normalised and
level-1 variants before promoting it.

`v_t=(M_t−M_{t−1})/Δt` is **market-wide matched currency per second**. Acceleration
is `(v_t−v_{t−1})/Δt` using only past observations. A reset or negative increment is
unavailable, not negative trading. Do not sum the same market total over runners.

There is no universal derivation of **0.65** from WOM alone. The economic threshold
would be the `w*` where conditional maker-versus-taker utility above crosses zero.
Estimating it requires queue/fill and price-transition data. For the archive,
0.35/0.65 are **predeclared exploratory bins**, not an optimised trading threshold.
Test signed future changes at fixed horizons against neutral-bin behavior, using
only timely observed targets, and report fixtures/days as clusters rather than
treating every minute of each runner as an independent bet. A single Saturday does
not identify a stable threshold. No WOM or volume signal changes the prototype's
schedule, so the policy comparison does not hide an in-sample signal fit.

## 7. Archive comparison protocol and interpretation

1. Identify an explicit live slate UUID; reconcile its own planned, filled and
   settled risk/P&L before counterfactual simulation. Keep distinct accounts apart.
2. Use identical ledger orders for TouchOnly, MultiLevelSweep and StagedTWAP. A
   legacy TouchOnly reproduction and a new guarded TouchOnly are different controls:
   the latter additionally enforces the 2% net risk hurdle and Kelly derivative.
   Snapshot and per-level binding diagnostics separately identify depth, reservation,
   slippage, Kelly, target and policy-level limits. They are sequential constraints,
   not an additive attribution of missing pounds; total unfilled risk must never be
   called a pure liquidity shortage.
3. Compare one-shot policies at T−25; staged decisions every minute through T−5.
   Same as-of selection and quote-age cutoff for each. Outcomes are used only after
   fills are frozen, for settlement. The archive timestamp is the filtration key;
   minute-bucket upserts may not prove receipt-time availability within the minute.
   Until collector timestamp semantics are independently verified, this remains an
   archive-timestamp replay, not an exact intra-minute executable-fill proof.
4. Track depth by **market, runner, side and absolute price**, not level number.
   Persistent resting volume can move between levels without being new liquidity.
   Share depletion across all orders touching the same runner/side.
5. Report a no-replenishment scenario (cumulative consumed size subtracted at each
   price) and a refreshed-display scenario. The former can miss genuine refill;
   the latter can spend the same historical resting pounds repeatedly. Both assume
   displayed liquidity is executable at the sampled instant. Neither is an actual
   probabilistic lower/upper bound on live P&L; even the no-refill scenario is not a
   guaranteed fill floor. Order collisions and processing priority must be explicit.
6. Paired liquidity growth compares the **same runners** with valid T−25 and T−5
   quotes; report missing/stale counts and totals plus medians. It is not evidence
   that every intended leg's reservation-qualified capacity grows 2–3×.
7. Report nominal matched liability, venue stake, fill ratio, arithmetic book VWAP,
   price-specific cashflow P&L and commission convention. Full-fill at original odds
   is an unexecutable capacity reference, not another strategy or a proven alpha.

## 8. Integration roadmap and gates

Keep two concepts separate: **execution policy proposes child orders**;
**fill model simulates acknowledgements**. `AbstractExecutionPolicy` currently
exists only in this prototype module. Existing `MD.AbstractFillModel` is not a
scheduler. `submit_slate!` in `src/MatchDay/ledger/reservation.jl` completes a
single submission pass and releases unmatched remainder; repeatedly calling it is
not staged execution.

A production design should:

1. Reserve the parent slate once, preserving the existing unique RESERVE constraint.
2. Persist versioned child-order intentions, acknowledgements, actual fill cashflows,
   working liability and cancel/replace state. Use unique parent/tick/child intent IDs,
   venue execution IDs, transactional fill deduplication and restart reconciliation.
3. Freeze fills, refresh the free-leg payoff columns at currently executable prices,
   and condition the **whole slate** risk solve before emitting the next batch.
   `current_development/match_day_inference/replay_state.jl:2580–2860` contains the
   manual override design, not a ready-made partial-child scheduler. Its constrained
   scaling path is sequential SlateDrawdown only; do not silently substitute it for
   the simultaneous joint-mode constraint or assume frozen exposure is feasible.
4. Cancel confirmed outstanding orders at T−5, on invalid identities, stale inputs,
   suspensions, kill switch, or a failed risk certificate. Never release liability
   while a late venue fill remains possible.
5. Retain live/replay ledger isolation and require explicit production promotion.

**Allocation boundary:** preallocate concrete 3-tuples and mutable execution states
outside the hot loop. The prototype sweep is allocation-free when warmed; SQL,
DataFrames, depletion dictionaries, result collection and reporting are deliberately
outside that kernel. Preserve `OddsIndex` and `BookWorkspace`: map typed order IDs
at the boundary, refresh buffers in-place, and do not insert archive iteration,
network calls, dictionaries, or child persistence into Portfolio's score-grid loop.
The prototype measures its own allocation contract, not the entire application.

**Promotion gates:** exact price-specific liability and payoff tests; quote
filtration and shared-depth tests; replay determinism; unchanged baseline ledger
reproduction; full conditional-slate risk audit at every child batch; exchange tick
and minimum-size compliance; market-netted commission; crash/restart/idempotency
and late-fill races; paper_runbook isolation; then multi-day shadow validation. Before graduation, merge/rename the prototype
`TouchOnly` type to avoid an exported-name collision with `MD.TouchOnly`; prototype
call sites deliberately use the `MicrostructureExecution` namespace.
Until these pass, do not deploy the staged policy.

## 9. Verification and discovered production defect

The pure suite tests reservation identities independently, partial-level VWAP,
lay liability at actual odds, concave scalar Kelly optimality, quote filtration,
reused stamps, invalid/crossed books, deadline behavior, TWAP catch-up and warmed
kernel allocations. On local Julia **1.12.1**, **8,820/8,820** assertions passed,
including 8,634 independent feasible-grid utility comparisons and 12 warmed
zero-byte checks spanning both sides, all policies and residual-depth input. This
is the installed version, not the guide's stated 1.12.6. Run:

```sh
julia --project -t 8 current_development/market_microstructure_execution/test_microstructure_sweeper.jl
```

A production accounting mismatch was found while reading existing fill/settlement
code: lay slippage uses the back sign, deeper lay risk uses touch leverage,
settlement uses original odds, and CLV VWAP uses harmonic averaging. This is tracked,
**not fixed inline**, in [T008](../../docs/tickets/T008-multilevel-fill-price-accounting.md).
Prototype replay cashflow arithmetic must not silently be passed off as a literal
reproduction of those existing multi-level methods.
