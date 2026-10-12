# Manager decisions (overnight run, TODO 046)

Each gate decision the manager took for James, with its reason. James reviews these in the morning.

- **D0 (2026-10-11)** The S0 research uses gpt-6-astra, capped at about 25 web fetches and 120K context.
- **D1 (2026-10-11 ~23:00) S0 accepted.** It delivered 12 EDA questions, 3 prior forms (P1 cohort
  prior, P2 club carry plus mover shift, P3 tier rebase plus club jump) and a cited report. Reviewed
  for scope; no fits, no DB.
- **D2 (2026-10-11) S1 question list = S0's Q1–Q12.** They supersede the base set where they overlap.
  - Order: Q6 and Q12 (prerequisites), then ★ Q1–Q5, then Q7, Q10, Q8.
  - Q9 (cups vs bookmaker) and Q11 (squad discontinuity) are optional if time is short.
  - Reason: Q1–Q5 decide the cause verdict and the prior. Q9 and Q11 need new data plumbing (cup
    bookmaker odds, lineup minutes) for secondary questions.
- **D3 (2026-10-11) S1 runs on gpt-6.1-sol with thinking high** (diagnosis). Astra used about 24%
  of the 5h window in 10 minutes, so it is not used for the build stages.
- **D4 (2026-10-11 ~23:55) S1 accepted** (commits up to `e319eb1f`). Reproduction checks pass: W2
  cohorts 5 / 6 / 2, 1,070 and 710 fixtures, 8 DB audits unchanged, 0 chronology violations.
  - **Cause verdict as S1 reports it.** For the W2 bases, (b) tier transport is the main cause, with
    some (a) cold start for clubs without history. For QS on its shorter panel, the bias is small
    and **positive in both directions** (+1.50 relegated, +1.44 promoted). There is
    promotion-specific (d) lag (+1.68 pp/week). (c), a summer step, is not securely identified.
  - The cohorts are tiny (QS 3 / 4), so none of this is a powered mechanism test. That is why
    the S3 arms on W2's longer panel matter.
- **D5 Evidence policy for the market prior (S1's open blocker): an "entrant prior".**
  - The market prior applies **only to clubs with no in-scope history in the fold's training
    window**, i.e. clubs entering League One/Two from the Championship or Premiership.
  - It is built from their **out-of-scope (54/55) closes** of seasons strictly before the target
    season. QS's market likelihood is lower-scope only, so the overlap with the likelihood is zero
    **by construction**; S2 must verify that count = 0 as a hard check.
  - Clubs moving between League One and League Two already have in-scope history and get no market
    prior. They are handled by A2 (F3) and the likelihood.
  - Clubs with no out-of-scope market history fall back to the A2 tier position.
  - Reason: this is disjoint and exact, not empirical Bayes; it removes S1's double-counting
    blocker; and it targets exactly the clubs that start cold. An overlapping EB arm is not built.
- **D6 Prior form.** S1's A2-centred P2 mean transport: the club's last out-of-scope market q/s plus a
  direction shift, centred on A2's step, with parameters from movers in earlier seasons only. Unit
  carry first; no free q/s covariance. The SD comes from S1's prior-only residuals.
- **D7 S2 build list, in S1's order with D5 applied:**
  1. the control (qs_marketobs) on W2's panel;
  2. QS+A2 composition (F3);
  3. the entrant prior (F1+F2) with a provenance manifest;
  4. the QS target-season step (F4, opt-in);
  5. a check whether weekly refits (F6) can be separated from the weekly dynamics clock.

  Pooled scope, cups and pxG (F5/F7) are deferred, because S1 found the cup identification not
  cleared. The manager revisits one pooled arm (lower + Championship, goals only) at the S3 → S4
  gate, since James asked for scope arms.
- **D8 S3 arm design: one factor at a time** vs the control on W2's panel: F3, F1+F2 (entrant prior),
  F4, F6 (if separable). Combinations of the passing factors go to S4. James was unsure; this is the
  manager's recommendation from DESIGN §5.
- **D9 (2026-10-11 ~23:55, S2 question) Overlap rule for F1F2.** **Zero overlap applies to the
  entrant's own endpoint quotes** (its out-of-scope 54/55 closes). The **transport (shift) parameters
  keep S1's estimation from earlier movers' new-tier quotes**, in seasons before the target season.
  Their overlap with the fold's market likelihood is **reported separately**, and the arm is labelled
  "exact entrant endpoint, EB transport hyperparameters".
  - Reason: a shift into League One can only be estimated from movers' League One quotes. Using
    54/55 only would estimate a different transition (Premiership ↔ Championship).
  - Those quotes set a population-level hyperparameter, not the entrant club's own state. That is
    ordinary empirical-Bayes use, not double-counting a club's evidence.
  - No look-ahead either way.
- **D10 (2026-10-11 ~23:56, S2 question) The entrant prior lives in the model, at the club's entry
  node, not as an OOS override.**
  - pi proposed drawing the prior only at prediction time for clubs with no training rows. That acts
    only in the club's first fold. From the second fold the club has rows and QS restarts it at the
    league mean, so the cold start returns. S1 found the bias persists to games 11–20.
  - Rule: eligible = no in-scope matches in seasons **before the target season**. For every fold of
    that season, the club's first in-scope state (its entry node at its first target-season period,
    never backdated to the window start) gets the transported mean and SD instead of the zero-centred
    `z_init` draw.
  - Zero-centring stays over incumbents only (or the entrant offset is subtracted), proved by a test
    that incumbents' fitted states don't move when the entrant prior mean moves.
  - Opt-in, with all src gates. If it isn't feasible in S2, the stage stops PHASES2_BLOCKED.
- **D11 (2026-10-12 ~00:05, S2 question) Zero-allocation regression in the unchanged control.**
  - The harness smoke fails `tape_allocation`: 57,568 B per gradient on the 23/24 opening fold, and
    the original wave-2 config allocates 86,544 B. Wave 2 passed this same check at 0 B
    (`06_.../PHASE0.md:16`), so this is a regression in code or packages.
  - Ruling: investigate and repair, time-boxed to 60 minutes. First environment vs code (run the
    audit on the wave-2 src, compare Manifests), then bisect, or pin a package if that is the
    documented fix.
  - **Gates are not weakened:** no limit change, no skipped check, no gate-code change. If it isn't
    fixed in time, S2 stops PHASES2_BLOCKED with `build/ALLOCATION_BLOCKER.md`, and the chain stops
    there for James.
- **D12 (2026-10-12 ~00:40) The beast is unreachable over Tailscale.**
  - S2's control and QS+A2 smokes had passed all hard checks, which implies D11's allocation problem
    was resolved; see S2's report.
  - The new components (`e2e956cf`) are committed but ungated.
  - The manager diagnosed it: the beast is up (sshd answers on the public IP, key not authorised
    there), but the Tailscale path returns nothing (rx 0).
  - Ruling: wait until 03:30, with a low-cadence connectivity watch, then continue; otherwise
    `HANDOVER.md` and PHASES2_BLOCKED.
- **D13 (2026-10-12 ~09:55) S2 accepted** (`f471fd63`). Every source gate passed on the first resume
  attempt:
  - units 46 + 373 assertions;
  - exact AD ≤ 1.6e-15;
  - FD convergence;
  - 0 B tapes;
  - 3-seed recovery 86/90;
  - post-src legacy bit identity;
  - entrant overlap 0;
  - incumbent invariance.

  All four smokes pass their hard checks. Smoke R̂ of 1.05–1.14 at 2 × (200 + 200) is expected; S3
  uses 4 × (500 + 1,000).
- **D14 (James, 2026-10-12 ~10:00):**
  - archive the stale harness row `harness_checks.id=832` (back it up to CSV first; nothing else);
  - S3 runs **all four arms** sequentially, control first (about 22 h);
  - **build F6** (an opt-in fold-cadence field) **alongside S3**, as a fifth arm afterwards;
  - **plan a pooled scope arm** (League One/Two + Championship, goals only) for S4.
- **D15 (2026-10-12 ~10:17, F6 question) Weekly fold cadence goes in a backward-compatible scope
  wrapper, not a new `DataScope` field.**
  - Appending a field broke Julia deserialisation of saved fits: the wave-2 `load_fit` read the next
    object as the new field.
  - A wrapper keeps the legacy type, the serialized layout and the default recipe hashes exact for
    unset scopes. It needs only small dispatch changes in the splitter and harness.
  - The same lesson as the 2026-09-03 joint-obs artifact drift.
- **D16 (2026-10-12 ~10:50) F6 accepted** (`exp/qs-transition-f6` at `db9f089b`, pushed).
  - 505 units; 1,368 legacy split signatures unchanged; wave-2 bit identity; 115 weekly folds with
    the same 1,070 fixtures; strict filtration; the smoke passes its hard checks at 0 B.
  - Estimated weekly grid: 5.65 h.
  - **The weekly grid is queued after S3's four arms,** run from the F6 beast checkout
    `/root/BF_runs/qs_transition_f6`, so the S3 worktree and checkout stay untouched mid-run. It is
    scored against `qs_control` in the same harness namespace.
  - **The branch is merged into `exp/qs-transition` only after S3 finishes,** so S3's worktree isn't
    changed under a running agent.
- **D17 (2026-10-12 ~11:00) Pooled-scope arm design** (S4 prep, built in parallel on branch
  `exp/qs-transition-pooled`, 4 beast threads):
  - `qs_pooled_a2` = `qs_a2` trained on League One/Two **and Championship goals**, scored on
    56/57. Market observation stays 56/57 only.
  - A2's tier term is required, because W1 showed that pooling without tier structure compresses
    tiers and worsens the transition bias.
  - **Not combined with the entrant prior.** Pooling observes the Championship directly, so the
    entrant prior would double-count it. Pooling is an alternative route for clubs relegated from the
    Championship.
  - Cups and pxG stay out (S1 did not clear the cup identification).
- **D18 (2026-10-12 ~11:02, pooled-prep question)** The recovery runner failed twice before
  producing draws, both from bugs in its own synthetic generator (an empty jump array typed
  `Vector{Real}`; a reused `delta` variable). Neither was a model or gate failure. Ruling: fix the
  generator and continue. The stop-after-two-failures rule protects against weakening gates, not
  against fixing test-harness bugs.
- **D19 (James, 2026-10-12 ~11:30) Standing rule for stale allocation rows.**
  - A stale failed `tape_allocation` harness check may be removed only if all of these hold: it is
    backed up to CSV, its `run_id` is NULL, and a later smoke of the same recipe hash passes every
    hard check at 0 B. Each removal is logged here.
  - Removed under this rule: **row 890** (pooled arm's first attempt). Row 832 was removed under D14.
  - **Harness follow-up:** the first allocation measurement in a fresh REPL sometimes reports
    allocation, and a rerun is 0 B (twice now: D11/832 and 890). The check probably needs a warm-up
    call before measuring.
- **D20 (2026-10-12 ~14:45, S3 question)** The control and QS+A2 grids are complete and scored. The
  entrant grid failed twice before sampling with a dispatch error in the experiment-local prototype
  feature extractor (world age / loading order). Ruling: diagnose the loading and retry with the
  prototype preloaded, with no src or recipe change. If it fails again, document it and run summer
  first.
- **D21 (2026-10-12 ~14:40) Interim S3 result, and the pooled prep accepted.**
  - **Control (qs_marketobs) on W2's panel**, transition bias over the first 20 games:
    relegated into L1 **+0.07 pp** [−2.69, +2.48]; promoted into L1 **+1.29** [+0.78, +1.91].
    That is far smaller than W2's bases (−4 to −5 / +3 to +4.5). The market observation already
    removes most of the in-panel bias.
  - **QS+A2 fails gate 2.** Promoted improves (+0.54; Δ|bias| −0.75 [−1.16, −0.37]), but relegated
    worsens (+1.26; Δ|bias| +1.19 [−1.85, +1.53]). Gate 1 passes (ΔLL +0.0003 [−0.0004, +0.0011]).
  - **Pooled prep accepted** (`exp/qs-transition-pooled` at `c127e362`, pushed): panel 1,070;
    Championship goals market-masked; 3-seed tier recovery 83/90; smoke hard checks pass at 0 B;
    grid estimated at 4–6 h. Row 890 was removed under D19.
  - **Grid queue after S3:** weekly (F6 checkout), then pooled (pooled checkout). Both are scored
    against `qs_control`.
  - Note for S4: A2 failed gate 2 on QS, and the pooled arm includes A2 by design (D17). Interpret
    the pooled result with that in mind.
- **D22 (2026-10-12 ~17:25) S3 accepted** (`92400e87`). Every fix passes gate 1; **none passes gate 2.**
  - A2: relegated +1.26 / promoted +0.54.
  - Entrant: +0.76 / +1.17.
  - Summer: −0.46 / +0.68.
  - Control: +0.07 / +1.29.
  - S3 notes that the control's near-zero signed relegated bias comes from cancellation: early games
    are over-rated (+0.63 first 10), later games under-rated. The frozen |mean| gate therefore
    penalises any change. **The gate stays frozen.** A possible secondary metric (|bias| by game bin)
    is James's decision and must be labelled post hoc.
  - F6 and the pooled branch were merged into `exp/qs-transition` (`60d9a81b`, one trivial conflict in
    candidates.jl: both scopes kept).
  - **S3b launched:** the weekly and pooled grids in `%118`, scored identically vs `qs_control`.
- **D23 (James, 2026-10-12 ~17:45) Absolute transition-error metric adopted** (DESIGN §9). The
  primary measure is the mean per-club |model − market| over the first 20 games; the secondary
  measures are the worst club and the splits for games 1–10 and 11–20. It is post hoc on the panel
  and pre-registered for 26/27, and the frozen gate stays reported. S3b computes it for all six arms
  after its grids.
