QS transition experiment, Stage 0: literature and web research to shape the EDA
Role: research agent (gpt-6-astra) in .worktrees/BayesianFootball-qs-transition, branch exp/qs-transition.
Goal: find what is known about rating clubs that change division, and turn it into EDA questions and prior forms.
Done: §4 DONE WHEN; write research/RESEARCH_REPORT.md and research/EDA_QUESTIONS.md.
Stop: print PHASES0_DONE alone on its line when done. No Julia, no DB, no model fits, no src edits.

## 1. Why

James's Scottish League One/Two model (`QualityStyleGRW`, "QS") misprices clubs that change tier.
Relegated clubs are under-rated and promoted clubs over-rated. In the 26/27 paper trade, bets on
transition-club fixtures returned −39%. Read `experiments/scotland/11_qs_transition/DESIGN.md` §1–§3
and §6 (about 150 lines; this is the one file to read in full). It lists the candidate causes (cold
start, tier scale, no summer step, lag) and fixes (F1–F7).

The EDA (Stage 1, a separate Sol agent, tonight) will test those causes on data. Your job is to
make it ask the right questions and to give it candidate prior forms grounded in the literature.

## 2. Context card

- **The model.**
  - Rates: log λ_h = μ + γ + α_home + β_away; log λ_a = μ + α_away + β_home.
  - Rotation: q = (α − β)/2 (quality), s = (α + β)/2 (style).
  - Team states follow non-centred, zero-centred Gaussian random walks at three scales
    (initial, season, biweekly), with ratio r = σ_s/σ_q ≈ 0.3.
  - qs_marketobs also observes past matches' **market-implied goal rates**: per-match KL inversion
    of de-vigged Betfair closing books to a double Poisson.
- **What exists.**
  - W2's A2 tier term: tier step s ≈ 0.47 log-goals per tier, 48/52 attack/concession split,
    full-pyramid anchor, carry κ ~ Beta(1.5, 4), club jump η.
  - TODO 029 transition priors, e.g. relegated into L1: α N(+0.12, 0.20²), β N(−0.16, 0.26²). These
    leaked panel market data, which is why every market prior must be built per fold from earlier
    seasons.
  - The CLV napkin found that the market rates a club early in a season from roughly last-season
    goal difference plus the tier change (R² 0.70).
- **Data available to the EDA:**
  - Scottish tiers 54–57, 21/22–25/26 inverted closes (no Championship 25/26);
  - cups with bookmaker odds but no Betfair;
  - lineups;
  - results back to 2008 (`sofascore.events`).
- **Library:** James's research library (about 330 papers and books on football models, Bayesian
  statistics and state-space models). Use the helpers by full path:
  `~/.pi/agent/skills/research-library/bin/lib-grep PATTERN [-i]`, `lib-show DOC_ID PAGE`,
  `lib-sql "SELECT …"`. `lib-grep` has no `-E`. Cite as doc_id plus page.
- **Web:** `web_search` and `web_fetch`. Prefer papers (arXiv, journals, SSRN) and the betting
  literature (e.g. Dixon & Coles, Koopman & Lit, Rue & Salvesen, Constantinou, Hvattum & Arntzen,
  Egidi, Baio & Blangiardo, Elo/pi-ratings, promoted-team studies, market-efficiency studies of
  promoted clubs). Read abstracts and the relevant sections only.

## 3. Questions to research

1. **Promotion and relegation effects.** What does the literature find about the strength of
   promoted and relegated clubs relative to their new league, how quickly it settles, and whether it
   is asymmetric? Are there numbers (goal-rate or rating shifts) we can compare with A2's 0.47 step
   and TODO 029's priors?
2. **How rating systems handle movers:**
   - Elo-type systems (Club Elo, pi-ratings) across divisions;
   - dynamic Poisson / state-space models: initial-state priors, between-season variance inflation;
   - hierarchical league models: league effects, cross-league linking through cups and movers.
3. **Market-informed priors.** Using bookmaker or exchange prices as priors or observations for team
   strength (pre-season odds, outright odds, market-implied ratings). How do they avoid look-ahead?
   Are there results on whether markets misprice promoted or relegated clubs (an inefficiency we
   could exploit, or one we are on the wrong side of)?
4. **Cold-start and partial pooling:** priors for new entities in hierarchical time-series models,
   such as empirical Bayes from past cohorts, or group means by entry type. Which forms suit a
   zero-centred random walk?
5. **Between-season dynamics:** evidence on summer variance (squad turnover, manager change) and how
   models inflate state variance at the boundary.

## 4. DONE WHEN

1. `research/RESEARCH_REPORT.md`: a summary of at most 12 lines first, then one section per question.
   Every claim carries a citation (library doc_id plus page, or URL plus title). Separate what is
   established from what is suggestive.
2. `research/EDA_QUESTIONS.md`: **at most 12** concrete, testable questions for the EDA. For each
   question give:
   - the hypothesis;
   - the statistic to compute;
   - which data it uses;
   - what result would support or refute it;
   - which cause (a–d) or fix (F1–F7) it informs.

   Mark the 5 most important.
3. `research/PRIOR_FORMS.md`: 2–4 candidate prior forms for F1/F2, written as equations on q/s. Each
   gives:
   - its per-fold, no-look-ahead construction;
   - its fallback for clubs without market history;
   - its literature basis;
   - the parameters the EDA must estimate.
4. Commit all three on `exp/qs-transition` (local commits only; the manager pushes). Print
   `PHASES0_DONE`.

## 5. Budget and rules

- gpt-6-astra costs about 5× the Sol models. Keep context under about 120K. Don't read whole papers:
  read abstracts and the relevant pages, and use at most about 25 web fetches.
- Write findings to the files as you go, so a compaction loses nothing.
- Don't run Julia, query betdb, or edit anything outside `experiments/scotland/11_qs_transition/research/`.
- If you are stuck on a scope question, use `ask_user_question` (the manager answers).

**After a compaction** (your context starts with `[Session Goal]`): re-read this brief's §4, then
the files you have written in `research/`. Continue from where they stop; don't redo searches whose
results are already recorded.
