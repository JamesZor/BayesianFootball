Why does log loss not move when transition fixes change the bias? Theory research (gpt-6-astra)
Role: research agent in .worktrees/BayesianFootball-qs-transition. Read-only except the one report file you write.
Goal: explain, with theory and our own numbers, why target log loss is flat across the transition arms, and what would make transition learning matter.
Done: write research/LOGLOSS_THEORY_REPORT.md; print PHASELL_DONE alone on its line.
Stop: no Julia, no DB, no git commands that write, no edits outside that report. Another agent is running in this worktree.

## 1. The question (James)

Five fixes to QualityStyleGRW (QS) for clubs that change tier were each run as a 60-fold grid
against the control:
- A2 tier term;
- an entrant market prior;
- a summer step;
- weekly refits;
- (pooled scope still running).

Each shifts the transitioning clubs' win probabilities by up to about 1–2 pp. **None moves target
1X2/all-market log loss by more than about ±0.001.** Why? Specifically:
1. Is log loss simply insensitive here? Consider: transition fixtures are a small share of the panel;
   a proper score is flat (second-order) near the optimum; signed errors cancel. Quantify each.
2. Is the **likelihood too strong**? qs_marketobs observes past market-implied log rates through a
   Gaussian with one fitted σ. Does that pin forecasts to the market so tightly that priors and
   transition terms barely matter? What does that imply for where the remaining error comes from?
3. **Do we need a term** that helps the model learn promotion/relegation moves, e.g. a
   transition-specific parameter with its own hierarchical prior, a likelihood weight, or
   decision-focused training? What does the literature say, and what would it predict for our data?
4. **Is log loss even the right test** for a fix that targets about 300 of 1,070 fixtures? Consider
   power, subset-restricted proper scores, paired designs, and the absolute-error metric we just
   adopted (DESIGN §9).

## 2. Context card

Read in full: `experiments/scotland/11_qs_transition/DESIGN.md` (§1–§4 and §9) and
`REPORT_S3.md` §2–§3. Skim `REPORT_S1.md`'s summary only.

- **Model:**
  - log λ_h = μ + γ + α_h + β_a; q = (α − β)/2, s = (α + β)/2;
  - zero-centred non-centred GRW at three scales;
  - qs_marketobs adds a Gaussian likelihood on past fixtures' inverted market log-rates (one σ).
- **Panel:** 1,070 target fixtures (23/24–25/26, Scottish League One/Two).
  - Transition cohorts (first 20 games): relegated into L1, 94 fixtures, 5 clubs; promoted into L1,
    112 fixtures, 6 clubs; plus League One/League Two moves and entered-SPFL.
- **Our numbers, all read-only** (use `duckdb -c` over CSVs; keep output under 40 lines):
  - `grid/scores_qs_{control,a2,entrant,summer,weekly}.csv`: harness scores, including
    `delta_logloss_vs_control` by subset and market, and transition bias;
  - `grid/observations_qs_*.csv` (where present): per-fixture probabilities. Use them to compute
    the actual per-fixture Δp and ΔLL contributions, and how much of the total ΔLL the transition
    fixtures carry;
  - `grid/bias_game_bins.csv`: bias by game bin per arm;
  - `eda/results/q1_club_bias.csv`: per-club bias (S1).
- **Library:** `~/.pi/agent/skills/research-library/bin/lib-grep PATTERN [-i]`, `lib-show DOC_ID PAGE`,
  `lib-sql "SELECT …"` (full paths; `lib-grep` has no `-E`). Topics: proper scoring rules and their
  sensitivity (Gneiting & Raftery), forecast evaluation power and Diebold–Mariano, local proper scores,
  hierarchical shrinkage and prior-vs-likelihood dominance, promoted-team literature.
- **Web:** `web_search`, `web_fetch`. Prefer papers; read abstracts and relevant sections only.

## 3. Deliverable: `research/LOGLOSS_THEORY_REPORT.md`

1. A summary of at most 12 lines that directly answers questions 1–4.
2. **Back-of-envelope analysis** with our numbers:
   - the second-order approximation ΔLL ≈ Σ δ²/(2p(1−p)) for a per-fixture shift δ;
   - the share of ΔLL from transition fixtures;
   - the size of improvement detectable at our n (a paired power estimate);
   - per-fixture Δp distributions by arm.

   Show each formula and the CSV it came from.
3. The likelihood-strength analysis: how the market observation's σ (compare the fitted values if
   they appear in any committed CSV; otherwise reason from the structure) limits how far the
   transition terms can move a forecast.
4. **Recommendations**, ranked: model terms (with equations), training or weighting changes, and
   evaluation changes. For each, the expected effect and how we would test it. Mark which are
   established in the literature and which are proposals.
5. Citations: library doc_id plus page, or URL plus title.

## 4. Budget and rules

- gpt-6-astra costs about 5× Sol. Keep context under about 120K and use at most about 20 web fetches.
- Write the report as you go, so a compaction loses nothing.
- **Don't run `git add`, `git commit` or any write command except creating your report file.**
  The manager commits it.
- If a scope question blocks you, use `ask_user_question`.

**After a compaction:** re-read §3, then your report so far; continue from where it stops.
