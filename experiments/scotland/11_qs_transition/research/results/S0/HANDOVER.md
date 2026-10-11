# Stage 0 handover

Status: DONE. Deliverables committed locally on `exp/qs-transition` in **3e34aaf8**. No push/merge.

## Outputs

- `research/RESEARCH_REPORT.md`: <=12-line opening summary, five question sections, verified source register and evidence limitations.
- `research/EDA_QUESTIONS.md`: 12 concrete tests; priorities Q1–Q5 cover bias decay, chronological prior-only value, tier identification, non-mover summer variance and refit lag. Run Q6 coverage and Q12 provenance as prerequisites.
- `research/PRIOR_FORMS.md`: three q/s candidates, each with chronology, uncertainty, no-history fallback, literature basis and EDA parameters.

## Decisions for the manager / Stage 1

1. Compare cohort-only P1, club carry + net empirical shift P2, and explicitly linked tier rebase + residual jump P3. No numerical prior or winning form chosen in Stage 0.
2. Check units: under the brief's split convention A2 total step 0.47 corresponds to q=0.235, not q=0.47. Dixon–Coles historical English division contrasts are q≈0.245/0.268/0.138, not Scottish mover effects.
3. One match quote identifies quality difference and style sum, not an isolated club's q/s. Prior-only opening tests need frozen pre-quote opponents or direct match-level scoring; disconnected leagues need links.
4. Freeze market prior inputs to seasons < target season. Do not backdate an S−1-informed mean to the beginning of a much earlier training window. Audit historical market-row reuse in both prior and QS likelihood.
5. No robust Scottish mover betting inefficiency or universal settling time established. Mean correction, summer variance and refit lag remain distinct hypotheses.

## Verification and limitations

Structural assertions and arithmetic checks passed; details in `logs/verification.md`. `git diff --cached --check` passed before the deliverable commit. Mandatory `./scripts/todo.sh check` failed on an existing invalid Assignee in TODO 046; the manager explicitly approved committing research-only with that failure documented. No out-of-scope metadata changes were made.

Source limits: ClubElo's original method page redirected, so its mover algorithm is not asserted; SinceAWin is a separate practitioner example. SAGE/SSRN efficiency pages were blocked; only a verified RePEc abstract is cited for that paper. Ridall's converted forgetting-factor table headers are ambiguous, so no numeric factor is imported. Egidi's fetched arXiv inversion equations have a complementary-constraint caveat. Sources distinguish established methods, limited observations and our proposals.

All work is under `research/` by manager-approved records exception. The absent hub experiment-card path is replaced by these deliverables. Untracked `eda/` appeared during work and was left untouched. No Julia, betdb, fits, src edits or 26/27 outcome analysis. No open jobs/panes.

Next action: manager incorporates selected questions into Stage 1; this worker stops with `PHASES0_DONE`.
