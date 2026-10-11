# Phase A handover — market APM on closes

## Status

Phase A is complete. Phase B was not started. Required CSVs, all league figures and the pooled figure, and `REPORT_A.md` are committed on `eda/market-apm-closes`.

Analysis/results commit: `fd3171b443e57b94e1c3117676e079ab051badc2`.
Earlier steps: `30b05507` (scope/hashes), `5947c8ff` (coverage).

**Push remains a manager action.** The user explicitly approved local-only commits because worker rules prohibit pushing. The user also approved using this report/handover as the experiment card instead of creating the absent `~/bet_project/hub/experiments/` directory.

## What is done

- Verified all three pinned datastore hashes; reused the original accepted inverted closes without database access or reinversion.
- Per-league × season fixture/close/XI coverage, failed accepted match IDs and reasons, and a stage-separated failure summary.
- Player starts, club movers, team-season usage concentration, consecutive-XI overlap, and absorbed two-rate starter-design rank.
- Team-season attack/defence least squares with a league home effect; residual supremacy/log-total headroom.
- Strictly historical regular definitions, current-absence slopes, immediate-next-XI placebo, immediate-previous-close form control, common-placebo-sample and complete-history sensitivities.
- Both match-HC1 and small-season-block CR1 uncertainty; typical-line double-Poisson probability translations.
- Binned figures and their CSV data; joined-match audit columns; numerical rank cross-checks and environment versions.

## Main findings and Phase B recommendation

Read the summary first in `../../REPORT_A.md`. All scientific numbers trace to the committed CSVs.

- In 56/57, extra home missing regulars associate with −0.02593 / −0.03154 log supremacy, approximately −0.84 / −1.01 home-win percentage points. England is similar.
- Previous-close form control attenuates the association, but **the next-match placebo is not generally null**. Do not label these player effects.
- Joined Scottish samples are 809 / 647; absorbed starter dimensions are 1,456 / 1,206. League Two's design saturates the team-adjusted observation dimension, so raw rank does not imply stable player ratings.
- Primary total-absence tests do not establish a consistent effect.

**Recommendation:** only a conditional, strongly regularized exploratory Phase B after addressing persistence/team-form confounding. First compare current and next absences on common samples with explicit time-varying controls. Then consider mover-aware Scottish tier pooling with shared player identity and separate league/team baselines. Require low-start uncertainty, regularization sensitivity, and temporal holdout stability. Do not proceed directly to an unrestricted player model or betting claim.

## Verification

- `r01_checks.jl`: 11 synthetic causal-history checks plus 97 output-contract checks, all pass; job `013549-7917`.
- `r01_rank_checks.jl`: 16 tolerance/dense-rank checks, all pass; job `013135-6144`.
- Final analysis job `013213-31593` and coverage-summary job `013327-19065`: ok.
- All nine SVGs parse as XML. `git diff --check` and `./scripts/todo.sh check` pass.
- Logs are committed under `logs/`; environment versions are in `environment.csv`.

## Runtime and exact resumption information

The owned beast REPL **mapm is stopped**. No active jobs or owned open panes remain. No other tmux session or checkout was touched.

Retained owned compute checkout: `/root/BF_runs/market_apm`, branch `eda/market-apm-closes`. It was initially cloned from a bundle at `30b05507`; code and generated results were subsequently mirrored there by explicit scp/rsync, not committed remotely. The local committed branch is the canonical source; synchronize it before a future rerun. The three verified cache copies and the copied beast Julia manifest remain in that owned checkout; no `.env` was copied.

For an authorized reproduction after synchronization:

```sh
jl start mapm --beast --threads 16 --dir /root/BF_runs/market_apm
jl run mapm eda/market_apm_closes/r01_market_apm.jl
jl run mapm eda/market_apm_closes/r01_rank_checks.jl
jl run mapm eda/market_apm_closes/r01_checks.jl
jl stop mapm
```

Retrieve only generated outputs, excluding `PROGRESS.md`, `DECISIONS_LOG.md`, and `HANDOVER.md`; otherwise stale remote notes can overwrite canonical local progress. The earlier overwrite was repaired before committing.

## Open questions / limitations

- How much current-absence signal remains after controlling persistence rather than merely previous residual form? No clean causal identification is demonstrated.
- Regular-count proxies do not distinguish player quality or position. Missing earlier XIs affect historical regular rankings; complete-history sensitivity is provided.
- Fixture ordering is by date/match ID; history quality is limited to the pinned recorded fixtures. A mover is any multi-club starter in that league's joined sample, not necessarily a contemporaneous transfer.
- The two-rate attack/defence design rank is not a signed-supremacy-only rank. Some individual coefficients remain unidentified even when the fitted span is large.
- Season-block intervals have few blocks; residual-response uncertainty and multiple testing are not fully accounted for.
- The response-only residualization follows the brief, not full FWL partial regression. A later rating design needs an explicit estimand.
- Starting XIs are safe for this close regression, not automatically for an earlier forecast.

Stop here. No Phase B work is authorized by this brief.
