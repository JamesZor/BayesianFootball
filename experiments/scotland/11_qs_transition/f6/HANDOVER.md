# F6 handover — PHASEF6_DONE

Branch/worktree: exp/qs-transition-f6, /home/james/bet_project/.worktrees/BayesianFootball-qs-transition-f6. Local commits only; no push/merge/grid.

## Complete — do not repeat

- Opt-in public `fold_col`, default equal to dynamics. Approved backward-compatible `FoldCadenceScope` wrapper preserves original nine-field DataScope artifacts/default representation and positional API.
- Weekly held-out/boundary clock; unchanged biweekly QS state indices. Same-biweek forecasts read the fitted state without new innovations. No new model parameters.
- `qs_weekly` is the exact control model/sampler plus weekly scope cadence. Final505 unit assertions pass; all1368 default-off split signatures identical across W2/wave2 panels and four existing policies, all three clocks.
- Wave2 reference81fdf817-f82d-4f66-9bfd-5f456c4fe29f:396000 chains /160000 latents byte-identical, maxdiff0, original seeds. Final reproduction job102604-1090.
- Frozen weekly panel115 folds /1070 same fixtures; all115 strict day filtration checks pass. 55 folds /531 fixtures reuse an observed biweek.
- Normal two-fold smoke103140-600 passes all6 hard checks. UUID9346d3b3-874c-49eb-9bc3-0db9f07de477; recipeaaed32f47a4c97e032bfb067b6debd2e4ccc14d934df58301955f4bccffae952. Receipt103819-27315 proves exactly0 B tapes and DB readiness.
- Convergence/performance REVIEW failures are disclosed, not bypassed: Rhat1.06250; ESS67.14/75.45;0 divergences; efficiency ratio2.08675, GC9.72%.
- REPORT_F6.md has ≤10-line summary, gate evidence, actual fold count, exact manager-only grid command, timing estimates and assumptions. Ideal16-thread sampling estimate115-fold5.65h;60-fold2.95h, not measured grid timing.
- Owned jl f6 STOPPED; no open jobs/watches or S3 checkout/session changes. No gate failed twice. Failed first legacy load and wrapper-panel test introspection attempts are logged, with their fixes/approval in DECISIONS_LOG.md.

## Provenance / retained outputs

Source commitsbb12dee7 (initial), fbfb86a1 (compatibility wrapper); gate/runner-fix evidence91bcd7d8. Final smoke/report evidence is in the following local commit on this branch. All CSVs and selected logs are under f6/; raw arrays are not committed.
Owned beast checkout/root/BF_runs/qs_transition_f6 is synced from local bundles, not pushed. Legacy checkpoints moved to/root/BF_runs/logs/f6_legacy_checkpoints; manifest copied here. Dirty run metadata reflects generated CSVs/untracked checkpoints, not unsampled source changes.
Frozen Pyramid SHAa1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12 reverified. Source defaults remain off.

## Manager next

Review retained REVIEW flags and schedule the weekly grid only after S3, using REPORT_F6.md's exact command on the final synced checkout. Normal harness gating remains enabled. No result/promotion claim is made for the weekly arm until that grid is run and evaluated. Records live here (hub directory absent).
