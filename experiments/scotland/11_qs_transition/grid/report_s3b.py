"""Render S3b's report from committed CSVs; never changes scores or gates."""
import csv
from pathlib import Path
P = Path(__file__).resolve().parent

def load(name):
    with (P / name).open() as f:
        return list(csv.DictReader(f))

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])

def signed(value, digits=3):
    return f'{float(value):+.{digits}f}'

def interval(row, value, lo, hi, digits=3):
    return f'{signed(row[value], digits)} [{signed(row[lo], digits)},{signed(row[hi], digits)}]'

names = {'qs_control': 'Control', 'qs_a2': 'A2, post-rerun', 'qs_entrant': 'Entrant',
         'qs_summer': 'Summer', 'qs_weekly': 'Weekly', 'qs_pooled_a2': 'Pooled A2, post-rerun'}
runs = load('runs.csv')
gates = load('gate_table.csv')
abs_rows = load('abs_error_table.csv')
cohorts = load('cohort_bias.csv')
bins = load('bias_game_bins.csv')
conv = load('convergence.csv')
timing = load('timing.csv')
assert len(runs) == len(gates) == len(conv) == 6
assert len(abs_rows) == 54
sections = ['''# Stage 3b — weekly and pooled grids complete; neither clears both gates

## Summary (9 lines)
- Weekly uses its unchanged 115-fold recipe; James approved the clarification to the brief's 60-fold title.
- Pooled A2 uses 60 folds; both arms score the frozen 1,070 lower-tier fixtures in 23/24–25/26.
- Weekly and pooled both pass LogLoss non-inferiority, but both fail frozen transition-bias Gate 2.
- Weekly needs no reruns; pooled folds 41,45,51,53 each received exactly one seeded rerun.
- Final weekly and pooled fits have no offending folds under James's R̂/divergence rule.
- All six arms also fail the separate **post hoc** DESIGN §9 absolute-error criterion.
- Weekly has the largest primary relegation MAE point reduction (−0.329 pp), but its CI crosses zero.
- Entered-SPFL remains one priced club-season; degenerate bootstrap CIs do not imply certainty.
- No recipes, priors, sampler, frozen gates or src changed; no 26/27 outcomes scored.

## 1. Frozen gate table — combined six arms

Scoring is S3's v1.2, B=10,000, panel `56+57|23/24,24/25,25/26|n=1070`,
paired against `qs_control` `866a947c-be65-467d-8dfd-c58aed0ad1f9`.
Gate 1: fixture-clustered all-market ΔLL upper 95% CI <0.002.
Frozen Gate 2: both first-20 direction absolute signed-mean biases must decrease,
with at least one paired club-season-clustered Δ|bias| CI excluding zero. Bias units pp.
The control's signed aggregate relegation bias is near zero by cancellation; this gate
is intentionally retained, not substituted after seeing results.
''']
gate_rows = []
for r in gates:
    arm = r['arm']
    bias_cols = []
    for d in ['relegated_into_L1', 'promoted_into_L1']:
        bias_cols.append(signed(r[d+'_bias_pp']) + '; ' +
                         interval(r,d+'_delta_abs_pp',d+'_lo',d+'_hi'))
    is_control = arm == 'qs_control'
    gate_rows.append([names[arm],interval(r,'delta_logloss','ll_lo','ll_hi',6),
                     'reference' if is_control else ('PASS' if r['gate1']=='True' else 'FAIL'),
                     *bias_cols, 'reference' if is_control else ('PASS' if r['gate2']=='True' else 'FAIL'),
                     '—' if is_control else ('PASS' if r['both_gates']=='True' else 'FAIL')])
headers = ['Arm','ΔLL [CI]','Gate1','Relegated bias; Δ\\|bias\\| [CI], pp',
           'Promoted bias; Δ\\|bias\\| [CI], pp','Gate2','Both']
sections.append(table(headers,gate_rows))
sections.append('### S3b rows only\n\n'+table(headers,gate_rows[-2:]))
sections.append('''## 2. Post hoc DESIGN §9 absolute transition error (D23)

James approved D23 after S3. This is **post hoc on 23/24–25/26**, not a replacement
for the frozen table above; it is pre-registered only for the later 26/27 test (not run here).
Primary `MAE_club` = mean across club-seasons of each club-season's mean
`|P_model(win) − P_market(win)|` over its first 20 league games. Absolute value is taken
**before** the per-club mean; club-seasons receive equal weight, regardless of priced game count.
Δ vs control uses paired resampling of club-season clusters, B=10,000, percentile 95% CI,
seed 20260927. Resample whole cluster summaries, never independent arm samples.
Post-hoc criterion: neither relegated/promoted Δ point is positive and at least one CI upper <0;
Gate 1 is still required. Entered-SPFL is descriptive, not a third gate direction.

All six fail this criterion. A2, entrant, weekly and pooled improve both direction points,
but no direction CI upper is negative. Summer worsens both primary points.
Worst-club is the maximum club-season `|mean signed error|`, **not** maximum MAE.
Units throughout are pp. Full precision, UUIDs, cluster/observation counts and all splits:
`grid/abs_error_table.csv`; per-game paired inputs: `grid/transition_errors.csv`.
''')
primary = []
for r in abs_rows:
    if r['game_split'] == '1-20':
        primary.append([names[r['arm']],r['direction'],f"{float(r['mae_club_pp']):.3f}",
                        interval(r,'delta_mae_club_pp','lo','hi'),
                        f"{float(r['worst_club_abs_mean_pp']):.3f}",r['n_club_seasons']])
sections.append(table(['Arm','Cohort','MAE_club','ΔMAE [CI]','Worst-club magnitude','Club-seasons'],primary))
split_rows = []
for run in runs:
    for d in ['relegated_into_L1','promoted_into_L1','entered_spfl']:
        pair = [next(r for r in abs_rows if r['arm']==run['label'] and r['direction']==d and r['game_split']==s)
                for s in ['1-10','11-20']]
        split_rows.append([names[run['label']],d,*[
            f"{float(r['mae_club_pp']):.3f}; "+interval(r,'delta_mae_club_pp','lo','hi') for r in pair]])
sections.append('### Post hoc early/late primary metric\n\n'+table(
    ['Arm','Cohort','Games1–10 MAE; Δ [CI]','Games11–20 MAE; Δ [CI]'],split_rows))
sections.append('''Only 5 relegated and 6 promoted club-seasons are priced. Entered-SPFL has one
club-season (7 first10 /15 first20 observations), so all its paired cluster intervals
collapse to a point. That is a sample limitation, not strong evidence of an effect.
''')
sections.append('## 3. First-10 and entered-SPFL signed-bias diagnostics\n')
diag = []
for run in runs:
    a = run['label']
    def bias(direction,n):
        return next(r['value'] for r in cohorts if r['run_id']==run['run_id'] and
                    r['subset']==f'transition_{direction}_first{n}')
    diag.append([names[a],signed(bias('relegated_into_L1',10)),signed(bias('promoted_into_L1',10)),
                 signed(bias('entered_spfl',10))+' / '+signed(bias('entered_spfl',20))])
sections.append(table(['Arm','Relegated first10, pp','Promoted first10, pp','Entered-SPFL first10 /first20, pp'],diag))
sections.append('First10 priced observations: relegated46 (44 fixtures), promoted55 (53 fixtures). '
                'Full CIs/counts remain in `grid/cohort_bias.csv` and per-arm score exports.')
sections.append('## 4. Bias-by-game-bin curves\n\nSigned model-minus-market win probability, pp. '
                'Game numbers count **all league games before intersecting the priced panel**, as S3; '
                'no renumbering around missing prices. `grid/bias_game_bins.csv` includes '
                'B=10,000 clustered CIs/counts for all six arms and all three cohorts.')
curve_rows = []
for d in ['relegated_into_L1','promoted_into_L1','entered_spfl']:
    for run in runs:
        values = [next(r['bias_pp'] for r in bins if r['arm']==run['label'] and r['direction']==d and r['game_bin']==b)
                  for b in ['1','2-5','6-10','11-20']]
        curve_rows.append([d+' / '+names[run['label']],*['—' if v=='NaN' else signed(v) for v in values]])
sections.append(table(['Direction /arm','Game1','Games2–5','Games6–10','Games11–20'],curve_rows))
sections.append('Promotion game1 has5 priced club-seasons (later bins6); relegation bins each5. '
                'Entered-SPFL game1 has no priced observation. These are descriptive, not causal curves.')
sections.append('## 5. UUIDs, convergence and timing\n')
sections.append(table(['Arm','Final scored UUID'],[[names[r['label']],f"`{r['run_id']}`"] for r in runs]))
sections.append('Pooled initial immutable UUID `0fa6219a-97cc-47c9-95bf-aec0235c89ef` is preserved. '
                'Final fit retains56 original folds and replaces41,45,51,53 once using seeds2026101200+fold '
                '(replacement divergences0,0,1,2). `reruns_qs_pooled_a2.csv` records diagnostics; '
                '`*_qs_pooled_a2_initial.csv` retains initial scores/folds/checks. Initial pooled maxR̂1.0236 '
                'and215/240000 divergences passed aggregate review but triggered four per-fold reruns. '
                'The final fit passed latent/target coverage and save/load parity checks. '
                'Pooled has1605 total latent fixtures, including Championship monitor fixtures; '
                'all inferential tables here use only the frozen1070 lower-tier targets.')
sections.append(table(['Final arm','Folds','Max R̂','Min bulk/tail ESS','Divergences /draws','Fold review'],[
    [names[r['arm']],r['n_folds'],f"{float(r['max_rhat']):.4f}",
     f"{float(r['min_ess_bulk']):.0f} /{float(r['min_ess_tail']):.0f}",
     r['divergences']+' /'+r['draws'],'PASS' if r['fold_review_pass']=='True' else 'FAIL: '+r['offending_folds']]
    for r in conv]))
sections.append('Weekly requires no reruns. Final pooled requires no further reruns. '
                'A2 from S3 still has its existing fold53 blocker, unchanged. ESS<400 is informational '
                'under James\'s rule; the stricter fit audit can print FAIL despite passing harness review.')
weekly_wall = float(next(r['fit_elapsed_seconds'] for r in timing if r['arm']=='qs_weekly'))
pooled_wall = float(load('timing_pooled_initial.csv')[0]['fit_elapsed_seconds'])
replay_wall = float(next(r['fit_elapsed_seconds'] for r in timing if r['arm']=='qs_pooled_a2'))
sections.append(table(['Work','Measured fit wall','Grid /rerun job'],[
    ['Weekly115',f'{weekly_wall:.2f}s /{weekly_wall/60:.2f}min','172312-4833'],
    ['Pooled60 initial',f'{pooled_wall:.2f}s /{pooled_wall/60:.2f}min','195701-7327'],
    ['Pooled four seeded reruns','Approximately10min watcher duration (includes replay/persistence)','223346-26600'],
    ['Pooled final replay only',f'{replay_wall:.2f}s; NOT grid wall','223346-26600']]))
sections.append('''Both original grids use beast16, pinned cores/BLAS1, four chains × (500 warmup +1,000
retained), one grid at a time, unchanged recipes. Two initial fit walls total
''' + f'{(weekly_wall+pooled_wall)/3600:.3f}h' + '''; rerun/scoring/export overhead is additional.
James may have been attached to `s3` during reruns, so rerun timing is not claimed as
isolated-core throughput. Manager explicitly required leaving `s3` untouched and using fresh
owned `s3r`; the phase ends with `s3r` stopped and `s3` left for James.
''')
sections.append('''## 6. Reproduction and evidence

- Owned checkout `/root/BF_runs/qs_transition` synced by bundle/fast-forward to pushed52c87cba
  (merged60d9a81b plus brief/D22 records). No push; source/recipes/Project/Manifest diff clean.
- Both lower and pooled-smoke cache hashes are
  `a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12`.
  Pooled smoke copied this from `/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls`
  to its owned checkout; identical bytes already existed in the S3 checkout, so no new copy,
  SQL refresh or cache regeneration was needed. Scope selects Championship goals from that cache.
- Readiness172219-11336 confirms approved weekly hash
  `aaed32f47a4c97e032bfb067b6debd2e4ccc14d934df58301955f4bccffae952` and pooled hash
  `0db99f621d0e430127f62404c530f07bdbdb22a459f8a345856193e0e09df887`, both passing smoke.
  Full histories: `s3b_smoke_history_*.csv`; no S3b stale rows removed.
- Absolute runner include and a separate `l01_entrant_prior.jl` preload precede each arm's `main`.
  Grid logs: `/root/BF_runs/logs/qs_transition_s3b_{weekly,pooled}_grid`.
- `grid/r02_score_export.jl`: same S3 scoring, only weekly fold assertion extended to115.
  `grid/r04_bias_bins.jl`: unchanged curves. Final score/curve job224438-5675 completed these
  exports before the first posthoc exporter panel assertion failed on monitored Championship rows.
- `grid/r05_transition_errors.jl` now explicitly restricts raw observations to the control priced
  lower-tier panel before pairing; successful224813-23506 exports1326 arm/cohort/game errors.
  Every one of18 arm/cohort signed means and counts matches frozen scoring within1e-10.
- `python3 grid/absolute_error.py --test`: synthetic unequal-game-count weighting, absolute-before-mean,
  paired resampling, self-reference and one-cluster tests PASS. Normal invocation exports54 rows.
  `summarize_s3.py` builds frozen six-arm tables; `report_s3b.py` renders this report from CSVs.
- No frozen gates/priors/sampler/recipes/src edits and no26/27 outcome scoring. Dirty fit telemetry
  reflects owned research outputs, not changes to model source.

## Decision

Neither S3b arm qualifies under the frozen gates; no arm qualifies under D23's posthoc criterion
on this panel either. Retain the control as the comparison baseline. The manager and James decide
any next-stage action; this report does not authorise a combination, gate relaxation or26/27 run.
''')
(P.parent / 'REPORT_S3B.md').write_text('\n\n'.join(sections)+'\n')
print('REPORT_S3B_RENDERED')
