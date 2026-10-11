#!/usr/bin/env python3
"""Artifact checks and compact CSV-backed report tables. No DB or Julia execution."""
from pathlib import Path
from collections import defaultdict
from datetime import date
import csv,hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;O=P/'results';F=P/'figures'
def read(n):
    with open(O/n,newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(O/n,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
checks=[]
def check(name,passed,detail):
    checks.append(dict(check=name,pass_=bool(passed),detail=detail))
    if not passed:raise AssertionError(name+': '+detail)
fixtures=read('fixtures.csv')
check('date embargo',all(r['match_date']<'2026-07-01' for r in fixtures),'all outcome-bearing fixtures precede 2026-07-01')
check('W2 cohort gate',[(r['direction'],int(r['n_club_seasons'])) for r in read('q6_cohort_gate.csv')]==[('relegated_into_L1',5),('promoted_into_L1',6),('entered_spfl',2)],'canonical 5/6/2')
q=read('accepted_quotes.csv');expected={54:956,55:477,56:835,57:693}
check('accepted inventory',all(sum(int(r['tournament'])==t for r in q)==n for t,n in expected.items()),str(expected))
check('unique accepted IDs',len({r['match_id'] for r in q})==len(q),'one quote inversion per fixture')
audits=list(O.glob('*_db_audit.csv'))
check('all DB audits unchanged',all(len(read(p.name))==10 and all(r['unchanged']=='true' for r in read(p.name)) for p in audits),f'{len(audits)} audits; ten tables each')
for model in ['td_lower_base','td_lower_a2full_carry_jump','grw_base','grw_step_a2_carry_jump','m12_td_ref','qs_marketobs']:
    ff=read('folds_'+model+'.csv');expected_n=710 if model=='qs_marketobs' else 1070
    target={r['match_id'] for r in ff if r['target_season'] in ['23/24','24/25','25/26'] and any(f['match_id']==r['match_id'] and f['tournament_id'] in ['56','57'] for f in fixtures)}
    check('heldout vintage '+model,len({r['match_id'] for r in ff})==len(ff) and len(target)==expected_n and all(r['valid']=='true' for r in ff),f'lower target {len(target)} fixtures, all assigned exactly once')
    for direction in ['relegated_into_L1','promoted_into_L1']:
        ours=next(r for r in read('q1_bias_first20.csv') if r['model']==model and r['group']==direction)
        harness=next(r for r in read('scores_'+model+'.csv') if r['subset']=='transition_'+direction+'_first20' and r['metric']=='transition_bias_pp')
        check('bias parity '+model+' '+direction,abs(float(ours['mean'])-float(harness['value']))<1e-10,'CSV decomposition equals harness gate mean')
check('prior source chronology',all(int(r['violations'])==0 for r in read('q12_source_manifest.csv')),'all source seasons/dates precede boundary')
check('prior parameter chronology',all(all(s<r['target_season'] for s in r['training_seasons'].split(',') if s) for r in read('q2_prior_parameters.csv')),'expanding-season hyperparameter labels only')
check('every question artifact',all(any(O.glob(f'q{i}_*.csv')) for i in range(1,13)),'Q9/Q11 optional scope explicitly recorded')
# Hash actual source values plus fitted parameters per season, repeated across QS folds.
params=read('q2_prior_parameters.csv');hashes={}
for s in ['24/25','25/26']:
    payload=dict(quotes=[r for r in q if r['season']<s],parameters=[r for r in params if r['target_season']==s],reference='s common mu_ref=0; q division sum-zero; late80 endpoint')
    hashes[s]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
foldhash=[dict(fold=r['fold'],target_season=r['target_season'],prior_value_sha256=hashes[r['target_season']]) for r in read('q12_cutoff_check_qs_marketobs.csv')]
write('q12_fold_prior_hashes.csv',foldhash)
check('season-frozen prior hashes',all(len({r['prior_value_sha256'] for r in foldhash if r['target_season']==s})==1 for s in hashes),'source values and parameter values, not IDs alone')
# Deduplicated headline: one fixture once. Keep all club-season inventory separately.
pr=read('q2_prior_predictions.csv')+read('q2_first5_predictions.csv');dedup={}
for r in pr:dedup[(r['method'],r['coordinate'],r['checkpoint'],r['match_id'])]=r
rr=list(dedup.values());headline=[]
for method in ['league_mean','A2_step','P1','P2','P2_A2shrink','P2_carry_est']:
    for coord in ['D','U']:
        for cp in ['1','5','mean_first5']:
            rows=[r for r in rr if r['method']==method and r['coordinate']==coord and r['checkpoint']==cp]
            err=np.array([float(r['error']) for r in rows]);headline.append(dict(method=method,coordinate=coord,checkpoint=cp,n_fixtures=len(rows),n_club_seasons=len({(r['club'],r['season']) for r in pr if r['method']==method and r['coordinate']==coord and r['checkpoint']==cp}),bias=err.mean(),mae=abs(err).mean(),rmse=math.sqrt((err**2).mean()),coverage95=np.mean([r['covered95']=='True' for r in rows]),negative_log_score=np.mean([float(r['log_score']) for r in rows]),calibration_slope=np.polyfit([float(r['prediction']) for r in rows],[float(r['actual']) for r in rows],1)[0]))
write('q2_prior_headline.csv',headline)
# Paired harness-style Bernoulli observation score: common fixture/selection rows, fixture clusters.
rng=np.random.default_rng(20261015);score=[]
for candidate,control in [('td_lower_a2full_carry_jump','td_lower_base'),('grw_step_a2_carry_jump','grw_base')]:
    ca=read('observations_'+candidate+'.csv');co={(r['match_id'],r['selection']):r for r in read('observations_'+control+'.csv')};ids={r['match_id'] for r in fixtures if r['tournament_id'] in ['56','57'] and r['season'] in ['23/24','24/25','25/26']}
    for market in ['all','1X2']:
        gg=defaultdict(list)
        for r in ca:
            c=co.get((r['match_id'],r['selection']))
            if not c or r['match_id'] not in ids or (market!='all' and r['family']!=market):continue
            gg[r['match_id']].append(float(r['ll_model'])-float(c['ll_model']))
        vv=list(gg.values());sums=np.array([sum(v) for v in vv]);ns=np.array([len(v) for v in vv]);ix=rng.integers(len(vv),size=(4000,len(vv)));boot=sums[ix].sum(1)/ns[ix].sum(1);lo,hi=np.quantile(boot,[.025,.975])
        score.append(dict(candidate=candidate,control=control,market=market,n_fixtures=len(vv),n_observations=int(ns.sum()),delta_logloss=sums.sum()/ns.sum(),lo=lo,hi=hi,method='paired common observation Bernoulli log loss; fixture-clustered bootstrap'))
write('q10_paired_model_scores.csv',score)
# Additional figures from persisted diagnostics.
plt.figure(figsize=(6,4));d=read('q4_dispersion.csv');labels=['summer','within'];x=np.arange(2)
for coord in ['dq','ds']:
    rows=[next(r for r in d if r['type']==t and r['tier']=='all' and r['coordinate']==coord) for t in labels];plt.plot(x,[float(r['noise_adjusted_sd']) for r in rows],marker='o',label=coord)
plt.xticks(x,labels);plt.ylabel('Noise-adjusted equal-quote-window SD');plt.title('Calendar gaps differ; see paired diffusion CSV');plt.legend();plt.tight_layout();plt.savefig(F/'q4_summer_dispersion.png',dpi=150);plt.close()
plt.figure(figsize=(6,4));d=[r for r in read('q5_conditional_age.csv') if r['model']=='qs_marketobs' and r['value']=='absolute_delta'];x=np.arange(len(d));y=np.array([float(r['age_slope_pp_per_week']) for r in d]);lo=np.array([float(r['lo']) for r in d]);hi=np.array([float(r['hi']) for r in d]);plt.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o',capsize=3);plt.axhline(0,color='grey');plt.xticks(x,[r['broad'] for r in d]);plt.ylabel('Absolute bias slope (pp/week since data cutoff)');plt.title('Game-bin + club-season controls; no causal refit estimate');plt.tight_layout();plt.savefig(F/'q5_conditional_age.png',dpi=150);plt.close()
# Pin every input/output identity; no credential files.
files=list(O.glob('*.csv'))
write('artifact_sha256.csv',[dict(path=str(p.relative_to(P)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in sorted(files) if p.name not in ['artifact_sha256.csv','verification.csv']])
write('verification.csv',checks)
print('VERIFICATION_OK checks=',len(checks),'audit_runs=',len(audits),'csv_artifacts=',len(files))
