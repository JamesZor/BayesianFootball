#!/usr/bin/env python3
"""Clustered uncertainty and sensitivity from S1 CSV outputs only."""
from pathlib import Path
from collections import defaultdict
import csv, math
import numpy as np
P=Path(__file__).resolve().parent;O=P/'results';rng=np.random.default_rng(20261013)
def read(n):
    with open(O/n,newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(O/n,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def bootstrap_mean(values,keys,B=4000):
    groups=defaultdict(list)
    for v,k in zip(values,keys):groups[k].append(v)
    groups=list(groups.values());s=np.array([sum(v) for v in groups]);n=np.array([len(v) for v in groups]);ix=rng.integers(len(groups),size=(B,len(groups)))
    return np.quantile(s[ix].sum(1)/n[ix].sum(1),[.025,.975])

# Prior calibration/error bootstrap: one fixture once, even when both clubs move.
rows=read('q2_prior_predictions.csv')+read('q2_first5_predictions.csv')
unique={}
for r in rows:unique[(r['method'],r['coordinate'],r['checkpoint'],r['match_id'])]=r
rows=list(unique.values());groups=defaultdict(list)
for r in rows:
    period='pre_panel' if r['season']<'23/24' else 'panel'
    groups[(r['method'],r['coordinate'],r['checkpoint'],period)].append(r)
summary=[]
for key,rr in sorted(groups.items()):
    err=np.array([float(r['error']) for r in rr]);pred=np.array([float(r['prediction']) for r in rr]);actual=np.array([float(r['actual']) for r in rr]);lo,hi=bootstrap_mean(err,[r['match_id'] for r in rr])
    summary.append(dict(zip(['method','coordinate','checkpoint','period'],key),n_fixtures=len(rr),n_club_seasons=len({(r['club'],r['season']) for r in rr}),bias=err.mean(),bias_lo=lo,bias_hi=hi,mae=abs(err).mean(),rmse=math.sqrt((err**2).mean()),coverage95=np.mean([r['covered95']=='True' for r in rr]),negative_log_score=np.mean([float(r['log_score']) for r in rr]),calibration_slope=np.polyfit(pred,actual,1)[0] if len(rr)>2 and pred.std()>1e-8 else float('nan'),old_history_missing=sum(int(r['old_quotes'])==0 for r in rr)))
write('q2_prior_unique_fixture.csv',summary)
lookup={(r['method'],r['coordinate'],r['checkpoint'],r['match_id']):r for r in rows};paired=[]
for method in ['P1','P2','P2_A2shrink','P2_carry_est','A2_step']:
    for base in ['league_mean','A2_step']:
        for coord in ['D','U']:
            for checkpoint in ['1','5','mean_first5']:
                for period in ['all','pre_panel','panel']:
                    rr=[r for r in rows if r['method']==method and r['coordinate']==coord and r['checkpoint']==checkpoint and (period=='all' or (r['season']<'23/24')==(period=='pre_panel'))]
                    dif=[];ids=[]
                    for r in rr:
                        b=lookup.get((base,coord,checkpoint,r['match_id']))
                        if b:dif.append(float(r['error'])**2-float(b['error'])**2);ids.append(r['match_id'])
                    if not dif:continue
                    lo,hi=bootstrap_mean(dif,ids)
                    paired.append(dict(method=method,baseline=base,coordinate=coord,checkpoint=checkpoint,period=period,n_fixtures=len(dif),delta_mse=np.mean(dif),lo=lo,hi=hi))
write('q2_paired_error.csv',paired)

# Rate residuals from relational mean lambdas: not posterior mean q_i/s_i.
quotes={int(r['match_id']):r for r in read('accepted_quotes.csv')};bias=read('q1_club_observations.csv');models=sorted(set(r['model'] for r in bias));ratelook={}
for model in models:ratelook[model]={int(r['match_id']):r for r in read('rates_'+model+'.csv')}
rates=[]
for r in bias:
    mid=int(r['match_id']);q=quotes.get(mid);m=ratelook[r['model']].get(mid)
    if not q or not m:continue
    dh=.5*(math.log(float(m['mean_lambda_h']))-math.log(float(q['lambda_h'])));da=.5*(math.log(float(m['mean_lambda_a']))-math.log(float(q['lambda_a'])))
    rates.append(dict(model=r['model'],club=r['club'],season=r['season'],match_id=mid,group=r['group'],game=r['game'],game_bin=r['game_bin'],rate_D_residual=(dh-da)*(1 if r['side']=='home' else -1),rate_U_residual=dh+da,label='log of posterior mean lambda; D includes home-effect mismatch; U is fixture style sum, not isolated club state'))
write('q1_rate_residuals.csv',rates)
rrsummary=[]
for model in models:
    for group in sorted(set(r['group'] for r in rates)):
        for gamebin in ['1','2-5','6-10','11-20']:
            rr=[r for r in rates if r['model']==model and r['group']==group and r['game_bin']==gamebin]
            if not rr:continue
            for value in ['rate_D_residual','rate_U_residual']:
                vals=[r[value] for r in rr];lo,hi=bootstrap_mean(vals,[(r['club'],r['season']) for r in rr])
                rrsummary.append(dict(model=model,group=group,game_bin=gamebin,coordinate=value,mean=np.mean(vals),lo=lo,hi=hi,n_obs=len(rr)))
write('q1_rate_summary.csv',rrsummary)

# Q4 paired summer/within-season diffusion comparison; resample whole league-season blocks.
w=read('q4_window_changes.csv');ww=defaultdict(dict)
for r in w:ww[(r['club'],r['season'])][r['type']]=r
pairedwindows=[v for v in ww.values() if set(v)=={'summer','within'} and all(int(r['overlap_quotes'])==0 for r in v.values())]
q4=[]
for coord,var in [('dq','vq'),('ds','vs')]:
    keys=[(v['summer']['tier'],v['summer']['season']) for v in pairedwindows]
    blockids=sorted(set(keys));byblock=[[i for i,k in enumerate(keys) if k==b] for b in blockids]
    def stat(indices):
        ans={}
        for typ in ['summer','within']:
            rr=[pairedwindows[i][typ] for i in indices];z=np.array([float(r[coord]) for r in rr]);v=np.mean([float(r[var]) for r in rr]);variance=max(0,float(z.var(ddof=1))-v)
            gap=np.mean([float(r['midpoint_gap_days']) for r in rr]);ans[typ]=(math.sqrt(variance),math.sqrt(variance/gap),z.mean(),gap)
        return ans['summer'][0],ans['within'][0],ans['summer'][1]/ans['within'][1] if ans['within'][1] else float('nan'),ans['summer'][2],ans['within'][2],ans['summer'][3],ans['within'][3]
    actual=stat(list(range(len(pairedwindows))));boots=[]
    for _ in range(2000):
        ids=[i for k in rng.integers(len(blockids),size=len(blockids)) for i in byblock[k]];boots.append(stat(ids))
    bb=np.array(boots);lo,hi=np.nanquantile(bb[:,2],[.025,.975])
    q4.append(dict(coordinate=coord,n_pairs=len(pairedwindows),n_league_season_blocks=len(blockids),summer_adjusted_sd=actual[0],within_adjusted_sd=actual[1],calendar_diffusion_sd_ratio=actual[2],ratio_lo=lo,ratio_hi=hi,summer_mean=actual[3],within_mean=actual[4],summer_midpoint_days=actual[5],within_midpoint_days=actual[6],limitation='calendar-time correction uses window midpoints, not equal-gap observations; approximate reconstruction noise; tier-season block bootstrap'))
write('q4_paired_diffusion.csv',q4)

# Q5 conditional age regression excludes season-opening vintages and >21-day stale gaps.
folds={m:{int(r['match_id']):r for r in read('folds_'+m+'.csv')} for m in models}
q5=[]
for model in models:
    for broad in ['nonmover','promoted','relegated']:
        rr=[r for r in bias if r['model']==model and r['broad']==broad and int(r['game'])<=20 and 0<int(r['age_days'])<=21 and int(folds[model][int(r['match_id'])]['time_step'])>0]
        if len(rr)<10:continue
        clusters=sorted({(r['club'],r['season']) for r in rr});gamebins=sorted({r['game_bin'] for r in rr})
        x=[]
        for r in rr:x.append([1,float(r['age_days'])/7]+[float((r['club'],r['season'])==c) for c in clusters[1:]]+[float(r['game_bin']==g) for g in gamebins[1:]])
        x=np.array(x)
        for val in ['delta','absolute_delta']:
            y=np.array([float(r[val]) for r in rr]);beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta;bread=np.linalg.pinv(x.T@x);meat=np.zeros((x.shape[1],x.shape[1]))
            for c in clusters:
                ix=np.array([(r['club'],r['season'])==c for r in rr]);score=x[ix].T@res[ix];meat+=np.outer(score,score)
            cov=bread@meat@bread*(len(clusters)/max(1,len(clusters)-1));se=math.sqrt(max(0,cov[1,1]))
            q5.append(dict(model=model,broad=broad,value=val,age_slope_pp_per_week=100*beta[1],lo=100*(beta[1]-1.96*se),hi=100*(beta[1]+1.96*se),n_obs=len(rr),n_club_seasons=len(clusters),rank=np.linalg.matrix_rank(x),n_parameters=x.shape[1],limitation='club-season sandwich normal interval; game-bin and club-season fixed effects; descriptive cross-fixture, no paired vintages'))
write('q5_conditional_age.csv',q5)
# Sequential club event comparison, explicitly not identical-fixture counterfactuals.
events=[]
for model in models:
    gg=defaultdict(list)
    for r in bias:
        if r['model']==model and int(r['game'])<=20:gg[(r['club'],r['season'])].append(r)
    for (club,s),rr in gg.items():
        rr=sorted(rr,key=lambda r:int(r['game']))
        for a,b in zip(rr,rr[1:]):
            if a['fold']==b['fold']:continue
            events.append(dict(model=model,club=club,season=s,group=b['group'],match_before=a['match_id'],match_id=b['match_id'],game_before=a['game'],game_after=b['game'],age_before=a['age_days'],age_after=b['age_days'],absolute_change_pp=100*(float(b['absolute_delta'])-float(a['absolute_delta'])),signed_change_pp=100*(float(b['delta'])-float(a['delta'])),limitation='different opponents/fixtures across refit; exploratory event, not weekly-refit effect'))
write('q5_refit_events.csv',events)
print('DIAGNOSTICS_OK prior_unique=',len(rows),'paired_windows=',len(pairedwindows),'age_regressions=',len(q5),'refit_events=',len(events))
