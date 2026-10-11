#!/usr/bin/env python3
"""Descriptive cross-tier cups and source bridge; no DB access, no production fits."""
from pathlib import Path
from collections import defaultdict
from datetime import date
import csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;O=P/'results';F=P/'figures'
rng=np.random.default_rng(20261012)
def read(n):
    with open(O/n,newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(O/n,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def season(s):
    d=date.fromisoformat(s[:10]);y=d.year-(d.month<7);return f'{y%100:02d}/{(y+1)%100:02d}'
fixtures={int(r['match_id']):r for r in read('fixtures.csv')}
tiers={}
for r in fixtures.values():
    if int(r['tournament_id']) in range(54,58):
        for c in [r['home_team'],r['away_team']]:tiers[(c,r['season'])]=int(r['tournament_id'])
ratings={(r['club'],r['season']):r for r in read('market_team_reconstruction.csv') if r['window']=='whole'}
quotes={int(r['match_id']):r for r in read('accepted_quotes.csv')}
raw=defaultdict(dict)
for r in read('q3_bookmaker_raw.csv'):
    if r['fraction_num'] and r['fraction_den'] and float(r['fraction_den'])>0:
        raw[int(r['match_id'])][r['choice_name']]=1/(1+float(r['fraction_num'])/float(r['fraction_den']))
book={m:{k:v/sum(p.values()) for k,v in p.items()} for m,p in raw.items() if set(p)=={'1','X','2'} and all(v>0 for v in p.values())}
bridge=[]
for mid,q in quotes.items():
    if mid not in book:continue
    pp=book[mid];z=.5*math.log(pp['1']/pp['2']);d=.5*math.log(float(q['lambda_h'])/float(q['lambda_a']))
    bridge.append(dict(match_id=mid,season=q['season'],tier=int(q['tournament']),book_z=z,market_raw_D=d,p_home_book=pp['1'],source='SofaScore final fractions vs Betfair KL rates; quote timestamps unavailable'))
write('q9_source_bridge_rows.csv',bridge)
x=np.column_stack([np.ones(len(bridge)),[r['market_raw_D'] for r in bridge]])
y=np.array([r['book_z'] for r in bridge]);beta=np.linalg.lstsq(x,y,rcond=None)[0]
bridge_slope=float(beta[1]);bridge_intercept=float(beta[0]);res=y-x@beta
write('q9_source_bridge.csv',[dict(n_matches=len(bridge),intercept=bridge_intercept,slope=bridge_slope,residual_sd=float(np.std(res,ddof=2)),correlation=float(np.corrcoef(x[:,1],y)[0,1]),timestamp_alignment='unverified; not a source-identification pass')])
goals={int(r['match_id']):r for r in read('q3_cup_regulation.csv')}
rows=[];coverage=defaultdict(Counter if False else dict)
counts=defaultdict(lambda:defaultdict(int))
for mid,r in fixtures.items():
    cup=int(r['tournament_id'])
    if cup not in [73,982,1520]:continue
    s=season(r['match_date']);h=r['home_team'];a=r['away_team'];th=tiers.get((h,s));ta=tiers.get((a,s))
    counts[cup]['fixtures']+=1
    if not th or not ta:counts[cup]['unlinked_or_outside_spfl']+=1;continue
    counts[cup]['both_spfl']+=1
    if mid in book:counts[cup]['bookmaker_1x2']+=1
    rh=ratings.get((h,s));ra=ratings.get((a,s))
    if not rh or not ra:counts[cup]['missing_league_rating']+=1;continue
    counts[cup]['both_reconstructed']+=1
    qh,qa=float(rh['q']),float(ra['q']);sh,sa=float(rh['s']),float(ra['s'])
    gr=goals.get(mid,{});gh=gr.get('home_regulation');ga=gr.get('away_regulation')
    if gh and ga:counts[cup]['regulation_scores']+=1
    pp=book.get(mid)
    rows.append(dict(match_id=mid,competition=cup,season=s,home=h,away=a,home_tier=th,away_tier=ta,gap=abs(th-ta),q_home=qh,q_away=qa,s_home=sh,s_away=sa,book_z=.5*math.log(pp['1']/pp['2']) if pp else '',home_reg=int(gh) if gh else '',away_reg=int(ga) if ga else '',old_firm=('Celtic' in [h,a] or 'Rangers' in [h,a]),date=r['match_date'],label='same-season league ratings, descriptive only; no cup prior production input'))
write('q3_cup_rows.csv',rows)
write('q3_cup_coverage.csv',[dict(competition=k,**v) for k,v in sorted(counts.items())])
# Quotient slope is learned on same-source league bridge, not pretending 1X2 identifies U.
results=[]
def cupfit(rr,name,separate=True):
    rr=[r for r in rr if r['book_z']!=''];cross=[r for r in rr if r['gap']>0]
    if len(rr)<10:return
    x=[];y=[]
    for r in rr:
        contrasts=[float(r['home_tier']<=t)-float(r['away_tier']<=t) for t in [54,55,56]]
        if not separate:contrasts=[sum(contrasts)]
        x.append([1]+contrasts);y.append(float(r['book_z'])/bridge_slope-(r['q_home']-r['q_away']))
    x=np.array(x);y=np.array(y);rank=np.linalg.matrix_rank(x)
    beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta
    cov=np.linalg.pinv(x.T@x)*float(res@res)/max(1,len(y)-len(beta))
    for i,b in enumerate(beta[1:]):
        se=math.sqrt(cov[i+1,i+1]);boots=[]
        for k in range(2000):
            ix=rng.integers(len(rr),size=len(rr));xb=x[ix];yb=y[ix]
            if np.linalg.matrix_rank(xb)==rank:boots.append(np.linalg.lstsq(xb,yb,rcond=None)[0][i+1])
        lo,hi=np.quantile(boots,[.025,.975]) if boots else [float('nan')]*2
        results.append(dict(source='bookmaker_1x2',sensitivity=name,adjacent='constant' if not separate else f'{54+i}_to_{55+i}',q_step=float(b),total_step=2*float(b),lo_q=float(lo),hi_q=float(hi),n_matches=len(rr),n_cross_tier=len(cross),rank=rank,n_parameters=x.shape[1],condition=float(np.linalg.cond(x)),residual_rmse=math.sqrt(float(res@res)/len(res)),cup_home_intercept=float(beta[0]),bridge_slope=bridge_slope,old_firm_excluded=all(not r['old_firm'] for r in rr),limitations='fixture bootstrap only; bridge/rating/lineup/source uncertainty not propagated; descriptive') )
cupfit(rows,'all',True);cupfit(rows,'all_constant',False)
cupfit([r for r in rows if not r['old_firm']],'exclude_old_firm',True)
for cup in [73,982,1520]:
    cupfit([r for r in rows if r['competition']!=cup],f'exclude_competition_{cup}',True)
    cupfit([r for r in rows if r['competition']==cup],f'competition_{cup}',True)
# Goal contrast on regulation scores only, with within-division q/s exposures removed.
# Home coefficient learned from same-tier cup fixtures; no likelihood optimisation.
for excluded in [None,'old_firm',73,982,1520]:
    rr=[r for r in rows if r['home_reg']!='' and (excluded!='old_firm' or not r['old_firm']) and (not isinstance(excluded,int) or r['competition']!=excluded)]
    same=[r for r in rr if r['gap']==0]
    gh=sum(r['home_reg'] for r in same);ga=sum(r['away_reg'] for r in same)
    eh=sum(math.exp(r['s_home']+r['s_away']+r['q_home']-r['q_away']) for r in same)
    ea=sum(math.exp(r['s_home']+r['s_away']-r['q_home']+r['q_away']) for r in same)
    gamma=math.log((gh+.5)/(eh+.5))-math.log((ga+.5)/(ea+.5)) if same else 0
    for gap in [1,2,3]:
        cc=[r for r in rr if r['gap']==gap];gs=gw=es=ew=0.
        for r in cc:
            home_strong=r['home_tier']<r['away_tier']
            exph=math.exp(r['s_home']+r['s_away']+r['q_home']-r['q_away']+gamma)
            expa=math.exp(r['s_home']+r['s_away']-r['q_home']+r['q_away'])
            gs+=r['home_reg'] if home_strong else r['away_reg'];gw+=r['away_reg'] if home_strong else r['home_reg']
            es+=exph if home_strong else expa;ew+=expa if home_strong else exph
        if not cc:continue
        q=.5*math.log(((gs+.5)/(es+.5))/((gw+.5)/(ew+.5)));se=.5*math.sqrt(1/(gs+.5)+1/(gw+.5))
        results.append(dict(source='regulation_goals',sensitivity='all' if excluded is None else f'exclude_{excluded}',adjacent=f'gap_{gap}',q_step=q/gap,total_step=2*q/gap,lo_q=(q-1.96*se)/gap,hi_q=(q+1.96*se)/gap,n_matches=len(cc),n_cross_tier=len(cc),rank='',n_parameters='',condition='',residual_rmse='',cup_home_intercept=gamma,bridge_slope='',old_firm_excluded=excluded=='old_firm',strong_goals=gs,weak_goals=gw,strong_exposure=es,weak_exposure=ew,limitations='aggregate corrected scoring contrast; Poisson-count interval; not team-clustered; not intrinsic causal tier estimate'))
write('q3_tier_cup_estimates.csv',results)
# Movers identify only net rebase + jump. Report adjacent contrasts without calling them pure tier effects.
shifts=read('q7_mover_shifts.csv');movers=[]
for old,new in [(54,55),(55,56),(56,57),(55,54),(56,55),(57,56)]:
    rr=[r for r in shifts if int(r['old_tier'])==old and int(r['new_tier'])==new]
    if not rr:continue
    values=np.array([float(r['dq']) for r in rr]);sign=1 if new>old else -1
    ix=rng.integers(len(rr),size=(4000,len(rr)));boot=sign*values[ix].mean(1);lo,hi=np.quantile(boot,[.025,.975])
    movers.append(dict(old_tier=old,new_tier=new,n_club_seasons=len(rr),q_step_equivalent=sign*float(values.mean()),total_step_equivalent=2*sign*float(values.mean()),lo_q=lo,hi_q=hi,mean_style_shift=float(np.mean([float(r['ds']) for r in rr])),label='net destination-relative seasonal mover shift; tier and free jump are confounded'))
write('q3_tier_mover_estimates.csv',movers)
write('q3_a2_units.csv',[dict(total_step=.47,attack_share=.48,concession_share=.52,attack_improvement=.2256,concession_improvement=.2444,q_improvement=.235,s_change=-.0094,warning='compare q contrasts with .235, never with .47')])
write('q9_status.csv',[dict(diagnostic='bookmaker vs Betfair source bridge',status='run descriptively',n_matches=len(bridge),limitation='timestamp alignment unavailable'),dict(diagnostic='cup source/competition sensitivity',status='run descriptively',n_matches=len(rows),limitation='same-season reconstructed anchors; no timestamped prior eligibility'),dict(diagnostic='lineup retention bins',status='not run (optional)',n_matches=0,limitation='squad data not exported; cannot isolate rotation')])
plt.figure(figsize=(7,4))
rr=[r for r in results if r['source']=='bookmaker_1x2' and r['sensitivity']=='all']
xx=np.arange(len(rr));v=np.array([r['q_step'] for r in rr]);lo=np.array([r['lo_q'] for r in rr]);hi=np.array([r['hi_q'] for r in rr])
plt.errorbar(xx,v,yerr=[v-lo,hi-v],fmt='o',capsize=3);plt.axhline(.235,color='red',label='A2 q step');plt.xticks(xx,[r['adjacent'] for r in rr]);plt.ylabel('Descriptive cup quality step');plt.legend();plt.tight_layout();plt.savefig(F/'q3_cup_steps.png',dpi=150);plt.close()
print('CUPS_OK bridge=',len(bridge),'reconstructed_cup_rows=',len(rows),'estimates=',len(results))
