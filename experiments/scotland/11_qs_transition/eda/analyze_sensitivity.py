#!/usr/bin/env python3
"""Coverage thresholds, source linkage, and explicit remaining-diagnostic status."""
from pathlib import Path
from collections import defaultdict
import csv,math
import numpy as np
P=Path(__file__).resolve().parent;O=P/'results';rng=np.random.default_rng(20261014)
def read(n):
    with open(O/n,newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(O/n,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
q=read('accepted_quotes.csv');gg=defaultdict(list)
for r in q:
    for c in [r['home'],r['away']]:gg[(c,r['season'])].append(r)
from datetime import date
counts=[]
for r in read('q6_coverage.csv'):
    rr=gg[(r['club'],r['season'])]
    dates=[date.fromisoformat(x['date'][:10]) for x in rr]
    if dates:
        last=max(dates);w=np.array([math.exp(-math.log(2)*(last-d).days/180) for d in dates]);ess=w.sum()**2/(w@w)
    else:ess=0
    counts.append(dict(r,in_market_era=r['season']>='21/22',effective_n_decay180=ess,rejection_rate=int(r['rejected'])/max(1,int(r['inversion_rows'])),recency_rule='diagnostic half-life 180d, not tuned or used in prior reconstruction'))
write('q6_coverage.csv',counts)
write('q6_quote_totals.csv',[dict(tier=t,accepted=sum(int(r['accepted']) for r in counts if int(r['tournament_id'])==t and r['in_market_era'])//2,n_club_seasons=sum(int(r['tournament_id'])==t and r['in_market_era'] for r in counts),no_history_club_seasons=sum(int(r['tournament_id'])==t and int(r['accepted'])==0 and r['in_market_era'] for r in counts),source_seasons='21/22-25/26') for t in range(54,58)])
# Q6 prior and bias coverage cuts, explicit counts.
rows=read('q1_club_observations.csv');out=[]
for model in sorted(set(r['model'] for r in rows)):
    for group in ['relegated_into_L1','promoted_into_L1','entered_spfl']:
        for threshold in [0,10,20]:
            rr=[r for r in rows if r['model']==model and r['group']==group and int(r['game'])<=20 and int(r['old_previous_quotes'])>=threshold]
            if rr:out.append(dict(model=model,group=group,min_old_quotes=threshold,n_obs=len(rr),n_club_seasons=len({(r['club'],r['season']) for r in rr}),bias_pp=100*np.mean([float(r['delta']) for r in rr])))
write('q6_bias_thresholds.csv',out)
# Prior evaluation in the actual L1 transition gate, not just pooled full-pyramid panel.
pr=read('q2_prior_predictions.csv');out=[]
for method in sorted(set(r['method'] for r in pr)):
    for group in ['relegated_into_L1','promoted_into_L1','entered_spfl']:
        for coord in ['D','U']:
            for checkpoint in ['1','5']:
                rr=[r for r in pr if r['method']==method and r['group']==group and r['coordinate']==coord and r['checkpoint']==checkpoint and r['season']>='23/24']
                if rr:out.append(dict(method=method,group=group,coordinate=coord,checkpoint=checkpoint,n_obs=len(rr),n_fixtures=len({r['match_id'] for r in rr}),n_club_seasons=len({(r['club'],r['season']) for r in rr}),bias=np.mean([float(r['error']) for r in rr]),rmse=math.sqrt(np.mean([float(r['error'])**2 for r in rr])),coverage95=np.mean([r['covered95']=='True' for r in rr])))
write('q2_lower_gate_prior.csv',out)
# Leave one linking club out of cup graph, without expensive repeated bootstrap.
rr=[r for r in read('q3_cup_rows.csv') if r['book_z']!=''];b=float(read('q9_source_bridge.csv')[0]['slope']);clubs=sorted({c for r in rr if int(r['gap'])>0 for c in [r['home'],r['away']]});sens=[]
for club in [None]+clubs:
    part=[r for r in rr if club not in [r['home'],r['away']]]
    x=np.array([[1]+[float(int(r['home_tier'])<=t)-float(int(r['away_tier'])<=t) for t in [54,55,56]] for r in part]);y=np.array([float(r['book_z'])/b-(float(r['q_home'])-float(r['q_away'])) for r in part]);rank=np.linalg.matrix_rank(x);beta=np.linalg.lstsq(x,y,rcond=None)[0]
    for i in range(3):sens.append(dict(excluded_club=club or 'none',adjacent=f'{54+i}_to_{55+i}',q_step=beta[i+1],rank=rank,expected_rank=4,n_matches=len(part),identified=rank==4))
write('q3_bridge_club_sensitivity.csv',sens)
# Source slopes by tier and missing timestamp audit.
br=read('q9_source_bridge_rows.csv');src=[]
for tier in range(54,58):
    rr=[r for r in br if int(r['tier'])==tier];x=np.column_stack([np.ones(len(rr)),[float(r['market_raw_D']) for r in rr]]);y=np.array([float(r['book_z']) for r in rr]);beta=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@beta
    src.append(dict(tier=tier,n_matches=len(rr),slope=beta[1],intercept=beta[0],residual_sd=np.std(res,ddof=2),timestamp_verified=False))
write('q9_source_by_tier.csv',src)
# Q8 source-strength overlap with non-movers: no causal extrapolation.
ratings={(r['club'],r['season']):r for r in read('market_team_reconstruction.csv') if r['window']=='late80'};trans=read('q6_transitions.csv');source=[]
for r in trans:
    if r['season']<'22/23':continue
    s=r['season'];old=f'{int(s[:2])-1:02d}/{int(s[3:])-1:02d}';a=ratings.get((r['club'],old))
    if not a:continue
    oldq=float(a['q']);candidates=[]
    for (club,season),v in ratings.items():
        if season!=old or v['tier']!=r['old_tier'] or (club,s) not in ratings:continue
        dest=ratings[(club,s)]
        if dest['tier']!=v['tier']:continue
        distance=abs(float(v['q'])-oldq)
        if distance<=.05:candidates.append((club,distance))
    source.append(dict(club=r['club'],season=s,broad=r['broad'],old_tier=r['old_tier'],old_q=oldq,n_strength_matched_nonmovers=len(candidates),closest_distance=min((d for c,d in candidates),default=''),interpretation='diagnostic old-q overlap within .05; unmatched mover effects cannot be called causal'))
write('q8_nonmover_overlap.csv',source)
write('q8_scope_status.csv',[dict(check='carry=0/1/estimated forward scoring',status='run'),dict(check='measurement reliability sensitivity',status='run descriptive seasonal labels'),dict(check='source-strength-matched non-movers',status='overlap inventory only; no causal comparison'),dict(check='whole-old-season endpoint sensitivity',status='not run; chosen late80 endpoint fixed'),dict(check='results-only cohorts back to 2008',status='not run; separate optional historical robustness panel')])
print('SENSITIVITY_OK coverage=',len(counts),'bridge_clubs=',len(clubs),'oldstrength_overlap=',len(source))
