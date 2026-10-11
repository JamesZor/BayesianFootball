#!/usr/bin/env python3
"""Q8 fixed endpoint-window sensitivity, chronology unchanged. CSV-only."""
from pathlib import Path
from collections import defaultdict
import csv,math
import numpy as np
P=Path(__file__).resolve().parent;O=P/'results'
def read(n):
    with open(O/n,newline='') as f:return list(csv.DictReader(f))
def write(n,rows):
    rows=list(rows);fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(O/n,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def prev(s):return f'{int(s[:2])-1:02d}/{int(s[3:])-1:02d}'
fixtures={int(r['match_id']):r for r in read('fixtures.csv')};tiers={}
for r in fixtures.values():
    if int(r['tournament_id']) in range(54,58):
        for c in [r['home_team'],r['away_team']]:tiers[(c,r['season'])]=int(r['tournament_id'])
ratings=read('market_team_reconstruction.csv');states={(r['club'],r['season'],r['window']):r for r in ratings};quotes=read('accepted_quotes.csv');qby={r['match_id']:r for r in quotes}
def broad(c,s):
    a=tiers.get((c,prev(s)),0);b=tiers[(c,s)]
    return 'entrant' if not a else 'relegated' if a<b else 'promoted' if a>b else 'nonmover'
def state(c,s,window):
    r=states.get((c,prev(s),window))
    if r:return float(r['q']),float(r['s'])
    # Same historical common-reference fallback as analyze_s1.
    vals={ (r['season'],r['tier']):float(r['mu'])/2 for r in ratings if r['window']=='whole' and r['season']<s and int(r['tier'])==tiers[(c,s)]}
    return 0,float(np.mean(list(vals.values()))) if vals else 0
G=['relegated','promoted','entrant']
def feature(q,coord,window):
    s=q['season'];h=q['home'];a=q['away'];hs=state(h,s,window);aa=state(a,s,window);sign=-1 if coord=='D' else 1;j=0 if coord=='D' else 1
    return hs[j]+sign*aa[j],np.array([float(broad(h,s)==g)+sign*float(broad(a,s)==g) for g in G])
def target(q,coord):
    s=q['season'];t=int(q['tournament']);vals={ (r['season'],r['tier']):float(r['gamma']) for r in ratings if r['window']=='whole' and r['season']<s and int(r['tier'])==t};gamma=float(np.mean(list(vals.values()))) if vals else 0
    lh=math.log(float(q['lambda_h']));la=math.log(float(q['lambda_a']))
    return (lh-la-gamma)/2 if coord=='D' else (lh+la-gamma)/2
opening=[q for q in quotes if q['season']>='22/23' and ((broad(q['home'],q['season'])!='nonmover' and int(q['game_home'])<=5) or (broad(q['away'],q['season'])!='nonmover' and int(q['game_away'])<=5))]
# Exact same validation fixture/checkpoint identities as the main prior check.
validation={}
for r in read('q2_prior_predictions.csv')+read('q2_first5_predictions.csv'):
    if r['method']=='P2_A2shrink' and str(r['checkpoint']) in ['1','mean_first5']:validation[(r['match_id'],r['coordinate'],str(r['checkpoint']))]=r
pred=[]
for s in ['22/23','23/24','24/25','25/26']:
    train=[q for q in opening if q['season']<s]
    for coord in ['D','U']:
        for window in ['late80','whole']:
            center=np.array([.235,-.235,0]) if coord=='D' else np.zeros(3)
            if train:
                bf=[feature(q,coord,window) for q in train];x=np.array([f for b,f in bf]);y=np.array([target(q,coord)-b for q,(b,f) in zip(train,bf)]);beta=np.linalg.solve(x.T@x+2*np.eye(3),x.T@y+2*center)
            else:beta=center
            for (mid,c,cp),r in validation.items():
                if c!=coord or r['season']!=s:continue
                q=qby[mid];b,f=feature(q,coord,window);forecast=b+f@beta;y=target(q,coord)
                pred.append(dict(window=window,coordinate=coord,checkpoint=cp,season=s,match_id=mid,prediction=forecast,actual=y,error=forecast-y,n_training_fixtures=len(train),max_training_season=max([q['season'] for q in train],default='')))
write('q8_endpoint_predictions.csv',pred)
summary=[]
for window in ['late80','whole']:
    for coord in ['D','U']:
        for cp in ['1','mean_first5']:
            rr=[r for r in pred if r['window']==window and r['coordinate']==coord and r['checkpoint']==cp];e=np.array([r['error'] for r in rr]);summary.append(dict(window=window,coordinate=coord,checkpoint=cp,n_fixtures=len(rr),bias=e.mean(),rmse=math.sqrt((e**2).mean()),mae=abs(e).mean(),selection='fixed sensitivity only; not used to retune main candidate'))
write('q8_endpoint_sensitivity.csv',summary)
# Exact point-forecast parity with primary candidate for late80.
assert all(abs(r['prediction']-float(validation[(r['match_id'],r['coordinate'],r['checkpoint'])]['prediction']))<1e-10 for r in pred if r['window']=='late80')
status=read('q8_scope_status.csv')
for r in status:
    if r['check']=='whole-old-season endpoint sensitivity':r['status']='run fixed late80/whole forward comparison; late80 retained without retuning'
write('q8_scope_status.csv',status)
print('ENDPOINT_OK predictions=',len(pred),'late80 point parity verified')
