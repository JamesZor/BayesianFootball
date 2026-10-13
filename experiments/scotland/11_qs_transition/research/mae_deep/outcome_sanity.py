#!/usr/bin/env python3
"""Not a QS arm: coherent 1X2-only projection of diagnostic residual corrections.
Keeps draw probability fixed, shifts non-draw mass between home/away. This cannot
pass the all-market gate because it is not produced by a coherent score tensor.
"""
import analyze as a
import numpy as np
from collections import defaultdict
obs=a.rows(a.P.parent.parent/'grid/observations_qs_control.csv')
fixtures=defaultdict(dict)
for r in obs:
 if r['selection'] in ('home','away','draw'):fixtures[r['match_id']][r['selection']]=r
idx={(r['match_id'],r['is_home']):i for i,r in enumerate(a.raw)}
summaries=[];club_rows=[]
for block in ['context','recent_results','recent_shots','form_pair','all_nonmarket','all_plus_market']:
 losses=[];pred=a.preds[block];corrected=np.full(a.n,np.nan);clipped=0
 for mid,z in fixtures.items():
  if len(z)!=3 or (mid,'1') not in idx or (mid,'0') not in idx:continue
  h=idx[(mid,'1')];away=idx[(mid,'0')]
  if a.years[h]<2024:continue
  raw_p=np.array([float(z[s]['p_model']) for s in ['home','draw','away']]);y=np.array([float(z[s]['y']) for s in ['home','draw','away']])
  assert .98<raw_p.sum()<1.00001 and y.sum()==1
  # A few exported score grids lose tail mass; explicitly normalize both sides
  # of this proper-score comparison, never silently treat sub-probabilities as a simplex.
  p=raw_p/raw_p.sum()
  shift=(pred[h]-pred[away])/200
  hp=np.clip(p[0]-shift,1e-8,1-p[1]-1e-8)
  clipped+=int(abs(hp-(p[0]-shift))>1e-9)
  q=np.array([hp,p[1],1-p[1]-hp]);
  corrected[h]=100*(raw_p[0]-q[0]);corrected[away]=100*(raw_p[2]-q[2])
  ll0=-np.sum(y*np.log(p));ll1=-np.sum(y*np.log(q))
  losses.append(dict(mid=mid,sy=a.years[h],mover=int((a.game[h]<=20 and a.directions[h] in ('promoted_into_L1','relegated_into_L1')) or (a.game[away]<=20 and a.directions[away] in ('promoted_into_L1','relegated_into_L1'))),ll0=ll0,ll1=ll1))
 for cohort in ['all','L1_mover_fixture']:
  for sy in [0,2024,2025]:
   rs=[r for r in losses if (cohort=='all' or r['mover']) and (not sy or r['sy']==sy)]
   d=np.array([r['ll1']-r['ll0'] for r in rs]);rng=np.random.default_rng(20261013)
   boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
   summaries.append(dict(block=block,cohort=cohort,test_season=sy or 'both',fixtures=len(rs),baseline_multiclass_ll=np.mean([r['ll0'] for r in rs]),delta_multiclass_ll=d.mean(),fixture_bootstrap_lo=np.quantile(boot,.025),fixture_bootstrap_hi=np.quantile(boot,.975),clipped_fixtures_total=clipped))
 for direction in ['promoted_into_L1','relegated_into_L1','nonmover']:
  mask=(a.game<=20)&(a.years>=2024)&(a.directions==direction)&np.isfinite(corrected)
  d,lo,hi=a.boot_delta(mask,corrected)
  club_rows.append(dict(block=block,group=direction,rows=int(mask.sum()),mae_club=a.metric(mask,corrected),delta=d,lo=lo,hi=hi))
a.save('outcome_sanity.csv',summaries);a.save('coherent_1x2_mae.csv',club_rows)
print('Coherent 1X2-only sanity check saved; not an all-market gate or a QS arm.')
