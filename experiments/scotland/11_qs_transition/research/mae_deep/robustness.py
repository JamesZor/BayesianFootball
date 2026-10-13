#!/usr/bin/env python3
"""Fixed, transparent robustness checks. Imports main analysis (regenerates CSVs).
No current-match price as input, no 26/27 data, no QS fits.
"""
import analyze as a
import numpy as np
from pathlib import Path

out=[]
def evaluate(label,pred,arm='qs_control'):
 for sy in [0,2024,2025]:
  for d in ['L1_movers','relegated_into_L1','promoted_into_L1','nonmover']:
   mask=(a.game<=20)&(a.years>=2024)&np.isfinite(pred)
   mask &= np.isin(a.directions,['promoted_into_L1','relegated_into_L1']) if d=='L1_movers' else a.directions==d
   if sy:mask &= a.years==sy
   if not mask.any():continue
   delta,lo,hi=a.boot_delta(mask,pred)
   out.append(dict(check=label,arm=arm,test_season=sy or 'both',group=d,rows=int(mask.sum()),club_seasons=len(set(a.keys[mask])),mae_club=a.metric(mask,pred),delta=delta,lo=lo,hi=hi,r2=1-np.mean((a.y[mask]-pred[mask])**2)/np.mean(a.y[mask]**2)))

def fit(names,restriction=None,alpha=10.):
 pred=np.full(a.n,np.nan)
 for sy in [2024,2025]:
  train=a.years<sy;test=a.years==sy
  if restriction is not None:train &= restriction
  pred[test]=a.ridge(names,train,test,alpha=alpha)[0][test]
 return pred

names=a.base+a.blocks['form_pair']
for alpha in [1.,10.,100.]:evaluate(f'form_pair_ridge_{alpha:g}',fit(names,alpha=alpha))
evaluate('form_pair_train_nonmovers',fit(names,a.directions=='nonmover'))
evaluate('form_pair_train_2023_only',fit(names,a.years==2023))
counts={}
for r in a.rows(a.P.parent.parent/'grid/observations_qs_control.csv'):
 if r['family']=='1X2':counts[r['match_id']]=counts.get(r['match_id'],0)+1
complete=np.array([counts[r['match_id']]==3 for r in a.raw])
evaluate('form_pair_train_complete_1x2',fit(names,complete))
# Wealth always quarantined; biological age is computed at historical date.
# Combined additions must show incremental value over the tiny form-pair block.
for b in ['all_nonmarket','all_plus_market']:
 evaluate(b,fit(a.base+a.blocks[b]))

# Delayed information, at strictly earlier timestamp than target minus d days.
delayed=a.rows(a.P/'delayed_features.csv')
for days in [7,14,28]:
 lookup={(r['match_id'],r['club']):r for r in delayed if int(r['days'])==days}
 for feature,field in [('gd','gd'),('shots','shots'),('sot','sot')]:
  vals=[]
  for r in a.raw:
   h=lookup[(r['match_id'],r['club'])]
   vals.append(a.num(h['own_'+field])-a.num(h['opp_'+field]))
   assert not h['own_latest'] or h['own_latest']<r['start_timestamp']
   assert not h['opp_latest'] or h['opp_latest']<r['start_timestamp']
  a.x[f'delay_{days}_{feature}']=np.array(vals)
 evaluate(f'form_pair_delay{days}d',fit(a.base+[f'delay_{days}_{f}' for f in ['gd','shots','sot']]))

# Paired six-arm robustness: features identical, arm probability and residual
# change together. The fit does not secretly train on control labels for all arms.
original_y=a.y.copy(); original_p=a.x['p'].copy(); original_p2=a.x['p2'].copy()
grid=a.P.parent.parent/'grid'
for arm in ['qs_control','qs_a2','qs_entrant','qs_summer','qs_weekly','qs_pooled_a2']:
 obs=a.rows(grid/f'observations_{arm}.csv')
 lookup={(r['match_id'],r['selection']):r for r in obs if r['selection'] in ('home','away')}
 for i,r in enumerate(a.raw):
  z=lookup[(r['match_id'],'home' if int(r['is_home']) else 'away')]
  a.y[i]=100*(float(z['p_model'])-float(z['p_market']))
  a.x['p'][i]=float(z['p_model']);a.x['p2'][i]=(float(z['p_model'])-.5)**2
 evaluate('six_arm_form_pair',fit(names),arm)
a.y[:]=original_y;a.x['p'][:]=original_p;a.x['p2'][:]=original_p2
a.save('robustness_results.csv',out)

# Leave-one-club-season influence on the held-out form-pair MAE improvement.
pred=a.preds['form_pair'];influence=[]
for d in ['L1_movers','relegated_into_L1','promoted_into_L1','nonmover']:
 mask=(a.game<=20)&(a.years>=2024)&(np.isin(a.directions,['promoted_into_L1','relegated_into_L1']) if d=='L1_movers' else a.directions==d)
 for k in np.unique(a.keys[mask]):
  m=mask&(a.keys!=k);delta,lo,hi=a.boot_delta(m,pred)
  influence.append(dict(group=d,excluded=k,remaining_club_seasons=len(set(a.keys[m])),delta=delta,lo=lo,hi=hi))
a.save('leave_one_cluster_sensitivity.csv',influence)

# Relative-vs-own features distinguish opponent contributions.
for f in ['gd','shots','sot']:
 field={'gd':'form_gd6','shots':'form_shots6','sot':'form_sot6'}[f]
 a.x['own_'+f]=a.col[field];a.x['opponent_'+f]=a.col['opp_'+field]
for label,ns in [('form_own_only',['own_gd','own_shots','own_sot']),('form_opponent_only',['opponent_gd','opponent_shots','opponent_sot']),('form_sides_separate',['own_gd','own_shots','own_sot','opponent_gd','opponent_shots','opponent_sot'])]:
 evaluate(label,fit(a.base+ns))
a.save('robustness_results.csv',out)

# A literal oracle median offset is the maximum MAE benefit any constant
# per-club-season correction could achieve on this exact panel.
oracle=[]
for d in ['relegated_into_L1','promoted_into_L1','nonmover']:
 mask=(a.game<=20)&(a.directions==d);p=np.zeros(a.n)
 for k in np.unique(a.keys[mask]):
  km=mask&(a.keys==k);p[km]=np.median(a.y[km])
 oracle.append(dict(group=d,baseline=a.metric(mask,np.zeros(a.n)),oracle_constant_median_mae=a.metric(mask,p),maximum_constant_gain=a.metric(mask,np.zeros(a.n))-a.metric(mask,p)))
a.save('oracle_constant_bound.csv',oracle)
print(f'Robustness rows: {len(out)}; all six arms checked. Fixed windows/delays only, no tuning.')
