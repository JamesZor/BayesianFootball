#!/usr/bin/env python3
"""Read-only local diagnostic regressions, explicitly approved by James.
No QS fit, future-market feature, hyperparameter search or random CV.
Run /tmp/qs-s1-venv/bin/python .../analyze.py from repository root.
"""
from pathlib import Path
import csv, json, hashlib
from collections import defaultdict
import numpy as np
import duckdb

P = Path(__file__).resolve().parent
ROOT = P.parents[4]
con = duckdb.connect()

def rows(path):
    with open(path, newline='') as f: return list(csv.DictReader(f))
def save(name, data):
    if not data: return
    with open(P/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader();w.writerows(data)
def num(s):
    if s in ('','NULL',None): return np.nan
    if s in ('true','false'): return float(s=='true')
    return float(s)

raw=rows(P/'panel.csv'); n=len(raw)
col={k:np.array([num(r[k]) for r in raw]) for k in raw[0] if k not in
 ('start_timestamp','club','opponent','manager','venue','previous_manager','previous_date','lineup_date','direction','opp_direction','opp_lineup_date','opp_previous_date')}
y=col['residual']; years=col['sy'].astype(int); game=col['game']; keys=np.array([r['club']+'|'+r['sy'] for r in raw]); directions=np.array([r['direction'] for r in raw]);
assert years.min()==2023 and years.max()==2025
assert len(set((r['match_id'],r['club']) for r in raw))==n
for r in raw:
    for k in ('previous_date','lineup_date','opp_lineup_date','opp_previous_date'):
        assert not r[k] or r[k]<r['start_timestamp'], (r['match_id'],k)
# Exact reproduction of canonical first-20 records, not just headline averages.
canonical=rows(P.parent.parent/'grid/transition_errors.csv')
ours={(r['match_id'],r['club']):r for r in raw if float(r['game'])<=20 and r['direction'] in ('relegated_into_L1','promoted_into_L1','entered_spfl')}
control=[r for r in canonical if r['arm']=='qs_control']
assert len(control)==len(ours)==221
assert max(abs(100*float(r['delta'])-float(ours[(r['match_id'],r['club'])]['residual'])) for r in control)<1e-10
assert all(int(r['game'])==int(ours[(r['match_id'],r['club'])]['game']) for r in control)

def grouped(mask, values):
    return np.array([np.mean(values[mask & (keys==k)]) for k in np.unique(keys[mask])])
def metric(mask,pred):
    return grouped(mask,np.abs(y-pred)).mean()
def boot_delta(mask,pred,ref=None):
    if ref is None: ref=np.zeros(n)
    d=grouped(mask,np.abs(y-pred)-np.abs(y-ref)); rng=np.random.default_rng(20261013)
    b=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
    return d.mean(),*np.quantile(b,[.025,.975])

summary=[]
for d in np.unique(directions):
 for year in [0,2023,2024,2025]:
  mask=(directions==d)&(game<=20)&((years==year) if year else True)
  if not mask.any():continue
  summary.append(dict(direction=d,season=year or 'all',rows=int(mask.sum()),fixtures=len(set(col['match_id'][mask])),club_seasons=len(set(keys[mask])),mae_club=metric(mask,np.zeros(n)),mae_row=np.abs(y[mask]).mean(),signed_club=grouped(mask,y).mean()))
save('baseline_summary.csv',summary)

# Candidate feature blocks, signed relative-to-opponent contrasts where appropriate.
x={}
def add(name,a): x[name]=np.asarray(a,dtype=float)
def diff(out,a,b): add(out,col[a]-col[b])
add('home',col['is_home']); add('p',col['p_model']); add('p2',(col['p_model']-.5)**2)
add('game',game/20); add('L1',col['tier']==3)
add('promoted',directions=='promoted_into_L1'); add('relegated',directions=='relegated_into_L1')
add('relegated_L2',directions=='relegated_into_L2'); add('entered',directions=='entered_spfl')
# Valuation timestamps failed audit: do not compute residual associations or
# diagnostic regressions from the unversioned log_value columns.
diff('age','age_mean','opp_age_mean')
diff('retention','retained_share','opp_retained_share');diff('higher_arrivals','higher_arrival_share','opp_higher_arrival_share')
add('own_retention',col['retained_share']);add('own_higher_arrivals',col['higher_arrival_share'])
diff('manager','manager_change3','opp_manager_change3'); add('own_manager',col['manager_change3'])
diff('old_gd','previous_gd','opp_previous_gd'); add('own_old_ppg',col['previous_ppg'])
add('old_shots',col['previous_shotdiff']);add('old_sot',col['previous_sotdiff'])
diff('form_gd','form_gd6','opp_form_gd6');diff('form_shots','form_shots6','opp_form_shots6');diff('form_sot','form_sot6','opp_form_sot6')
diff('upper_history','prior_upper_share','opp_prior_upper_share')
add('opponent_old_gd',col['opp_previous_gd']);add('opponent_form_gd',col['opp_form_gd6'])
diff('market_level','market_d_lag5','opp_market_d_lag5')
add('market_trend',(col['market_d_last']-col['market_d_lag5'])-(col['opp_market_d_last']-col['opp_market_d_lag5']))
add('lag_innovation',(col['residual_lag3']-col['opp_residual_lag3'])/2)
add('own_lag_innovation',col['residual_lag3'])
add('market_style',col['market_s_lag5']+col['opp_market_s_lag5'])
base=['home','p','p2','game','L1','promoted','relegated','relegated_L2','entered']
blocks={'context':[], 'squad_age':['age'],
 'squad_continuity':['retention','own_retention'], 'higher_tier_arrivals':['higher_arrivals','own_higher_arrivals'],
 'manager':['manager','own_manager'], 'last_season_results':['old_gd','own_old_ppg'],
 'last_season_shots':['old_shots','old_sot'], 'recent_results':['form_gd'],
 'recent_shots':['form_shots','form_sot'], 'resources':['upper_history'],
 'opponent':['opponent_old_gd','opponent_form_gd'],
 'market_history':['market_level','market_trend','market_style'],
 'lag_residual':['lag_innovation','own_lag_innovation'],
 'form_pair':['form_gd','form_shots','form_sot'],
 'all_nonmarket':['age','retention','higher_arrivals','manager','old_gd','old_shots','old_sot','form_gd','form_shots','form_sot','upper_history'],
 'all_plus_market':['age','retention','higher_arrivals','manager','old_gd','old_shots','old_sot','form_gd','form_shots','form_sot','upper_history','market_level','market_trend','market_style','lag_innovation','own_lag_innovation']}
extra={(r['match_id'],r['club']):r for r in rows(P/'extra_features.csv')}
for f in ['own_xi_continuity','opp_xi_continuity','end_season_gd6','rest_days','opp_rest_days']:
    add(f,[num(extra[(r['match_id'],r['club'])][f]) for r in raw])
add('xi_continuity_diff',x['own_xi_continuity']-x['opp_xi_continuity'])
add('rest_diff',np.minimum(x['rest_days'],30)-np.minimum(x['opp_rest_days'],30))
blocks['lineup_churn']=['xi_continuity_diff','own_xi_continuity']
blocks['end_season_form']=['end_season_gd6']
blocks['league_rest']=['rest_diff']
# Coverage and raw associations are descriptive, NOT forecast performance.
assoc=[]
for f,v in x.items():
 if f in base:continue
 for group,gm in [('L1_movers',np.isin(directions,['promoted_into_L1','relegated_into_L1'])),('nonmover',directions=='nonmover')]:
  mask=gm&(game<=20)&np.isfinite(v)
  if mask.sum()<3:continue
  a=v[mask];b=y[mask];sd=np.std(a)
  r=np.corrcoef(a,b)[0,1] if sd else np.nan
  assoc.append(dict(feature=f,group=group,available=int(mask.sum()),total=int((gm&(game<=20)).sum()),club_seasons=len(set(keys[mask])),mean=float(a.mean()),sd=float(sd),correlation=float(r),in_sample_univariate_r2=float(r*r),slope_pp_per_unit=float(np.cov(a,b,ddof=0)[0,1]/(sd*sd)) if sd else np.nan))
save('feature_associations.csv',assoc)

def ridge(names,train,test,alpha=10.):
    """Fixed shrinkage; train-only centering/imputation and no test tuning."""
    a=np.column_stack([x[f] for f in names]); miss=~np.isfinite(a)
    means=np.array([np.mean(a[train & ~miss[:,j],j]) if (train & ~miss[:,j]).any() else 0 for j in range(len(names))])
    a=np.where(miss,means,a)
    std=a[train].std(0);std=np.where(std>1e-8,std,1.)
    z=(a-means)/std
    # Explicit missingness masks, only if training has variation; no future stats.
    midx=[j for j in range(len(names)) if 0<miss[train,j].sum()<train.sum()]
    z=np.column_stack([np.ones(n),z,miss[:,midx].astype(float)])
    labs=['intercept']+names+[names[j]+'_missing' for j in midx]
    # Each club-season has equal training weight; sum weights = number of rows.
    count={k:int((train&(keys==k)).sum()) for k in np.unique(keys[train])}
    w=np.array([1/count[k] for k in keys[train]]); w*=len(w)/w.sum()
    pen=np.eye(z.shape[1])*alpha;pen[0,0]=0
    coef=np.linalg.solve(z[train].T@(w[:,None]*z[train])+pen,z[train].T@(w*y[train]))
    return z@coef, list(zip(labs,coef))

preds={'zero':np.zeros(n)}; coeff=[]
for name,features in blocks.items():
 pred=np.full(n,np.nan)
 for sy in [2024,2025]:
    train=years<sy;test=years==sy
    p,c=ridge(base+features,train,test)
    pred[test]=p[test]
    for f,b in c:coeff.append(dict(block=name,test_season=sy,feature=f,standardized_coefficient=b,training_rows=int(train.sum())))
 preds[name]=pred
# Fixed corrections are transparently untrained sensitivity arms.
for alpha in [.25,.5,1.]:
 preds[f'fixed_lag_{alpha}']=alpha*np.nan_to_num(x['lag_innovation'])
save('diagnostic_coefficients.csv',coeff)

results=[]
for name,p in preds.items():
 for sy in [0,2024,2025]:
  for d in ['L1_movers','relegated_into_L1','promoted_into_L1','nonmover','relegated_into_L2','entered_spfl']:
   mask=(game<=20)&(years>=2024)&np.isfinite(p)
   mask&= np.isin(directions,['promoted_into_L1','relegated_into_L1']) if d=='L1_movers' else directions==d
   if sy:mask&=years==sy
   if not mask.any():continue
   delta,lo,hi=boot_delta(mask,p)
   inc,ilo,ihi=boot_delta(mask,p,preds['context'])
   results.append(dict(block=name,test_season=sy or 'both',group=d,rows=int(mask.sum()),club_seasons=len(set(keys[mask])),mae_club=metric(mask,p),delta_vs_zero=delta,delta_lo=lo,delta_hi=hi,delta_vs_context=inc,increment_lo=ilo,increment_hi=ihi,r2_vs_zero=1-np.mean((y[mask]-p[mask])**2)/np.mean(y[mask]**2)))
save('chronological_diagnostics.csv',results)
save('residual_predictions.csv',[dict(match_id=r['match_id'],club=r['club'],season=r['sy'],direction=r['direction'],game=r['game'],residual=y[i],**{k:v[i] for k,v in preds.items() if k!='zero'}) for i,r in enumerate(raw)])

# Direct error decomposition: row-weighted total MSE versus within-club demeaned MSE.
decomp=[]
for d in ['L1_movers','relegated_into_L1','promoted_into_L1','nonmover']:
 mask=(game<=20)&(np.isin(directions,['promoted_into_L1','relegated_into_L1']) if d=='L1_movers' else directions==d)
 demean=y.copy(); oracle=np.zeros(n)
 for k in np.unique(keys[mask]):
  k_mask=mask&(keys==k);oracle[k_mask]=y[k_mask].mean();demean[k_mask]-=oracle[k_mask]
 decomp.append(dict(group=d,rows=int(mask.sum()),club_seasons=len(set(keys[mask])),mse=np.mean(y[mask]**2),within_club_variance=np.mean(demean[mask]**2),fraction_mse_club_offset=1-np.mean(demean[mask]**2)/np.mean(y[mask]**2),mae_club=metric(mask,np.zeros(n)),oracle_demeaned_mae=metric(mask,oracle)))
save('error_decomposition.csv',decomp)

# Temporal/binary/quantile strata; equal-club average within each stratum.
strata=[]
stratifiers={'venue':np.where(col['is_home']==1,'home','away'),
 'game_bin':np.where(game<=5,'01-05',np.where(game<=10,'06-10','11-20')),
 'opponent_prior_gd':np.where(col['opp_previous_gd']>.3,'strong_gt0.3',np.where(col['opp_previous_gd']<-.3,'weak_lt-0.3','middle')),
 'own_qs_probability':np.where(col['p_model']>.55,'favourite_gt55',np.where(col['p_model']<.30,'outsider_lt30','middle')),
 'manager_last3':np.where(col['manager_change3']==1,'changed','not_observed_changed'),
 'continuity':np.where(~np.isfinite(col['retained_share']),'missing',np.where(col['retained_share']<.5,'below_half','at_least_half')),
 'higher_arrivals':np.where(~np.isfinite(col['higher_arrival_share']),'missing',np.where(col['higher_arrival_share']>0,'any','none')),
 'age':np.where(~np.isfinite(col['age_mean']),'missing',np.where(col['age_mean']<25,'below25',np.where(col['age_mean']>28,'over28','25to28')))}
for s,v in stratifiers.items():
 for d in ['relegated_into_L1','promoted_into_L1','nonmover','relegated_into_L2']:
  for level in np.unique(v):
   mask=(directions==d)&(game<=20)&(v==level)
   if not mask.any():continue
   strata.append(dict(stratifier=s,group=d,level=level,rows=int(mask.sum()),club_seasons=len(set(keys[mask])),mae_club=metric(mask,np.zeros(n)),signed_club=grouped(mask,y).mean(),mean_p_model=100*col['p_model'][mask].mean(),mean_p_market=100*col['p_market'][mask].mean()))
save('stratified_errors.csv',strata)

clubs=[]
for k in np.unique(keys[game<=20]):
 mask=(keys==k)&(game<=20);i=np.where(mask)[0][0]
 record=dict(club=raw[i]['club'],season=int(years[i]),direction=directions[i],rows=int(mask.sum()),bias=y[mask].mean(),mae=np.abs(y[mask]).mean())
 for binname,bm in [('g1_5',game<=5),('g6_10',(game>5)&(game<=10)),('g11_20',game>10)]:
  q=mask&bm
  record[binname+'_n']=int(q.sum());record[binname+'_bias']=y[q].mean() if q.any() else np.nan;record[binname+'_mae']=np.abs(y[q]).mean() if q.any() else np.nan
 clubs.append(record)
save('club_trajectories.csv',clubs)

validation=dict(rows=n,fixtures=len(set(col['match_id'])),canonical_transition_rows=len(control),canonical_max_difference_pp=max(abs(100*float(r['delta'])-float(ours[(r['match_id'],r['club'])]['residual'])) for r in control),target_years=sorted(set(years.tolist())),chronology='all previous/lineup dates strictly earlier',ridge_penalty=10,training='all earlier-season club-fixtures; equal club-season weights',evaluation='first 20 league games; equal club-season weights',seed=20261013)
(P/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
print('Wrote diagnostics; no QS or outcome model fitted.')
