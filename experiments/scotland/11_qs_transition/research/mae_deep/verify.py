#!/usr/bin/env python3
"""Small deterministic checks: data filtration, known denominators and findings."""
from pathlib import Path
import csv,json,math,datetime
P=Path(__file__).resolve().parent

def rows(name):
 with open(P/name,newline='') as f:return list(csv.DictReader(f))
checks=[]
def check(name,ok):
 checks.append({'check':name,'pass':bool(ok)})
 assert ok,name
v=json.loads((P/'validation.json').read_text())
check('canonical_first20_rows_and_values',v['canonical_transition_rows']==221 and v['canonical_max_difference_pp']==0)
check('win_selection_denominators',v['rows']==1795 and v['fixtures']==898)
check('target_seasons_only',v['target_years']==[2023,2024,2025])
raw=rows('panel.csv')
check('strict_historical_feature_dates',all(not r[k] or r[k]<r['start_timestamp'] for r in raw for k in ['previous_date','lineup_date','opp_previous_date','opp_lineup_date']))
lookup={(r['match_id'],r['club']):r for r in raw}
check('delayed_features_strict',all(not r[k] or datetime.datetime.fromisoformat(r[k])<datetime.datetime.fromisoformat(lookup[(r['match_id'],r['club'])]['start_timestamp'])-datetime.timedelta(days=int(r['days'])) for r in rows('delayed_features.csv') for k in ['own_latest','opp_latest']))
check('baseline_primary_reproduced',all(abs(float(r['mae_club'])-{'relegated_into_L1':5.8018667256592185,'promoted_into_L1':4.952105747587713}[r['direction']])<1e-9 for r in rows('baseline_summary.csv') if r['season']=='all' and r['direction'] in ['relegated_into_L1','promoted_into_L1']))
rr=rows('robustness_results.csv')
arms=[r for r in rr if r['check']=='six_arm_form_pair' and r['test_season']=='both' and r['group']=='L1_movers']
check('six_arms_same_direction',len(arms)==6 and all(float(r['delta'])<0 for r in arms))
main=[r for r in rows('chronological_diagnostics.csv') if r['block']=='form_pair' and r['group']=='L1_movers' and r['test_season']=='both'][0]
check('main_holdout_denominator',int(main['rows'])==130 and int(main['club_seasons'])==7)
check('main_gain_numerical',abs(float(main['delta_vs_zero'])-(-0.8075068269))<.001)
check('no_unversioned_valuation_regressions',not any(r['feature'].startswith('wealth') for r in rows('diagnostic_coefficients.csv')))
check('valuation_vintage_failure_recorded',rows('value_vintage_audit.csv')[0]['players']=='575' and rows('value_vintage_audit.csv')[0]['distinct_positive_values']=='1')
check('outcome_caution_preserved',any(r['block']=='form_pair' and r['test_season']=='both' and r['cohort']=='all' and float(r['delta_multiclass_ll'])>0 for r in rows('outcome_sanity.csv')))
with open(P/'verification.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['check','pass']);w.writeheader();w.writerows(checks)
print(f'{len(checks)} verification checks passed.')
