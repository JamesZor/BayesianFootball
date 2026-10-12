"""Build the S3 gate/diagnostic tables from committed v1.2 score exports."""
import csv
from pathlib import Path
P = Path(__file__).resolve().parent
ARMS = ['qs_control', 'qs_a2', 'qs_entrant', 'qs_summer']

def load(name):
    with (P / name).open() as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (P / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

scores = {a: load(f'scores_{a}.csv') for a in ARMS}

def score(a, subset, market, metric):
    rows = [r for r in scores[a] if (r['subset'], r['market'], r['metric']) == (subset, market, metric)]
    assert len(rows) == 1, (a, subset, market, metric)
    r = rows[0]
    assert r['scorecard_version'] == 'v1.2', r['scorecard_version']
    assert r['panel'] == '56+57|23/24,24/25,25/26|n=1070', r['panel']
    return r

gates = []; diagnostics = []; cohorts = []
for arm in ARMS:
    ll = score(arm, 'target', 'all', 'delta_logloss_vs_control')
    bias = {}; delta = {}
    for direction in ['relegated_into_L1', 'promoted_into_L1']:
        subset = f'transition_{direction}_first20'
        bias[direction] = score(arm, subset, '1X2', 'transition_bias_pp')
        delta[direction] = score(arm, subset, '1X2', 'delta_abs_bias_vs_control')
    point_falls = all(float(r['value']) < 0 for r in delta.values())
    excludes_zero = any(float(r['hi']) < 0 or float(r['lo']) > 0 for r in delta.values())
    gate1 = float(ll['hi']) < .002
    gate2 = point_falls and excludes_zero
    row = dict(arm=arm, run_id=ll['run_id'], control_run_id=ll['control_run_id'], bootstrap_B=10000,
               delta_logloss=float(ll['value']), ll_lo=float(ll['lo']), ll_hi=float(ll['hi']), gate1=gate1)
    for direction in bias:
        row.update({f'{direction}_bias_pp': float(bias[direction]['value']),
                    f'{direction}_delta_abs_pp': float(delta[direction]['value']),
                    f'{direction}_lo': float(delta[direction]['lo']), f'{direction}_hi': float(delta[direction]['hi'])})
    row.update(gate2=gate2, both_gates=gate1 and gate2)
    gates.append(row)
    folds = load(f'fold_diagnostics_{arm}.csv')
    assert len(folds) == 60
    bad = [int(r['fold_idx']) for r in folds if float(r['r_hat_max']) > 1.05 or int(r['divergences']) / 4000 > .001]
    diagnostics.append(dict(arm=arm, run_id=ll['run_id'], n_folds=60,
        max_rhat=max(float(r['r_hat_max']) for r in folds),
        min_ess_bulk=min(float(r['ess_bulk_min']) for r in folds),
        min_ess_tail=min(float(r['ess_tail_min']) for r in folds),
        divergences=sum(int(r['divergences']) for r in folds), draws=240000,
        offending_folds=';'.join(map(str,bad)), fold_review_pass=not bad))
    for r in scores[arm]:
        if r['metric'] == 'transition_bias_pp':
            cohorts.append({k:r[k] for k in ['model','run_id','subset','value','lo','hi','n_obs','n_fixtures']})
write('gate_table.csv', gates)
write('convergence.csv', diagnostics)
write('cohort_bias.csv', cohorts)
metrics=[]
for arm in ARMS:
    for market, metric in [('all','logloss'),('all','ece'),('1X2','logloss'),('1X2','ece'),('1X2','compression_slope'),('OU2.5','logloss'),('BTTS','logloss')]:
        r=score(arm,'target',market,metric)
        metrics.append({k:r[k] for k in ['model','run_id','market','metric','value','n_obs','n_fixtures']})
write('target_metrics.csv',metrics)
print('S3_SUMMARY_CSV_DONE: all fixes gate1 pass, gate2 fail' if all(r['gate1'] and not r['gate2'] for r in gates[1:]) else 'S3_SUMMARY_CSV_DONE')
