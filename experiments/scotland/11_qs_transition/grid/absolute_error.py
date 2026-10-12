"""DESIGN §9 / D23: post-hoc equal-club-season MAE, paired cluster bootstrap.

Input is the same priced first-20 cohort as the frozen scorer, with game numbers
counted on all league fixtures. All output probabilities are in percentage points.
Run `python3 absolute_error.py --test` for exact synthetic weighting/pairing checks.
"""
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

P = Path(__file__).resolve().parent
B = 10000
SEED = 20260927
CONTROL = 'qs_control'
DIRECTIONS = ['relegated_into_L1', 'promoted_into_L1', 'entered_spfl']
SPLITS = [('1-20', 1, 20), ('1-10', 1, 10), ('11-20', 11, 20)]


def percentile(values, q):
    ordered = sorted(values)
    i = (len(ordered) - 1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    return ordered[lo] + (i - lo) * (ordered[hi] - ordered[lo])


def stats(paired, bootstrap_B=B):
    """Each cluster is one club-season; resample identical clusters in both arms."""
    keys = sorted(paired)
    if not keys:
        return dict(mae_club_pp=math.nan, control_mae_club_pp=math.nan,
                    delta_mae_club_pp=math.nan, lo=math.nan, hi=math.nan,
                    worst_club_abs_mean_pp=math.nan, worst_club='', n_club_seasons=0)
    model_mae = [100 * mean(abs(a) for a, _ in paired[k]) for k in keys]
    control_mae = [100 * mean(abs(c) for _, c in paired[k]) for k in keys]
    differences = [a - c for a, c in zip(model_mae, control_mae)]
    signed = [100 * abs(mean(a for a, _ in paired[k])) for k in keys]
    rng = random.Random(SEED)
    draws = [mean(differences[rng.randrange(len(keys))] for _ in keys)
             for _ in range(bootstrap_B)]
    worst = max(range(len(keys)), key=lambda i: signed[i])
    return dict(mae_club_pp=mean(model_mae), control_mae_club_pp=mean(control_mae),
                delta_mae_club_pp=mean(differences), lo=percentile(draws, .025),
                hi=percentile(draws, .975), worst_club_abs_mean_pp=signed[worst],
                worst_club='|'.join(keys[worst]), n_club_seasons=len(keys))


def test():
    # Unequal game counts must NOT cause fixture-weighted MAE.
    paired = {('A', '24/25'): [(.1, .2)],
              ('B', '24/25'): [(-.3, -.4), (.3, .4), (.3, .4)]}
    s = stats(paired)
    assert math.isclose(s['mae_club_pp'], 20)
    assert math.isclose(s['control_mae_club_pp'], 30)
    assert math.isclose(s['delta_mae_club_pp'], -10)
    assert math.isclose(s['lo'], -10) and math.isclose(s['hi'], -10)
    assert math.isclose(s['worst_club_abs_mean_pp'], 10)
    assert math.isclose(stats({('A', '24/25'): [(.1, .2)]})['lo'], -10)
    self_stats = stats({k: [(a, a) for a, _ in v] for k, v in paired.items()})
    assert self_stats['delta_mae_club_pp'] == self_stats['lo'] == self_stats['hi'] == 0
    print('ABS_ERROR_TEST_PASS: equal-club weighting, abs-before-mean, pairing, single-cluster CI, self-reference')


def main():
    runs = list(csv.DictReader((P / 'runs.csv').open()))
    rows = list(csv.DictReader((P / 'transition_errors.csv').open()))
    by_arm = defaultdict(dict)
    for r in rows:
        key = (r['direction'], r['club'], r['season'], int(r['match_id']))
        assert key not in by_arm[r['arm']], ('duplicate observation', r)
        by_arm[r['arm']][key] = (int(r['game']), float(r['delta']))
    control = by_arm[CONTROL]
    results = []
    for run in runs:
        arm = run['label']
        assert by_arm[arm].keys() == control.keys(), ('unpaired observations', arm)
        for direction in DIRECTIONS:
            for label, low, high in SPLITS:
                clusters = defaultdict(list)
                fixtures = set()
                for key, (game, delta) in by_arm[arm].items():
                    cgame, cdelta = control[key]
                    assert game == cgame
                    if key[0] == direction and low <= game <= high:
                        clusters[(key[1], key[2])].append((delta, cdelta))
                        fixtures.add(key[3])
                results.append(dict(arm=arm, run_id=run['run_id'],
                    control_run_id=next(r['run_id'] for r in runs if r['label'] == CONTROL),
                    direction=direction, game_split=label, status='post_hoc', units='pp',
                    bootstrap_B=B, seed=SEED, **stats(clusters),
                    n_obs=sum(map(len, clusters.values())), n_fixtures=len(fixtures)))
    # This amendment is a separate criterion; NEVER overwrite frozen gate flags.
    passes = {}
    for run in runs:
        direction_rows = [r for r in results if r['arm'] == run['label'] and
                          r['game_split'] == '1-20' and r['direction'] in DIRECTIONS[:2]]
        assert len(direction_rows) == 2
        passes[run['label']] = (all(r['delta_mae_club_pp'] <= 0 for r in direction_rows)
                               and any(r['hi'] < 0 for r in direction_rows))
    for r in results:
        r['posthoc_error_criterion_pass'] = passes[r['arm']]
    with (P / 'abs_error_table.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    assert len(results) == len(runs) * 9
    print('ABS_ERROR_TABLE_DONE rows=', len(results), 'posthoc_pass=', passes)


if __name__ == '__main__':
    test() if '--test' in sys.argv else main()
