#!/usr/bin/env python3
"""Offline invariant checks and SHA256 manifest; no DB or statistical fitting."""
from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def club_key(name):
    # Only documented spelling aliases; no fuzzy matching of distinct clubs.
    aliases = {'heart-of-midlothian': 'hearts', 'the-spartans': 'the-spartans',
               'inverness-caledonian-thistle': 'inverness-caledonian-thistle'}
    key = name.lower().replace("'", '').replace('.', '').replace(' ', '-')
    if key.endswith('-fc'):
        key = key[:-3]
    return aliases.get(key, key)


def main():
    status = rows(ROOT / 'data/spfl_club_operational_status.csv')
    expected = {f'{y}/{y+1}' for y in range(21, 27)}
    assert {r['season'] for r in status} == expected
    assert Counter(r['season'] for r in status) == Counter({s: 42 for s in expected})
    assert len({(r['season'], club_key(r['club_name'])) for r in status}) == 252
    assert {r['operational_status'] for r in status} <= {'Unknown', 'Full-Time', 'Part-Time', 'Hybrid'}
    sources = rows(ROOT / 'data/status_sources.csv')
    assert len({r['source_id'] for r in sources}) == len(sources)
    by_source = {r['source_id']: r for r in sources}
    for r in status:
        if r['operational_status'] == 'Unknown':
            assert r['evidence_level'] == 'Unknown'
        else:
            for source_id in r['source_id'].split(';'):
                assert source_id in by_source, r
                assert by_source[source_id]['fetch_status'] == 'Fetched'
                assert by_source[source_id]['quote']
    web = rows(ROOT / 'data/spfl_membership_web_2627.csv')
    db_membership = {(int(r['tournament_id']) - 53, club_key(r['club_name']))
                     for r in status if r['season'] == '26/27'}
    web_membership = {(int(r['tier']), club_key(r['club_name'])) for r in web}
    assert db_membership == web_membership, (db_membership - web_membership, web_membership - db_membership)

    ledger = rows(ROOT / 'data/slate_2026-09-19.csv')
    assert len({r['order_id'] for r in ledger}) == len(ledger), 'multiplicative ledger join'
    m12 = [r for r in ledger if r['run_name'] == 'm12_joint_hybrid_synergy']
    assert len(m12) == 13
    assert all(not r['model_run_id'] for r in m12)
    focus = [r for r in m12 if r['match_id'] in {'16362442', '16362450'} and r['market_group'] == '1X2']
    assert len(focus) == 4
    D = Decimal
    assert sum(D(r['risk']) for r in focus) == D('20.35')
    assert sum(D(r['risk_filled']) for r in focus) == D('16.29')
    assert sum(D(r['net_pnl']) for r in m12) == D('-5.09')
    assert all(abs(D(r['edge']) - (D(r['p_model']) - D(r['p_market']))) <= D('0.000001') for r in ledger)

    manifest = {}
    for directory in ('data', 'results'):
        for path in sorted((ROOT / directory).glob('*.csv')):
            content = path.read_bytes()
            manifest[str(path.relative_to(ROOT))] = {
                'sha256': hashlib.sha256(content).hexdigest(),
                'bytes': len(content), 'rows': len(rows(path)),
            }
    output = ROOT / 'results/artifact_manifest.json'
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    print('PASS: 252 unique membership rows; evidence references; 42 official current-tier matches; ledger arithmetic.')
    print('Status evidence counts:', dict(Counter(r['evidence_level'] for r in status)))
    print(f'Wrote {len(manifest)} CSV checksums to {output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
