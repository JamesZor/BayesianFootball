#!/usr/bin/env python3
"""Recover publication dates omitted by readable web extraction (no DB access)."""
import csv
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'S006': 'https://arbroathfc.co.uk/transfer-deadline-day-action-at-arbroath/',
    'S008': 'https://www.bbc.com/sport/football/articles/c97z27mqjmro',
    'S009': 'https://www.theterrace.scot/news/26254406.stewart-petrie-montroses-big-summer-recruitment/',
}

def main():
    rows = []
    for source_id, url in SOURCES.items():
        html = urllib.request.urlopen(url, timeout=20).read().decode('utf-8')
        dates = sorted({x.rstrip('\\') for x in re.findall(
            r'(?:datePublished|article:published_time)[^0-9]{1,25}(\d{4}-\d{2}-\d{2}[^"< ]*)', html)})
        if len(dates) != 1:
            raise ValueError(f'{source_id}: expected unambiguous publication date, found {dates}')
        rows.append(dict(source_id=source_id, url=url, published_timestamp=dates[0]))
    path = ROOT / 'data/status_publication_metadata.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['source_id', 'url', 'published_timestamp'])
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {len(rows)} source publication dates.')

if __name__ == '__main__':
    main()
