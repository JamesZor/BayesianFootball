#!/usr/bin/env python3
"""Idempotently append the second fetched web-evidence tranche (2026-09-23).

The quotes were checked with web_fetch by the status-evidence researcher. Season
context is not a precise publication timestamp. This deliberately adds no
inferred-by-silence club labels. Source files remain manually curated evidence.
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
# ID, club, publisher, URL, exact excerpt, year, status, interpretation
RECORDS = [
 ('S010', "Queen's Park", "Queen's Park FC",
  'https://queensparkfc.co.uk/history',
  'A full-time professional club about to play our second season in the Scottish Championship.',
  2023, 'Full-Time', 'Official history explicitly anchors statement to summer2023; not the webpage publication date.'),
 ('S011', 'Falkirk', 'Falkirk FC',
  'https://falkirkfc.co.uk/wp-content/uploads/2025/10/FFC-2025-COMMERCIAL-OPPORTUNITIES-A4_v7.pdf',
  'we are the only full-time professional football club in Falkirk District',
  2025, 'Full-Time', 'Brochure cover explicitly2025/2026 season; does not verify preceding LeagueOne seasons.'),
 ('S012', 'Arbroath', 'The Herald',
  'https://www.heraldscotland.com/sport/25659884.arbroath-became-premiership-promotion-contenders-playing-part-time/',
  'full-time opponents like Queen’s Park, Ross County and St Johnstone',
  2025, 'Part-Time', 'Article explicitly describes Arbroath part-time status and2025/26 context; quoted excerpt identifies opposing FT clubs.'),
 ('S013', 'Peterhead', 'Press and Journal',
  'https://www.pressandjournal.co.uk/fp/sport/football/peterhead-fc/6359066/caleb-goldie-life-post-celtic-time-at-peterhead-so-far/',
  'being part-time at Peterhead',
  2023, 'Part-Time', '2023/24 context: first season after Celtic release; January2024 photographs. Date unresolved.'),
 ('S014', 'Peterhead', 'Press and Journal',
  'https://www.pressandjournal.co.uk/fp/sport/football/peterhead-fc/6490643/max-barry-reveals-why-peterhead-was-the-ideal-move/',
  'remaining part-time with League Two Peterhead is the best option right now',
  2024, 'Part-Time', 'Incoming2024/25 context following explicitly described2023-24 Buckie season.'),
 ('S015', 'Alloa Athletic', 'Rangers FC',
  'https://www.rangers.co.uk/article/loan-rangers-making-an-impact-at-alloa-athletic/5etFv4HG7W6T8t9GjXbRf',
  'It’s easy to come from a club like Rangers to a part-time team',
  2025, 'Part-Time', 'Manager quote;2025/26 inferred from named loan cohort and DannyRohl. Context-dated, not precise PIT evidence.'),
 ('S016', 'Kelty Hearts', 'Dunfermline Press',
  'https://www.dunfermlinepress.com/sport/25825177.kelty-hearts-american-owners-best-part-time-club-aim/',
  'we’re going to be the best part-time club there is',
  None, 'Part-Time', 'Ownership testimony; season/date unresolved. NOT assigned to panel until dated.'),
 ('S017', 'Stenhousemuir', 'BBC Sport',
  'https://www.bbc.com/sport/football/live/c617jwdk42nwt',
  'the part-time hosts were unable to repeat the feat',
  2026, 'Part-Time', 'Championship-era2026/27 match context; no backward verification.'),
 ('S018', 'Queen of the South', 'BBC News',
  'https://www.bbc.co.uk/news/articles/cpvxpgpdw9ro',
  'The Doonhamers were relegated from the Championship in 2022 and have remained full-time in League One since then.',
  2022, 'Full-Time', 'Explicit retrospective continuity22/23 through25/26; report reviews24/25 accounts ahead of1May2026 AGM. NOT historical PIT knowledge.'),
 ('S019', 'Montrose', 'The Courier',
  'https://www.thecourier.co.uk/fp/sport/football/2784072/stewart-petrie-on-5-years-at-montrose-still-striving-for-improvement-greatest-memory-and-why-jobs-elsewhere-have-never-been-considered/',
  'We’re only here part-time as manager and players',
  2021, 'Part-Time', 'December2021 fifth-anniversary context. Opponent FT list conflicts with other sources and is NOT used to classify opponents.'),
 ('S020', 'Alloa Athletic', 'BBC Sport',
  'https://www.bbc.co.uk/sport/football/60006014',
  'That fuelled the part-time hosts with hope',
  2021, 'Part-Time', 'Alloa1-2Celtic ScottishCup report22January2022; supports21/22 club status.'),
]


def load(name):
    with (DATA/name).open(newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save(name, fields, records):
    with (DATA/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main():
    sf, sources = load('status_sources.csv')
    ef, evidence = load('operational_status_evidence.csv')
    ids = {r[0] for r in RECORDS}
    sources = [r for r in sources if r['source_id'] not in ids]
    evidence = [r for r in evidence if r['source_id'] not in ids
                and r['club_name'] != 'Queen of the South']  # superseded by explicit S018 continuity
    for sid, club, publisher, url, quote, year, status, note in RECORDS:
        sources.append(dict(source_id=sid, club_name=club, source_type='fetched article',
                            publisher=publisher, published_date='', url=url, quote=quote,
                            evidence_class='Verified', fetched_utc='2026-09-23',
                            fetch_status='Fetched', scope='first-team operational model', notes=note))
        if year is None:
            continue
        clubs = [(club, status)]
        if sid == 'S012':
            clubs += [("Queen's Park", 'Full-Time'), ('Ross County', 'Full-Time'), ('St Johnstone', 'Full-Time')]
        for name, label in clubs:
            end_year = 2026 if sid == 'S018' else year + 1
            level = 'Inferred' if sid == 'S015' else 'Verified'
            evidence.append(dict(club_name=name, club_id='', operational_status=label,
                                 evidence_level=level, effective_from=f'{year}-07-01',
                                 effective_to=f'{end_year}-06-30', source_id=sid, classification_note=note))
    save('status_sources.csv', sf, sources)
    save('operational_status_evidence.csv', ef, evidence)
    print(f'Curated {len(sources)} fetched sources; {len(evidence)} evidence intervals.')

if __name__ == '__main__':
    main()
