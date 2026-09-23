# Operational-status evidence and coverage

As of 2026-09-23. This is a **partial evidence-backed panel**, not a completed ground-truth census. `r03_status_construction.jl` joins SQL season membership to `data/operational_status_evidence.csv`; `Unknown` is never imputed to Part-Time. `Verified` means a fetched source supports the stated season; it does **not** mean the source was available before every match of that season. `Inferred` denotes explicitly bounded continuity, excluded from the strict analysis.

## Existing evidence

The source register `data/status_sources.csv` carries exact short excerpts and URLs. The following are source-supported statements, not automatic all-season labels:

- **Cove:** [18 May 2023 announcement](https://coverangersfc.com/2023/05/18/club-statement-full-time-football/) describes the preceding Championship campaign as part-time and the coming 23/24 season as a hybrid transition. [1 January 2024 statement](https://coverangersfc.com/2024/01/01/chairmans-new-year-message-3/) corroborates implementation. Consequently the prompt's undated PT classification cannot be carried into 2026 without new evidence.
- **Airdrie:** [18 April 2019 announcement](https://www.airdriefc.com/1819-news/180419/hybrid-model-the-future-for-airdrieonians) explicitly describes a mixture of FT/PT players for 19/20. Its extension to 21/22–26/27 is labelled inferred, not verified.
- **Hamilton:** [BBC, 12 June 2023](https://www.bbc.com/sport/football/65881664) confirms staying FT after relegation for 23/24. The prior-season classification is inferred from “remain”.
- **Inverness:** [21 May 2024 report](https://www.inverness-courier.co.uk/sport/ict-confirm-they-will-remain-full-time-351275/) confirms FT for 24/25 despite considering other options. Insolvency alone does not establish a change of training regime.
- **Arbroath:** [Ryan Flynn signing article, 31 August 2024](https://arbroathfc.co.uk/transfer-deadline-day-action-at-arbroath/) explicitly describes moving from FT to PT. This retrospectively supports the 24/25 season label.
- **Queen of the South:** [BBC appointment report, 9 May 2024](https://www.bbc.com/sport/football/articles/c97z27mqjmro) quotes Peter Murphy welcoming FT football. The subsequently fetched [BBC financial review](https://www.bbc.co.uk/news/articles/cpvxpgpdw9ro) explicitly states FT continuity since2022 relegation, so22/23–25/26 now have retrospectively verified labels, superseding the earlier inferred23/24 row.
- **Montrose:** [5 July 2026 preview](https://www.theterrace.scot/news/26254406.stewart-petrie-montroses-big-summer-recruitment/) explicitly identifies PT recruitment and Tuesday/Thursday training for 26/27.

The last three exact publication dates were omitted by readable-page extraction and recovered from HTML `datePublished` metadata using `r09_fetch_status_metadata.py`, saved in `data/status_publication_metadata.csv`. Retrieval dates are day-precision; no exact retrieval time is claimed.

## Final evidence tranche

`r11_extend_status_evidence.py` records the later fetched sources, including official Queen's Park history and Falkirk brochure, explicit25/26 Arbroath/FT-opponent reporting, Peterhead23/24–24/25 PT reports, Alloa21/22 PT reporting and Montrose21/22 first-person testimony. The conflicting opponent FT list in the latter is quarantined rather than used to label Alloa/Cove. Alloa's later loan-report season placement remains inferred; Kelty's ownership quotation has no panel assignment until dated. These distinctions survive in the source notes and interval CSV.

Final panel: **252 rows,45 clubs;21 Verified,8 Inferred,223 Unknown**, from20 fetched source records (two records may refer to different excerpts of one document). The full classification criterion remains unmet. See final README for season-level counts and strictly filtered analyses.

## Membership and analytical use

`data/spfl_membership_web_2627.csv` independently transcribes the official [SPFL current tables](https://spfl.co.uk/clubs/ross-county/fixtures), updated 21–22 September 2026 and fetched 23 September. `r07` checks all 42 current club/tier assignments against the SQL-derived panel. Membership is not operational-status evidence.

The interval table describes research season labels with inclusive end dates. The constructor selects the interval at 1 July. This is **not** a match-level point-in-time feature builder, and cannot represent an unrecorded mid-season transition. Production must use effective dates **and** publication/knowledge timestamps, preserve revisions, and refuse ambiguous/missing status. Retrospective verification cannot simply become a historical model input.

Do not use current league, generic professional registration, payroll, club size, lack of a change announcement, academy working hours or part-time administrative vacancies as direct proof of first-team operational status. Unknown rows are unfinished research, not an assertion that evidence does not exist.
