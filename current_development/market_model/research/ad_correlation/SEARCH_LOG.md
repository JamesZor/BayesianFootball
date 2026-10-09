# Search log — attack/defence correlation research (2026-10-06)

Each row: time (BST), source, query or chase (API keys omitted), hit count, notes.
OpenAlex calls used `api_key` and `mailto` parameters, which are not logged.

| Time | Source | Query / chase | Hits | Notes |
|---|---|---|---:|---|
| 11:53 | library lib-grep --docs | `correlat[a-z]* between (the )?attack and defen` | 0 | no paper states an attack–defence correlation in those words |
| 11:53 | library lib-grep --docs | `bivariate normal` | 19 hits / 5 docs | benz2021 (11), ghosh2020, ruppert2010, jondeau2007, mchale2011 |
| 11:53 | library lib-grep --docs | `LKJ` / `Wishart` / `Cholesky` | 0 / 3 docs / 6 docs | none football team-effect covariance |
| 11:53 | library lib-grep --docs | `exchangeab` / `overshrink` / `sum.to.zero` | 9 / 3 / 5 docs | baio2010, karlis2003, owen2011 |
| 11:53 | library lib-grep --docs | `team strength` | 70 hits / 11 docs | glickman1998, benz2021, foulley2015, holmes2023, ... |
| 11:53 | library lib-sql | all 140 kind='paper' rows listed | 140 | screened by title |
| 11:54 | library lib-grep per doc | `correlat` in 10 core football papers | — | koopman2013, koopman2019, owen2011, baio2010, ridall2024, glickman1998, karlis2003/2008, macridemartino2026, dixon1997 |
| 11:54 | library lib-grep per doc | `sum.to.zero|identif|constrain|restrict|covariance|independen` in 6 core papers | — | constraint and covariance statements located |
| 11:55 | OpenAlex `/works?search=` | 8 keyword queries (attack defence correlation; correlated attack and defence parameters; bivariate normal random effects attack defence soccer; promoted relegated multiple divisions; multivariate random walk; latent factor; betting odds prior; ratings from odds) | 1641 / 1135 / 41 / 1715 / 413 / 575 / 253 / 1148 | relevance-ranked top 25 screened by title; mostly noise; found Egidi 2018, Wunderlich 2018, Duffield 2024, cross-league 2026 |
| 11:56 | OpenAlex forward citations | `filter=cites:W2109802089` (Baio 2010) | 163 | all titles screened |
| 11:56 | OpenAlex forward citations | `filter=cites:W1480768621` (Koopman & Lit 2015) | 18 | all screened |
| 11:56 | OpenAlex forward citations | `filter=cites:W2788508817` (Egidi et al. 2018) | 21 | all screened |
| 11:56 | OpenAlex forward citations | `filter=cites:W2113003261` (Owen 2011) | 40 | all screened |
| 11:57 | OpenAlex forward citations | `filter=cites:W2754826543` (Koopman & Lit 2019) and `cites:W2035378550` (Rue & Salvesen 2000) | 58 / 208 | all screened |
| 11:57 | OpenAlex forward citations | `filter=cites:W2172229041&search=attack defence` (Karlis & Ntzoufras 2003) | (subset) | merged; 429 unique citing works in total, title-filtered |
| 11:58 | OpenAlex `/works?search=` | offensive and defensive strengths correlated random effects; offensive defensive ratings state space NFL; Rue Salvesen; Knorr-Held dynamic rating; multivariate random walk team abilities; promotion relegation priors | 4580 / 289 / 112 / 117 / 1753 / 50 | top 15 each; found Lopez et al. 2018 |
| 11:59 | arXiv API | first attempt over http returned empty bodies (redirect); rerun over https | — | — |
| 11:59 | arXiv API | `abs:attack AND abs:defence AND abs:correlation`; `abs:football AND abs:"attack and defence" AND abs:hierarchical`; `abs:soccer AND abs:"team strength" AND abs:dynamic`; `abs:football AND abs:"betting odds" AND abs:prior`; `abs:"attacking and defensive" AND abs:football`; `abs:football AND abs:"bivariate Poisson"`; `abs:football AND abs:"state space"`; `abs:soccer AND abs:hierarchical AND abs:Bayesian AND abs:teams` | 15 / 0 / 0 / 0 / 2 / 0 / 13 / 1 | phrase search is brittle |
| 12:00 | arXiv API | `cat:stat.AP AND (all:football OR all:soccer) AND all:poisson`; `… AND all:dynamic`; `(all:football OR all:soccer) AND all:attack AND all:defence`; `… all:defense` | 31 / 27 / 1 / 18 | found Broatch & Karl 1710.05284, Tsokos 1807.01623, Ley 1705.09575, Egidi 1802.08848 |
| 12:01 | library lib-ask | synthesis question on attack/defence correlation, covariance/factor/mixture structures, 1-D vs 2-D comparisons | — | timed out after 600 s ("still working"); pane not touched; not retried |
| 12:03 | OpenAlex `/works/doi:` | verified DOIs for Tsokos, Egidi, Wunderlich, Baker & McHale, Rue & Salvesen, Crowder; Lopez (my remembered DOI was wrong; corrected by title search); Ley (title search); Knorr-Held (only the LMU preprint DOI found) | — | corrections applied to candidates.csv and BIBLIOGRAPHY.md |
| 12:04 | Web (WebSearch) | `Thompson soccer predictions Bayesian mixed effects models attack defense correlation wjakethompson`; `Rue Salvesen 2000 … pdf`; `Stan hierarchical football model attack defense LKJ correlation multi_normal_cholesky teams` | ~9 each | search engine returned generic hits; Thompson's site fetched directly instead; no OA copy of Rue & Salvesen |
| 12:05 | Web (curl + pandoc) | https://wjakethompson.github.io/soccer/ pages define-model, diagnostic-output, conclusion, fit-model, predict, gather-data, index | 7 pages | independent effects only |
| 12:06 | citation chase (backward) | Broatch & Karl reference list → Karl, Yang & Lohr (2014) on arXiv (`ti:"correlated random effects"`, `au:Karl`) | 2 | 1403.7676 downloaded |
| 12:08 | OpenAlex `/works/doi:` | LKJ 2009, Frühwirth-Schnatter & Wagner 2010, Self & Liang 1987, Vehtari 2017, Papaspiliopoulos 2007, Bitto 2019; Stram & Lee by title; Maher 1982, Dixon & Coles 1997 | — | verified |
| 12:09 | OpenAlex forward citations with search | `filter=cites:W2136744117` (Maher), `cites:W1975926620` (Dixon & Coles), `cites:W2172229041` (Karlis & Ntzoufras) × {correlated random effects, attack defence correlation, multivariate random walk, latent factor, promoted relegated} | 27 / 26 / 34 / 24 / 35 (the other 10 calls failed with a jq parse error on control characters) | no new correlated-effects football paper |
| 12:09 | Web (curl) | OA PDFs: PLOS (Wunderlich), LMU epub (Knorr-Held, paper_98.pdf), arXiv 1701.05976 (Lopez); OUP qnad007 for Basini returned HTML (not staged) | — | — |
| 12:10 | OpenAlex + arXiv | Ruiz & Pérez-Cruz 2015 (title search; no arXiv copy found with `ti:"college basketball" AND ti:generative`); Guo et al. 2012 Springer PDF returned HTML | 1 / 0 | abstract only |
| 12:12 | library lib-sql | books/theses/notes with sport, football, soccer, hierarchical, state space or multilevel in the title | 16 | egidi2025 book, kharrat2016 thesis, chan2018 handout selected |
| 12:13 | library lib-show + rg | full extracted text of egidi2025 (262 pp.), kharrat2016 (124 pp.), chan2018 (7 pp.); regex `correlat…(attack|defen|offen)`, `multivariate normal`, `bivariate normal`, `LKJ`, `Wishart`, `Cholesky` | 0 relevant hits | footBayes prior read at PDF pp. 147–148 |
| 12:13 | library lib-grep | `attack and defen[cs]e strength` in koopman2013 results pages | 33 | Table 2 scale estimates; model (d) |
| 12:15 | OpenAlex search | `Approximate leave-future-out cross-validation for Bayesian time series models` | 2 | DOI verified |
| 12:18 | OpenAlex search | `Estimating team strength in the NFL` with author Glickman; free-text variant | 0 matching | recalled 2017 handbook chapter could not be verified; removed from PAYWALLED.md |
| 12:19 | OpenAlex `/works/doi:` biblio | volume/issue/pages for 19 cited works | 19 | all heading details match; Ley 19(1) 55–73 confirmed; Lopez page range absent |
