# Research brief: attack/defence correlation and hierarchical team-strength models

**For:** a Claude Code research agent (Opus 5.5) running in tmux.
**Manager:** the Claude session that launched you. It reviews your output; James set the scope.
**Time box:** about 3 hours. Start 2026-10-06 ~12:00 laptop time. **Write the final files by 15:00.**
Check the time with `date` at each phase boundary.

## 1. Why this research exists

James's market model (TODO 039) inverts Betfair closing odds for Scottish League One/Two into
per-match double-Poisson rates, then fits team strengths with a weekly Gaussian random walk:

    log λ_home = μ + α[home] + β[away] + γ
    log λ_away = μ + α[away] + β[home]

- α is attack. **β means "concedes more"** (a defensive weakness), so a better team has α up and
  β down. Many papers use the opposite sign for defence (e.g. Koopman & Lit, Maher). **Normalise
  every reported correlation to our convention and say when you flipped a sign.**
- "Dixon–Coles" in James's vocabulary means this attack/defence/home parameterisation, not the
  ρ low-score correction.

**What the market data showed** (sources in §6):

| Finding | Value |
|---|---|
| Correlation of same-team weekly **innovations** (Δα, Δβ), model R6 | ρ_αβ = −0.82 [−0.87, −0.76] (retrospective); −0.86 [−0.94, −0.74] (honest) |
| Rotated axes q = (α − β)/2 "quality", s = (α + β)/2 "style": SD ratio σ_s/σ_q | 0.32 [0.26, 0.37]; 0.28 [0.19, 0.40] |
| σ_q, σ_s per week (R6) | 0.032 / 0.010 (retrospective); 0.026 / 0.007 (honest) |
| corr(Δq, Δs) | ≈ 0 (−0.02 [−0.25, 0.22]) |
| Correlation of season-average **levels** of α and β | −0.82 to −0.86, **even in R2, where innovations are independent** |
| Allowing the correlation (R6 vs independent R2) | fixes 90% interval coverage on both axes; honest predictive density improves |
| A Gaussian copula for (Δα, Δβ) | adequate: no posterior-predictive check is extreme (Phase B3) |
| Home advantage | ≈ 0.18 on log supremacy; no sign it scales with team quality (κ ≈ 0) |

The production **goal** model (`src/models/pregame/components/dynamics/team_level/multiscale.jl`,
`MultiScaleGRW`, fitted in Turing with NUTS) uses **independent** attack and defence walks at three
time scales: an initial level z₀, a season step zₛ and a weekly step zₖ, each with its own Gamma
prior on the scale. Read that file (read only) so your specs fit it.

**James's question:** is the attack/defence correlation evidence of a hierarchical or latent
structure, and how should a goal model use it?

## 2. Research questions

Answer each one in the report, citing every claim.

- **RQ1. Empirical record.** What attack/defence correlation have published models reported, for
  static levels or dynamic innovations, and with what method, league and era? Keep these apart:
  - an estimated correlation **hyperparameter**, versus a correlation of **fitted** team effects;
  - **levels**, versus **innovations** (steps).
- **RQ2. How the correlation has been modelled.** Cover each structure James named:
  1. **correlated team effects:** (α_t, β_t) drawn from a bivariate distribution with a learned
     correlation (LKJ, inverse-Wishart or Cholesky priors);
  2. **latent factor / rank reduction:** one overall-quality factor plus a small style dimension;
     compare with our quality/style rotation;
  3. **groups or mixtures of teams:** tiers, as in Baio & Blangiardo's three-group mixture;
  4. **multilevel by league or tier:** pooling across divisions and seasons, promoted and
     relegated teams, cross-league priors, and how the correlation behaves at each level.

  Include dynamic versions: correlated random walks, multivariate state-space or VAR team
  dynamics, and score-driven models.
- **RQ3. Does a correlation imply a hierarchy?** When is it a real latent structure, and when is
  it an artefact? Candidate artefacts include:
  - sum-to-zero or corner constraints (n teams under sum-to-zero induce about −1/(n−1));
  - a shared μ;
  - shrinkage;
  - the data simply containing good teams that score more and concede less.

  Explain why our **levels** correlate even with independent innovations. Then: how do papers
  test one factor against two (Bayes factors, LOO, nested zero-variance nulls, which sit on the
  boundary)?
- **RQ4. Does it help prediction?** Any out-of-sample comparison of correlated against
  independent attack/defence, or one-dimensional (Elo-like) against two-dimensional ratings.
- **RQ5. Practice.** Which priors have been used for the correlation and the scales (e.g. LKJ(η))?
  Non-centred parameterisations, HMC geometry, Kalman/state-space implementations. Also any use of
  **market-implied ratings as priors** for goal models.

## 3. Deliverables (all in this folder: `current_development/market_model/research/ad_correlation/`)

1. **`REPORT.md`**
   - A summary of at most 12 lines at the top.
   - Then RQ1–RQ5.
   - Then **2–3 candidate model specs** for the goal model. Examples, not prescriptions:
     - correlated innovations through a 2×2 Cholesky factor at each `MultiScaleGRW` time scale;
     - a quality/style rotation with independent σ_q, σ_s;
     - a correlated static level plus independent steps.

     For each spec give:
     - the equations, priors (with reasons; may be centred on the market numbers above) and
       identifiability constraints;
     - what result would falsify it;
     - expected cost and complexity in Turing/NUTS;
     - which papers support it.
   - Then open questions.
   - **No code.**
2. **`BIBLIOGRAPHY.md`**, an annotated bibliography. Each entry gives:
   - the citation and DOI or arXiv ID;
   - where you read it (library `doc_id`, or URL);
   - **how deeply you read it**: full text, specific sections, or abstract only;
   - what it says that bears on RQ1–RQ5, with page, section or equation;
   - relevance (high / medium / low).
3. **`candidates.csv`**, every work you screened, with these columns:
   `id, title, first_author, year, doi, arxiv_id, openalex_id, in_library (doc_id or empty), found_via, screened (title|abstract|full), relevance, notes`.
4. **`SEARCH_LOG.md`**: each query or citation chase, its source and its hit count, so the search
   can be repeated.
5. **Staged PDFs** in `~/library_inbox/ad_correlation/`, plus `manifest.csv` there with columns
   `file, doi, arxiv_id, title, authors, year, source_url, sha256, why`.
   - **Open access only:** arXiv, OpenAlex `best_oa_location` / `oa_url`, or author or
     institutional pages.
   - Never use shadow libraries, and never get around a paywall.
   - Name files `firstauthorYEAR-short-title.pdf`.
   - Stage only papers rated high or medium relevance that aren't in the library already.
   - Write paywalled papers worth getting to `PAYWALLED.md` in this folder.

**Write progressively.** Create the files early and append as you go, so the work survives an
interruption.

## 4. Sources and how to use them

**Order:**
1. the library;
2. OpenAlex citation chasing and author following;
3. arXiv;
4. web and blogs.

### a. James's library (about 140 converted papers)

Read `~/library-system/wiki/guides/research-library-for-agents.md` first. Use:
- `~/.pi/agent/skills/research-library/bin/lib-grep`, `lib-show` and `lib-sql`, called by full
  path;
- `lib-ask` for synthesis questions, one at a time; never type into the `lib-agent` pane yourself.

Cite library sources as `(doc_id, p. N)`. **Never write to `~/library` or its database.**

Relevant papers already present include `baio2010-bayesian-hierarchical-model-prediction-football`,
`koopman2013-dynamic-bivariate-poisson-model-analysing` and
`macridemartino2026-bayesian-weighted-discrete-time-dynamic`. Find the rest.

### b. OpenAlex

James authorised his key for this task.
- Read it **without printing it**:
  `KEY=$(grep '^OPENALEX_API_KEY=' ~/.config/library-system/.env | cut -d= -f2-)`.
  `OPENALEX_MAILTO` is read the same way.
- Pass it as the `api_key` query parameter.
- **Never echo the key, never write it to a file, and never include it in a logged URL.** Log
  queries without it.
- Useful calls:
  - `https://api.openalex.org/works?search=…`
  - `/works/doi:<doi>`
  - `/works?filter=cites:<W-id>` (forward citations)
  - the `referenced_works` field (backward)
  - `/works?filter=author.id:<A-id>` (follow an author)
- Use `select=` to keep responses small, and `jq` to parse.
- Stay at or below about 5 requests per second. **On HTTP 429, back off; if it repeats, stop using
  OpenAlex and note it in the log.**
- **Seeds to chase:** the library papers above, plus the authors who recur. **Follow authors**
  whose work keeps coming up, e.g. Baio, Blangiardo, Egidi, Torelli, Karlis, Ntzoufras, Koopman,
  Lit, Owen, Rue, Ley, Glickman, Held and McHale. Verify each yourself: this list is from memory.

### c. arXiv API

`http://export.arxiv.org/api/query?search_query=…`, at most one request every 3 seconds. Relevant
categories are stat.AP, stat.ME and stat.CO.

### d. Web

Use WebSearch/WebFetch for Stan and PyMC case studies (e.g. the rugby/football hierarchical
examples), the Stan forums, the opisthokonta.net blog, Gelman's blog, sports-analytics blogs and
conference talks. **Blogs are evidence of practice, not of fact.** Label them so.

**Seed works**, to verify rather than trust:
- Maher (1982); Dixon & Coles (1997);
- Rue & Salvesen (2000); Crowder et al. (2002); Knorr-Held (2000); Glickman & Stern (1998);
- Karlis & Ntzoufras (2003, 2009); Baio & Blangiardo (2010); Owen (2011);
- Koopman & Lit (2015, 2019);
- Egidi, Pauli & Torelli (2018), on combining historical data with bookmaker odds;
- Ley, Van de Wiele & Van Eetvelde (2019);
- Wunderlich & Memmert (2018), on odds-based ratings;
- Constantinou & Fenton (2013), on pi-ratings;
- Hvattum & Arntzen (2010), on Elo;
- Tsokos et al. (2019).

## 5. Rules

- **Citations are checkable or absent.**
  - Don't fabricate a reference, DOI, page or quote. If you can't verify something, say
    "unverified" or leave it out.
  - Use verbatim quotes only after checking them against the source: `lib-grep -F` for the
    library, the fetched text for the web.
  - Keep **what a paper reports** separate from **your inference**. Label inferences.
- **Read only, outside your outputs.** You may write only to this folder and to
  `~/library_inbox/ad_correlation/`.
  - Don't edit code, run Julia, use git (no commit, no branch), ssh to mcmc-beast, or touch any
    other tmux pane except through `lib-ask`.
- You may read anything in this worktree. The useful files are:
  - `current_development/market_model/DESIGN.md`;
  - `results/B2/PHASE_B2_REPORT.md` and `results/B3/PHASE_B3_REPORT.md`;
  - `MORNING_SUMMARY.md`.
- Work alone. **Don't ask the user questions.** Resolve an ambiguity by stating your assumption in
  the report.
- **If context use passes about 70%**, write `HANDOVER.md`: what's done, what's left and the
  current candidates. Then stop with the handover sentinel.

## 6. Sources for the numbers in §1

`results/B2/PHASE_B2_REPORT.md` covers:
- "What the correlations are";
- "Human addendum: quality/style hierarchy";
- the levels correlation table.

`results/B3/PHASE_B3_REPORT.md` covers the copula check. `MORNING_SUMMARY.md` covers home
advantage.

## 7. Finish

**Plan your time:**

| Time | Work |
|---|---|
| 12:00–12:30 | library sweep |
| 12:30–13:30 | citation chasing and author following |
| 13:30–14:00 | arXiv and web |
| 14:00–14:40 | close reading of the top 10–15 |
| 14:40–15:00 | specs and report |

Adjust as you need, but **leave at least 30 minutes for the report and specs.**

When done, print exactly one of these on a line by itself as your last output:

    RESEARCH_AD_DONE
    RESEARCH_AD_BLOCKED
    RESEARCH_AD_HANDOVER

Before the sentinel, give a 5-line summary: files written, works screened, works read in full,
PDFs staged and the headline finding.
