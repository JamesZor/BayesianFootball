# Annotated bibliography — attack/defence correlation

Sign convention: our β = "concedes more". Correlations quoted from papers that use a defensive
*strength* (β enters log-rate with a minus sign, or "defence" means good defence) are flipped
and marked **[sign flipped]**.

Depth codes: **full** = full text read; **sections** = named sections/pages read;
**abstract** = abstract only.

## Library sources

### Koopman, S. J. & Lit, R. (2015). A dynamic bivariate Poisson model for analysing and forecasting match results in the English Premier League. *JRSS A* 178(1), 167–186.
- DOI 10.1111/rssa.12042 · library `koopman2013-dynamic-bivariate-poisson-model-analysing` (working-paper version; page numbers below are the source PDF's).
- **Depth:** sections — §2.2–2.3 (pp. 7–8), estimation restrictions (pp. 12–15), model-comparison list (pp. 18–19).
- **Bears on RQ1/RQ2/RQ5.** Intensities are λx = exp(δ + α_i − β_j), λy = exp(α_j − β_i) (eq. 3, p. 7): β is a defensive *strength*, opposite to ours. Attack and defence are AR(1) processes whose "disturbances … are normally distributed error terms which are independent of each other" (eq. 4, p. 7); in state-space form H = diag(σ²_α…, σ²_β…) (p. 8). The paper says of the teams' strengths: "we do not relate to each other" (p. 7). It **assumes** independence and reports **no estimated** attack/defence correlation. Estimation is Monte Carlo maximum likelihood by importance sampling on a linear Gaussian state-space approximation (§2.4, p. 10); the AR coefficients and variances are restricted to be common across teams because of the parameter count (pp. 12–15).
- **Relevance: high** (the canonical dynamic attack/defence state-space model; the specific independence assumption our R6 result contradicts on market data).

### Koopman, S. J. & Lit, R. (2019). Forecasting football match results in national league competitions using score-driven time series models. *International Journal of Forecasting* 35(2), 797–809.
- DOI 10.1016/j.ijforecast.2018.10.011 · library `koopman2019-forecasting-football-match-results-national`.
- **Depth:** sections — §3.1–3.2 (pp. 5–6), §4.5 and Tables 3–4 (pp. 10–11).
- **Bears on RQ2 (score-driven), RQ4 (1-D vs 2-D).** Score-driven (GAS) updates f_{t+1} = ω + B f_t + A s_t (eq. 11, p. 5) for f_t = (α, β) of all teams; A and B are block-diagonal, a₁I_N / a₂I_N and b₁I_N / b₂I_N, so "there are no spillover effects", and the score scaling is the identity (p. 6). Any α–β coupling in a step comes only through the joint score of a match, not a covariance parameter. **RQ4:** the dynamic ordered-probit model has a single strength per team ("we cannot separate the strength in defence and attack", p. 6). Against the best 2-D model (bivariate Poisson or Skellam, f₁ static init), the 1-D ordered probit is worse in ARPS in five of six leagues, e.g. England 0.1996 vs 0.1981–0.1984 (DM 2.32), Spain 0.1929 vs 0.1916 (DM 3.09); Netherlands is the exception, 0.1934 best (Tables 3–4, p. 11). The paper attributes the gap to "condensation of data" (p. 10): the 1-D model also discards the score, so this **confounds** dimension with data granularity (my inference).
- **Relevance: high.**

### Owen, A. (2011). Dynamic Bayesian forecasting models of football match outcomes with estimation of the evolution variance parameter. *IMA Journal of Management Mathematics* 22(2), 99–113.
- DOI 10.1093/imaman/dpq018 · library `owen2011-dynamic-bayesian-forecasting-models-football`.
- **Depth:** sections — §2.1–2.3 (pp. 4–6).
- **Bears on RQ2/RQ3/RQ5.** Same sign convention as ours: log μ = α_i + β_j + γ_H, log λ = α_j + β_i + γ_A (eqs. 3–4, p. 4), so β is "concedes more" — **no flip needed**. Sum-to-zero on α and on β separately at every round (eqs. 5–6). Random walks with one common evolution variance σ² for attack and defence, independent (eqs. 7–8, p. 5). To keep the constraint in a random walk the evolution covariance becomes R = nσ²/(n−1)(I − 11ᵀ/n) (eq. 14, p. 5): the **between-team** correlation of same-side steps is exactly −1/(n−1), while the within-team α–β covariance is untouched (eq. 14 is applied to each side separately). Data: Scottish Premier League 2003/04–2005/06, 12 teams (p. 6) — the only Scottish dynamic attack/defence paper found. Implementation: WinBUGS, sampling n−1 unconstrained innovations (eqs. 16–17).
- **Relevance: high** (constraint-induced covariance, made explicit; Scottish data).

### Baio, G. & Blangiardo, M. (2010). Bayesian hierarchical model for the prediction of football results. *Journal of Applied Statistics* 37(2), 253–264.
- DOI 10.1080/02664760802684177 (from OpenAlex/publisher; library row has none) · library `baio2010-bayesian-hierarchical-model-prediction-football` (preprint pagination).
- **Depth:** full text for §2–4 (pp. 3–9).
- **Bears on RQ2.1/RQ2.3/RQ3.** log θ_g1 = home + att_h + def_a, log θ_g2 = att_a + def_h (p. 3): same sign as ours ("good teams are associated with negative defense effect", p. 8) — **no flip**. Basic model: att and def exchangeable, each from its own Normal(μ, τ), "hyper-priors … modelled independently" (p. 4), sum-to-zero on each. §4 (pp. 6–8): "overshrinkage" penalises extreme teams; the remedy is a three-component mixture (bottom/mid/top) of non-central t (ν = 4) for att and def, with **separate** latent group labels grp^att(t) and grp^def(t), each with its own Dirichlet(1,1,1) weights (p. 7). The anticorrelation is built in only through **sign-truncated group means**: bottom μ₁^att ∈ (−3,0), μ₁^def ∈ (0,3); top the reverse (p. 7). No joint (att, def) group label and no correlation parameter. Fig. 3 (p. 9) plots average attack vs defence and shows clusters; Table 3 (p. 8) shows the basic model under-predicting Inter at 69 points vs 85 observed, mixture 76.
- **Relevance: high** (the three-group mixture James named).

### Benz, L. S. & Lopez, M. J. (2021). Estimating the change in soccer's home advantage during the Covid-19 pandemic using bivariate Poisson regression. *AStA Advances in Statistical Analysis* 107, 205–232.
- DOI 10.1007/s10182-021-00413-9 · arXiv 2012.14949 · library `benz2021-estimating-change-soccers-home-advantage-during`.
- **Depth:** sections — §3.1 model (pp. 5–7), priors (p. 8), §4 simulation (p. 9), §5 results (p. 14), appendix (p. 22).
- **Bears on RQ1/RQ2.1.** Fitted model: log λ₁ = μ + T + α_H + δ_A, log λ₂ = μ + α_A + δ_H (eq. 2, p. 5): δ is "concedes", our sign — **no flip**. Fitted effects are **independent** normals, α ~ N(0, σ²_att), δ ~ N(0, σ²_att) (p. 5; the paper writes σ_att for both). The **simulation** draws (α, δ) from a bivariate normal with SD 0.35 and ρ* ∈ {−0.8, −0.4, 0}, "reflecting the range of correlations in scoring and defending strength (negative correlations infer that teams that score more goals also give up fewer goals)" (appendix, p. 22), following Thompson (2018), a blog. The range is asserted, not estimated in the paper. Their own Bundesliga 2015/16 posterior means: "the correlation between offensive and defensive team strength estimates is weak" (p. 14) — no number. Home-advantage bias showed no pattern across ρ* (p. 11).
- **Relevance: medium** (practice: practitioners assume ρ ≈ −0.4 to −0.8 for levels; and one counter-observation).

### Macrì Demartino, R., Egidi, L. & Torelli, N. (2026). Bayesian weighted discrete-time dynamic models for association football prediction. *JRSS C*, qlag032.
- DOI 10.1093/jrsssc/qlag032 · arXiv 2508.05891 · library `macridemartino2026-bayesian-weighted-discrete-time-dynamic`.
- **Depth:** sections — §2.1, §2.4, §3 (pp. 3, 5–6, 8–9).
- **Bears on RQ2/RQ5.** Standard dynamic prior: independent random walks for att and def with one common evolution precision (eq. 5, p. 5); initial level N(μ_att, 1/σ) (eq. 6, p. 6); zero-sum within each period (eq. 7). Their contribution: separate commensurate precisions φ_att,τ and φ_def,τ per period with a continuous spike-and-slab half-normal mixture prior (eqs. 10–11, p. 8), "similarly to Egidi et al. (2018)". Stan through the R package footBayes (p. 9). No attack–defence covariance anywhere.
- **Relevance: medium** (state of the art in footBayes keeps attack and defence independent; per-period scale mixture is an alternative to a correlation for season jumps).

### Broatch, J. E. & Karl, A. T. (2018). Multivariate generalized linear mixed models for joint estimation of sporting outcomes. *Italian Journal of Applied Statistics* 30(2).
- arXiv 1710.05284 (read v1, 2017; OpenAlex W2765209148). The journal venue in the heading is from memory and **unverified**; cite the arXiv version. Staged PDF.
- **Depth:** sections — §1–2 (pp. 1–6), §4.4–4.6 and Table 5 (pp. 16–21).
- **Bears on RQ1/RQ2.1 (the only estimated attack/defence covariance hyperparameter found for team sports).** E[y_home] = f(b_o,h − b_d,a) (p. 4): b_d enters with a minus sign, so it is a defensive *strength*. Team effects (b_o, b_d, b_w) ~ N₃(0, G*) with G* **unstructured**, fitted by first-order or fully exponential Laplace ML in R package mvglmmRank (pp. 5, 10). For scores, "the correlation between offensive and defensive team score ratings … ranges from 0.77 for American college football data to −0.3 for the professional basketball (NBA) data" (p. 19); 2005 college football: 0.71 (Table 5, p. 20). **[sign flipped]** In our convention (β = concedes more): **≈ −0.71 to −0.77** for college football and **≈ +0.3** for the NBA. These are static, within-season **levels**. Authors' explanation: recruiting builds both units in college football; NBA players play both ends (pp. 19–21). No football (soccer) data.
- **Relevance: high.**

### Tsokos, A., Narayanan, S., Kosmidis, I., Baio, G., Cucuringu, M., Whitaker, G. & Király, F. (2019). Modeling outcomes of soccer matches. *Machine Learning* 108, 77–95.
- DOI 10.1007/s10994-018-5741-1 (verified, OpenAlex W2809797215; journal version is open access) · arXiv 1807.01623 (read v2). Staged PDF.
- **Depth:** sections — §4.1 model (pp. 8–9), Table 5 and §6–7 (pp. 13–14).
- **Bears on RQ2/RQ4.** HPL model: η₁ = … + α_h + ξ_a + γ_{h,s} + δ_{a,s} (eq. 7, p. 8): ξ is "concedes", our sign. α_t ~ N(0, σ²_α) and ξ_t ~ N(0, σ²_ξ) **independent**; season deviations γ_ts, δ_ts are AR(1) with **one shared innovation SD σ_ε** for attack and defence (p. 9). INLA with R-INLA default priors. **RQ4:** the 2-D goal model (HPL) has validation RPS 0.2073 (SE 0.0025) against 0.2079–0.2083 for the best 1-D Bradley–Terry outcome models (Table 5, p. 14) — inside one SE; "similar performance … marginally better" (p. 14). Like Koopman & Lit (2019) this confounds dimension with data (goals vs results).
- **Relevance: medium.**

### Egidi, L., Pauli, F. & Torelli, N. (2018). Combining historical data and bookmakers' odds in modelling football scores. *Statistical Modelling* 18(5–6), 436–459.
- DOI 10.1177/1471082X18798414 (verified, OpenAlex W2788508817) · arXiv 1802.08848 (read v1). Staged PDF.
- **Depth:** sections — §1, §3 (pp. 4, 10–14), §4.1.
- **Bears on RQ2/RQ5 (market information in a goal model).** log θ₁ = μ + att_h + def_a (eq. 3.4, p. 12): def is "concedes", our sign ("the lower is the team-defence parameter … the better is the defence"). Seasonal random walks with drift, att_{t,τ} ~ N(μ_att + att_{t,τ−1}, σ²_att), def likewise, **independent**, separate σ_att, σ_def ~ half-Cauchy(0, 2.5) (eq. 3.5, p. 12; p. 13); zero-sum within season (p. 13). **Market:** per bookmaker, implied rates θ̂ solved from the 1X2 probabilities under the Skellam (eq. 3.2, p. 11) — a direct analogue of our Phase A inversion — enter as extra data with truncated-normal noise around bookmaker parameters λ (eqs. 3.7–3.8, p. 13), and the scoring rate is a convex combination p θ + (1 − p) λ with p ~ Beta (eq. 3.3, p. 11). The market enters the **rate**, not as a prior on α/β, and no attack–defence covariance is used. On the mean probability of the observed result, "our model is very close to the bookmakers' probabilities" (Table 3, arXiv v1 p. 25).
- **Relevance: high** (closest precedent for combining market-implied rates with a goal model).

### Ley, C., Van de Wiele, T. & Van Eetvelde, H. (2019). Ranking soccer teams on the basis of their current strength: a comparison of maximum likelihood approaches. *Statistical Modelling* 19(1), 55–73.
- DOI 10.1177/1471082X18817650 (verified, OpenAlex W2785347929) · arXiv 1705.09575 (read v3; page numbers are the arXiv version). Volume 19(1), 55–73 verified via OpenAlex biblio. Staged PDF.
- **Depth:** full text §2–4 (pp. 3–11).
- **Bears on RQ4 (the cleanest like-for-like 1-D vs 2-D comparison found).** "Independent Poisson" with **one** strength per team: λ_i = exp(c + (r_i + h) − r_j), λ_j = exp(c + r_j − (r_i + h)) (p. 7). In our terms that is α_i = r_i, β_i = −r_i: a **perfectly anticorrelated** attack/defence pair, i.e. style variance σ_s = 0 (my mapping). The "Def. & Att." variants use Maher's 2T parameters, λ_i = exp(c + (o_i + h) − d_j) (p. 8). Same goal data, same time-decay weighted ML. **Out of sample RPS**, EPL 2008/09–2017/18, 3300 matches: Bivariate Poisson 1-param 0.1953, Independent Poisson 1-param 0.1954, Independent Poisson Def&Att 0.1961, Bivariate Poisson Def&Att 0.1961; Bradley–Terry/Thurstone 0.1985–0.2003 (Table 1, p. 10). National teams 2008–2017, 4268 matches: 0.1651 / 0.1653 vs 0.1656 / 0.1656 (Table 2, p. 11). Authors: "parsimony in terms of parameters to estimate is important" (p. 10). No standard errors or tests are reported for these gaps.
- **Relevance: high.**

### Karl, A. T., Yang, Y. & Lohr, S. L. (2014). Computation of maximum likelihood estimates for multiresponse generalized linear mixed models with non-nested, correlated random effects. *Computational Statistics & Data Analysis* 73, 146–162.
- DOI 10.1016/j.csda.2013.11.019 (verified, OpenAlex W2076079619) · arXiv 1403.7676 (read v1; pages are the arXiv version). Staged PDF.
- **Depth:** sections — §2.2 model (p. 3), §5.1 application and Table 1 (pp. 12–13).
- **Bears on RQ1/RQ2.1.** Same model family as Broatch & Karl: score of A vs B depends on "team A's 'offensive rating' and team B's 'defensive rating'" with −1 on the defensive effect (p. 3), so b_d is a defensive *strength*. Unstructured 3×3 G for (offence, defence, win propensity), ML by Laplace/EM. 2012 NCAA FBS, 844 games, 126 teams: G[o,o] = 0.0920, G[d,d] = 0.1025, G[o,d] = 0.0473 (fully exponential, joint model, Table 1, p. 13), so **r = 0.49 (my arithmetic)**; scores-only model 0.0460 / √(0.0908·0.1015) = 0.48. Text: "The offensive and defensive ratings are positively correlated, suggesting teams with good (bad) offenses tend to also have good (bad) defenses" (p. 12). **[sign flipped]** In our convention **≈ −0.48**. Static levels, one season, estimated hyperparameter. Note the two variances are nearly equal (0.092 vs 0.103).
- **Relevance: high.**

### Glickman, M. E. & Stern, H. S. (1998). A state-space model for National Football League scores. *JASA* 93(441), 25–35.
- DOI 10.1080/01621459.1998.10474084 · library `glickman1998-state-space-model-national-football-league`.
- **Depth:** sections — model (pp. 3–5), posterior summaries (pp. 10–11).
- **Bears on RQ2 (dynamic, multi-scale) and RQ4/RQ5.** One strength per team (point differential, normal). Two time scales: between-season θ_(k+1,1) ~ N(β_s G θ_(k,g_k), (φω_s)⁻¹ I) and between-week θ_(k,j+1) ~ N(β_w G θ_(k,j), (φω_w)⁻¹ I), where G centres the vector (p. 4) — i.e. regression to the mean plus a re-centring at both scales. Posterior σ_w = 0.88 (0.52, 1.36) points vs σ_s = 2.35 (1.14, 3.87); β_w = 0.99 (0.96, 1.02) (Table 3, p. 10). This is the template for a multi-scale walk with season-boundary shrinkage, and it is **one-dimensional**: the analogue of modelling only our quality q.
- **Relevance: medium.**

### Knorr-Held, L. (2000). Dynamic rating of sports teams. *The Statistician* 49(2), 261–276.
- Read the LMU preprint: SFB 386 discussion paper 98, DOI 10.5282/ubm/epub.1490 (OpenAlex W1685803780). The JRSS D citation is from memory and the DOI was **not** verified. Staged PDF (preprint).
- **Depth:** sections — §2.1 and §3.1–3.2 (printed pp. 4–7).
- **Bears on RQ3/RQ5.** Constrained random-walk prior θ_t = θ_{t−1} + u_t, u_t ~ N(0, Q) with Q = σ²(I − 11ᵀ/n), rank n − 1, the unique exchangeable choice; it "implies that components of u_t are a priori negatively correlated" (printed p. 6, transcribed from the extracted text, whose Greek letters were lost). Also: "the posterior mode estimator in dynamic models is not invariant with respect to the identifiability constraint" (p. 6), unlike the static case where ML estimates under corner and sum-to-zero constraints map into each other (pp. 4–5). One strength per team (ordinal paired comparison). Our `MultiScaleGRW` centres raw states after the fact (`raw .- mean(raw, dims = 1)`), which is the same projection (I − 11ᵀ/n) (my inference).
- **Relevance: high** for RQ3.

### Lopez, M. J., Matthews, G. J. & Baumer, B. S. (2018). How often does the best team win? A unified approach to understanding randomness in North American sport. *Annals of Applied Statistics* 12(4) (page range not in OpenAlex; **unverified**).
- DOI 10.1214/18-AOAS1165 (verified, OpenAlex W2579988816) · arXiv 1701.05976 (read; pages are the arXiv version). Staged PDF.
- **Depth:** sections — §2–3 (pp. 7–11).
- **Bears on RQ5 (market-implied data in a state-space strength model).** The observations are **closing money-line implied probabilities**, logit p_ij ~ N(θ_i − θ_j + α₀ (+ α_i*), σ²_game) (p. 8). Strengths are centred to sum to zero every week (p. 8). θ evolves with a season-to-season AR (γ_season, σ_season) and a week-to-week AR (γ_week, σ_week) "similar to Glickman and Stern (1998)" (p. 10). Priors: Uniform(0, 1000) on precisions, γ_season ~ U(0, 1), γ_week ~ U(0, 1.5) (p. 11). One-dimensional (win probability only). It is the closest published analogue of our Phase B: a market-inversion state-space model, but without an attack/defence split.
- **Relevance: high.**

### Wunderlich, F. & Memmert, D. (2018). The Betting Odds Rating System: using soccer forecasts to forecast soccer. *PLoS ONE* 13(6), e0198668.
- DOI 10.1371/journal.pone.0198668 (verified, OpenAlex W2807514546). Staged PDF.
- **Depth:** sections — "Rating systems" and "Statistical framework" (pp. 5–7), results (pp. 9–10).
- **Bears on RQ5.** ELO-Odds: an Elo rating updated with the market's expected result a_H = p_H + 0.5 p_D in place of the actual result (p. 6); "no match result is ever directly used" (p. 6). Promoted teams "carry on the ratings of the relegated teams" (p. 6). ELO-Odds beats goal- and result-based Elo "on a highly significant level" by paired t-tests, while the betting odds themselves remain best (pp. 8–9). One-dimensional; it gives no attack/defence split.
- **Relevance: medium.**

### Thompson, W. J. (2018). *Soccer predictions using Bayesian mixed effects models* (online book). https://wjakethompson.github.io/soccer/ — **blog/practice, not peer-reviewed.**
- **Depth:** the "Defining the model" chapter (Stan code) and the conclusion, fetched 2026-10-06.
- **Bears on RQ1/RQ5 (practice).** Cited by Benz & Lopez (2021, p. 22) as the source of their bivariate-normal team-strength simulation. The Stan models shown use **independent** non-centred effects: alpha = raw_alpha * sigma_a, delta = raw_delta * sigma_d, sigma ~ normal(0, 10) (`define-model.html`). No correlation is estimated on the pages read. So the ρ* ∈ {−0.8, −0.4, 0} range in Benz & Lopez is not traceable to an estimate in this source (my inference from the pages read).
- **Relevance: low.**

### Huber, F., Koop, G. & Onorante, L. (2021). Inducing sparsity and shrinkage in time-varying parameter models. *Journal of Business & Economic Statistics* 39(3), 669–683.
- DOI 10.1080/07350015.2020.1713796 · library `huber2020-inducing-sparsity-shrinkage-time-varying`.
- **Depth:** sections — §2 (pp. 4–5).
- **Bears on RQ3 (testing a zero-variance state) and RQ5.** Uses the non-centred parameterisation of Frühwirth-Schnatter & Wagner (2010): β_t = β₀ + √V β̃_t with β̃_t a standard random walk, so the signed √v_j enters the observation equation as an ordinary regression coefficient (eq. 6, pp. 4–5). A zero-variance (static) state is then an interior point √v_j = 0, which a shrinkage or spike-and-slab prior can test or select. Applied to σ_s it would give a non-boundary test of "no style dynamics" (my inference).
- **Relevance: medium.**

### Secondary-only and abstract-only seeds
- **Crowder, M., Dixon, M., Ledford, A. & Robinson, M. (2002).** Dynamic modelling and prediction of English Football League matches for betting. *JRSS D* 51(2), 157–168. DOI 10.1111/1467-9884.00308 (verified, OpenAlex W1989661440). **Abstract only** (paywalled): attack and defence "evolve through time … according to some unobserved bivariate stochastic process", fitted to all 92 English Football League teams 1992–1997 with a fast approximation versus MCMC. Whether the bivariate process has a non-zero attack–defence correlation could **not be verified**. Relevance **high** if obtained (only multi-division dynamic attack/defence model found). → PAYWALLED.md.
- **Rue, H. & Salvesen, Ø. (2000).** Prediction and retrospective analysis of soccer matches in a league. *JRSS D* 49(3), 399–418. DOI 10.1111/1467-9884.00243 (verified, OpenAlex W2035378550). **Not read.** Via Owen (2011, p. 12): continuous-time Brownian evolution with evolution and prior variances "common to the attack and defence abilities for all teams", no identifiability constraints, prior means equally spaced on [−0.2, 0.2] from a prior ranking (Owen 2011, pp. 12–13). → PAYWALLED.md.
- **Maher, M. J. (1982).** Modelling association football scores. *Statistica Neerlandica* 36(3), 109–118. DOI 10.1111/j.1467-9574.1982.tb00782.x (verified, OpenAlex W2136744117). **Not read**; its attack/defence form is reproduced in Ley et al. (2019, p. 8) and Koopman & Lit (2015, p. 4).
- **Ruiz, F. J. R. & Pérez-Cruz, F. (2015).** A generative model for predicting outcomes in college basketball. *JQAS* 11(1). DOI 10.1515/jqas-2014-0055 (OpenAlex W2091706878). **Abstract only**: extends "a classical model for soccer" with conference-specific behaviour (a multilevel-by-conference structure) and "different strategies of teams and conferences". Relevance medium (RQ2.2/2.4).

### Methodological references (verified DOIs; abstract or known content only — cited for standard results, not read in this session)
- **Lewandowski, Kurowicka & Joe (2009)**, Generating random correlation matrices based on vines and extended onion method, *J. Multivariate Analysis* 100, 1989–2001. DOI 10.1016/j.jmva.2009.04.008 (W2084045976). Origin of the LKJ(η) prior on correlation matrices; for a 2×2 matrix LKJ(η) is ρ ~ (1+ρ)/2 ~ Beta(η, η) (standard result, **not re-checked against the paper in this session**).
- **Frühwirth-Schnatter & Wagner (2010)**, Stochastic model specification search for Gaussian and partial non-Gaussian state space models, *J. Econometrics* 154, 85–100. DOI 10.1016/j.jeconom.2009.07.003 (W2056536708). Non-centred ±√θ parameterisation for testing whether a state varies (see Huber et al. above for the form used).
- **Bitto & Frühwirth-Schnatter (2019)**, Achieving shrinkage in a time-varying parameter model framework, *J. Econometrics* 210, 75–97. DOI 10.1016/j.jeconom.2018.11.006 (W2550542038). Shrinkage priors on the signed process SD.
- **Self & Liang (1987)**, Asymptotic properties of ML estimators and likelihood ratio tests under nonstandard conditions, *JASA* 82, 605–610. DOI 10.1080/01621459.1987.10478472 (W2046076301), and **Stram & Lee (1994)**, Variance components testing in the longitudinal mixed effects model, *Biometrics* 50, 1171–1177. DOI 10.2307/2533455 (W2063605818). Boundary nulls: the LRT for a single variance = 0 is a 50:50 mixture of χ²₀ and χ²₁.
- **Papaspiliopoulos, Roberts & Sköld (2007)**, A general framework for the parametrization of hierarchical models, *Statistical Science* 22(1), 59–73. DOI 10.1214/088342307000000014 (W1997628954). Centred vs non-centred: non-centred mixes better when the data are weak relative to the prior.
- **Vehtari, Gelman & Gabry (2017)**, Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC, *Statistics and Computing* 27, 1413–1432. DOI 10.1007/s11222-016-9696-4 (W2203714058). PSIS-LOO; for time series, leave-future-out is the relevant variant (my note).

### Egidi, L., Karlis, D. & Ntzoufras, I. (2025). *Predictive Modelling for Football Analytics*. Chapman & Hall/CRC.
- DOI 10.1201/9781003186496 (OpenAlex W4413469118) · library `egidi2025-predictive-modelling-football-analytics` (catalogued, not converted; read through `lib-show` from the PDF, so page numbers are PDF pages with printed pages in brackets).
- **Depth:** sections — table of contents (PDF pp. 8–10); §4.4.4.4 "Prior distributions and STZ constraints" (PDF pp. 147–148, printed 130–131). I also searched the full extracted text for correlation, multivariate-normal, LKJ, Wishart and Cholesky terms.
- **Bears on RQ2/RQ5 (state of practice).** The footBayes dynamic prior is att_{k,t} ~ N(att_{k,t−1}, σ²_att), def_{k,t} ~ N(def_{k,t−1}, σ²_def), separate SDs, with sum-to-zero at each match time (eqs. 4.24–4.25, printed p. 130); `dynamic_type = "weekly"` or `"seasonal"`. The search of the extracted text found **no** attack–defence correlation, multivariate-normal team-effect prior or LKJ prior; the multivariate-normal hits are about HMC momentum. The current reference text for Bayesian football models (by the Baio/Karlis/Ntzoufras school) keeps attack and defence independent.
- **Relevance: medium.**

### Addendum to Koopman & Lit (2015): scale of weekly steps
- Table 2 (p. 16): φ_α = 0.9985, φ_β = 0.9992, σ²_α = 0.000206, σ²_β = 0.000143 (M = 1000), i.e. weekly SDs **0.0144 (attack) and 0.0120 (defence)** on EPL goals 2003/04–2009/10; "the attack and defense strengths are highly persistent and behave almost as random walks" (p. 16). Model (d) on p. 19 keeps one strength per team but **deletes the opponent's defence** (λ_it = exp(θ_it)); it is not a quality-only model, so its in-sample rejection (p. 20) says nothing about one vs two dimensions (my reading).
- For comparison (my arithmetic): R6's σ_q = 0.032 and σ_s = 0.010 per week with corr(Δq, Δs) ≈ 0 imply a per-side weekly SD of about √(0.032² + 0.010²) ≈ 0.034, about 2.3–2.8 times Koopman & Lit's goal-based estimates. Different leagues and data (closing prices vs goals), so this is a scale indication only.

### Bürkner, P.-C., Gabry, J. & Vehtari, A. (2020). Approximate leave-future-out cross-validation for Bayesian time series models. *Journal of Statistical Computation and Simulation* 90(14), 2499–2523.
- DOI 10.1080/00949655.2020.1783262 (verified, OpenAlex W2951464221). **Title only** this session: cited for the existence of PSIS-based leave-future-out CV, the appropriate predictive comparison for walk models (standard LOO leaks the future). Relevance medium.

