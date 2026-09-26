# Stage 5 — pedigree, operational-status and tier-prior design

## Decision frame

The current time-decay dynamics component samples attack and defence independently
and then centres each vector:

\[
\alpha_i=\sigma_\alpha(z_i-\bar z),\qquad
\beta_i=\sigma_\beta(u_i-\bar u).
\]

Thus its league-average attack and defence are exactly zero, but its team
contrasts are **not bounded**. This is important for the incident motivating this
EDA: the prompt's alleged “mathematical cap” of approximately 0.76 log goals (or
54--60% win probability) is false. Normal random effects have unbounded support,
and centring only removes a common location. Further, a log-rate difference does
not map to a universal 1X2 probability: the draw probability also depends on the
common scoring level. A numerical ceiling can be an early-season shrinkage
outcome, a feature/extraction issue, or a market/model comparison issue; it cannot
be inferred from zero-sum identification alone.

No production conclusion should be made from the illustrative component in
`l04_pedigree_tier_components.jl`. It has deliberately no database access and no
MCMC result. Its deterministic AD gate **did** run successfully on 2026-09-23:

```bash
julia --project=/home/james/bet_project/BayesianFootball -e \
  'include("experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/r04_pedigree_tier_components.jl")'
```

The synthetic 4-team / 6-fixture check recorded 31 ReverseDiff tape
instructions, compiled-versus-fresh ReverseDiff relative error `0.0`,
compiled-versus-ForwardDiff relative error `1.0748029516984209e-16`, and
perturbed compiled-versus-fresh error `0.0`. Its warmed measurement was 2432.0
bytes/gradient. These results validate only this branch-free, vectorised mock's
local calculus; they neither validate a Turing integration nor measure production
allocation behaviour.

## A. Informative prior on team supremacy (recommended first test)

Give club \(i\), at fixture time \(t\), a causal pedigree score \(p_i(t)\), an
operational-status category \(c_i(t)\in\{\mathrm{PT},\mathrm{Hybrid},\mathrm{FT}\}\),
and tier rank \(r_i(t)\), with 1 the highest SPFL tier. Define a latent
supremacy prior mean

\[
 m_i(t)=\underbrace{\tau_{r_i(t)}}_{\text{tier}}+
        \underbrace{\delta_{c_i(t)}}_{\text{operational status}}+
        \underbrace{\kappa\,p_i(t)}_{\text{pedigree}},\qquad
 \delta_{\mathrm{PT}}=0.
\]

Use a categorical Hybrid coefficient, rather than assigning Hybrid an arbitrary
FT share such as 0.5: there is no contract-level evidence that its football effect
is halfway between PT and FT. An initial **illustrative, not empirically
calibrated** prior specification is

\[
 \delta_{\mathrm{FT}}\sim\mathcal N(0,0.20^2),\quad
 \delta_{\mathrm{Hybrid}}\sim\mathcal N(0,0.20^2),\quad
 \kappa\sim\mathcal N(0,0.30^2),\quad
 \sigma_\alpha,\sigma_\beta\sim\operatorname{HalfNormal}(0.30).
\]

These scales are sensitivity-analysis starting points, not estimates of an FT/PT
premium. They must be revised only from a frozen causal status panel and held-out
comparison.

The bars are computed over **teams in the fitted competition and fold**, not over
future fixtures. They remove only location; all between-club contrasts remain.
Use adjacent-tier increments rather than four unrelated effects:

\[
 \tau_4=0,\quad d_j\sim\mathcal N^+(0,s_d^2),\quad
 \tau_r=\sum_{j=r}^{3}d_j\quad (r=1,2,3).
\]

A reference tier fixes one location but does **not** identify tier gaps if tiers
are disjoint: the observed comparison graph must connect tiers through cup/cross-
tier fixtures, promotions/relegations under an explicitly invariant cross-season
model, or another defensible bridge. For a League 1/2-only fit, the current-tier
term is common to every same-tier fixture and cancels from its home-away contrast.
It therefore cannot solve same-tier cold starts. Causally accumulated *pedigree*
and status contrasts can; tier effects principally transfer information when a
fixture or connected historical graph actually spans tiers. Estimating free tier
effects from a single-tier sample is not evidence of cross-tier strength.

Map supremacy to the existing attack/defence convention as

\[
 \alpha_i \sim \mathcal N(+m_i/2,\sigma_\alpha^2),\qquad
 \beta_i \sim \mathcal N(-m_i/2,\sigma_\beta^2),
\]

followed by separate zero-sum centring of \(\alpha\) and \(\beta\). The factor
one-half is required: in
\(\eta_h=\cdots+\alpha_h+\beta_a\) and
\(\eta_a=\cdots+\alpha_a+\beta_h\), this gives expected log-rate difference
\(m_h-m_a\), not twice that value. Centring preserves this contrast exactly.

A more conservative transition prior decays imported pedigree after a club enters
the observed tier:

\[
 p_i(t)=p_{i,0}\exp[-\log(2)\,D_i(t)/h_p],
\]

where \(D_i(t)\) is days since the *known before-kickoff* promotion/relegation
transition and \(h_p\) has a prior or is fixed in a preregistered sensitivity
analysis. It should decay toward zero, not toward a result estimated using later
League 1 matches. A promoted/relegated club's tier and status must be looked up
as-of kickoff, with effective timestamps and source/version provenance.

### Dynamic residuals and collinearity

The intended interpretation is `prior mean + residual`, not two competing team
ratings. Write \(s_i=m_i+r_i\), with \(r_i\) zero-mean and time-decay likelihood
updating it. Strongly regularise \(r_i\) initially and do not also include the
same tier/status contrast as an unconstrained covariate. Evaluate incremental OOS
performance by transition age (matches 1--5, 6--10, 11--20) against a flat-prior
control. If a dynamic state model is later used, apply the pedigree decay to its
state mean/initialisation rather than repeatedly injecting it as a fixture effect.

## B. Linear fixture covariate (ablation, not simultaneous default)

Keep flat team priors and add a supremacy term:

\[
 \eta_h=\ldots+w x_i,\qquad \eta_a=\ldots-w x_i.
\]

For a categorical status contrast, construct one fitted-fold design per
non-reference category: \(x_{\mathrm{FT},i}=(I_{h,\mathrm{FT}}-
I_{a,\mathrm{FT}})/2\) and equivalently for Hybrid. If the desired effect is
\(w_z(z_h-z_a)\) in the **log-rate difference**, the composable
`SupremacyRole` design must divide this contrast by two. Its \((+q,-q)\)
placement changes \(\eta_h-\eta_a\) by \(2wx_i\); using the raw difference
silently doubles the estimand. The same rule applies to tier/pedigree contrasts. Standardise fitted-fold columns using only
training rows, retain those constants for OOS, and impute an unavailable linear
covariate to 0.0 only when “unknown = league-neutral” is an explicit policy.

This form is tape-compatible: the feature builder emits a finite
`Vector{Float64}` and the `@model` only performs `w .* x` and broadcast additions.
It is **not** a claim of zero allocations overall: building feature vectors,
creating `VarInfo`, compiling a tape, and many distribution/Turing operations may
allocate. The relevant claim is that the vectorised likelihood shape has no
per-observation scalar loop or value branch.

Do not use Approach A and B with free coefficients for the same signal in the
first comparison: they are nearly non-identifiable with one another and with
short-window dynamic ratings. Compare (i) flat control, (ii) tier/status prior,
(iii) covariate only, then use shrinkage/projection diagnostics before considering
a combined model.

## Feature/extraction seam and filtration

A production feature extractor would provide per-fitted-match `Float64` vectors:
`flat_pedigree_supremacy`, `flat_status_supremacy`, plus team-level as-of metadata
for informative-prior construction. `required_features` must declare that
extractor. Its SQL/data contract must select the record whose `effective_from <=
kickoff < effective_to` (or an explicit open-ended successor), and must reject
ambiguous revisions. Tier history, status transitions, and season registration
must be versioned as known at fixture time; later announcements, final league
positions, valuations, and results cannot backfill past features.

The supplied mock uses caller-provided vectors precisely so it cannot accidentally
cross this boundary. It is a prototype-local demonstration, not a `src` component.

### Integration seam (sketch only; not drop-in production code)

The existing composable prototype establishes the relevant API: a covariate
subtypes `AbstractCovariateConfig`, implements `covariate_name`,
`covariate_role`, `covariate_prior`, `covariate_features`, `covariate_column`,
and `covariate_oos`, then is attached through `add(...)`; `SupremacyRole()` maps
its sampled `q` to `(q, -q)`. A real status covariate would have the following
shape after a point-in-time feature extractor exists:

```julia
# Suite-local sketch; names are the verified composable-builder hooks, but this
# does not define an extractor or constitute a src/models change.
struct FullTimeStatusCovariate{F,D,R} <: AbstractCovariateConfig
    feature::F
    prior::D
    role::R
end
covariate_name(::FullTimeStatusCovariate) = :full_time_status
covariate_role(c::FullTimeStatusCovariate) = c.role
covariate_prior(c::FullTimeStatusCovariate) = c.prior
covariate_features(c::FullTimeStatusCovariate) = [c.feature]
covariate_column(::FullTimeStatusCovariate, fs) =
    Vector{Float64}(fs.data[:flat_full_time_status_design])
```

The categorical Hybrid treatment needs **two** distinct covariates (or a future
vector-weight component): `:full_time_status` and `:hybrid_status`, each with its
own `Normal(0, 0.20)` coefficient and half-difference design. An informative
prior instead belongs in a new dynamics config/submodel that replaces the
zero-mean `TimeDecayDynamics` mean before its existing separate attack/defence
centring; it cannot be correctly introduced by merely adding the fixture
covariate above. This is intentionally not implemented in `src`.

## Acceptance path

1. Freeze and audit the time-stamped status/pedigree source first.
2. Run a no-MCMC deterministic density/gradient test (the supplied `r04` does
   this on synthetic finite data).
3. Build a single causal fold and verify perturbation/no-future-feature gates.
4. Compare the three ablations OOS, stratified by transition age, and inspect
   posterior residual-versus-pedigree correlation before any portfolio test.
5. Only then use the remote MCMC protocol for a single fold; never infer market
   calibration or a Kelly fix merely from a prior mechanism.
