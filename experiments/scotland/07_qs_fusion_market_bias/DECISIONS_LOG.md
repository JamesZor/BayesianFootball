# Wave 3 — decisions

## 2026-10-10 — scope and Phase 0

- Scope set by James 2026-10-10 (brief `experiments/claude_qs_fusion_bias_brief.md`). Implementer: Claude CLI; manager: the launching Claude session.
- Wave-2 manager rulings carry over: C2-pending t05 child set excluded from acceptance; FD audit = step-size convergence with Richardson ≤ 1e-8 plus exact AD ≤ 1e-10; recovery = 3 seeds, no parameter misses all three, pooled coverage ≥ 80%; control forwarding; deterministic control rescoring with exact-equality audit; wave-1 market module loaded at top level (namespace provenance).
- Phase 0 reuses wave 2's gate list, adds `test/test_market_rate_observation.jl` (its outputs redirected to `/root/BF_runs/qs_experiment_w3_out/phase0/<gate>` through `QSX2_TEST_OUTPUT`, so wave-2 outputs are untouched) and drops the xG coverage gate (unchanged snapshot, already audited for all 40 folds in wave 2). Pins are checked in shell before any Julia.
- Phase 0 PASS at `122cbc37` (`PHASE0.md`): all 16 gates pass; C2-pending 29/39 excluded as ruled; pins match. Owned panes closed.

## 2026-10-10 — Phase 1 design (frozen before any fit)

- Manager accepted Phase 0. The 3815 vs 3831 count is fully accounted for (PHASE0.md addendum): 14 seed-1 truth anchors + 2 fold-count checks in the two seed-1 refits, both skipped because Phase 0 ran with `QSX2_RERUN_RECOVERY_FIRST=false` and wave 2 with `true`.
- Component: `JointMarketFusionObservation(; joint, market, kappa_D_prior = LogNormal(0, 0.2), delta_D_prior = Normal(0, 0.1), delta_M_prior = Normal(0, 0.1))`. `joint` is a `SharedKappa` `JointGammaPoissonObservation` (type-restricted; wave-2 config), `market` a `MarketRateObservation` (frozen feature, σ_obs prior HalfNormal(0.20)). Existing components unchanged.
- **The market observes the TRUE goal log-rates `ζ = η + log κ`** (the joint goals arm's rates, which are also what predictions price), not the xG latent η. So D = ζ_h − ζ_a (log κ cancels) and the market level includes log κ; δ_M is the market's offset from the goals model's level, separately identified from log κ because xG pins η.
- Off switches: each bias prior may be `nothing` → that term is fixed (κ_D = 1, δ_D = 0, δ_M = 0) and has no site. All three off dispatches (type `UnbiasedMarketFusion`) to wave 2's `_market_rate_ll` on ζ, so test 3 is bitwise equality, not a tolerance. Goals and Gamma arms are the joint's expressions term for term (the joint's `_observe` is not refactored).
- Biased market likelihood is evaluated exactly in supremacy/total coordinates (r_h² + r_a² = (r_h − r_a)²/2 + (r_h + r_a)²/2), with κ_D as an unfused scalar–array product and δ shifts applied after the reductions (AD guide Rule 7).
- Sites, in θ order: `obs.ν, obs.log_κ, obs.σ_obs, obs.κ_D, obs.δ_D, obs.δ_M` (off terms absent). Validation requires a finite ClampGuard (joint Gamma arm), MatchProxyXGFeature, the market σ prior rule, κ_D prior with nonnegative support and finite upper tail.
- Recipe display spells out the joint feature fields and priors, the market table's canonical SHA256 and σ prior, and each bias prior or `off`.
- FD audit: new helper `test/helpers/market_fusion_fd_audit.jl`, same method as wave 2's, minus wave 2's hard-coded historical failure counts. Random-sample seeds 20261121 (grw_bias), 20261122 (qs_bias), 20261123 (qs_nobias). Exact-AD perturbation points `MersenneTwister(20261112)`, deltas (0, 0.001, −0.002), fold 40.
- **Recovery, pre-declared.** Arms: fusion_grw_bias (index 1), fusion_qs_bias (2), fusion_qs_nobias (3), fold 40 geometry and real masks (pxG availability, full books), 2×(200+200), accept 0.65, depth 10, 90% intervals. Seeds s = 1,2,3: data `20261113 + index + 10(s−1)`, sampler `20261117 + index + 10(s−1)`. Truths: dynamics as wave 2 (GRW α σ [.12,.06,.03], β σ [.20,.11,.024]; QS τ [.16,.085,.027], r [.4,.4,.3]); intercept log 1.2, home 0.15 (not scored, as wave 2); ν = 4.0, log κ = 0.05, σ_obs = 0.07; bias arms κ_D = 0.85, δ_D = 0.03, δ_M = −0.05; the nobias arm is generated and fitted with bias off. Gate (wave 2): no parameter misses all 3 seeds; pooled coverage ≥ 80%.
- Outputs to `/root/BF_runs/qs_experiment_w3_out/phase1` through `QSX3_TEST_OUTPUT` (default is the w3 path, never wave 2's).
- Phase 1 PASS at `01f42219` attempt a1 (`PHASE1.md`): fusion 4256/4256, tape 184 (24 new, 0 B), regressions unchanged. Recovery 84/99 pooled (gate met); qs_nobias alone 21/27, reported, not re-run or re-tuned.

## 2026-10-10 — Phase 2 smoke plan (frozen before any fit)

- Manager accepted Phase 1 and asked for Phase 2 smoke on all three arms, then AskUserQuestion before the grid.
- `candidates.jl`: wave 2's scope `lower`, W2 sampler 4×(500+1000) accept .65 depth 10, joint config `make_joint_observation()` and weak-r QS, both unchanged; the market table comes from wave 2's `l01_market_table.jl` (SHA-checked; loads `Main.QSMarketArm` at top level). Arms: `fusion_qs_bias` (**control**), `fusion_qs_nobias` (all three bias priors `nothing`), `fusion_grw_bias` (MultiScaleGRW). Bias priors as the brief.
- Smoke = `Harness.smoke`, unchanged: folds 1 and 40 at 2×(200+200), hard checks (gradient, tape_allocation, filtration, latents, score_grid_coherence, fit_parity), 16-chain 50+50 performance probe. Adapter `l03_smoke.jl` is wave 2's with wave-3 seeds/tags/TODO only. Fresh beast REPL per arm, order qs_bias → qs_nobias → grw_bias; root seeds 20261151/52/53, pre-declared, no smoke seed tuning.
- ETA = probe window × (40×4×1500)/(16×100), as wave 2 (largest fold, excludes overhead).
- Phase 2 smoke a1 at `79f003e0` (`PHASE2.md`): 6/6 hard checks pass on every arm, 0 divergences. Short-smoke convergence review fails on all three (R̂ 1.081 / 1.103 / 1.059; bulk ESS 19.7 / 24.3 / 65.0; tail 54.6 / 24.0 / 46.5), in wave 2's smoke range. Probe ETAs 4.314 + 4.423 + 4.606 = 13.343 h. No grid code run; asking manager.
- Manager (AskUserQuestion, 2026-10-10): **"Approve grid as frozen"** — Phase 3 in order qs_bias → qs_nobias → grw_bias, W2 sampler 4×(500+1000), control forwarding, one seeded rerun per R̂ > 1.05 fold; no budget/prior/threshold changes. All short-smoke flags and the 13.343 h ETA were disclosed.
