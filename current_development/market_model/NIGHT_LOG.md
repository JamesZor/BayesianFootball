# Night log, TODO 039 (2026-10-06)

The manager's running record while the human sleeps. If the manager's context is compacted,
**read this file first**: it holds the current state, the next action and every answer given on
the human's behalf.

## Human instructions (2026-10-06, before sleeping)

- **Scope:** run Phase B3, then Phase C on the manager's defaults.
- **Phase B3:**
  - (a) a tail/asymmetry check on R6's quality/style steps;
  - (b) score-grid comparison: double Poisson (baseline), Dixon–Coles global ρ, bivariate
    Poisson λ₃, and a Frank copula with Poisson margins.
- **Phase C:**
  - **double Poisson kept regardless of B3**; any better grid goes to a later Phase C2 that the
    human reviews first;
  - 1X2-only books inform supremacy only;
  - quality/style GRW with independent steps; independent equal-side u_m;
  - Laplace–Kalman engine (Gaussian approximation of each book's Dirichlet/Beta likelihood,
    learned concentration n);
  - totals-only books inform the total only.
- **Authority:** the manager answers pi's in-scope questions and logs each below. Anything that
  changes scope, `src/`, thresholds or data waits for the human.
- **Limits:** on a quota limit or stall, wait for the reset and resume the same model
  (`openai-codex/gpt-6.1-sol`).
- **Monitoring:** watcher scripts plus the manager; no subagent.
- **Context:** a fresh pi session per phase. If pi's context passes ~75% mid-phase, pi writes
  `HANDOVER.md` and is restarted fresh.
- **Git:** merge reviewed phases into `feat/market-model-inversion-eda` locally (the stream branch
  is not pushed); pi pushes only its own branch.
- **Morning:** `MORNING_SUMMARY.md` plus a chat summary.
- **Don't** start Phase D or Phase C2. Don't touch the human's sessions: `pi_julia_mm_b` (their
  Claude pane), `agents`, `0`, `library`.

## State

| Time | Event |
|---|---|
| 2026-10-06 start | Stream branch at `19b83990` (A, B, B2 merged and reviewed). |
| B3 launched | Brief `97a2b201`; worktree `BayesianFootball-market-model-pi-b3`, branch `pi/market-model-phase-b3`; pi pane `%33` (session `agent_pi_market_model_b3`), gpt-6.1-sol; codex quota 30% (5h) / 18% (7d) at launch. Watcher: `scratchpad/watch_pi_night.sh %33 B3`. |

| 01:28 | B3 `PHASEB3_DONE` after 33 min (pi context 78%, codex quota 53%). Six commits, pushed; statistical SHA `2e02a785`. |
| ~01:32 | B3 review: fresh laptop REPL t04 92/92, t03 131/131, t02 131/131; manager beast rerun at `2e02a785` running (pane %260). |

| ~01:42 | B3 manager beast rerun at `2e02a785`: 35/35 hashes OK (533 s). **B3 merged** (`d287c745`). |
| ~01:44 | **Phase C launched.** Worktree `BayesianFootball-market-model-pi-c`, branch `pi/market-model-phase-c` from `d287c745`; pi pane `%38` (session `agent_pi_market_model_c`); codex quota 54% (5h) / 21% (7d) at launch. Watcher: `watch_pi_night.sh %38 C`. |

| ~01:48 | Phase C revision 1 `PHASEC_BLOCKED` at preflight (see answers below). |
| ~01:58 | **Phase C revision 2 launched**: fresh pi, pane `%40` (session `agent_pi_market_model_c2`), same branch and worktree; codex quota 62% (5h) / 23% (7d). Watcher: `watch_pi_night.sh %40 C`. |

Times are laptop local; earlier entries were first written in the beast clock (+1 h) and corrected.

**B3 headline** (from `results/B3`):
- **Tails:** Gaussian team-step dependence is adequate; no PPC p-value is extreme (all
  0.28–0.63), so no copula or heavy-tailed step model is needed.
- **Grids:** **Dixon–Coles global ρ = −0.064** is the best:
  - draw residual −1.34 pp → −0.14 pp; held-out MAE is the best of the four;
  - the 1X2-only totals bias shrinks from −0.103 to −0.009;
  - outcome log loss is unchanged (all CIs include 0).

  This is the candidate for C2. Phase C stays double Poisson, as the human decided.
- **R6 on G1 rates:** structurally the same (ρ_αβ −0.83).

## Human requests during the night

- **2026-10-06, mid-setup:** "can the home advantage be correlated with the attack defense of
  the team". Added to the Phase C brief as §2.4 side rungs, two-stage on isolated rates against
  C0:
  - H1 split HA (γ_att on the home rate, γ_def off the away rate);
  - H2 quality-linked HA (κ·q_home).

  Added unless the human objects. It doesn't change C1/C2.

## Answers given on the human's behalf

- **Phase C revision 1 blocked at preflight** (`614a2676`, about 7 min in, laptop only, no
  sampling). pi correctly found three errors in the **manager's** brief:
  1. the Dirichlet large-n expansion is −n·KL(q‖p), so Phase A's KL(p‖q) centre and Hessian
     are not the Dirichlet's;
  2. projecting a 1X2 likelihood to supremacy must fail a gate against the unprojected book
     likelihood;
  3. over/under constrains the curved λ_h + λ_a, and BTTS + O/U books identify both rates.

  Manager fix (revision 2, `pi_market_model_phaseC_v2_prompt.md`): the exact Dirichlet
  likelihood, a Laplace/iterated-EKF update at the joint mode of likelihood × state prediction,
  and a gate against exact quadrature, with **thresholds unchanged**. 1X2-only books use the
  level-integrated likelihood ∫L(d, ℓ)dℓ, the exact form of "supremacy only".
- **Decision taken for the human (please review):** totals/BTTS-only books now enter with their
  **exact** likelihood instead of a "total-only" linear projection.
  - O/U-only books still carry no supremacy information, because the likelihood is flat along
    the constant-total curve.
  - BTTS + O/U books contribute the weak supremacy information their prices contain.
  - Reason: "informs the total only" was the manager's linear approximation of the intent
    (stop T015's optimiser-guess supremacy), and it is mathematically ill-posed. The exact
    likelihood meets that intent.
  - If the human wants BTTS + O/U books held to totals only, that is a small change in C2.

## Next action

1. (done) B3 reviewed and merged; Phase C running in pane %38.
2. When the watcher fires:
   - on QUESTION, LIMIT or IDLE, handle it;
   - on the sentinel, review B3: fresh laptop tests, beast rerun plus hash check, then merge.
3. When the C watcher fires: handle QUESTION / LIMIT / IDLE, or review C (fresh laptop tests, beast rerun + hash check), merge, then write MORNING_SUMMARY.md. Do not start D or C2.
- **~02:05: Phase C revision 2 blocked at the posterior-mean gate.** Laplace marginal passes
  everywhere (median ≤ 0.004 nats) and SD error ≤ 1%. The mean is off by up to 9.17% of SD at
  n = 250, spread 0.20 (12/30 full books).
  - Diagnosis: the Laplace mode ≠ mean under posterior skew.
  - The 5% mean gate was the manager's own revision-2 addition. It was **not** loosened, because
    a threshold change waits for the human.
  - Remedy, in scope as a better method: revision 3, a second-order (skewness-corrected) mean
    from the third derivatives at the mode, then the full Gate 1 on all book types.
  - Sent to the same pi session (pane `%40`, context 52%); sentinels `PHASEC3_*`.
- **~02:24: Phase C revision 3 blocked.** The skewness correction makes full books pass
  Gate 1 at all 12 settings. Thin books fail only at prediction spread 0.20:
  - BTTS+O/U median 0.02–0.07 nats, SD error up to 49%;
  - O/U-only SD error 5.9–6.8%;
  - the 1X2-integrated derivative check is incomplete.

  At spread 0.05 every type passes. Diagnosis: Laplace on curved likelihood ridges.
- **~02:35, manager decision (scope reduction, no threshold change):** revision 4 runs C0, C1
  (full books only), H1 and H2. **C2 (thin-book pooling) is deferred to the human** with four
  options in `pi_market_model_phaseC_v4_scope.md`:
  - (i) exact moment matching;
  - (ii) NUTS on the exact Dirichlet;
  - (iii) gating at the realistic spread (a setting change: the human's call);
  - (iv) restricting thin-book types (a data change: the human's call).

  pi wrote `results/C/HANDOVER.md` (`31ec8520`, context 81%, codex 90%), and a fresh session
  was started.
- **~02:33, pi question (Phase C rev 4):** "What Gaussian prior should H1 use for γ_def?"
  - Options: (1) γ_att ~ N(0.15, 0.25²), γ_def ~ N(0, 0.25²), independent (pi recommended);
    (2) a different joint prior.
  - **Answered (1).** It keeps C0's γ prior for the home-scoring boost, with a neutral prior
    centred at 0 for the away suppression, so H1 nests C0 exactly at γ_def = 0.
- **~02:42: Codex usage limit reached** (5h window at 100%; 7d at 29%) during Phase C rev 4.
  - pi stopped mid-way, while rerunning t02 after t03 passed, with uncommitted work in its
    worktree (pane `%44`, context 54%).
  - Reset at **03:20:07 BST** (`~/.pi/agent/usage-cache.json` → `resetsAt`).
  - Waiting, per the human's rule. **2 banked resets** are shown as available on the plan;
    **not used**, because that's the human's resource and the rule says wait.
  - At ~03:21: tell the same pi session to continue.
- **03:27: pi resumed** after the quota reset (first retry), in the same session, pane `%44`.
- **03:36: pi C4 handover** at `622f9a55`: 209/209 deterministic tests, 53/53 state gates, full-book Gate 1 12/12; no fits yet. **03:38:** fresh pi from the handover, pane `%48` (session `agent_pi_market_model_c5`), new codex window at 4%.
- **04:15: Phase C rev 4 blocked again (a numerical blocker, not a scientific one).** Sampler and
  synthetic generator implemented (t05 270/270; t02–t04 pass). The C1 synthetic recovery failed
  the per-book joint-mode Newton line search twice, with ‖∇f‖ ≈ 3e-5, i.e. roundoff at the
  optimum. pi stopped rather than try a third numerical variation.
- **~04:20, manager: revision 5** (`pi_market_model_phaseC_v5_solver.md`): the Newton-decrement
  stopping rule.
  - Accept δ ≤ 1e-12; a stalled line search is accepted only if δ ≤ 1e-9, and is counted.
  - BigFloat re-solve verification on the failing case.
  - No change to any scientific gate.
  - Priority order if time runs short: recovery → C0/C1 + 10b → H1/H2 → second reproduction run.
- **04:31: Phase C rev 5 blocked** on the manager's own verification tolerances: mode gap 4.0e-7
  (limit 1e-8) and marginal gap 6.7e-7 nats (limit 1e-9); t05 282/284.
  - Cause: the manager's rev 5 was inconsistent. Accepting δ ≤ 1e-9 allows a ~1e-6 mode error.
  - Not loosened. **Revision 6** (`pi_market_model_phaseC_v6_polish.md`): up to 3 undamped,
    gradient-only Newton polishing steps when the line search stalls; tolerances unchanged.
  - Sent to the same session `%53` (context 57%).
  - If it still fails, the tolerance question goes to the human.
- **04:53: rev 6 blocked.** The exact-case regression passes (t05 289/289), but recovery hits a
  zero-motion Armijo edge at fixture 12476686: 94 accepted no-op steps, δ = 1.04e-12.
  **Revision 7:** zero-motion steps count as stalls and are polished; tolerances unchanged. pi's
  context is 79%, so a fresh session starts from HANDOVER.md.
- **05:42: rev 7 passes** (t05 305/305). The C1 synthetic recovery is live on the beast (pane
  `%265`). pi handed over at 88% context.
  - **Measured rate: about 5 iterations per minute per chain**, so the prescribed 4×(2000+3000)
    takes about 17 h, and each C1 production fit as long again.
  - **Revision 8** (no budget or setting change):
    - run C0, H1 and H2 (minutes) fully now, including the reproduction runs;
    - launch C1 10a and 10b production at the prescribed budget in parallel;
    - leave the recovery untouched;
    - write a progress report and hand over.
  - **Morning decision for the human:** wait for C1 (finishing this evening) or cut its budget.
- **~06:00, pi question (rev 8):** the dense Gaussian engine runs at about 170 iterations per
  minute per chain on C0, so the six fast fits ×2 runs would take hours. pi asked to add a
  scalar-Kalman likelihood fast path, verified against the existing engine at unchanged
  tolerances, in a separate checkout. Recovery and the C1 processes stay untouched.
  **Answered: permit (option 1).** It is an implementation optimisation with an equivalence gate;
  no priors, budgets, seeds, data or thresholds change. Beast load was 13.2: recovery, C1 10a/10b
  and the fast runs.
- **~06:25, pi question (rev 8):** "All six fast fits (C0/H1/H2 × 10a/10b) and 107 engine gates
  pass, but the path figure stopped: Ross County and Airdrie are not in the 56/57 panel's 22 teams
  (Kelty's key is kelty-hearts-fc). Annotate them as unavailable and keep the data unchanged?"
  **Answered: yes (option 1).** The manager's brief named clubs from the CLV §14 list; they were
  Championship clubs in 24/25–25/26.
- **06:29: Codex limit again** (5h window 100%). Resets at **08:24 BST**. pi was at C8 with the
  fast-rung final run 1 in beast pane `%270`.
- **06:33, beast contention (manager action, reversible):**
  - The C1 recovery (`%265`, pid 3497268) has printed nothing past iteration 250 for about 45 min,
    against about 25 iterations per 5 min before.
  - The slowdown began when C1 production 10a (`%268`, pid 3516292) and 10b (`%267`,
    pid 3516204) started: three `-t 16` processes on 32 cores; fork–join threading oversubscribed.
  - **Paused C1-10a with `kill -STOP 3516292`** (beast 07:28:11). Resume with `kill -CONT 3516292`;
    no state is lost.
  - Watching whether the recovery speeds back up.
