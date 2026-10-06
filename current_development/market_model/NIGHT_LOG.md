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
