# 010 — Design and Prototype Autonomous Agent Bayesian Model Search Loop

| Field | Value |
|---|---|
| ID | 010 |
| Title | Design and Prototype Autonomous Agent Bayesian Model Search Loop |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [src/models/pregame/components/](../src/models/pregame/components/); [src/models/pregame/interfaces.jl](../src/models/pregame/interfaces.jl); [src/training/](../src/training/); [src/evaluation/](../src/evaluation/); [docs/architecture/ai_agent_infrastructure_and_execution_context.md](../docs/architecture/ai_agent_infrastructure_and_execution_context.md) |

## Context & Problem Statement

With ReverseDiff executing full 40-fold walk-forward CV grids in 5–8 minutes (and 2-fold preflights in <30 seconds), modeling progress is no longer limited by sampling latency. We can build an autonomous Bayesian Discovery Loop (inspired by DeepMind's FunSearch and AlphaTensor paradigms) where an AI agent programmatically formulates mathematical hypotheses, generates candidate model architectures using the `CountModelBuilder` DSL, compiles ReverseDiff tapes, executes MCMC on `mcmc-beast`, parses convergence/posterior diagnostics, and iterates toward lower LogLoss and higher portfolio return.

However, this requires deep architectural scoping:
1. **Agent Harness & Substrate**: Should the search loop be driven by `pi` (pi.dev harness), Antigravity/Claude Code subagents, or a repo-native Julia/Python daemon calling LLM APIs directly?
2. **Search Space Representation**: Defining a rigorous grammar over composable components (`Interception`, `Dynamics`, `HomeAdvantage`, `Covariates`, `Observations`) and prior hyperparameter bounds.
3. **Diagnostic Feedback Channel**: Translating MCMC convergence failures (funnels, low ESS, label switching) and posterior shrinkage (credible intervals covering zero) into actionable prompt feedback for the model-generation LLM.
4. **Safety & Compute Budgeting**: Preventing unconstrained MCMC forks, ensuring strict deduplicated PostgreSQL run tracking, and killing stuck sampling tasks.

## Acceptance Criteria

- [ ] Author an architectural RFC (`docs/architecture/rfc_autonomous_bayesian_model_search.md`) addressing:
  - Harness trade-offs (`pi` vs native daemon vs Claude/Antigravity).
  - State machine design (Hypothesis $\to$ DSL Synthesis $\to$ Tape Preflight $\to$ Grid Sampling $\to$ Proper Scoring $\to$ Reflection $\to$ Branching).
  - Resource safety protocols (timeout limits, thread pinning, DB deduplication).
- [ ] Implement a structured Diagnostic Feedback Reporter (`src/evaluation/agent_feedback.jl` or standalone prototype loader) that outputs JSON/Markdown summaries of:
  - Convergence health: $\hat{R}_{\max}$, $\text{ESS}_{\min}$, divergence count, and geometry flags.
  - Posterior parameter significance: parameters whose 95% Credible Interval overlaps 0 with high density (pruning candidates).
  - Out-of-sample proper scores vs current champion baseline (Δ LogLoss, Δ Brier, Δ RPS, CLV).
- [ ] Implement a standardized candidate evaluation CLI (`experiments/autonomous_search/run_candidate.jl`) that takes a model recipe specification, executes preflight + grid, persists to `mcmc_experiments`, and returns evaluation telemetry.
- [ ] Prototype an end-to-end 3-generation search demonstration showing autonomous hypothesis generation, compilation, execution, and selection.

## Ideas & Candidate Solutions

- **Harness Substrate Options**:
  - *Option A: `pi` (pi.dev harness)*:
    - *Pros*: Native terminal and RPC interface, already proven on complex multi-stage tasks (TODO 005, TODO 006), handles bash/tmux/ssh transparently, preserves complete conversation/decision transcripts.
    - *Cons*: Agent session management, token consumption over long-running loops.
  - *Option B: Repo-native Julia daemon*:
    - *Pros*: Fast in-process execution, programmatic HTTP calls to Claude/GPT-4/Gemini APIs, tight loop control.
    - *Cons*: Requires managing LLM API keys and building custom code-editing/sandboxing tools in Julia.
  - *Option C: Hybrid (Recommended)*:
    - A Julia harness (`rXX_eval_candidate.jl`) outputs structured machine-readable JSON telemetry (`evaluation.json`). An autonomous `pi` agent running in a detached tmux session acts as the scientist, reading the telemetry, formulating code modifications via `CountModelBuilder`, and committing winning candidates to git.
- **Genetic / Island Search Topology (FunSearch style)**:
  - Instead of a simple hill-climbing search that gets stuck in local minima, maintain a "pool" of diverse candidate architectures (e.g. pure Poisson with rich covariates, vs minimal Joint Gamma-Poisson, vs high-order player RAPM). The agent picks candidates from the frontier to recombine.
- **Divergence Debugger Prompting**:
  - When Turing NUTS reports divergences $>0$, the diagnostic reporter explicitly highlights the offending parameter coordinates (e.g. $\sigma_{\text{attack}} \to 0$), instructing the agent to switch that component to a non-centered parameterization or tighten prior bounds.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task in BACKLOG following user brainstorm. Outlined architectural questions, harness trade-offs, and acceptance criteria.

## Verification & Findings

Not run yet. Record architectural RFC decisions, harness prototype benchmarks, and multi-generation candidate search performance.
