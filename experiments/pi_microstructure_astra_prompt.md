# WORK PACKAGE: Market Microstructure & Dynamic Execution Engine for MatchDay
# Target Model: openai-codex/gpt-6-astra (thinking: high)
# Working Branch: feat/market-microstructure-execution
# Working Directory: current_development/market_microstructure_execution/
# Remote Compute: root@mcmc-beast:/root/BF_microstructure

> **Agent Role**: Principal Quantitative Researcher & Market Microstructure Architect  
> **Model**: `openai-codex/gpt-6-astra` (Reasoning: High, Context: 272K)  
> **Mission**: Design, empirically validate, and prototype the **Next-Generation Market Microstructure & Multi-Level Staged Execution Engine** for the Scottish Lower execution desk.

---

## 1. Executive Context & The Saturday Operational Leak

On Saturday 2026-09-05, we deployed the validated **Option B** calibrated MatchDay system on the 10-fixture Scottish Lower slate (Tournaments 56 & 57) priced from **Fold 43** of `m12_joint_hybrid_synergy`.

### The Core Finding
* **Alpha is Proven**: With confirmed pre-match lineups, our predictive engine generated **18 positive-edge legs totaling £451.09 risk** on a £2,400 bankroll ($k_{\text{risk}} = 0.681$).
* **Execution is the Bottleneck**: MatchDay currently executes as an **atomic single-shot dump at the touch (`TouchOnly`) at T−25 (13:35 UTC)**. Because Scottish Lower top-of-book depth is thin 25 minutes before kickoff, only **£165.65 of risk actually matched** at the touch. **£261.43 (~58%) of Kelly risk volume was dropped or left unmatched!**
* **The PnL Cost**:
  * Realised PnL at the top-of-book touch: **+£19.14 (Saturday actual) / +£30.09 (BBC confirmed lineups)**.
  * Theoretical PnL under full fill: **+£54.82 PnL improvement** (from -£65.52 up to -£10.70).
  * Dropping 60% of calculated Kelly volume leaves massive positive expectation on the table.

### The Market Microstructure Reality
1. **Betfair Live Order Books Have 3 Levels**: The database (`betfair_live.order_book_1m`) archives 3 levels of back/lay prices and sizes every minute.
2. **Liquidity Dynamically Thickens**: In lower leagues, institutional and retail flow does not sit on the book at T−25; liquidity **doubles or triples between T−25 and T−5** as kickoff approaches.
3. **Weight of Money (WOM)**:
   $$\text{WOM} = \frac{\text{Size}_{\text{Back}}}{\text{Size}_{\text{Back}} + \text{Size}_{\text{Lay}}}$$
   Order book imbalance signals whether price is under shortening pressure ($\text{WOM} > 0.65$) or drifting pressure ($\text{WOM} < 0.35$).

---

## 2. Infrastructure, Data & Compute Access

### A. Operational Database (`betdb` on `archpc:5433`)
Accessible via `ENV["BF_DB_URL"]` or `BayesianFootball.MatchDay.paper_connection()`.
Key tables to research:
* `betfair_live.order_book_1m`: 1-minute historical snapshots:
  * Columns: `market_id`, `ts`, `market_matched`, `best_back_price`, `best_back_size`, `back_level2_price`, `back_level2_size`, `back_level3_price`, `back_level3_size`, `best_lay_price`, `best_lay_size`, `lay_level2_price`, etc.
* `betfair.match_meta`: Identity crosswalk linking SofaScore `match_id` to `betfair_event_id` and market IDs.
* `paper_runbook.paper_orders`, `paper_fills`, `paper_settlements`, `paper_slates`: Saturday's live execution ledger.

### B. High-Performance Remote Compute (`mcmc-beast`)
You have full access to run heavy Julia scripts, order book replays, and simulations on the dedicated 32-core compute server:
* **Host**: `root@mcmc-beast` (AMD Ryzen 9, 32 cores, 64 GB RAM, currently 100% idle).
* **Worktree Directory**: `/root/BF_microstructure` (tracking branch `feat/market-microstructure-execution`).
* **Execution Command**:
  ```bash
  ssh root@mcmc-beast "export PATH=/root/.juliaup/bin:\$PATH; cd /root/BF_microstructure && julia --project -t 32 <script.jl>"
  ```
* **Git Sync SOP**:
  1. Make edits locally in `current_development/market_microstructure_execution/`.
  2. Commit and push: `git push origin feat/market-microstructure-execution`.
  3. Pull on beast: `ssh root@mcmc-beast "git -C /root/BF_microstructure pull origin feat/market-microstructure-execution"`.
  4. Run code on beast with 32 threads.

---

## 3. Your Research Mandate & Creative Freedom

You have complete creative and quantitative freedom to investigate the order book data, formulate the mathematical algorithms, and design the execution engine. Specifically address:

### Problem A: Multi-Level Ladder Sweeping vs. Price Slippage
* Formulate the mathematical optimization for sweeping down the 3-level ladder:
  * Given model probability $p_{\text{model}}$ and hurdle rate $\alpha_{\text{min}}$ (e.g. 2.0% minimum edge), what is the strict reservation price $r_i^*$ beyond which no fill is acceptable?
  * What is the slippage budget $\Delta_{\text{slip}}$?
  * Derive the volume allocation across Level 1, Level 2, and Level 3 that maximizes expected log-utility (Kelly growth) subject to the maximum volume-weighted average price (VWAP).

### Problem B: Staged Working Window (T−25 to T−5)
* Formulate an execution strategy across time:
  * When should an order be an aggressive taker (sweeping immediate liquidity) vs a passive maker (placing limit orders inside the spread)?
  * How can Weight of Money (WOM), spread width, and traded volume acceleration $\frac{d(\text{matched})}{dt}$ be used as execution signals?
  * If an order is 40% filled at T−15, how does the remaining risk adapt dynamically across subsequent minutes?

### Problem C: Integration with MatchDay & Portfolio Architecture
* How should this execution engine fit into `src/MatchDay/` as an `AbstractExecutionPolicy` (e.g. `MultiLevelSweep(max_slip = 0.01)`, `StagedTWAP(window = Minute(20))`)?
* Ensure that the execution engine preserves `src/Portfolio/` zero-allocation guarantees (`BookWorkspace`, `OddsIndex`).

---

## 4. Required Deliverables

In `current_development/market_microstructure_execution/`:
1. `REPORT.md`: Detailed research report containing:
   * Empirical analysis of Scottish Lower order books (`betfair_live.order_book_1m`).
   * Mathematical derivations for the reservation price, multi-level VWAP allocation, and WOM signal threshold.
   * Backtested simulation comparing `TouchOnly` vs `MultiLevelSweep` vs `StagedTWAP` on Saturday's slate.
2. `l01_microstructure_sweeper.jl`: Loader module defining structs, sweeping math, WOM metrics, and execution policies.
3. `r01_microstructure_sweeper.jl`: Research runner script demonstrating the execution policy on historical Scottish Lower order books.
4. `README.md`: High-level summary of the engine and recommended implementation roadmap.
