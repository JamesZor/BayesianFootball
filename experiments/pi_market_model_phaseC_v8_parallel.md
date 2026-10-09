# Phase C, revision 8: run the fast rungs now and start C1 in parallel (manager, ~05:55)

**No setting, budget, seed or threshold changes.**

**Situation:**
- Revision 7 passes (t05 305/305, workflow 85/85).
- The C1 synthetic recovery (frozen source `6335316f`) is running in beast pane `%265`, at about
  5 iterations per minute per chain. Its prescribed 4×(2000+3000) budget needs about 17 h.
- The C1 production fits at the same budget need about as long each. Nothing in C1 can finish by
  morning.
- C0, H1 and H2 are exact Kalman fits and take minutes.

**Do, in this order:**
1. **Leave the recovery alone** in pane `%265`. Don't kill it, relaunch it, check out code under
   it, or reload code into its REPL. Only read its pane or log.
2. **C0, H1 and H2: the full pipeline now, on the beast.** Fresh checkout at the current workflow
   SHA in a new path, for example `/root/BF_runs/market_model_c_fast`, with your own
   pane-ID-only sessions.
   - Both protocols (10a/10b), the prescribed budgets, and gates 2–5 for these rungs.
   - The revision 1 §3 measures for C0/H1/H2: one-step, coverage, paired H1/H2 vs C0 with
     fixture SE, and C0 vs B2's R6.
   - The home-advantage table (`home_advantage_rungs.csv`).
   - **Two fresh byte-identical runs** for these rungs.
3. **C1 production (10a and 10b): launch now, at the prescribed budget,** in their **own** beast
   sessions, in parallel with the recovery. The load is about 4.4 on 32 cores; check `uptime`
   first.
   - **Don't promote or interpret C1** until the recovery has passed its gates. Record each
     session's pane ID, start time, SHA and log path, plus an ETA from the measured iteration
     rate.
4. **A progress report:** `results/C/PHASE_C_PROGRESS_REPORT.md`, giving
   - C0/H1/H2 results in plain words (does home advantage act through away suppression? does it
     scale with quality?);
   - C0 against R6;
   - the live runs with their ETAs, and what remains.

   Update the README with the same. Commit and push.
5. Then print **`PHASEC8_HANDOVER`** and stop. Don't wait idly on the long runs; the manager
   monitors them. If something fails, print `PHASEC8_BLOCKED` with numbers.
