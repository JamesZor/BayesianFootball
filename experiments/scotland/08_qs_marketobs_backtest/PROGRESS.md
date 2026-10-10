# Wave 4 progress

- Phase 0: PASS, definition of done met. STOP at boundary: PHASEBT_HANDOVER.
- Historical gate: source72cd5496 reproduces 2.515204600600047 unit wealth /1127 bets/99 slates/608 books, matching the committed experiment07 m12/raw/B0_canonical/prod headline exactly; converged=true, zero build errors. Required tolerance unchanged. Recipe/source lines in HISTORICAL_REPRODUCTION_RECIPE.md.
- Canonical measurement retained without rerun: source8018dacf, wealth2.773493294562897/1132. Wrong target association superseded under ruling3. Historical is validation only, not another Phase3 policy.
- Fixed Option B raw m12 reference: wealth2.207736087761667/1172. No fitted calibration, tuning, fitting or new src changes.
- Three snapshot books complete at90-minute bound, fixed710 denominator: any archive fixtures T−25=611, T−6h=295, T−24h=53; complete1X2=545/182/47. T−24h/360-minute sensitivity complete:190 fixtures,170 complete1X2. Coverage/staleness CSV has36 market-line rows across4 snapshot/bound settings, zeros included.
- Safety: static PASS locally/on beast; actual SHOW off logged once; before/after audit byte-identical,10/10 unchanged. Same input IDs/T−25 book byte-identical to committed canonical inputs. No betdb connection, cache SQL fallback, DB writes or PGOPTIONS debugging.
- New output: results/phase0_historical/; policy role/source comparison in results/phase0_policy_reference.csv; log logs/phase0_historical.log; local exact-headline/daily/snapshot checks logs/historical_result_checks.log. Previous attempts/artifacts remain preserved.
- Owned pane%453/pi_qsbt_phase0 CLOSED and verified absent. Prior%451/%452 already closed. No remaining owned Julia job; unrelated sessions untouched. Beast worktree /root/BF_runs/qs_backtest_phase0 @72cd5496 detached, Julia1.12.4/16 threads/core-pinned/BLAS1.
- Checkout collision resolved without force: nine owned untracked canonical outputs hash-matched to committed copies, archived at /root/BF_runs/qsbt_canonical_artifacts_8018dacf. Cache hash unchanged: c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4.
- Current README/REPORT/HANDOVER and chronology/arm statuses updated; TODO043 Phase0 criterion checked, overall IN_PROGRESS/paused for manager. Tracker/diff/result checks pass; final evidence commit/push recorded by git log.
- Price-source caveat: last-traded archive prices, not verified executable touch/depth. Historical policy selected using these same seasons. Phase0 references use existing KellyLogUtility, not yet validated scenario-native allocation.
- Later phases NOT_RUN: allocator audit/change, five-model grid, CLV/liquidity, bootstrap and final questions. No qs_marketobs profitability claim. Await manager-authorised Phase1 continuation.
