# Phase 4 — scoring/report complete

Saved fits only; no sampling/reference refits or score upserts. Fresh beast REPL
%401,16threads,pinned cores,BLAS1. Scorer v2 sourcec7123f17 after manager
explicitly authorised the one-parenthesis fix in wave2 only. Failed v1/source
f9b6cdd7 and partial outputs remain in results/phase4_v1 and gate_logs.

- Exact independent wave1 harness reference parity for control_grw, qs_weak_r,
  market_c0; reused1992 reference score rows also byte-identical to committed
  wave1 CSV lines. Exact710 fixture goal-reference rows per reference arm.
- All14 prescribed pairs×4market families=56comparisons;10non-close pairs×
  3goal channels=30comparisons. Every comparison has noncircular90%, circular90%
  and unchanged harness clustered95% paths.999block/10000cluster replicates,
  unchanged seeds; noncircular classification, circular contradictions explicit.
- Unchanged wave1 r05_block_check and r05_goal_cluster reused. All30goal-cluster
  point estimates match; standalone cluster output bytes equal main scorer CSV.
- Posterior r236fold/site rows, sigma120arm/fold rows,6panel native SVG;
  full observations and fixture goal scores committed. REPORT summary10lines.
- Record/board audit atdb9bb8fe: all4arms×664rows/9960fields match existing
  Phase3 scores EXACT,maxdiff0; no new score writes. Register status scored,
  reproduction pending. Canonical generator adds only4headline rows/24Markdown
  rows and one register row; existing rows unchanged. Run IDs merge existing
  smoke provenance; final grid UUIDs are explicitly mapped in RUNS/README.

Decision: no promotion. All primary noncircular1X2 intervals cross zero;
marketarms−GRWjoint circular intervals classify better and are flagged.
Noncircular centring drift0.004564; every interval/metric traceable to CSV.
GRWjoint tail324.47,GRWmarket tail339.34/bulk395.44 review flags retained.
No ROI/staking or favourable subset. Phase5 still required before TODO completion.

Read-only wave1 provenance follow-up is recorded in TODO041/DECISIONS_LOG:
54a22ee7 introduced the current scorer's parse defect; base results came from
727af1d4, repaired r-only extraction81c77a42/7f3ff52b, saved base CSVb14e98b9.
Final clustered goal columns came from standalone r05_goal_cluster at54a22ee7,
committedd16a5800. Wave1 source/outputs remain untouched.

Evidence: results/phase4/*.csv,posterior_r_by_fold.svg,report_tables.sql;
results/gate_logs/phase4_{score_v2,interval,record}.log. All160original grid
checkpoint SHA256 digests frozen before reproduction. Scoring pane closed;
next is ONLY frozen-seed fold1 reproduction per newarm, fresh REPL each.
