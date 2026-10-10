# Phase 2 — smoke hard gates PASS (2026-10-10); grid awaits manager approval

Source `79f003e0`, beast checkout detached at that commit, attempt `a1`, 00:42:07–00:58:01 UTC.
Launcher `phase2_gate.sh a1` (pins checked in shell first: snapshot SHA + mtime, frozen table SHA,
all match); one fresh `claude_qsx3_p2_<arm>` REPL per arm, 16 threads, pinned cores, BLAS 1.
`Harness.smoke` unchanged: folds 1 and 40 at 2×(200+200), accept .65, depth 10, plus the
16-chain 50+50 performance probe on fold 40. Pre-declared root seeds 20261151/52/53. Raw logs
`/root/BF_runs/logs/qs_experiment_w3/phase2/a1/`; CSVs `results/phase2/`. All owned panes closed.

## Hard checks (6/6 per arm)

gradient, tape_allocation (0 B on both folds), filtration (2 folds, 1780 fitted / 30 held out),
latents, score_grid_coherence, fit_parity (persisted under `scottish_lower_qs_wave3_2426_smoke`).

| Arm | Smoke UUID | Params (fold 40) | Grad ms | ForwardDiff err (fold 1) | Wall s |
|---|---|---:|---:|---:|---:|
| fusion_qs_bias (control) | `084b2906-91b1-44b1-a700-13614c41c4ec` | 980 | 0.334 | 9.3e-16 | 274 |
| fusion_qs_nobias | `f74d00ed-d56b-49aa-a290-2f7c21925f7c` | 977 | 0.338 | 7.6e-16 | 258 |
| fusion_grw_bias | `807672a7-9632-4efd-a13e-9b9334e77883` | 980 | 0.323 | 5.6e-16 | 273 |

## Convergence at the short smoke budget (review, not hard)

| Arm | max R̂ | min bulk ESS | min tail ESS | Divergences | min BFMI | Treedepth |
|---|---:|---:|---:|---:|---:|---:|
| fusion_qs_bias | 1.0811 | 19.7 | 54.6 | 0 | 0.47 | 0 |
| fusion_qs_nobias | 1.1029 | 24.3 | 24.0 | 0 | 0.76 | 0 |
| fusion_grw_bias | 1.0588 | 65.0 | 46.5 | 0 | 0.56 | 0 |

All three convergence rows fail review (R̂ > 1.05 at 400 draws over 2 chains). Wave 2 at the
same budget: grw_joint 1.0541 / 182.9 / 69.7, qs_joint 1.0745 / 133.1 / 79.9, grw_marketobs
1.1379 / 16.1 / 57.5, qs_marketobs 1.0625 / 67.1 / 74.3, all 0 divergences — the fusion arms sit
in that range. Grid budget is 4×(500+1000).

## Performance probe (review)

| Arm | GC share | GC-stall share | ms/leapfrog | Efficiency ratio | Probe ETA h |
|---|---:|---:|---:|---:|---:|
| fusion_qs_bias | 0.289 | 0.382 | 1.50 | 4.49 | 4.314 |
| fusion_qs_nobias | 0.304 | 0.484 | 1.47 | 4.36 | 4.423 |
| fusion_grw_bias | 0.296 | 0.529 | 1.54 | 4.77 | 4.606 |

Tapes are 0 B; the GC is AdvancedHMC-side, as wave 2 (market arms 4.20/4.23 h).
**Full ETA 13.343 h** (≈ 13 h 21 min) = probe window × (40×4×1500)/(16×100) summed over the three
arms; excludes setup, scoring, checkpoints and any R̂ > 1.05 reruns.
