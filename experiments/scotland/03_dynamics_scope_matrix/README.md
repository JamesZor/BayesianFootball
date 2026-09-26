# 03 — Scottish Pyramid Dynamics × Scope × Observation Matrix (TODO 031 Wave 1)

## Question

How do team dynamics (`TimeDecayDynamics(180)` vs `MultiScaleGRW()`), data scope (`lower` vs `spfl` vs `spfl_cups`), and likelihood (`Poisson` vs `JointGammaPoissonObservation`) interact on market-on-model compression and out-of-sample log-loss across the Scottish league pyramid?

## Candidates (Wave 1: 12 Factorial Arms)

| Candidate | Dynamics | Scope | Likelihood | Role |
|---|---|---|---|---|
| `td_lower_poisson` | TimeDecay(180) | lower (56/57) | Poisson | candidate |
| `td_lower_joint` | TimeDecay(180) | lower (56/57) | Joint pxG | **control** |
| `td_spfl_poisson` | TimeDecay(180) | SPFL (54–57) | Poisson | candidate |
| `td_spfl_joint` | TimeDecay(180) | SPFL (54–57) | Joint pxG | candidate |
| `td_spfl_cups_poisson` | TimeDecay(180) | SPFL + senior cup ties | Poisson | candidate |
| `td_spfl_cups_joint` | TimeDecay(180) | SPFL + senior cup ties | Joint pxG | candidate |
| `grw_lower_poisson` | MultiScaleGRW | lower (56/57) | Poisson | candidate |
| `grw_lower_joint` | MultiScaleGRW | lower (56/57) | Joint pxG | candidate |
| `grw_spfl_poisson` | MultiScaleGRW | SPFL (54–57) | Poisson | candidate |
| `grw_spfl_joint` | MultiScaleGRW | SPFL (54–57) | Joint pxG | candidate |
| `grw_spfl_cups_poisson` | MultiScaleGRW | SPFL + senior cup ties | Poisson | candidate |
| `grw_spfl_cups_joint` | MultiScaleGRW | SPFL + senior cup ties | Joint pxG | candidate |

## Decision Rule (from TODO 031)

- Screen stage (`--stage screen`): fast MAP ranking across all 40 walk-forward folds to verify feature builder integrity, clock stability, and initial calibration ordering.
- Grid selection on the canonical 56/57 walk-forward book (710 held-out matches across 24/25 and 25/26).
- Primary selection criteria:
  1. Decompression: market-on-model compression slope closest to 1.0 (controlling under-confidence vs over-confidence).
  2. Target 1X2 and all-market LogLoss vs control (`td_lower_joint`).
  3. Monitor reporting on 54/55 Premiership and Championship tiers.
