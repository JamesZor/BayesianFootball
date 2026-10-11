# Manager decisions (overnight run, TODO 046)

Each gate decision the manager took for James, with its reason. James reviews these in the morning.

- **D0 (2026-10-11)** The S0 research uses gpt-6-astra, capped at about 25 web fetches and 120K context.
- **D1 (2026-10-11 ~23:00) S0 accepted.** It delivered 12 EDA questions, 3 prior forms (P1 cohort
  prior, P2 club carry plus mover shift, P3 tier rebase plus club jump) and a cited report. Reviewed
  for scope; no fits, no DB.
- **D2 (2026-10-11) S1 question list = S0's Q1–Q12.** They supersede the base set where they overlap.
  - Order: Q6 and Q12 (prerequisites), then ★ Q1–Q5, then Q7, Q10, Q8.
  - Q9 (cups vs bookmaker) and Q11 (squad discontinuity) are optional if time is short.
  - Reason: Q1–Q5 decide the cause verdict and the prior. Q9 and Q11 need new data plumbing (cup
    bookmaker odds, lineup minutes) for secondary questions.
- **D3 (2026-10-11) S1 runs on gpt-6.1-sol with thinking high** (diagnosis). Astra used about 24%
  of the 5h window in 10 minutes, so it is not used for the build stages.
