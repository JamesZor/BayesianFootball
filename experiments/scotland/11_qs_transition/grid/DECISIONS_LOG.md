# S3 decisions

- 2026-10-12: D13–D14 authorise four sequential grids and removal of only stale hard check832 after full CSV archive is committed. No gate bypass, recipe/sampler tuning, src edits, F6 work or 26/27 outcomes.
- Owned checkout had no origin remote. Synced latest pushed exp/qs-transition d3f86b29 using a local git bundle and fast-forward in /root/BF_runs/qs_transition only (no push). Cache SHA matched a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12.
- Fresh owned s3 beast REPL16 started. Read-only archive job095752-9899 succeeded; full row832 copied to CSV before any deletion. Cleanup will assert archived row equality and recipe/global count decrements of exactly1.
