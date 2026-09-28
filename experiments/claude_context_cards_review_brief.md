# Review brief — TODO 037 context cards, branch `docs/context-cards`

You are the **independent reviewer**. A pi agent on Gemini 3.8 Flash wrote these cards.
- Earlier Gemini documentation in this repo had wrong line numbers and defaults, so **verify
  claims against the code; do not trust the prose**.
- Do not fix the docs yourself; report findings. No subagents.

## Inputs

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-context-cards`. Pull first.
  The diff under review is `git diff <base>...docs/context-cards`, where `<base>` is the
  `feat/w2-tier-components` commit named in the TODO 037 Work Log.
- **Contract:** `experiments/pi_context_cards_prompt.md` (the agreed design; don't re-litigate it)
  and the TODO 037 acceptance criteria.
- **Builder report:** `docs/architecture/context_cards_report.md`.
- **Rules:** read-only everywhere except your review file. No Julia, no DB, no credentials.

## Checks

1. **Accuracy, every card.**
   - For each frontmatter `symbols` entry, confirm it exists in the listed `sources`.
   - For every signature, keyword default, threshold, constant, table or column name and CLI flag
     quoted in a card body, grep the code and confirm it is **verbatim**. List every mismatch.
   - Read the documented function for at least one behavioural claim per card, and confirm the
     claim holds.
   - Confirm there are no line numbers anywhere (`rg -n ':[0-9]+|line [0-9]+|L[0-9]{2,}' docs/context`
     and inspect the hits).
2. **Shape.**
   - Every card is ≤ 150 lines, uses the template sections, and has valid frontmatter (`id` equals
     its path; `related` ids exist).
   - ASCII blocks are fenced as `text` and ≤ 100 columns.
   - The INDEX files list every card.
3. **Slimming (no information loss).** For each of the four guides, compare the base and head
   versions.
   - Every removed paragraph, table, code block and rule must appear in a card, or be marked kept,
     and the ledger row must be correct.
   - The seven rules in the DB guide §0 must be verbatim.
   - Every anchor linked from outside `docs/context` still resolves. Check the headings, not just
     the ledger.
   - List every lost or altered fact.
4. **"Why" sections.** Each links a real TODO, README or report, and the link resolves.
5. **Stale check.**
   - Run `scripts/context_stale.sh` and `--strict`. Show that it exits 0 in normal mode.
   - Show that `./scripts/todo.sh check` still passes with a stale card: make a scratch local edit
     to a source, then revert it with no commit.
   - Confirm the symbol and missing-file warnings fire.
6. **`UNVERIFIED:` items.** Resolve each one against the code if you can; list the rest.
7. **Wiring.** Check the AGENTS.md row, that AGENTS.md is under 22,000 bytes, the `docs/README.md`
   pointer, and that `./scripts/todo.sh check` passes.

## Output

Write `docs/architecture/context_cards_review.md`:
- `VERDICT: ACCEPT` or `VERDICT: CHANGES_REQUIRED`;
- a findings table: ID, severity (blocker / major / minor / nit), file and section, what is
  wrong, what the code actually says, and the fix;
- the counts: cards checked, tokens verified, mismatches, lost facts.

A wrong fact on a card or a lost fact from a guide is at least **major**. Commit and push that
file only, print exactly `CTX_REVIEW_DONE` on its own line, and stop. For a re-review, append
"Re-review N" and print `CTX_REREVIEW<N>_DONE`.
