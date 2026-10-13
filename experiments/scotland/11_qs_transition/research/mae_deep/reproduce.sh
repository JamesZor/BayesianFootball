#!/usr/bin/env bash
# Run from the repository root. Offline by default: no credentials or DB needed.
# PYTHON=/tmp/qs-s1-venv/bin/python bash .../reproduce.sh
set -euo pipefail
P=experiments/scotland/11_qs_transition/research/mae_deep
PYTHON=${PYTHON:-python3}
tar -xzf "$P/extract_snapshot.tar.gz" -C "$P"
for f in features sensitivity extra_features; do
  duckdb -batch -bail < "$P/$f.sql"
done
"$PYTHON" "$P/robustness.py" > "$P/robustness.log"
"$PYTHON" "$P/outcome_sanity.py" > "$P/outcome_sanity.log"
duckdb -batch -bail < "$P/final_tables.sql"
"$PYTHON" "$P/verify.py"
# Remove only our explicitly named, reproducible, large raw CSV extracts.
for f in matches lineups statistics bbc_shots bbc_meta cup_matches cup_lineups; do
  rm -- "$P/$f.csv"
done
printf 'Offline reproduction and validation complete. Raw extracts remain in extract_snapshot.tar.gz.\n'
