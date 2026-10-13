#!/usr/bin/env bash
# Invoke from repository root. Never echo credential-bearing input or raw errors.
set -euo pipefail
set +x
set -a
. /home/james/bet_project/BayesianFootball/.env
set +a
{ printf "LOAD postgres; ATTACH '%s' AS db (TYPE postgres, READ_ONLY);\n" "${BF_DB_URL//\'/\'\'}"; printf '.read %s\n' "${1:-experiments/scotland/11_qs_transition/research/mae_deep/queries.sql}"; } |
 duckdb -batch -bail 2>&1 | sed -E 's#postgres(ql)?://[^ ]*#<url>#g'
