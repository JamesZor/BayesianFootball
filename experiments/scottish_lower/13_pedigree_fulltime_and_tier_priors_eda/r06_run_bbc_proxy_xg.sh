#!/usr/bin/env bash
# Run from any directory. BF_DB_URL must be supplied by the caller's environment.
# The Python extractor opens a read-only transaction and never prints the DSN.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/r06_extract_bbc_proxy_xg.py"
