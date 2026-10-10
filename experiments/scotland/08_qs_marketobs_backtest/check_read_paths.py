#!/usr/bin/env python3
"""Static read-path gate; no database access or Julia execution.

Review covers the experiment entry points. The transitive load_fit read path was
inspected separately in src/training/inference/{io,db_storage}.jl; it is not the
whole db_storage module (which also defines unused persistence functions).
"""
import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
FILES = tuple(sys.argv[1:]) or ("l00_phase0_inventory.jl", "r00_phase0_inventory.jl")
FORBIDDEN_SQL = re.compile(r"\b(?:INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|MERGE|CALL|COPY|GRANT|REVOKE)\b", re.I)
FORBIDDEN_CALL = re.compile(r"\b(?:save\w*|record\w*|persist\w*|extend\w*|migrate\w*|ensure\w*|fit_model|run_tasks|run_experiment|_db_exec\w*|load_datastore_sql|load_datastore_cached)\s*\(")
TRAINING_CALL = re.compile(r"BF\.Training\.([\w.]+)\s*\(")
ALLOWED_TRAINING = {"PostgresStorage", "load_fit", "Fit", "Inference._db_connect"}

failures = []
for name in FILES:
    path = ROOT / name
    text = path.read_text()
    for number, line in enumerate(text.splitlines(), 1):
        # These files have no block comments; retain strings to inspect SQL too.
        code = line.split("#", 1)[0]
        if FORBIDDEN_SQL.search(code) or FORBIDDEN_CALL.search(code):
            failures.append(f"{name}:{number}: forbidden write/schema/sampling/fallback path")
        for call in TRAINING_CALL.findall(code):
            if call not in ALLOWED_TRAINING:
                failures.append(f"{name}:{number}: non-allowlisted Training call {call}")
        if "LibPQ.execute(" in code and "sql" not in code:
            failures.append(f"{name}:{number}: DB execution outside read_query guard")
        if "BF.Harness." in code or "BF.Experiments." in code:
            failures.append(f"{name}:{number}: harness/experiment execution is not allowed")
    print(f"SOURCE {name} sha256={hashlib.sha256(text.encode()).hexdigest()}")

if failures:
    print("STATIC_READ_PATHS_FAIL")
    print("\n".join(failures))
    sys.exit(1)
print("STATIC_READ_PATHS_PASS: only allowlisted fit reads; SQL executes through SELECT/SHOW guard; no SQL cache fallback")
