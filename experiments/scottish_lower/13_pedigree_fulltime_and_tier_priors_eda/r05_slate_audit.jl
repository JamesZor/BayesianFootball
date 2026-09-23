# Read-only entry point for the 2026-09-19 paper_runbook provenance audit.
#
# The implementation is kept in Python because the available PostgreSQL drivers expose a
# portable transaction-read-only switch there. This runner intentionally has no fallback to
# reconstruction or sampling: the saved ledger probability is the operational evidence.

const R05_PYTHON_AUDIT = joinpath(@__DIR__, "r05_slate_audit.py")

function run_slate_audit()
    isfile(R05_PYTHON_AUDIT) || error("r05_slate_audit.py is missing beside this runner")
    python = get(ENV, "PYTHON", "python3")
    run(`$python $R05_PYTHON_AUDIT`)
    return nothing
end

run_slate_audit()
