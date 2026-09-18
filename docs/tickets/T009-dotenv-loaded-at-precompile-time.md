# T009 — `.env` is loaded at precompile time, so `BF_DB_URL` is unset at runtime

| | |
|---|---|
| **Status** | open |
| **Severity** | medium — no wrong numbers, but every `betdb` reader fails outside a shell that exports the variable itself |
| **Area** | `src/BayesianFootball.jl` (module body) |
| **Raised** | 2026-09-12, by Task 008 Phase 1 `r05_slate_repricing.jl` on `mcmc-beast` |

## Summary

`BayesianFootball` reads `.env` in the **top-level body** of the module:

```julia
# src/BayesianFootball.jl:6-12
# Loads .env (e.g. BF_DB_URL) into ENV at module init. ...
using DotEnv
const env_path = joinpath(pkgdir(@__MODULE__), ".env")
if isfile(env_path)
    DotEnv.load!(ENV, env_path)
end
```

Top-level code in a package module runs when the package is **precompiled**, not when it
is loaded from its cache. The `ENV` mutation therefore happens inside the precompile worker
and is lost; a session that loads the cached package never sees `.env`. The comment says
"at module init", but nothing here is in `__init__`.

## Evidence

On `mcmc-beast`, `/root/BF_hier_ha_slate/.env` defines `BF_DB_URL` (81-character quoted
value). A runner launched as `tmux new-window "script.sh"` (non-interactive, `.bashrc` not
sourced) failed at its first `betdb` read:

```
ERROR: LoadError: paper_connection: BF_DB_URL is not set. Export it, e.g.
```

Every runner that has worked on this host was started from an interactive shell, and
`/root/.bashrc:3` carries `export BF_DB_URL=…`. That export — not `.env` — is what has been
supplying the variable. The same script succeeded once `BF_DB_URL` was exported from `.env`
by the shell before `julia` started.

## Reproduction

```bash
cd /root/<any checkout with .env and a warm precompile cache>
env -u BF_DB_URL /root/.juliaup/bin/julia --project -e \
  'using BayesianFootball; println(haskey(ENV, "BF_DB_URL"))'     # prints false
```

## Blast radius

Anything that reaches `betdb` (`Data.load_datastore_sql`, `MatchDay.paper_connection`, the
live and replay consoles, the `ScottishLower` cache refresh) when started without the
variable in the parent environment: `tmux new-window` / `send-keys` into a non-login shell,
`systemd` units, `cron`, CI, `nohup` wrappers, and any fresh machine whose `.bashrc` lacks
the export. `mcmc_experiments` access is unaffected (`PostgresStorage` falls back to
`~/.pgpass`).

## Proposed fix

Move the load into `__init__`, which runs on every load, cached or not:

```julia
using DotEnv
function __init__()
    path = joinpath(pkgdir(@__MODULE__), ".env")
    isfile(path) && DotEnv.load!(ENV, path)
end
```

Check for an existing `__init__` in `src/BayesianFootball.jl` first and merge rather than
define a second one. Decide explicitly whether `.env` should **override** a variable already
in the parent environment; the current `load!` semantics should be kept and documented.

## Acceptance criteria

- [ ] The reproduction above prints `true` with a warm precompile cache.
- [ ] A shell that already exports `BF_DB_URL` keeps the documented precedence.
- [ ] The comment matches the behaviour.
- [ ] Full suite passes (`test/runtests.jl`).

## Scope guard

Do not change how credentials are stored, printed or masked. Do not touch
`PostgresStorage`'s `.pgpass` fallback. Do not add new environment variables.
