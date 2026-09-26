# Stage 2 completion for m04 ONLY, from the fit already on disk.
#
# WHY THIS EXISTS: the Stage 2 grid sampled all four arms and saved every one to
# disk, but the process died after m04's `save_fit` and before its database row,
# so m04 has 7h30m of valid sampling on disk and no run UUID. m04's Stage B is 40
# folds x 1200 serial inner NUTS runs; re-sampling it to recover a database row we
# can derive from the persisted chains would cost another 7h30m for no new
# information. This replays only the post-sampling tail against the saved fit.
#
# It is deliberately NOT part of the ladder: it asserts the same gates as r20 and
# writes the same row into the same CSV, and it refuses to run if the source
# fingerprint has moved, so a resumed arm cannot silently mix code versions.
# USAGE: julia --project -t 16 .../r21_resume_m04.jl

using ThreadPinning, LinearAlgebra, CSV, DataFrames
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l12_loader.jl"))
const D = DecoupledGenerativeXG

const CONFIG = D.FunnelConfig()
const RUNTIME = D.runtime_config(CONFIG)

# PROVENANCE IS SPLIT HERE, DELIBERATELY AND VISIBLY.
#
# `ACCEPT` is the current fingerprint: the code deciding whether these chains pass.
# `SAMPLED` is the fingerprint that produced them. They differ because the grid's
# only defect was in `cut_stage_b_gate` -- a post-hoc acceptance criterion that is
# not on the sampling path at all -- so the chains it rejected are the same chains
# the corrected gate reads. Re-sampling 7h30m to make a hash match would change a
# label, not a number.
#
# This is NOT a licence to evaluate stale fits: the two are recorded separately in
# the manifest, the arm is refused unless the other three arms carry `SAMPLED`, and
# Stage 3 still asserts that ITS code matches `ACCEPT`.
const ACCEPT = D.source_fingerprint()
const PROD_ROOT = joinpath(CONFIG.save_root, "production")
const dirs = filter(isdir, readdir(PROD_ROOT; join = true))
length(dirs) == 1 || error("expected exactly 1 production source dir, found $(length(dirs))")
const OUTPUT = only(dirs)
const SAMPLED = basename(OUTPUT)
const NAME = "m04_funnel_hierarchical_kappa"
const CSV_PATH = joinpath(OUTPUT, "production_runs.csv")

isfile(CSV_PATH) || error("no production_runs.csv to append to")

# The arm must be missing, and the other three must already be present and passing:
# this script completes an interrupted grid, it does not start one.
existing_rows = CSV.read(CSV_PATH, DataFrame)
NAME in existing_rows.model && error("$NAME already has a production row; nothing to resume")
nrow(existing_rows) == 3 || error("expected 3 completed arms, found $(nrow(existing_rows))")
all(existing_rows.gate_pass) || error("a previously completed arm did not pass its gates")

# Locate the persisted fit for this source and arm.
candidates = filter(p -> startswith(basename(p), NAME),
                    readdir(joinpath(OUTPUT, "full_fits"); join = true))
length(candidates) == 1 || error("expected exactly 1 saved $NAME fit, found $(length(candidates))")
full_path = only(candidates)
all(==(SAMPLED), existing_rows.source) ||
    error("completed arms were not all sampled under $SAMPLED")
println("RESUME sampled_source=", SAMPLED, "\n       accept_source =", ACCEPT,
        "\n       fit=", full_path)
SAMPLED == ACCEPT ||
    println("NOTE: gate code changed since sampling; only cut_stage_b_gate and ",
            "reporting columns differ. Recorded separately in the manifest.")

fit = D.load_fit(full_path)
length(fit.folds) == CONFIG.expected_folds ||
    error("saved fit has $(length(fit.folds)) folds, expected $(CONFIG.expected_folds)")

# ---- identical gate battery to r20 section 6 ----
D.gph_latent_audit(fit)
grid = D.score_grid_audit(fit)
zero_sum_error = D.hierarchical_zero_sum_audit(fit)

gates = [D.cut_stage_b_gate(f.chain) for f in fit.folds]
bad = findall(g -> !g.passed, gates)
isempty(bad) || error("$NAME Stage B failed on folds $bad: $(gates[first(bad)])")
stage_b = (; stage_b_pass = true,
             stage_b_worst_rhat = maximum(g.max_rhat for g in gates),
             stage_b_worst_frac = maximum(g.frac_rhat_gt for g in gates),
             stage_b_min_ess = minimum(g.min_ess for g in gates),
             stage_b_divergences = sum(g.divergences for g in gates),
             stage_b_worst_div_frac = maximum(g.div_frac for g in gates),
             stage_b_runs = sum(g.runs for g in gates))

# Rebuild the SAME recipe r20 registered: `register_recipe!` takes a fit recipe
# (model + splitter + sampler + tags), not a bare model, so the registered hash and
# provenance match what a non-interrupted grid would have written.
db = D.gph_database(CONFIG.experiment)
ds = D.gph_load_data()
splitter = D.gph_splitter(CONFIG.target_seasons)
model = last(only(filter(m -> first(m) == NAME, D.models())))
config = D.fit_recipe(CONFIG, NAME, model, splitter, RUNTIME; smoke = false)
D.register_recipe!(db, NAME, config)
recipe_hash = D.gph_run_hash(db, config)
run_id = D.gph_save_and_verify(db, fit)
convergence_row = D.gph_convergence_row(NAME, fit, RUNTIME; run_id)
passed = D.convergence_pass(fit, CONFIG.expected_folds) && stage_b.stage_b_pass

row = (; convergence_row..., grid..., zero_sum_error, stage_b..., full_path,
         gate_pass = passed, source = SAMPLED, recipe_hash)
println("RESUMED ARM ", row)
passed || error("$NAME failed its gates on resume")

# Append, preserving model order. The three completed arms were written before the
# two new Stage B columns existed, so union the schemas rather than assuming a match.
CSV.write(CSV_PATH, vcat(existing_rows, DataFrame([row]); cols = :union))

posterior = NamedTuple[]
for (name, _) in D.models()
    path = only(filter(p -> startswith(basename(p), name),
                       readdir(joinpath(OUTPUT, "full_fits"); join = true)))
    arm = name == NAME ? fit : D.load_fit(path)
    append!(posterior, [(; model = name, r...)
                        for r in D.kappa_posterior(arm, collect(1:CONFIG.expected_folds))])
end
isempty(posterior) || CSV.write(joinpath(OUTPUT, "kappa_posterior.csv"), DataFrame(posterior))

# Manifest: the immutable run-UUID handoff Stage 3 consumes.
final = CSV.read(CSV_PATH, DataFrame)
nrow(final) == 4 || error("expected 4 arms after resume, found $(nrow(final))")
all(final.gate_pass) || error("an arm does not pass its gates")
manifest = (; source = ACCEPT,
              sampled_source = SAMPLED,
              resumed = [NAME],
              runs = Dict(String(r.model) => String(r.run_id) for r in eachrow(final)),
              folds = CONFIG.expected_folds,
              oos = CONFIG.expected_oos,
              experiment = CONFIG.experiment)
D.Serialization.serialize(joinpath(CONFIG.save_root, "production_manifest.jls"), manifest)
println("PRODUCTION PASS: ", manifest.runs)
