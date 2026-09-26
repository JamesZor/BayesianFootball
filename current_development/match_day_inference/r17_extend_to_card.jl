# r17 — Extend a live run by the fold whose held-out block is the NEXT (unplayed) card.
#
#   R17_EXPERIMENT=scottish_lower_joint_player_2426 R17_RUN=132df5c2-c742-4e95-8693-3aeb2b2cbaef \
#   R17_FROM=2026-10-03 R17_TO=2026-10-04 [R17_PREVIEW=1] julia --project -t 16 r17_extend_to_card.jl
#
# Steps: fresh ScottishLower DataStore → inject the card (l17) → preview → extend_fit (writes the
# new fold back to the same run) → verify that MatchDay.select_split, given the same injected
# store, conditions the card on the NEW last fold.

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball, DataFrames, Dates, LibPQ, Printf, UUIDs, Serialization
const MD = BayesianFootball.MatchDay
const DD = BayesianFootball.Data
const TT = BayesianFootball.Training
const PG = BayesianFootball.Models.PreGame
include(joinpath(@__DIR__, "l17_upcoming_fixtures.jl"))

# Pre-2026-09-03 artefacts hold a 3-field JointGammaPoissonObservation (memory note / r68 §0).
if fieldcount(PG.JointGammaPoissonObservation) == 4
    function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                       T::Type{<:PG.JointGammaPoissonObservation})
        T isa DataType && return invoke(Serialization.deserialize,
                                        Tuple{Serialization.AbstractSerializer, DataType}, s, T)
        fields = Any[]
        for _ in 1:3
            tag = Int32(read(s.io, UInt8)::UInt8)
            push!(fields, Serialization.handle_deserialize(s, tag))
        end
        return PG.JointGammaPoissonObservation(fields[1], fields[2], fields[3], PG.SharedKappa())
    end
end

const EXPERIMENT = ENV["R17_EXPERIMENT"]
const RUN = ENV["R17_RUN"]
const FROM = DateTime(Date(ENV["R17_FROM"]))
const TO = DateTime(Date(ENV["R17_TO"]))
const PREVIEW = get(ENV, "R17_PREVIEW", "0") == "1"

splitter = DD.GroupedCVConfig(tournament_groups = [[56, 57]],
                              target_seasons = ["24/25", "25/26", "26/27"],
                              history_seasons = 2, dynamics_col = :match_biweek,
                              warmup_period = 0, stop_early = true)

db = TT.PostgresStorage(EXPERIMENT)
ds = DD.load_datastore_cached(DD.ScottishLower(); force = true)
played_latest = maximum(ds.matches.match_date)
card = inject_upcoming_fixtures!(ds, FROM, TO)
println("store: latest played ", played_latest, " | injected ", length(card), " card fixture(s) ", FROM, " → ", TO)
isempty(card) && error("no not-started 56/57 fixtures in the window; nothing to extend to")

b = DD.create_id_boundaries(ds, splitter)
last_next = Set(Int.(DD.get_next_matches(ds, b[end], splitter).match_id))
last_next == Set(card) || error("the last fold's held-out block is not exactly the card " *
                                "($(length(last_next)) vs $(length(card)) ids) — refusing")
train_ids = vcat(b[end][1].history_match_ids, b[end][1].target_match_ids)
latest_train = maximum(ds.matches.match_date[in.(Int.(ds.matches.match_id), Ref(Set(Int.(train_ids))))])
println("last fold #", length(b), ": trained through ", latest_train, ", holds out the card")

plan = TT.preview_extension(db, RUN, ds; splitter = splitter)
println("preview: ", plan.new_count, " new fold(s)")
PREVIEW && (println("R17_PREVIEW_DONE"); exit(0))

if plan.new_count > 0
    t0 = time()
    TT.extend_fit(db, RUN, ds; splitter = splitter, execution = TT.QueuedExecution(16))
    @printf("extended in %.1f min\n", (time() - t0) / 60)
end

# The live GRW slate (r16) loads Task 007 run f870dbb7 with require_converged = false: its
# fold 43 already fails that run's strict R̂ < 1.01 gate. Mirror whatever the slate uses.
cf = MD.canonical_fit(db, RUN; require_converged = get(ENV, "R17_REQUIRE_CONVERGED", "1") == "1")
cf.n_folds == length(b) || error("run has $(cf.n_folds) folds; store builds $(length(b))")
sel = MD.select_split(cf, b; exclude = card, ds = ds, config = splitter, fixture_ids = card)
sel.idx == length(b) || error("select_split picked fold $(sel.idx), not the new fold $(length(b))")
println("select_split → fold ", sel.idx, " (warning: '", sel.warning, "')")
@printf("R17_OK run=%s folds=%d trained_through=%s card=%d converged=%s\n",
        RUN, cf.n_folds, latest_train, length(card), cf.converged)
