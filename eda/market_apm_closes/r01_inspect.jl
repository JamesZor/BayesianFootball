# Phase A schema inspection only; no DB reads or writes.
# ===================================================================
# 1. Packages and stream-local segment
# ===================================================================
using BayesianFootball, LinearAlgebra
BLAS.set_num_threads(1)
module QualityStyleEDA
import BayesianFootball
struct MarketModelEnglish <: BayesianFootball.Data.DataTournemantSegment end
BayesianFootball.Data.tournament_ids(::MarketModelEnglish) = [1, 2, 3, 84]
end
# ===================================================================
# 2. Pinned snapshot schemas
# ===================================================================
for segment in (BayesianFootball.Data.ScottishLower(), BayesianFootball.Data.ScottishUpper(), QualityStyleEDA.MarketModelEnglish())
    ds = BayesianFootball.Data.load_datastore_cached(segment; max_age_hours=10^6)
    println(nameof(typeof(segment)))
    println("matches: ", names(ds.matches))
    println("lineups: ", names(ds.lineups))
    show(stdout, MIME("text/plain"), first(ds.matches, 1)); println()
    show(stdout, MIME("text/plain"), first(ds.lineups, 1)); println()
end
