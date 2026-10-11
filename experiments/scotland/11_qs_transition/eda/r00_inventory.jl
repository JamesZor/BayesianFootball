# Stage 1 read-only inventory. Execute only on beast via jl.
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, Serialization, LibPQ
BLAS.set_num_threads(1)
if !isdefined(Main, :QSBacktestPhase0)
    include(joinpath(@__DIR__, "../../08_qs_marketobs_backtest/l00_phase0_inventory.jl"))
end
const S1OUT = joinpath(@__DIR__, "results")
const S1CACHE = "/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls"
function s1audit(f, label)
    mkpath(S1OUT)
    conn = BayesianFootball.Training.Inference._db_connect(Training.PostgresStorage("scottish_lower_qs_wave2_2426"))
    try
        before = QSBacktestPhase0.audit_snapshot(conn)
        CSV.write(joinpath(S1OUT, label * "_db_before.csv"), before)
        try
            f(conn)
        finally
            after = QSBacktestPhase0.audit_snapshot(conn)
            CSV.write(joinpath(S1OUT, label * "_db_after.csv"), after)
            checks = DataFrame(table_name=sort(unique(before.table_name)))
            checks.unchanged = [isequal(filter(r->r.table_name==t,before),filter(r->r.table_name==t,after)) for t in checks.table_name]
            CSV.write(joinpath(S1OUT,label * "_db_audit.csv"),checks)
            println("AUDIT ", label, " unchanged=", all(checks.unchanged), " tables=", nrow(checks))
            all(checks.unchanged) || error("DB audit changed; STOP")
        end
    finally
        close(conn)
    end
end
s1audit("inventory") do conn
    global s1ds = Serialization.deserialize(S1CACHE)
    println("CACHE matches=", nrow(s1ds.matches), " fields=", join(string.(propertynames(s1ds.matches)),","))
    # Cup season labels can be calendar years; enforce the actual date embargo too.
    m = filter(r->String(r.season) <= "25/26" && Date(r.match_date) < Date(2026,7,1), s1ds.matches)
    CSV.write(joinpath(S1OUT,"fixtures.csv"),m)
    coverage = combine(groupby(s1ds.matches, [:season,:tournament_id]), nrow=>:n_fixtures)
    CSV.write(joinpath(S1OUT,"q6_fixture_inventory.csv"),coverage)
    schema = QSBacktestPhase0.read_query(conn,"SELECT table_name,column_name,data_type FROM information_schema.columns WHERE table_schema='public' AND table_name IN ('runs','harness_scores','fold_results','match_latents','fit_fold_artifacts') ORDER BY table_name,ordinal_position")
    CSV.write(joinpath(S1OUT,"db_schema.csv"),schema)
    println("RUN_FIELDS ", join(schema.column_name[schema.table_name .== "runs"],","))
    tiers = Harness.club_season_tiers(s1ds)
    panel = Int.(filter(r->String(r.season) in ["23/24","24/25","25/26"] && Int(r.tournament_id) in (56,57),m).match_id)
    gate = NamedTuple[]
    for (direction,expected) in [(:relegated_into_L1,5),(:promoted_into_L1,6),(:entered_spfl,2)]
        cohort = Harness.transition_cohort(s1ds,panel,tiers; first_n=20,direction)
        attrs = Dict(Int(r.match_id)=>String(r.season) for r in eachrow(m))
        pairs = Set((c,attrs[mid]) for (mid,clubs) in cohort.clubs for c in clubs)
        push!(gate,(;direction=String(direction),expected,n_club_seasons=length(pairs),n_fixtures=length(cohort.match_ids),pass=length(pairs)==expected))
    end
    CSV.write(joinpath(S1OUT,"q6_cohort_gate.csv"),DataFrame(gate))
    println(DataFrame(gate))
    all(r.pass for r in gate) || error("W2 cohort counts do not match 5/6/2; STOP")
end
