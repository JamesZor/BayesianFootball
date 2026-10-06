# Manager revision-4 exclusion, not a numerical waiver. No integrator change/retry.
# Run unchanged thin-book checks once and report, without accepting C2.
try
    @testset "C2-pending (excluded from Phase C4 acceptance)" begin
        @testset "1X2-only integrated derivative: unchanged tolerance" begin
            config = PC05.MM.scottish_lower_2425_2526()
            ds = BayesianFootball.Data.load_datastore_cached(config.segment; max_age_hours=10^6)
            book,_ = PC05.MM.gated_close(ds,config)
            rates = CSV.read(joinpath(@__DIR__,"results","A","rates.csv"),DataFrame)
            full = sort(filter(r->r.accepted && r.n_selections>=config.min_selections_ladder,rates),:match_id)
            id = first(full.match_id)
            one = filter(r->r.match_id == id && r.market_name == "1X2",book)
            markets = PC05.PF.market_vectors(one)
            d = log(first(full.lambda_h)/first(full.lambda_a))
            L32 = PC05.level_integral(d,markets,1000.0; order=32)
            L64 = PC05.level_integral(d,markets,1000.0; order=64)
            @test abs(L32.marginal-L64.marginal) <= 1e-8
            @test max(L64.lower_relative_logdensity,L64.upper_relative_logdensity) <= -30
            f = x -> PC05.level_integral(x[1],markets,1000.0; order=64).marginal
            T = PC05.third_ad(f,[d])
            discrepancy = norm(T-PC05.third_fd(f,[d]))/norm(T)
            println("C2_PENDING integrated third relative discrepancy=$discrepancy limit=1e-6")
            @test discrepancy <= 1e-6
        end
        @testset "Archived thin-book Gate 1: reported, NOT regenerated" begin
            # Historical measurements remain the evidence; no third numerical variation.
            summary = CSV.read(joinpath(@__DIR__,"results","C","v3_gate","laplace_gate.csv"),DataFrame)
            thin = filter(:book_type=>!=("full"),summary)
            for r in eachrow(thin)
                println("C2_PENDING $(r.book_type) n=$(r.n) sd=$(r.spread) offset=$(r.offset): " *
                    "median=$(r.median_abs_error), p95=$(r.p95_abs_error), " *
                    "mean/SD=$(r.max_mean_error), SDrelative=$(r.max_sd_error), pass=$(r.gate_pass)")
                @test r.gate_pass
            end
        end
    end
    println("C2_PENDING_REPORTED: checks passed; still excluded, no C2 fit")
catch exception
    exception isa Test.TestSetException || rethrow()
    println("C2_PENDING_REPORTED: thin-book failures retained; excluded from C4 acceptance")
end
