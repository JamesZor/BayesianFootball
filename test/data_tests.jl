# test/data_tests.jl

using Test
using BayesianFootball
using DataFrames
using Dates
using InlineStrings

@testset "Data Module" begin
    
    @testset "Fractional to Decimal Parsing" begin
        # Valid cases
        @test BayesianFootball.Data.parse_fractional_to_decimal("1/1") == 2.0
        @test BayesianFootball.Data.parse_fractional_to_decimal("1/2") == 1.5
        @test BayesianFootball.Data.parse_fractional_to_decimal("5/2") == 3.5
        @test BayesianFootball.Data.parse_fractional_to_decimal("100/30") ≈ 4.333 atol=1e-3
        
        # Invalid cases
        @test BayesianFootball.Data.parse_fractional_to_decimal("SP") == 0.0
        @test BayesianFootball.Data.parse_fractional_to_decimal("1") == 0.0
        @test BayesianFootball.Data.parse_fractional_to_decimal(missing) == 0.0
    end

    @testset "Lineup market values" begin
        raw = DataFrame(
            tournament_id = [56, 56],
            season_id = [1, 1],
            match_id = [1001, 1001],
            team_side = ["home", "away"],
            player_id = [101, 202],
            proposed_market_value = Union{Missing, Int64}[250_000, missing],
            proposed_market_value_currency = Union{Missing, String}["EUR", missing],
            totalPass = Union{Missing, Float64}[42.0, missing],
        )

        processed = BayesianFootball.Data.process_data(
            raw, BayesianFootball.Data.LineUpsData())

        @test eltype(processed.proposed_market_value) == Union{Missing, Int64}
        @test isequal(processed.proposed_market_value, Union{Missing, Int64}[250_000, missing])
        @test eltype(processed.proposed_market_value_currency) == Union{Missing, String3}
        @test processed.proposed_market_value_currency[1] == "EUR"
        @test ismissing(processed.proposed_market_value_currency[2])
        @test "total_passes" in names(processed)
        @test eltype(processed.total_passes) == Union{Missing, Float64}
        @test isequal(processed.total_passes, Union{Missing, Float64}[42.0, missing])
    end

    @testset "Betfair Grading" begin
        # 1X2
        @test BayesianFootball.Data.grade_selection("1X2", 0.0, :home, 2, 1) == true
        @test BayesianFootball.Data.grade_selection("1X2", 0.0, :home, 1, 1) == false
        @test BayesianFootball.Data.grade_selection("1X2", 0.0, :draw, 1, 1) == true
        @test BayesianFootball.Data.grade_selection("1X2", 0.0, :away, 1, 2) == true
        
        # Over/Under
        @test BayesianFootball.Data.grade_selection("OverUnder", 2.5, :over_25, 2, 1) == true
        @test BayesianFootball.Data.grade_selection("OverUnder", 2.5, :over_25, 1, 1) == false
        @test BayesianFootball.Data.grade_selection("OverUnder", 2.5, :under_25, 1, 1) == true
        
        # BTTS
        @test BayesianFootball.Data.grade_selection("BTTS", 0.0, :btts_yes, 1, 1) == true
        @test BayesianFootball.Data.grade_selection("BTTS", 0.0, :btts_yes, 2, 0) == false
        @test BayesianFootball.Data.grade_selection("BTTS", 0.0, :btts_no, 2, 0) == true
        
        # Correct Score
        @test BayesianFootball.Data.grade_selection("CorrectScore", 0.0, :cs_21, 2, 1) == true
        @test BayesianFootball.Data.grade_selection("CorrectScore", 0.0, :cs_21, 1, 2) == false
        @test BayesianFootball.Data.grade_selection("CorrectScore", 0.0, :cs_any_other_home, 4, 1) == true
        @test BayesianFootball.Data.grade_selection("CorrectScore", 0.0, :cs_any_other_home, 3, 1) == false
        
        # Missing data
        @test ismissing(BayesianFootball.Data.grade_selection("1X2", 0.0, :home, missing, 1))
    end

    @testset "Betfair summaries retain closing-only fixtures (T005)" begin
        D = BayesianFootball.Data
        selections = [:home, :draw, :away]

        betfair_ticks = DataFrame(
            match_id = Int[],
            market_name = String[],
            market_line = Float64[],
            selection = Symbol[],
            traded_price = Float64[],
            minutes_to_kickoff = Float64[],
        )
        # Mirror the ticket's Scottish Lower OOS inventory: 360 requested fixtures,
        # 324 with ticks, 322 with a valid close, but only 30 with an opening observation.
        for match_id in 1:322
            if match_id <= 30
                for (selection, price) in zip(selections, (2.50, 3.20, 3.00))
                    push!(betfair_ticks,
                          (match_id, "1X2", 0.0, selection, price, -1400.0))
                end
            end
            for (selection, price) in zip(selections, (2.40, 3.40, 3.10))
                push!(betfair_ticks, (match_id, "1X2", 0.0, selection, price, -10.0))
            end
        end
        for match_id in 323:324, (selection, price) in zip(selections, (2.50, 3.20, 3.00))
            push!(betfair_ticks, (match_id, "1X2", 0.0, selection, price, -2000.0))
        end

        matches = DataFrame(
            match_id = collect(1:360),
            home_score = fill(2, 360),
            away_score = fill(1, 360),
            match_date = [Date("2024-08-01") + Day(i - 1) for i in 1:360],
        )
        empty = DataFrame()
        ds = D.DataStore(D.ScottishLower(), matches, empty, empty, empty, empty,
                         betfair_ticks)

        summary = D.summarize_betfair_market(ds)
        @test Set(summary.match_id) == Set(1:322)
        @test length(unique(summary.match_id)) / nrow(matches) == 322 / 360
        @test length(unique(summary.match_id)) / nrow(matches) >= 0.85

        closing_only = summary[summary.match_id .== 31, :]
        @test nrow(closing_only) == 3
        for column in (:odds_open, :overround_open, :prob_implied_open,
                       :prob_fair_open, :fair_odds_open, :vig_open)
            @test all(ismissing, closing_only[!, column])
        end
        @test all(!ismissing, closing_only.odds_close)
        @test all(!ismissing, closing_only.prob_fair_close)

        no_open = D.summarize_betfair_market(ds; open_window=(-5000.0, -4000.0))
        @test Set(no_open.match_id) == Set(1:322)
        @test all(ismissing, no_open.odds_open)

        strict = D.summarize_betfair_market(ds; require_open=true)
        @test Set(strict.match_id) == Set(1:30)
        @test all(!ismissing, strict.odds_open)

        sparse_ticks = filter(:match_id => in(1:2), betfair_ticks)
        for (selection, price) in zip(selections, (2.50, 3.20, 3.00))
            push!(sparse_ticks, (3, "1X2", 0.0, selection, price, -2000.0))
        end
        sparse_ds = D.DataStore(D.ScottishLower(), first(matches, 3), empty, empty,
                                empty, empty, sparse_ticks)
        @test_logs (:warn, r"Betfair closing-summary coverage is below 85%") begin
            D.summarize_betfair_market(sparse_ds)
        end
    end

    @testset "Preprocessing: Match Week Logic" begin
        df = DataFrame(
            tournament_id = [1, 1, 1, 1],
            season = [2022, 2022, 2022, 2023],
            # Matches on Mon, Tue, and next Mon
            match_date = [Date("2022-08-01"), Date("2022-08-02"), Date("2022-08-08"), Date("2023-08-01")] 
        )
        
        processed_df = BayesianFootball.Data.add_match_week_column(df)
        
        @test "match_week" in names(processed_df)
        # Aug 1 and Aug 2 are in the same week for the 2022 season (Week 1)
        @test processed_df.match_week[1] == 1
        @test processed_df.match_week[2] == 1
        # Aug 8 is the next week (Week 2)
        @test processed_df.match_week[3] == 2
        
        # 2023 season restarts the week counter
        @test processed_df.match_week[4] == 1
    end
end
