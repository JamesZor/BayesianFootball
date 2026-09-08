using Test
using DataFrames, Dates, JSON3
using HTTP
using BayesianFootball

const BBC_MD = BayesianFootball.MatchDay

function bbc_test_payload(; home_starters = 11, away_starters = 11)
    function player(side, i; position = i == 1 ? "Goalkeeper" : i == 11 ? "Striker" : "Midfielder")
        surname = "$(side)Last$(i)"
        return Dict("urn" => "urn:bbc:sportsdata:football:player:s-$(side)-$(i)",
                    "name" => Dict("first" => "$(side)First$(i)", "last" => surname),
                    "displayName" => "$(side[1]). $surname",
                    "shirtNumber" => i, "position" => position)
    end
    function team(side, n)
        Dict("players" => Dict("starters" => [player(side, i) for i in 1:n],
                               "substitutes" => [player(side, 20; position = "Substitute")]))
    end
    return JSON3.read(JSON3.write(Dict(
        "homeTeam" => team("home", home_starters),
        "awayTeam" => team("away", away_starters))))
end

const BBC_FIXTURE = BBC_MD.Fixture(99001, "home", "away", DateTime(2026, 9, 5, 14), 56)
const BBC_AS_OF = DateTime(2026, 9, 5, 13, 35)

@testset "BBCLineupSource" begin
    @testset "parses a complete confirmed 11 v 11 payload" begin
        lineup = BBC_MD.parse_bbc_lineup(bbc_test_payload(), BBC_FIXTURE, BBC_AS_OF)
        @test lineup isa BBC_MD.Lineup
        @test lineup.confirmed
        @test lineup.source === :bbc
        @test count(p -> !p.substitute, lineup.home) == 11
        @test count(p -> !p.substitute, lineup.away) == 11
        @test count(p -> p.substitute, lineup.home) == 1
        @test all(p -> p.player_id < 0, vcat(lineup.home, lineup.away))
        @test lineup.home[1].position === :M  # synthetic fallback deliberately neutral
    end

    @testset "rejects an unannounced or incomplete XI" begin
        @test BBC_MD.parse_bbc_lineup(bbc_test_payload(home_starters = 10),
                                      BBC_FIXTURE, BBC_AS_OF) === nothing
        @test BBC_MD.parse_bbc_lineup(bbc_test_payload(away_starters = 10),
                                      BBC_FIXTURE, BBC_AS_OF) === nothing
        @test BBC_MD.parse_bbc_lineup("not json", BBC_FIXTURE, BBC_AS_OF) === nothing
    end

    @testset "resolves DB, shirt+surname, and synthetic tiers" begin
        mapped = DataFrame(bbc_player_id = ["s-home-1", "s-home-11"],
                           sofascore_player_id = [1234, 4321],
                           sofascore_name = ["Known", "Known Striker"])
        historical = DataFrame(match_id = [1], player_id = [5678], team_side = ["home"],
                               shirt_number = [2], player_name = ["homeFirst2 homeLast2"])
        ds = (lineups = historical,)
        lineup = BBC_MD.parse_bbc_lineup(bbc_test_payload(), BBC_FIXTURE, BBC_AS_OF;
                                          ds = ds, player_map = mapped)
        @test lineup.home[1].player_id == 1234
        @test lineup.home[2].player_id == 5678
        @test lineup.home[11].player_id == 4321
        @test lineup.home[11].position === :F
        @test lineup.away[1].player_id < 0
    end

    @testset "404, 5xx retry, and timeout failures are fail-soft" begin
        calls = Ref(0)
        handler(request) = begin
            path = HTTP.URI(request.target).path
            occursin("/missing/", path) && return HTTP.Response(404, "missing")
            if occursin("/error/", path)
                calls[] += 1
                return HTTP.Response(503, "unavailable")
            end
            occursin("/slow/", path) && sleep(2)
            return HTTP.Response(200, "{}")
        end
        port = 20_000 + Int(rand(UInt16)) % 10_000
        server = HTTP.serve!(handler, "127.0.0.1", port; verbose = false)
        try
            base = "http://127.0.0.1:$port/"
            source = BBC_MD.BBCLineupSource(timeout_seconds = 0.01, max_retries = 1)
            @test BBC_MD._bbc_lineup_from_event(
                source, BBC_FIXTURE, BBC_AS_OF, "mock";
                base_url = base * "missing/") === nothing
            @test BBC_MD._bbc_lineup_from_event(
                source, BBC_FIXTURE, BBC_AS_OF, "mock";
                base_url = base * "error/") === nothing
            @test calls[] == 2
            no_retry = BBC_MD.BBCLineupSource(timeout_seconds = 0.01, max_retries = 0)
            @test BBC_MD._bbc_lineup_from_event(
                no_retry, BBC_FIXTURE, BBC_AS_OF, "mock";
                base_url = base * "slow/") === nothing
        finally
            close(server)
        end

        # An unsupported tournament has no discovery endpoint and must not throw or
        # accidentally fall through to an unbounded network request.
        unsupported = BBC_MD.Fixture(99002, "home", "away", BBC_AS_OF, 999)
        @test BBC_MD.lineup(BBC_MD.BBCLineupSource(), unsupported, BBC_AS_OF) === nothing
    end
end
