# Offline evidence-join regression checks. No SQL and no output-panel mutation.
# 1. Packages and implementation
using Test
include(joinpath(@__DIR__, "r03_status_construction.jl"))

# 2. Synthetic evidence: exercise IDs, names, time boundaries and conflicts
@testset "Operational evidence join" begin
    evidence = DataFrame(
        club_name = ["Cove Rangers"], club_id = Union{Missing,Int}[8062],
        operational_status = ["Hybrid"], evidence_level = ["Verified"],
        effective_from = [Date(2023, 7, 1)], effective_to = [Date(2024, 6, 30)],
        source_id = ["S002"], classification_note = ["synthetic"],
    )
    membership = DataFrame(club = ["cove-rangers"], club_id = [8062], season = ["23/24"])
    @test status_for_membership(evidence, membership[1, :])[1] == "Hybrid"
    membership.club_id .= 999
    @test status_for_membership(evidence, membership[1, :])[1] == "Unknown"
    no_id = DataFrame(club = ["cove-rangers"], season = ["23/24"])
    @test status_for_membership(evidence, no_id[1, :])[1] == "Hybrid"
    evidence.club_id .= missing
    @test status_for_membership(evidence, membership[1, :])[1] == "Hybrid"
    @test nrow(active_evidence_rows(evidence, no_id[1, :], Date(2023, 6, 30))) == 0
    @test nrow(active_evidence_rows(evidence, no_id[1, :], Date(2023, 7, 1))) == 1
    @test nrow(active_evidence_rows(evidence, no_id[1, :], Date(2024, 6, 30))) == 1
    @test nrow(active_evidence_rows(evidence, no_id[1, :], Date(2024, 7, 1))) == 0
    @test_throws ErrorException status_for_membership(vcat(evidence, evidence), no_id[1, :])
    @test season_start("2023/2024") == Date(2023, 7, 1)
    @test_throws ErrorException season_start("2023")
    @test canonical_club_key("Queen's Park FC") == "queens-park"
end
