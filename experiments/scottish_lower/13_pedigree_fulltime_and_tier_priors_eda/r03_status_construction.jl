#!/usr/bin/env julia

# Stage 1 constructor for the versioned club operational-status panel.
# It uses the database-derived membership extract, never a current 42-club list.
# Missing or non-applicable evidence becomes Unknown; inferred continuity remains explicit.

using CSV
using DataFrames
using Dates

const EXPERIMENT_DIR = @__DIR__
const DATA_DIR = joinpath(EXPERIMENT_DIR, "data")
const MEMBERSHIP_PATH = joinpath(DATA_DIR, "r01_club_tier_membership.csv")
const EVIDENCE_PATH = joinpath(DATA_DIR, "operational_status_evidence.csv")
const OUTPUT_PATH = joinpath(DATA_DIR, "spfl_club_operational_status.csv")
const REQUIRED_MEMBERSHIP_COLUMNS = [:tournament_id, :season, :club, :first_match_date, :last_match_date]
const REQUIRED_EVIDENCE_COLUMNS = [
    :club_name, :club_id, :operational_status, :evidence_level,
    :effective_from, :effective_to, :source_id, :classification_note,
]

function require_columns(df::DataFrame, required::Vector{Symbol}, path::String)
    absent = setdiff(required, Symbol.(names(df)))
    isempty(absent) || error("$(path) is missing required columns $(absent)")
    return nothing
end

function season_start(season::AbstractString)
    match_result = match(r"^(\d{2}|\d{4})/(\d{2}|\d{4})$", season)
    isnothing(match_result) && error("unrecognised season $(repr(season)); expected YY/YY or YYYY/YYYY")
    first_year = match_result.captures[1]
    year = length(first_year) == 2 ? 2000 + parse(Int, first_year) : parse(Int, first_year)
    return Date(year, 7, 1)
end

function parse_evidence_date(value, field::Symbol)
    ismissing(value) && return missing
    value isa Date && return value
    parsed = tryparse(Date, strip(String(value)))
    isnothing(parsed) && error("invalid $(field) date $(repr(value)) in $(EVIDENCE_PATH)")
    return parsed
end

function canonical_club_key(name::AbstractString)
    key = lowercase(strip(name))
    key = replace(key, "-fc" => "", " fc" => "", "'" => "", " " => "-")
    return key
end

function active_evidence_rows(evidence::DataFrame, membership_row::DataFrameRow, date::Date)
    starts = parse_evidence_date.(evidence.effective_from, :effective_from)
    any(ismissing, starts) && error("every evidence row requires a valid effective_from date")
    ends = [ismissing(value) ? Date(9999, 12, 31) : value for value in parse_evidence_date.(evidence.effective_to, :effective_to)]
    membership_id = hasproperty(membership_row, :club_id) ? membership_row.club_id : missing
    membership_key = canonical_club_key(membership_row.club)
    # Compare stable IDs when BOTH sides have them. Otherwise use the documented
    # canonical-name fallback; a known ID mismatch must never match by name.
    matches = map(eachrow(evidence)) do row
        if !ismissing(membership_id) && !ismissing(row.club_id)
            return membership_id == row.club_id
        end
        return canonical_club_key(row.club_name) == membership_key
    end
    return evidence[matches .& (starts .<= date) .& (date .<= ends), :]
end

function status_for_membership(evidence::DataFrame, membership_row::DataFrameRow)
    candidates = active_evidence_rows(evidence, membership_row, season_start(membership_row.season))
    nrow(candidates) == 0 && return ("Unknown", "Unknown", missing, "No fetched effective-dated evidence")
    nrow(candidates) == 1 || error("conflicting active evidence for $(membership_row.club) in $(membership_row.season)")
    candidate = candidates[1, :]
    return (candidate.operational_status, candidate.evidence_level, candidate.source_id, candidate.classification_note)
end

function build_status_panel()
    isfile(MEMBERSHIP_PATH) || error("membership extract missing: $(MEMBERSHIP_PATH)")
    isfile(EVIDENCE_PATH) || error("evidence interval file missing: $(EVIDENCE_PATH)")
    membership = CSV.read(MEMBERSHIP_PATH, DataFrame; missingstring = ["", "NA"])
    evidence = CSV.read(EVIDENCE_PATH, DataFrame; missingstring = ["", "NA"])
    require_columns(membership, REQUIRED_MEMBERSHIP_COLUMNS, MEMBERSHIP_PATH)
    require_columns(evidence, REQUIRED_EVIDENCE_COLUMNS, EVIDENCE_PATH)
    membership = filter(row -> row.season in ["21/22", "22/23", "23/24", "24/25", "25/26", "26/27"], membership)

    status_rows = map(eachrow(membership)) do membership_row
        status, evidence_level, source_id, note = status_for_membership(evidence, membership_row)
        (; season = membership_row.season, tournament_id = membership_row.tournament_id,
           club_id = missing, club_name = membership_row.club, operational_status = status,
           evidence_level, source_id, classification_note = note,
           first_match_date = membership_row.first_match_date, last_match_date = membership_row.last_match_date)
    end
    output = DataFrame(status_rows)
    sort!(output, [:season, :tournament_id, :club_name])
    nrow(output) == 252 || error("expected 252 observed club-season rows for 21/22–26/27; found $(nrow(output))")
    CSV.write(OUTPUT_PATH, output)
    println("Wrote $(nrow(output)) club-season rows to $(OUTPUT_PATH)")
    println("Unknown rows: $(count(==("Unknown"), output.operational_status))")
    println("Verified rows: $(count(==("Verified"), output.evidence_level)); inferred rows: $(count(==("Inferred"), output.evidence_level))")
    return output
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_status_panel()
end
