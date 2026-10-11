# Frozen S1 transport inputs; opt-in feature, no SQL or fitting.
module QSTransitionInputs
using BayesianFootball, CSV, DataFrames, Dates, SHA, Statistics
const F = BayesianFootball.Features
const ROOT = @__DIR__
const EDA = joinpath(ROOT,"eda","results")
parseids(x) = ismissing(x) || isempty(string(x)) ? Int[] : parse.(Int,split(string(x),';'))
const FIXTURES = CSV.read(joinpath(EDA,"fixtures.csv"),DataFrame)
const QUOTES = CSV.read(joinpath(EDA,"accepted_quotes.csv"),DataFrame)
const RECON = CSV.read(joinpath(EDA,"market_team_reconstruction.csv"),DataFrame)
const PARAMS = CSV.read(joinpath(EDA,"q2_prior_parameters.csv"),DataFrame)
const SOURCES = CSV.read(joinpath(EDA,"q12_source_manifest.csv"),DataFrame)
const GRAPH = CSV.read(joinpath(EDA,"q3_reconstruction_graph.csv"),DataFrame)
const QUOTE_LOOKUP = Dict(Int(r.match_id)=>r for r in eachrow(QUOTES))
const MARKET_IDS = Set(Int.(filter(:full_book=>identity,CSV.read(joinpath(ROOT,"build","market_rates.csv"),DataFrame)).match_id))
const INPUT_FILES = ("fixtures.csv","accepted_quotes.csv","market_team_reconstruction.csv",
    "q2_prior_parameters.csv","q12_source_manifest.csv","q3_reconstruction_graph.csv")
input_hash() = bytes2hex(sha256(join([bytes2hex(open(SHA.sha256,joinpath(EDA,f))) for f in INPUT_FILES],":")))
struct EntrantPriorFeature <: F.AbstractFeatureConfig
    input_sha256::String
end
EntrantPriorFeature() = EntrantPriorFeature(input_hash())

function parameter(season,coordinate,group)
    rows = filter(r -> r.target_season==season && r.method=="P2_A2shrink" &&
        r.coordinate==coordinate && r.group==group,PARAMS)
    r = only(eachrow(rows))
    ismissing(r.training_seasons) || all(s -> s<season,split(r.training_seasons,',')) || error("future prior parameter season")
    return r
end
function prior(club,season,tier,inc_tier,style_reference)
    cutoff = Date(2000+parse(Int,season[1:2]),7,1)
    upper = filter(r -> r.club==club && r.tier in (54,55) && r.season<season && r.window=="late80",RECON)
    endpoint = isempty(upper) ? nothing : last(eachrow(sort(upper,:season)))
    state_ids = Int[]
    fallback = endpoint===nothing
    if endpoint!==nothing
        q = sort(filter(r -> r.tournament==endpoint.tier && r.season==endpoint.season,QUOTES),[:date,:match_id])
        q = last(q,min(80,nrow(q)))
        state_ids = Int.(q.match_id)
        graph = filter(r -> r.tier==endpoint.tier && r.season==endpoint.season && r.window=="late80",GRAPH)
        fallback = endpoint.n_own_quotes<3 || isempty(graph) || !only(graph.connected_identified) ||
            cutoff-Date(string(endpoint.last_date))>Day(548)
    end
    # The division transport already includes its rebase. Add the destination's
    # A2 reference relative to incumbent tier anchors ONCE, not an old-new step.
    tier_position = -0.235*((tier-53)-inc_tier)
    direction = endpoint===nothing ? "entrant" : "relegated"
    pq,ps = parameter(season,"D",direction),parameter(season,"U",direction)
    qmean = fallback ? tier_position : endpoint.q+pq.offset+tier_position
    smean = fallback ? 0.0 : endpoint.s+ps.offset-style_reference
    qsd = fallback ? 0.35 : sqrt(endpoint.q_se^2+pq.sigma_predictive^2+pq.offset_se^2)
    ssd = fallback ? 0.20 : sqrt(endpoint.s_se^2+ps.sigma_predictive^2+ps.offset_se^2)
    # Rejected/sparse/stale endpoint evidence is not attached at all.
    fallback && empty!(state_ids)
    all(id -> QUOTE_LOOKUP[id].tournament in (54,55) &&
        QUOTE_LOOKUP[id].season<season && Date(string(QUOTE_LOOKUP[id].date))<cutoff,state_ids) || error("entrant endpoint filtration")
    manifest = filter(r -> r.target_season==season,SOURCES)
    hyper_ids = sort!(unique(vcat([vcat(parseids(r.hyper_label_ids),parseids(r.normalization_source_ids)) for r in eachrow(manifest)]...)))
    all(id -> QUOTE_LOOKUP[id].season<season && Date(string(QUOTE_LOOKUP[id].date))<cutoff,hyper_ids) || error("hyper chronology")
    return (; club,season,tier,q_mean=qmean,q_sd=qsd,s_mean=smean,s_sd=ssd,fallback,state_ids,hyper_ids,
        offset_q=pq.offset,offset_s=ps.offset,sigma_q=pq.sigma_predictive,sigma_s=ps.sigma_predictive,
        offset_se_q=pq.offset_se,offset_se_s=ps.offset_se,style_reference,inc_tier)
end

function F.add_feature!(data::Dict,config::EntrantPriorFeature,ordered_ids,team_map::Dict,ds::Data.DataStore)
    config.input_sha256==input_hash() || error("frozen entrant inputs changed")
    season = String(data[:target_season])
    cutoff = Date(2000+parse(Int,season[1:2]),7,1)
    lower = filter(r -> r.tournament_id in (56,57),FIXTURES)
    past = filter(r -> r.season<season,lower)
    current = filter(r -> r.season==season,lower)
    oldclubs = Set(vcat(String.(past.home_team),String.(past.away_team)))
    currentclubs = sort!(unique(vcat(String.(current.home_team),String.(current.away_team))))
    entrants = setdiff(currentclubs,collect(oldclubs)) # D10: persists in every target-season fold
    for club in entrants
        haskey(team_map,club) || (team_map[club]=length(team_map)+1)
    end
    data[:n_teams] = length(team_map)
    incumbents = setdiff(collect(keys(team_map)),entrants)
    function tier_of(club,frame)
        rows = filter(r -> r.home_team==club || r.away_team==club,frame)
        isempty(rows) && error("missing structural tier for $club")
        return Int(last(eachrow(sort(rows,[:season,:match_date]))).tournament_id)
    end
    inc_tier = mean([tier_of(club,past)-53 for club in incumbents])
    # Earlier-season style reference is a declared gauge input, not endpoint evidence.
    oldstyle = filter(r -> r.club in incumbents && r.tier in (56,57) &&
        r.season<season && r.window=="whole",RECON)
    lastseason = isempty(oldstyle) ? nothing : maximum(oldstyle.season)
    style_reference = lastseason===nothing ? 0.0 : mean(filter(r -> r.season==lastseason,oldstyle).s)
    records = [prior(club,season,tier_of(club,current),inc_tier,style_reference) for club in entrants]
    likelihood_ids = intersect(Set(Int.(ordered_ids)),MARKET_IDS)
    manifest_rows = NamedTuple[]
    for p in records
        rows = filter(r -> r.home_team==p.club || r.away_team==p.club,current)
        node_date = minimum(Date.(string.(rows.match_date)))
        # Frozen Scottish panel entrants all start in the first biweekly period.
        anchor = minimum(Date.(string.(current.match_date)))
        node_date-anchor<Day(14) || error("entry after target period one needs an explicit per-club time map")
        endpoint_overlap = length(intersect(Set(p.state_ids),likelihood_ids))
        endpoint_overlap==0 || error("entrant endpoint overlaps market likelihood")
        maxdate = isempty(p.state_ids) ? "" : string(maximum(Date(string(QUOTE_LOOKUP[id].date)) for id in p.state_ids))
        sources = join(sort!(unique(String(QUOTE_LOOKUP[id].season) for id in p.state_ids)),";")
        push!(manifest_rows,(; fold=data[:grw_oos_seed],club=p.club,target_season=season,
            entry_node_date=string(node_date),input_sha256=config.input_sha256,
            source_ids=join(p.state_ids,";"),source_seasons=sources,
            max_source_date=maxdate,cutoff=string(cutoff),endpoint_overlap,
            hyper_source_ids=join(p.hyper_ids,";"),hyper_source_overlap=length(intersect(Set(p.hyper_ids),likelihood_ids)),
            q_mean=p.q_mean,q_sd=p.q_sd,s_mean=p.s_mean,s_sd=p.s_sd,offset_q=p.offset_q,
            offset_s=p.offset_s,sigma_q=p.sigma_q,sigma_s=p.sigma_s,
            offset_se_q=p.offset_se_q,offset_se_s=p.offset_se_s,
            fallback=p.fallback,incumbent_tier_reference=inc_tier,style_reference,
            evidence_policy="D9 endpoint disjoint; earlier-season hyperparameters may overlap",
            gauge="destination A2 tier reference once; incumbent-centred QS; earlier lower style reference"))
    end
    data[:qs_entry_prior] = (; indices=[team_map[p.club] for p in records],
        q_mean=Float64[p.q_mean for p in records],q_sd=Float64[p.q_sd for p in records],
        s_mean=Float64[p.s_mean for p in records],s_sd=Float64[p.s_sd for p in records])
    data[:qs_entry_manifest] = manifest_rows
    out = joinpath(ROOT,"build","prior_manifests")
    mkpath(out)
    isempty(manifest_rows) || CSV.write(joinpath(out,"$(replace(season,'/'=>'-'))_fold$(data[:grw_oos_seed]).csv"),DataFrame(manifest_rows))
    CSV.write(joinpath(out,"summary_$(data[:grw_oos_seed]).csv"),DataFrame([(;fold=data[:grw_oos_seed],season,
        entrants=length(records),endpoint_overlap=sum(r.endpoint_overlap for r in manifest_rows;init=0),
        hyper_overlap=sum(r.hyper_source_overlap for r in manifest_rows;init=0))]))
end
end
