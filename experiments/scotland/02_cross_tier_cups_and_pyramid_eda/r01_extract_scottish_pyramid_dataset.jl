# r01 — Extract and link every Scottish league + cup fixture, assign point-in-time tiers.
#
#   include("experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r01_extract_scottish_pyramid_dataset.jl")
#
# Read-only against betdb.  Writes:
#   data/r01_pyramid_fixtures.csv      one row per finished fixture 2008-07-01 .. AS_OF
#   data/r01_club_season_tiers.csv     club × football-season → league tier (from league events)
#   data/r01_bbc_shot_descriptors.csv  parsed BBC shot descriptors (pxG audit trail)
#   results/r01_sample_breakdown.csv   provenance table (tournament × window × enrichment)
#   results/r01_category_audit.csv     every club that received a B / GUEST / T5 label

# ── 1. Setup ────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "_common.jl"))
conn = db_connect()

# ── 2. Fixture universe: sofascore.events (matches is a narrower enrichment table) ─
# Normal-time scores come from the raw score object, so AET / penalty ties are scored
# at 90 minutes — the quantity every goals model in this repo predicts.
ev_sql = """
SELECT e.match_id, e.tournament_id, s.year AS season_label,
       to_timestamp(e.start_timestamp) AT TIME ZONE 'UTC' AS kickoff_utc,
       e.status_type, e.raw_data #>> '{status,description}' AS status_desc,
       COALESCE(e.raw_data #>> '{roundInfo,name}', '') AS round_name,
       (e.raw_data #>> '{homeTeam,id}')::int AS home_id, e.home_team,
       e.raw_data #>> '{homeTeam,name}' AS home_name,
       e.raw_data #>> '{homeTeam,country,name}' AS home_country,
       e.raw_data #>> '{homeTeam,parentTeam,slug}' AS home_parent,
       (e.raw_data #>> '{awayTeam,id}')::int AS away_id, e.away_team,
       e.raw_data #>> '{awayTeam,name}' AS away_name,
       e.raw_data #>> '{awayTeam,country,name}' AS away_country,
       e.raw_data #>> '{awayTeam,parentTeam,slug}' AS away_parent,
       NULLIF(e.raw_data #>> '{homeScore,normaltime}', '')::float8 AS home_goals,
       NULLIF(e.raw_data #>> '{awayScore,normaltime}', '')::float8 AS away_goals,
       (e.raw_data ? 'isAwarded') AS awarded
FROM sofascore.events e JOIN sofascore.seasons s ON s.season_id = e.season_id
WHERE e.tournament_id = ANY(\$1)
  AND to_timestamp(e.start_timestamp)::date BETWEEN \$2::date AND \$3::date
"""
ev = query(conn, ev_sql, ["{" * join(TOURNAMENTS, ",") * "}", string(Date(2006, 7, 1)), string(AS_OF + Day(400))])
ev.match_date = Date.(ev.kickoff_utc)
ev.fs = fseason.(ev.match_date)
println("events pulled: ", nrow(ev))

# ── 3. Point-in-time league membership ──────────────────────────────────────
# A club's tier in football season fs is the league it is scheduled in that season.
# All statuses count (fixtures are published before July cup ties), so a relegated
# club already carries its new tier in its first League Cup / Challenge Cup match.
lg = ev[in.(ev.tournament_id, Ref(keys(LEAGUE_TIER))), :]
sides = vcat(DataFrame(fs = lg.fs, team_id = lg.home_id, team = lg.home_team, tid = lg.tournament_id),
             DataFrame(fs = lg.fs, team_id = lg.away_id, team = lg.away_team, tid = lg.tournament_id))
mem = combine(groupby(sides, [:fs, :team_id]),
              :tid => (t -> LEAGUE_TIER[mode(t)]) => :tier, :team => first => :team,
              :tid => (t -> length(unique(t))) => :n_leagues, nrow => :league_fixtures)
@assert all(mem.n_leagues .== 1) "club in two leagues in one football season"
tier_of = Dict((r.fs, r.team_id) => r.tier for r in eachrow(mem))
ever_spfl = Set(mem.team_id)
CSV.write(joinpath(DATA, "r01_club_season_tiers.csv"), sort(mem[:, [:fs, :team_id, :team, :tier, :league_fixtures]], [:fs, :tier, :team]))

# ── 4. Finished, scoreable fixtures only ────────────────────────────────────
fx = ev[(ev.status_type .== "finished") .& in.(ev.status_desc, Ref(("Ended", "AET", "AP"))) .&
        .!ev.awarded .& .!ismissing.(ev.home_goals) .& .!ismissing.(ev.away_goals) .&
        (ev.match_date .>= LONG_START) .& (ev.match_date .<= AS_OF), :]
fx.home_goals = Int.(fx.home_goals); fx.away_goals = Int.(fx.away_goals)
fx.competition = [COMP_LABEL[t] for t in fx.tournament_id]
fx.is_cup = in.(fx.tournament_id, Ref(CUP_IDS))
fx.in_primary_window = fx.match_date .>= PRIMARY_START
fx.extra_time = in.(fx.status_desc, Ref(("AET", "AP")))

# ── 5. Category assignment (the Challenge Cup nuance lives here) ────────────
# B / U21 sides: sofascore's parentTeam link where present, else the slug suffix.
# The suffix is anchored so real clubs (cumbernauld-colts-fc, wick-academy-fc,
# edusport-academy-fc, hamilton-academical) are NOT caught.
const RESERVE_RX = r"(-b|-u2[0-3]|-b-u2[0-3])$"
is_reserve(slug, parent) = !ismissing(parent) || occursin(RESERVE_RX, slug)
# Guests: non-Scottish association.  Berwick Rangers (English town, Scottish pyramid,
# ex-SPFL, Lowland League) is a senior Scottish club, not a guest.
is_guest(slug, country) = !ismissing(country) && country != "Scotland" && slug != "berwick-rangers"

function category(fs, id, slug, country, parent)
    is_reserve(slug, parent) && return "B"
    is_guest(slug, country)  && return "GUEST"
    t = get(tier_of, (fs, id), 0)
    return t == 0 ? "T5" : "T$t"
end
fx.home_cat = category.(fx.fs, fx.home_id, fx.home_team, fx.home_country, fx.home_parent)
fx.away_cat = category.(fx.fs, fx.away_id, fx.away_team, fx.away_country, fx.away_parent)
tiernum(c) = c in SENIOR ? parse(Int, c[2:end]) : missing
fx.home_tier = tiernum.(fx.home_cat); fx.away_tier = tiernum.(fx.away_cat)
fx.tier_delta = fx.away_tier .- fx.home_tier        # + ⇒ home club is the higher-tier side
@assert all(skipmissing(fx.tier_delta[.!fx.is_cup]) .== 0) "league fixture across tiers"

# T5 heterogeneity marker: ex-SPFL clubs and Challenge-Cup invitees (Highland/Lowland
# champions and top sides since 2016/17) are genuine tier-5 clubs; everyone else in the
# Scottish Cup early rounds may be tier 6+ (East/West of Scotland, SJFA, amateur).
cc_invitees = Set(vcat(fx.home_id[(fx.tournament_id .== 1520) .& (fx.home_cat .== "T5")],
                       fx.away_id[(fx.tournament_id .== 1520) .& (fx.away_cat .== "T5")]))
t5sub(c, id) = c != "T5" ? missing : (id in ever_spfl || id in cc_invitees) ? "T5a" : "T6+"
fx.home_t5sub = t5sub.(fx.home_cat, fx.home_id); fx.away_t5sub = t5sub.(fx.away_cat, fx.away_id)

# ── 6. Neutral venues ───────────────────────────────────────────────────────
# Scottish Cup and League Cup semis + finals are at Hampden; Challenge Cup final at a
# neutral ground.  BBC venue confirms/extends this below.
bbc_meta = query(conn, "SELECT match_id, venue FROM bbc.match_meta WHERE match_id = ANY(\$1)",
                 ["{" * join(fx.match_id, ",") * "}"])
venue = Dict(r.match_id => r.venue for r in eachrow(bbc_meta))
fx.venue = [get(venue, m, missing) for m in fx.match_id]
round_neutral = ((fx.tournament_id .∈ Ref((73, 982))) .& in.(fx.round_name, Ref(("Semifinals", "Final")))) .|
                ((fx.tournament_id .== 1520) .& (fx.round_name .== "Final"))
hampden_neutral = [!ismissing(v) && occursin("Hampden", v) && h != "queens-park" && c
                   for (v, h, c) in zip(fx.venue, fx.home_team, fx.is_cup)]
fx.neutral = round_neutral .| hampden_neutral

# ── 7. Enrichment: BBC shots / shots on target (filled = imputed ⇒ missing) ─
st = query(conn, """
    SELECT match_id, stat_cat, stat_type, home_value, away_value, filled
    FROM bbc.match_stats WHERE match_id = ANY(\$1)
      AND stat_type IN ('shotsTotal','shotsOnTarget') AND stat_cat IN ('basic','attack')""",
    ["{" * join(fx.match_id, ",") * "}"])
st = st[.!coalesce.(st.filled, false), :]
# Prefer the 'basic' block; fall back to 'attack' (same quantity, different BBC layout era).
sort!(st, [:match_id, :stat_type, order(:stat_cat)])       # 'attack' < 'basic'
stb = combine(groupby(st, [:match_id, :stat_type]), [:home_value, :away_value] => ((h, a) -> (h = last(h), a = last(a))) => AsTable)
getstat(tp) = Dict(r.match_id => (r.h, r.a) for r in eachrow(stb[stb.stat_type .== tp, :]))
shots, sot = getstat("shotsTotal"), getstat("shotsOnTarget")
fx.home_shots = [haskey(shots, m) ? shots[m][1] : missing for m in fx.match_id]
fx.away_shots = [haskey(shots, m) ? shots[m][2] : missing for m in fx.match_id]
fx.home_sot   = [haskey(sot, m) ? sot[m][1] : missing for m in fx.match_id]
fx.away_sot   = [haskey(sot, m) ? sot[m][2] : missing for m in fx.match_id]

# ── 8. Enrichment: BBC commentary proxy xG ──────────────────────────────────
# Same measurement as TODO 027 r06 / Features.parse_shot: parse zone × body × context,
# empirical-Bayes cell conversion (pseudo-count 25 toward the open-play base rate),
# penalties at their empirical rate, sum per side.  The table is fitted on every
# parsed shot in this panel (league + cup), so cup and league pxG share one scale.
const ZONE_PATTERNS = [
    ("the left side of the six yard box", "six_yard_side"), ("the right side of the six yard box", "six_yard_side"),
    ("a difficult angle and long range", "difficult_long"), ("the centre of the box", "box_centre"),
    ("the left side of the box", "box_side"), ("the right side of the box", "box_side"),
    ("a difficult angle on the left", "difficult_angle"), ("a difficult angle on the right", "difficult_angle"),
    ("very close range", "six_yard_centre"), ("more than 35 yards", "very_long_range"),
    ("more than 40 yards", "very_long_range"), ("long range on the left", "long_range"),
    ("long range on the right", "long_range"), ("outside the box", "outside_box"),
    ("a free kick", "free_kick_zone")]
const BODY_PATTERNS = [("header", "header"), ("right footed", "right_foot"), ("left footed", "left_foot")]
const CONTEXT_PATTERNS = [("from a direct free kick", "direct_free_kick"), ("following a set piece situation", "set_piece"),
                          ("following a corner", "corner"), ("following a fast break", "fast_break")]
firstlabel(t, pats, def) = (i = findfirst(p -> occursin(p[1], t), pats); i === nothing ? def : pats[i][2])
function parse_shot(etype, text)
    ismissing(text) && return (zone = "unknown", body = "unknown", ctx = "open_play", pen = startswith(etype, "penalty"), parsed = false)
    t = lowercase(text)
    pen = startswith(etype, "penalty") || occursin("penalty", t)
    z = firstlabel(t, ZONE_PATTERNS, "unknown"); b = firstlabel(t, BODY_PATTERNS, "unknown")
    c = firstlabel(t, CONTEXT_PATTERNS, "open_play")
    z == "free_kick_zone" && ((z, c) = ("outside_box", "direct_free_kick"))
    return (zone = z, body = b, ctx = c, pen = pen, parsed = z != "unknown" || pen)
end
shots_ev = query(conn, """
    SELECT lt.match_id, lt.post_index, lt.event_type, lt.text,
           CASE WHEN regexp_replace(lt.team, '-fc\$', '') = regexp_replace(mm.bbc_home_slug, '-fc\$', '') THEN 1
                WHEN regexp_replace(lt.team, '-fc\$', '') = regexp_replace(mm.bbc_away_slug, '-fc\$', '') THEN 0
                ELSE -1 END AS side_home
    FROM bbc.live_text lt JOIN bbc.match_meta mm ON mm.match_id = lt.match_id
    WHERE lt.match_id = ANY(\$1)
      AND lt.event_type IN ('goal','attempt_missed','attempt_saved','attempt_blocked','post','penalty_missed','penalty_saved')""",
    ["{" * join(fx.match_id, ",") * "}"])
ps = [parse_shot(coalesce(e, ""), t) for (e, t) in zip(shots_ev.event_type, shots_ev.text)]
for k in (:zone, :body, :ctx, :pen, :parsed); shots_ev[!, k] = getproperty.(ps, k); end
shots_ev.is_goal = shots_ev.event_type .== "goal"
op = shots_ev[.!shots_ev.pen .& shots_ev.parsed, :]
base_rate = mean(op.is_goal); pen_xg = mean(shots_ev.is_goal[shots_ev.pen])
cells = combine(groupby(op, [:zone, :body, :ctx]), :is_goal => sum => :g, nrow => :n)
cell_xg = Dict((r.zone, r.body, r.ctx) => (r.g + 25base_rate) / (r.n + 25) for r in eachrow(cells))
shots_ev.xg = [s.pen ? pen_xg : s.parsed ? get(cell_xg, (s.zone, s.body, s.ctx), base_rate) : base_rate
               for s in eachrow(shots_ev)]
sided = shots_ev[shots_ev.side_home .>= 0, :]
pxg = combine(groupby(sided, :match_id),
              [:xg, :side_home] => ((x, s) -> (h = sum(x[s .== 1]), a = sum(x[s .== 0]),
                                                nh = count(==(1), s), na = count(==(0), s))) => AsTable)
pxgd = Dict(r.match_id => r for r in eachrow(pxg))
# A side with zero resolved shot events is treated as no-coverage only when BOTH sides are
# empty; one side with 0 shots in a covered match is a genuine 0.
fx.home_pxg = [haskey(pxgd, m) ? pxgd[m].h : missing for m in fx.match_id]
fx.away_pxg = [haskey(pxgd, m) ? pxgd[m].a : missing for m in fx.match_id]
CSV.write(joinpath(DATA, "r01_bbc_shot_descriptors.csv"),
          shots_ev[:, [:match_id, :post_index, :event_type, :zone, :body, :ctx, :pen, :parsed, :side_home, :xg]])
@printf("pxG table: %d parsed open-play shots, base %.4f, penalty %.3f, %d cells\n", nrow(op), base_rate, pen_xg, nrow(cells))

# ── 9. Enrichment: closing 1X2 odds ─────────────────────────────────────────
# (a) sofascore.match_odds 'Full time' fractional — the only source covering all three
#     cups (Challenge Cup has no Betfair archive).  `fraction` is the last pre-match
#     price SofaScore held (is_live = false); `initial_fraction` the opening price.
so = query(conn, """
    SELECT match_id, choice_name, fraction_num, fraction_den, initial_fraction_num, initial_fraction_den
    FROM sofascore.match_odds WHERE market_name = 'Full time' AND NOT is_live AND match_id = ANY(\$1)""",
    ["{" * join(fx.match_id, ",") * "}"])
so.dec  = 1 .+ so.fraction_num ./ so.fraction_den
so.dec0 = 1 .+ so.initial_fraction_num ./ so.initial_fraction_den
odds = Dict{Int, Dict{String, Tuple{Float64, Float64}}}()
for r in eachrow(so)
    get!(odds, r.match_id, Dict{String, Tuple{Float64, Float64}}())[r.choice_name] = (r.dec, r.dec0)
end
oget(m, c, i) = (haskey(odds, m) && haskey(odds[m], c) && length(odds[m]) == 3) ? odds[m][c][i] : missing
fx.odds_home = oget.(fx.match_id, "1", 1); fx.odds_draw = oget.(fx.match_id, "X", 1); fx.odds_away = oget.(fx.match_id, "2", 1)
fx.odds_home_open = oget.(fx.match_id, "1", 2); fx.odds_draw_open = oget.(fx.match_id, "X", 2); fx.odds_away_open = oget.(fx.match_id, "2", 2)

# (b) Betfair exchange MATCH_ODDS, last coherent snapshot at or before kick-off
#     (league + Scottish Cup + League Cup only).  Extracted in SQL so the JSON arrays
#     never leave the server.
bf = query(conn, """
    WITH k AS (
      SELECT oh.match_id, oh.odds_data, extract(epoch FROM mm.kickoff_time) * 1000 AS ko_ms
      FROM betfair.odds_history oh
      JOIN betfair.markets mk ON mk.market_id = oh.market_id AND mk.market_type = 'MATCH_ODDS'
      JOIN betfair.match_meta mm ON mm.match_id = oh.match_id AND mm.status = 'SUCCESS' AND mm.is_verified
      WHERE oh.match_id = ANY(\$1))
    SELECT k.match_id, x.ts, x.h, x.d, x.a, (x.ts - k.ko_ms) / 60000.0 AS min_to_ko
    FROM k CROSS JOIN LATERAL (
      SELECT t.ts::float8 AS ts, (k.odds_data->'home'->>(t.i::int - 1))::float8 AS h,
             (k.odds_data->'draw'->>(t.i::int - 1))::float8 AS d,
             (k.odds_data->'away'->>(t.i::int - 1))::float8 AS a
      FROM jsonb_array_elements_text(k.odds_data->'timestamps') WITH ORDINALITY AS t(ts, i)
      WHERE t.ts::float8 <= k.ko_ms
        AND (k.odds_data->'home'->>(t.i::int - 1))::float8 > 1
        AND (k.odds_data->'draw'->>(t.i::int - 1))::float8 > 1
        AND (k.odds_data->'away'->>(t.i::int - 1))::float8 > 1
      ORDER BY t.i DESC LIMIT 1) x""",
    ["{" * join(fx.match_id[fx.match_date .>= Date(2020, 7, 1)], ",") * "}"])
bfd = Dict(r.match_id => r for r in eachrow(bf))
function bf_fair(m)
    haskey(bfd, m) || return (missing, missing, missing, missing)
    r = bfd[m]; inv = (1 / r.h, 1 / r.d, 1 / r.a); s = sum(inv)
    return (inv[1] / s, inv[2] / s, inv[3] / s, r.min_to_ko)
end
bfv = bf_fair.(fx.match_id)
fx.bf_p_home = getindex.(bfv, 1); fx.bf_p_draw = getindex.(bfv, 2); fx.bf_p_away = getindex.(bfv, 3)
fx.bf_min_to_ko = getindex.(bfv, 4)
println("betfair 1X2 snapshots: ", nrow(bf), " | sofascore 1X2 triplets: ", count(!ismissing, fx.odds_home))

# ── 10. Persist ─────────────────────────────────────────────────────────────
keep = [:match_id, :tournament_id, :competition, :is_cup, :season_label, :fs, :match_date, :round_name,
        :status_desc, :extra_time, :neutral, :venue, :in_primary_window,
        :home_id, :home_team, :home_name, :home_cat, :home_tier, :home_t5sub,
        :away_id, :away_team, :away_name, :away_cat, :away_tier, :away_t5sub, :tier_delta,
        :home_goals, :away_goals, :home_shots, :away_shots, :home_sot, :away_sot, :home_pxg, :away_pxg,
        :odds_home, :odds_draw, :odds_away, :odds_home_open, :odds_draw_open, :odds_away_open,
        :bf_p_home, :bf_p_draw, :bf_p_away, :bf_min_to_ko]
sort!(fx, [:match_date, :match_id])
CSV.write(joinpath(DATA, "r01_pyramid_fixtures.csv"), fx[:, keep])

# ── 11. Provenance tables ───────────────────────────────────────────────────
function breakdown(d, label)
    combine(groupby(d, [:tournament_id, :competition]), nrow => :n_matches,
            :fs => (f -> fs_label(minimum(f))) => :first_season, :fs => (f -> fs_label(maximum(f))) => :last_season,
            [:home_cat, :away_cat] => ((h, a) -> count(in(SENIOR).(h) .& in(SENIOR).(a) .& (h .!= a))) => :senior_cross_tier,
            [:home_cat, :away_cat] => ((h, a) -> count((h .== "B") .| (a .== "B"))) => :with_B_team,
            [:home_cat, :away_cat] => ((h, a) -> count((h .== "GUEST") .| (a .== "GUEST"))) => :with_guest,
            [:home_cat, :away_cat] => ((h, a) -> count((h .== "T5") .| (a .== "T5"))) => :with_T5,
            :neutral => sum => :neutral, :extra_time => sum => :aet_or_pens,
            :home_shots => nnz_ => :with_bbc_shots, :home_pxg => nnz_ => :with_pxg,
            :odds_home => nnz_ => :with_sofa_1x2, :bf_p_home => nnz_ => :with_betfair_1x2) |>
        x -> (insertcols!(x, 1, :window => label); sort!(x, :tournament_id))
end
prov = vcat(breakdown(fx[fx.in_primary_window, :], "primary 21/22–26/27"), breakdown(fx, "long 08/09–26/27"))
save_csv("r01_sample_breakdown.csv", prov); save_md("r01_sample_breakdown.md", prov)

aud = vcat(DataFrame(team = fx.home_team, name = fx.home_name, cat = fx.home_cat, sub = fx.home_t5sub, tid = fx.tournament_id, fs = fx.fs),
           DataFrame(team = fx.away_team, name = fx.away_name, cat = fx.away_cat, sub = fx.away_t5sub, tid = fx.tournament_id, fs = fx.fs))
aud = aud[in.(aud.cat, Ref(("B", "GUEST", "T5"))), :]
aud = combine(groupby(aud, [:cat, :team, :name]), nrow => :appearances,
              :sub => (s -> join(unique(skipmissing(s)), "/")) => :t5_sub,
              :tid => (t -> join(sort(unique(t)), ",")) => :tournaments,
              :fs => (f -> fs_label(minimum(f)) * "–" * fs_label(maximum(f))) => :seasons)
sort!(aud, [:cat, order(:appearances, rev = true)])
save_csv("r01_category_audit.csv", aud)

pw = fx[fx.in_primary_window, :]
@printf("primary window: %d fixtures (%d league, %d cup); senior cross-tier %d; B-team %d; guest %d\n",
        nrow(pw), count(.!pw.is_cup), count(pw.is_cup),
        count(coalesce.(pw.tier_delta .!= 0, false)), count((pw.home_cat .== "B") .| (pw.away_cat .== "B")),
        count((pw.home_cat .== "GUEST") .| (pw.away_cat .== "GUEST")))
@printf("long window: %d fixtures; senior cross-tier %d\n", nrow(fx), count(coalesce.(fx.tier_delta .!= 0, false)))
close(conn)
