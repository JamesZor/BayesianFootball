# r08_score_grid.jl — is our score grid mis-weighted? (BRIEF_08)
#
# QUESTION
#   Our models price every scoreline from a posterior mixture of independent Poissons. Summed over
#   all held-out matches and folds, do the grids put the right mass on each scoreline, on draws,
#   on total goals and on its dispersion? Does a one-parameter reshape (holding each draw's goal
#   means) fix the misfit out of sample, and does it change the bets?
#
# DESIGN
#   runs     W0 m12_td 132df5c2, m05_joint_td ed541a7c, floor s12_m01_td_poisson de7fa956;
#            W1 grw_spfl_joint 82e01a3b; W2 td_lower_a2full_carry_jump 6c0aa3ea,
#            grw_step_a2_carry_jump c4a0fa94. All are CountLatents{Float64, Nothing}: Poisson.
#   panels   L1/L2 (56/57) held-out targets on each run's own panel (W0/W1 24/25–25/26, W2
#            23/24–25/26); 54/55 monitor for the two pooled GRW runs. Panels are never pooled.
#   grid     the model's own grid: src `compute_score_grid!` over EVERY posterior draw, averaged.
#   §1       O vs E over 0..4 + 5+; (O−E)/√E; Pearson χ² with E < 5 cells pooled and a parametric
#            bootstrap p (2,000 replicate seasons drawn from the models' own grids).
#   §2       shapes refitted per draw on 400 evenly thinned draws, means held; exact-score MLE on
#            training seasons: W2 train 23/24 → test 24/25–25/26; W0/W1 (no 23/24) two-way season
#            split, 24/25 → 25/26 and 25/26 → 24/25, pooled.
#   §3       the best shape's 1X2 / O-U 2.5 at T−25 / T−60 (§3/§10 LTP method) and §13's 26/27
#            order-book view (back price), against the unreshaped grid from the same draws.
#
# Read-only; no sampling. USAGE (mcmc-beast):
#   julia --project -t 16 current_development/clv_napkin/r08_score_grid.jl

# %%
# ===================================================================
# 1. Packages and configuration
# ===================================================================

const R8_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "l08_score_grid.jl"))
cn_include_prototype_loaders!(R8_ROOT)
LinearAlgebra.BLAS.set_num_threads(1)

const R8_OUT = joinpath(@__DIR__, "out")
const R8_B = 2_000
const R8_SEED = 20260928
const R8_RUNS = [
    (label = "m12_td", wave = "W0", exp = "scottish_lower_joint_player_2426", id = "132df5c2-c742-4e95-8693-3aeb2b2cbaef",
     seasons = ["24/25", "25/26"], monitor = false),
    (label = "m05_joint_td", wave = "W0", exp = "scottish_lower_joint_player_2426", id = "ed541a7c-01e2-447e-a771-783517728d47",
     seasons = ["24/25", "25/26"], monitor = false),
    (label = "FLOOR s12_m01_td_poisson", wave = "W0", exp = "scottish_lower_decoupled_xg", id = "de7fa956-87e8-418f-afb4-61ce01cb9f7d",
     seasons = ["24/25", "25/26"], monitor = false),
    (label = "grw_spfl_joint", wave = "W1", exp = "scottish_dynamics_scope_matrix_w1", id = "82e01a3b-d686-4e29-a269-3702b96fc5ba",
     seasons = ["24/25", "25/26"], monitor = true),
    (label = "td_lower_a2full_carry_jump", wave = "W2", exp = "scottish_pyramid_tier_transition_w2", id = "6c0aa3ea-7e0e-466b-a102-aa18346be8c0",
     seasons = ["23/24", "24/25", "25/26"], monitor = false),
    (label = "grw_step_a2_carry_jump", wave = "W2", exp = "scottish_pyramid_tier_transition_w2", id = "c4a0fa94-42bf-4c1c-99bd-7511f0272062",
     seasons = ["23/24", "24/25", "25/26"], monitor = true)]
println("threads: ", Threads.nthreads(), "  started ", now())
r8_write(name, df) = CSV.write(joinpath(R8_OUT, "r08_" * name * ".csv"), df)
r8_show(df) = (show(stdout, MIME"text/plain"(), df; allrows = true, allcols = true, truncate = 0); println())

# %%
# ===================================================================
# 2. Held-out inputs per run and panel
# ===================================================================

ds = deserialize(joinpath(C4_OUT, "r04_datastore_ScottishPyramid.jls"))
meta = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
panels = Dict{Tuple{String,String},Vector{C8Match}}()      # (run label, "L1/L2" | "Prem/Champ monitor")
ms_2627 = C8Match[]
for r in R8_RUNS
    t0 = time()
    fit = cn_load_fit(r.exp, UUID(r.id))
    panels[(r.label, "L1/L2")] = c8_extract(fit, cn_panel(ds, fit, r.seasons), meta)
    r.monitor && (panels[(r.label, "Prem/Champ monitor")] = c8_extract(fit, cn_panel(ds, fit, r.seasons; tournaments = (54, 55)), meta))
    if r.label == "grw_spfl_joint"          # §13's in-memory 26/27 extension
        ext = deserialize(joinpath(C4_OUT, "r04_ext_grw_spfl_joint.jls"))
        pseudo = (; latents = ext.latents, config = (; model = fit.config.model))
        ids = [m for m in Int.(ext.latents.match_ids) if haskey(meta, m) && Int(meta[m].tournament_id) in (56, 57)]
        append!(ms_2627, c8_extract(pseudo, ids, meta))
    end
    @printf("%-28s %-8s L1/L2 %4d  monitor %4d  (%s)  %.0fs\n", r.label, r.wave, length(panels[(r.label, "L1/L2")]),
            length(get(panels, (r.label, "Prem/Champ monitor"), C8Match[])), fit.config.name, time() - t0)
    fit = nothing; GC.gc()
end
println("26/27 grw_spfl_joint L1/L2 matches with a result: ", length(ms_2627))
pkeys = [(r.label, p) for r in R8_RUNS for p in ("L1/L2", "Prem/Champ monitor") if haskey(panels, (r.label, p))]

# %%
# ===================================================================
# 3. §1 Observed vs expected
# ===================================================================

cell_rows = NamedTuple[]; sum_rows = NamedTuple[]; tot_rows = NamedTuple[]; gd_rows = NamedTuple[]
r8_txt = IOBuffer()
for (lab, pan) in pkeys
    ms = panels[(lab, pan)]
    N = length(ms)
    E, O = c8_oe(ms)
    Z = (O .- E) ./ sqrt.(E)
    pool = BitMatrix(E .< 5)
    chi = c8_chi2(O, E, pool)
    bs = c8_bootstrap(ms, pool, E; B = R8_B, seed = R8_SEED)
    for j in 1:C8_TAB, i in 1:C8_TAB
        push!(cell_rows, (; run = lab, panel = pan, home = i == C8_TAB ? "5+" : string(i - 1),
                          away = j == C8_TAB ? "5+" : string(j - 1), O = O[i, j], E = E[i, j], z = Z[i, j],
                          o_minus_e_pp = 100 * (O[i, j] - E[i, j]) / N))
    end
    mom = [c8_moments(m.grid) for m in ms]
    obs_t = [m.hs + m.as for m in ms]
    E_draw = sum(x.pd for x in mom); O_draw = count(m -> m.hs == m.as, ms)
    var_model = mean(x.var for x in mom) + var([x.mean for x in mom]; corrected = false)
    push!(sum_rows, (; run = lab, panel = pan, n = N, chi2 = chi, pooled_cells = count(pool),
                     cells = C8_TAB^2 - count(pool) + (any(pool) ? 1 : 0), p_boot = mean(bs.chi .>= chi),
                     draws_O = O_draw, draws_E = E_draw, draws_lo = quantile(bs.draws, 0.025),
                     draws_hi = quantile(bs.draws, 0.975), draw_rate_O = O_draw / N, draw_rate_E = E_draw / N,
                     p_draws_boot = mean(abs.(bs.draws .- E_draw) .>= abs(O_draw - E_draw)),
                     mean_total_O = mean(obs_t), mean_total_E = mean(x.mean for x in mom),
                     var_total_O = var(obs_t; corrected = false), var_total_model = var_model,
                     var_within_model = mean(x.var for x in mom),
                     var_between_model = var([x.mean for x in mom]; corrected = false),
                     var_lo = quantile(bs.vars, 0.025), var_hi = quantile(bs.vars, 0.975),
                     dispersion_ratio = var(obs_t; corrected = false) / var_model))
    Et = sum(x.tot for x in mom); Ot = [count(m -> min(m.hs + m.as, C8_TAB - 1) == k, ms) for k in 0:C8_TAB-1]
    for k in 1:C8_TAB
        push!(tot_rows, (; run = lab, panel = pan, total = k == C8_TAB ? "5+" : string(k - 1), O = Ot[k], E = Et[k],
                         lo = quantile(bs.tots[:, k], 0.025), hi = quantile(bs.tots[:, k], 0.975),
                         o_minus_e_pp = 100 * (Ot[k] - Et[k]) / N))
    end
    Eg = sum(x.gd for x in mom); Og = [count(m -> clamp(m.hs - m.as, -4, 4) == d, ms) for d in -4:4]
    for (k, d) in enumerate(-4:4)
        push!(gd_rows, (; run = lab, panel = pan, gd = d == -4 ? "≤−4" : d == 4 ? "≥+4" : string(d), O = Og[k], E = Eg[k],
                        lo = quantile(bs.gds[:, k], 0.025), hi = quantile(bs.gds[:, k], 0.975),
                        o_minus_e_pp = 100 * (Og[k] - Eg[k]) / N))
    end
    println(r8_txt, c8_ascii(Z; title = "$lab — $pan (n = $N): (O − E)/√E"))
    println(r8_txt, c8_ascii(O; fmt = "%6.0f", title = "O"))
    println(r8_txt, c8_ascii(E; fmt = "%6.1f", title = "E"))
end
summ = DataFrame(sum_rows)
r8_show(summ)
print(String(take!(copy(r8_txt))))
write(joinpath(R8_OUT, "r08_residual_tables.txt"), String(take!(r8_txt)))
cells = DataFrame(cell_rows); totals = DataFrame(tot_rows); gds = DataFrame(gd_rows)
r8_show(unstack(select(totals, :run, :panel, :total, :o_minus_e_pp), :total, :o_minus_e_pp))
r8_show(unstack(select(gds, :run, :panel, :gd, :o_minus_e_pp), :gd, :o_minus_e_pp))
r8_write("summary", summ); r8_write("cells", cells); r8_write("totals", totals); r8_write("goal_difference", gds)

# side by side with the market's CS pattern (§14: market CS − Poisson fitted to its 1X2 + O/U 2.5)
mres = CSV.read(joinpath(R8_OUT, "r05_residuals.csv"), DataFrame)
mres = mres[(mres.grid .== "poisson") .& in.(mres.group, Ref(["all", "L1", "L2"])), :]
cellmap = Dict("00" => ("0", "0"), "10" => ("1", "0"), "01" => ("0", "1"), "11" => ("1", "1"), "22" => ("2", "2"))
vs_rows = NamedTuple[]
for (lab, pan) in pkeys
    c = cells[(cells.run .== lab) .& (cells.panel .== pan), :]
    s = summ[(summ.run .== lab) .& (summ.panel .== pan), :][1, :]
    for (key, (h, a)) in cellmap
        push!(vs_rows, (; source = lab * " — " * pan * " (O − E)", cell = key,
                        pp = c[(c.home .== h) .& (c.away .== a), :o_minus_e_pp][1]))
    end
    push!(vs_rows, (; source = lab * " — " * pan * " (O − E)", cell = "draw", pp = 100 * (s.draws_O - s.draws_E) / s.n))
end
for r in eachrow(mres)
    r.cell in ("00", "10", "01", "11", "22", "draw_cs") || continue
    push!(vs_rows, (; source = "market CS − Poisson (§14, " * r.group * ")", cell = r.cell == "draw_cs" ? "draw" : r.cell, pp = r.mean_pp))
end
vs = select(unstack(DataFrame(vs_rows), :source, :cell, :pp), :source, "00", "10", "01", "11", "22", "draw")
r8_show(vs)
r8_write("vs_market", vs)

# %%
# ===================================================================
# 4. §2 Keep the rates, change only the shape
# ===================================================================

"Train/test folds for a run's panel: W2 23/24 → later; otherwise the two-way season split."
function r8_folds(ms)
    ss = sort(unique(m.season for m in ms))
    if "23/24" in ss
        return [("23/24 → 24/25–25/26", [m for m in ms if m.season == "23/24"], [m for m in ms if m.season != "23/24"])]
    end
    return [("24/25 → 25/26", [m for m in ms if m.season == "24/25"], [m for m in ms if m.season == "25/26"]),
            ("25/26 → 24/25", [m for m in ms if m.season == "25/26"], [m for m in ms if m.season == "24/25"])]
end

# the market's §14 shapes, applied FIXED (not fitted): what swapping in the market's grid does
const R8_FIXED = [(:mkt_dc, :dc, -0.04, "(m) market DC ρ = −0.04, fixed"),
                  (:mkt_bpcom, :bpcom, (0.08, 1.08), "(m) market bivariate + COM (0.08, 1.08), fixed")]
for (key, _, _, lab) in R8_FIXED
    C8_SHAPE_LABEL[key] = lab
end
fit_rows = NamedTuple[]; test_rows = NamedTuple[]
test_scores = Dict{Tuple{String,String,Symbol},DataFrame}()
shape_theta = Dict{Tuple{String,String,Symbol,String},Any}()
for (lab, pan) in pkeys
    ms = panels[(lab, pan)]
    parts = Dict(k => DataFrame[] for k in (:poisson, C8_SHAPES..., first.(R8_FIXED)...))
    for (fname, tr, te) in r8_folds(ms)
        push!(parts[:poisson], insertcols!(c8_scores(te, :poisson, 0.0), :fold => fname))
        for k in C8_SHAPES
            t0 = time()
            θ, nll = c8_fit_shape(tr, k)
            shape_theta[(lab, pan, k, fname)] = θ
            nll0 = c8_nll(tr, :poisson, 0.0)
            push!(fit_rows, (; run = lab, panel = pan, fold = fname, shape = C8_SHAPE_LABEL[k], n_train = length(tr),
                             theta = θ isa Tuple ? θ[1] : θ, nu = θ isa Tuple ? θ[2] : NaN,
                             train_nll = nll, train_gain_vs_poisson = nll0 - nll))
            push!(parts[k], insertcols!(c8_scores(te, k, θ), :fold => fname))
            @printf("  %-26s %-18s %-22s %-28s θ = %s  (%.0fs)\n", lab, pan, fname, C8_SHAPE_LABEL[k], θ, time() - t0)
        end
        for (key, kind, θ, _) in R8_FIXED
            push!(parts[key], insertcols!(c8_scores(te, kind, θ), :fold => fname))
        end
    end
    base = vcat(parts[:poisson]...)
    test_scores[(lab, pan, :poisson)] = base
    push!(test_rows, (; run = lab, panel = pan, shape = C8_SHAPE_LABEL[:poisson], n_test = nrow(base),
                      ll_exact = mean(base.ll_exact), d_exact = 0.0, d_exact_lo = 0.0, d_exact_hi = 0.0,
                      ll_1x2 = mean(base.ll_1x2), d_1x2 = 0.0, d_1x2_lo = 0.0, d_1x2_hi = 0.0,
                      ll_ou25 = mean(base.ll_ou25), d_ou25 = 0.0, d_ou25_lo = 0.0, d_ou25_hi = 0.0,
                      draw_rate_pred = mean(base.p_draw), draw_rate_obs = mean(base.draw)))
    for k in (C8_SHAPES..., first.(R8_FIXED)...)
        sc = vcat(parts[k]...)
        test_scores[(lab, pan, k)] = sc
        dE = c7_paired(sc.ll_exact, base.ll_exact, sc.date; B = R8_B, seed = R8_SEED)
        d1 = c7_paired(sc.ll_1x2, base.ll_1x2, sc.date; B = R8_B, seed = R8_SEED)
        dO = c7_paired(sc.ll_ou25, base.ll_ou25, sc.date; B = R8_B, seed = R8_SEED)
        push!(test_rows, (; run = lab, panel = pan, shape = C8_SHAPE_LABEL[k], n_test = nrow(sc),
                          ll_exact = mean(sc.ll_exact), d_exact = dE.mean, d_exact_lo = dE.lo, d_exact_hi = dE.hi,
                          ll_1x2 = mean(sc.ll_1x2), d_1x2 = d1.mean, d_1x2_lo = d1.lo, d_1x2_hi = d1.hi,
                          ll_ou25 = mean(sc.ll_ou25), d_ou25 = dO.mean, d_ou25_lo = dO.lo, d_ou25_hi = dO.hi,
                          draw_rate_pred = mean(sc.p_draw), draw_rate_obs = mean(sc.draw)))
    end
end
shape_fit = DataFrame(fit_rows); shape_test = DataFrame(test_rows)
r8_show(shape_fit); r8_show(shape_test)
r8_write("shape_fit", shape_fit); r8_write("shape_test", shape_test)

# %%
# ===================================================================
# 5. §3 Does it matter for betting? (best shape per run, L1/L2 test matches)
# ===================================================================
#
# Best shape = the largest out-of-sample exact-score gain on the run's L1/L2 panel. The bets use
# each test match's out-of-sample θ; the comparison grid is the unreshaped Poisson from the same
# thinned draws, so only the shape differs.

close_book, _ = cn_close_book(ds)
entry = Dict("T−25" => first(cn_entry_book(ds, -25.0)), "T−60" => first(cn_entry_book(ds, -60.0)))
fam(r) = r.market_name == "1X2" ? "1X2" : "OU2.5"
bet_rows = NamedTuple[]; change_rows = NamedTuple[]
best_shape = Dict{String,Symbol}()
for r in R8_RUNS
    st = shape_test[(shape_test.run .== r.label) .& (shape_test.panel .== "L1/L2") .& (shape_test.shape .!= C8_SHAPE_LABEL[:poisson]), :]
    k = C8_SHAPES[argmin([st.d_exact[findfirst(==(C8_SHAPE_LABEL[s]), st.shape)] for s in C8_SHAPES])]
    best_shape[r.label] = k
    p0 = c8_probs_frame(test_scores[(r.label, "L1/L2", :poisson)])
    outc = cn_outcomes(ds, select(p0, CN_KEY))
    for kk in (k, :mkt_bpcom), (inst, eb) in entry
        k_ = kk
        p1 = c8_probs_frame(test_scores[(r.label, "L1/L2", k_)])
        f0 = cn_selection_frame(p0, eb, close_book, outc); f1 = cn_selection_frame(p1, eb, close_book, outc)
        for fm in ("1X2", "OU2.5")
            b0 = cn_bets(f0[fam.(eachrow(f0)) .== fm, :], 2.0); b1 = cn_bets(f1[fam.(eachrow(f1)) .== fm, :], 2.0)
            k0 = Set(Tuple.(eachrow(select(b0, CN_KEY)))); k1 = Set(Tuple.(eachrow(select(b1, CN_KEY))))
            s0 = cn_summary(b0; B = R8_B, seed = R8_SEED); s1 = cn_summary(b1; B = R8_B, seed = R8_SEED)
            for (g, s) in (("Poisson", s0), (C8_SHAPE_LABEL[k_], s1))
                push!(bet_rows, (; run = r.label, source = "archive LTP", instant = inst, family = fm, grid = g, n_bets = s.n_bets,
                                 clv_pp = s.clv_pp, clv_lo = s.clv_pp_lo, clv_hi = s.clv_pp_hi, ev2 = s.ev2, ev2_lo = s.ev2_lo, ev2_hi = s.ev2_hi))
            end
            push!(change_rows, (; run = r.label, source = "archive LTP", instant = inst, family = fm, shape = C8_SHAPE_LABEL[k_],
                                n_poisson = length(k0), n_reshaped = length(k1), dropped = length(setdiff(k0, k1)),
                                added = length(setdiff(k1, k0)), d_clv_pp = s1.clv_pp - s0.clv_pp, d_ev2 = s1.ev2 - s0.ev2))
        end
    end
end

# §13's 26/27 order book, grw_spfl_joint, back price, θ fitted on all of its 24/25–25/26 L1/L2 panel
kg = best_shape["grw_spfl_joint"]
θg, _ = c8_fit_shape(panels[("grw_spfl_joint", "L1/L2")], kg)
println("26/27 order book: grw_spfl_joint reshaped with ", C8_SHAPE_LABEL[kg], " θ = ", θg)
q0 = c8_probs_frame(c8_scores(ms_2627, :poisson, 0.0)); q0 = q0[q0.market_name .== "1X2", :]
markets = c4_markets(ds); snaps = c4_snapshots(markets)
close_mid = c4_close(snaps, :mid)
outc27 = cn_outcomes(ds, select(q0, CN_KEY))
for (kq, θq) in ((kg, θg), (:mkt_bpcom, (0.08, 1.08))), (inst, t) in (("T−60", -60.0), ("T−25", -25.0))
    q1 = c8_probs_frame(c8_scores(ms_2627, kq === :mkt_bpcom ? :bpcom : kq, θq)); q1 = q1[q1.market_name .== "1X2", :]
    eb = c4_entry_book(snaps, t)
    b0 = cn_bets(c4_frame(q0, eb, close_mid, outc27; price = :back), 2.0)
    b1 = cn_bets(c4_frame(q1, eb, close_mid, outc27; price = :back), 2.0)
    k0 = Set(Tuple.(eachrow(select(b0, CN_KEY)))); k1 = Set(Tuple.(eachrow(select(b1, CN_KEY))))
    s0 = cn_summary(b0; B = R8_B, seed = R8_SEED); s1 = cn_summary(b1; B = R8_B, seed = R8_SEED)
    for (g, s) in (("Poisson", s0), (C8_SHAPE_LABEL[kq], s1))
        push!(bet_rows, (; run = "grw_spfl_joint", source = "26/27 order book (back)", instant = inst, family = "1X2", grid = g,
                         n_bets = s.n_bets, clv_pp = s.clv_pp, clv_lo = s.clv_pp_lo, clv_hi = s.clv_pp_hi,
                         ev2 = s.ev2, ev2_lo = s.ev2_lo, ev2_hi = s.ev2_hi))
    end
    push!(change_rows, (; run = "grw_spfl_joint", source = "26/27 order book (back)", instant = inst, family = "1X2",
                        shape = C8_SHAPE_LABEL[kq], n_poisson = length(k0), n_reshaped = length(k1),
                        dropped = length(setdiff(k0, k1)), added = length(setdiff(k1, k0)),
                        d_clv_pp = s1.clv_pp - s0.clv_pp, d_ev2 = s1.ev2 - s0.ev2))
end
bets_tab = DataFrame(bet_rows); changes = DataFrame(change_rows)
r8_show(changes); r8_show(bets_tab)
r8_write("betting", bets_tab); r8_write("betting_changes", changes)
println("DONE r08_score_grid ", now())
