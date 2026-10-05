# Phase A: does an isolated KL fit reproduce the coherent close across lines?
# Control: SSE on the same complete, overround-gated book. This is NOT a forecast.
# Snapshot: pinned ScottishLower cache; outputs in results/A and results/figures/A_*.png.
# Usage: own tmux Julia REPL (-t 8), include this file; never one-shot Julia.

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball, DataFrames, CSV, Statistics, Dates, Plots
if !isdefined(@__MODULE__, :MarketModel)
    include("l01_market_model.jl")
end
const MM = MarketModel
const MD = MarketInverseDynamics
const AC = BayesianFootball.Calibration

# %%
# ===================================================================
# 2. Configuration and output: deterministic, replaceable diagnostics
# ===================================================================
const A_CONFIG = MM.scottish_lower_2425_2526()
const A_ROOT = get(ENV, "MM_A_ROOT", joinpath(@__DIR__, "results"))
const A_OUT = joinpath(A_ROOT, "A")
const A_FIG = joinpath(A_ROOT, "figures")
mkpath(A_OUT)
mkpath(A_FIG)
a_start = time()

# %%
# ===================================================================
# 3. Pinned data; no SQL refresh or database writes
# ===================================================================
a_ds = Data.load_datastore_cached(A_CONFIG.segment; max_age_hours = 10^6)
a_cache = stat(joinpath(pwd(), ".cache", "datastore_$(nameof(typeof(A_CONFIG.segment))).jls"))
CSV.write(joinpath(A_OUT, "snapshot.csv"), DataFrame(cache_mtime=[string(unix2datetime(a_cache.mtime))],
                                                    cache_bytes=[a_cache.size]))

# %%
# ===================================================================
# 4. Close: completeness and overround BEFORE de-vig
# ===================================================================
a_raw = MM.closing_book(a_ds, A_CONFIG)
a_book, a_ref = MM.gate_book(a_raw)
a_rates = MM.invert_panel(a_ds, a_book; config=A_CONFIG)
CSV.write(joinpath(A_OUT, "rates.csv"), a_rates)
CSV.write(joinpath(A_OUT, "coverage.csv"), MM.coverage(a_rates, A_CONFIG))
CSV.write(joinpath(A_OUT, "market_refusals.csv"), combine(groupby(a_ref, [:market_name,:reason]), nrow => :n))
CSV.write(joinpath(A_OUT, "market_refusal_details.csv"), a_ref)
a_old_groups = unique(select(a_raw, [:match_id,:market_name,:market_line]))
a_ref_panel = filter(:match_id => in(Set(a_rates.match_id)), a_ref)
CSV.write(joinpath(A_OUT, "book_effect.csv"), DataFrame(metric=["raw_panel_groups", "refused_panel_groups"],
    n=[nrow(filter(:match_id => in(Set(a_rates.match_id)), a_old_groups)), nrow(a_ref_panel)]))
MM.check_expected(A_CONFIG, :panel, nrow(a_rates))
MM.check_expected(A_CONFIG, :accepted, count(a_rates.accepted))

# %%
# ===================================================================
# 5. Same-book SSE control; ungated TODO 023 reproduction
# ===================================================================
a_sse = AC.inversion_frame(AC.invert_market_rates(a_book; match_ids=a_rates.match_id))
a_compare = innerjoin(select(filter(:accepted => identity,a_rates), :match_id,:lambda_h,:lambda_a),
                      select(filter(:accepted => identity,a_sse), :match_id,
                             :lambda_mkt_h => :sse_h,:lambda_mkt_a => :sse_a); on=:match_id)
a_compare.d_log_h = log.(a_compare.lambda_h ./ a_compare.sse_h)
a_compare.d_log_a = log.(a_compare.lambda_a ./ a_compare.sse_a)
a_compare.d_supremacy = a_compare.d_log_h .- a_compare.d_log_a
a_compare.d_log_total = log.((a_compare.lambda_h .+ a_compare.lambda_a) ./ (a_compare.sse_h .+ a_compare.sse_a))
a_compare.max_abs_delta = max.(abs.(a_compare.d_log_h),abs.(a_compare.d_log_a))
CSV.write(joinpath(A_OUT,"kl_vs_sse.csv"),a_compare)
a_quant = DataFrame(axis=String[],quantile=Float64[],value=Float64[],n=Int[])
for axis in (:d_log_h,:d_log_a,:d_supremacy,:d_log_total), p in (0.0,0.05,0.25,0.5,0.75,0.95,1.0)
    push!(a_quant,(string(axis),p,quantile(a_compare[!,axis],p),nrow(a_compare)))
end
CSV.write(joinpath(A_OUT,"kl_vs_sse_quantiles.csv"),a_quant)
CSV.write(joinpath(A_OUT,"kl_vs_sse_top20.csv"),first(sort(a_compare,:max_abs_delta,rev=true),20))
a_ungated = AC.inversion_frame(AC.invert_market_rates(a_raw; match_ids=a_rates.match_id))
a_reference = CSV.read(joinpath(@__DIR__,"..","market_inverse_dynamics","results","production","market_targets.csv"),DataFrame)
CSV.write(joinpath(A_OUT,"reference_columns.csv"),DataFrame(column=string.(names(a_reference))))
a_repro = innerjoin(select(a_ungated,:match_id,:lambda_mkt_h => :new_h,:lambda_mkt_a => :new_a),
                    select(a_reference,:match_id,:lambda_mkt_h => :old_h,:lambda_mkt_a => :old_a);on=:match_id)
a_repro = filter(r -> isfinite(r.old_h) && isfinite(r.old_a) && isfinite(r.new_h) && isfinite(r.new_a),a_repro)
CSV.write(joinpath(A_OUT,"ungated_reproduction.csv"), DataFrame(n_common=[nrow(a_repro)],
    max_abs_lambda_h=[maximum(abs.(a_repro.new_h.-a_repro.old_h))],
    max_abs_lambda_a=[maximum(abs.(a_repro.new_a.-a_repro.old_a))]))
a_prior = CSV.read(joinpath(@__DIR__,"..","market_inverse_dynamics","results","production","market_targets.csv"),DataFrame)
a_prior_ids=Set(a_prior.match_id)
CSV.write(joinpath(A_OUT,"previously_accepted_refused.csv"),select(filter(r -> r.match_id in a_prior_ids && !r.accepted,a_rates),:match_id,:reason))
CSV.write(joinpath(A_OUT,"optimum_check.csv"),DataFrame(n_solved=[count(isfinite,a_rates.start_spread)],
    max_abs_theta=[maximum(a_rates.start_spread[isfinite.(a_rates.start_spread)])]))

# %%
# ===================================================================
# 6. In-sample and held-out-line residuals; fixture-cluster bootstrap
# ===================================================================
a_in = MM.residual_rows(a_rates,a_book)
a_held = MM.residual_rows(a_rates,a_book;heldout=true)
a_all = vcat(a_in,a_held)
a_summaries = DataFrame[]
for scope in ("insample","heldout")
    source=filter(:scope => ==(scope),a_all)
    for split in (:overall,:tournament,:n_markets)
        copy_source=copy(source)
        if split != :tournament
            copy_source.tournament .= 0
        end
        if split != :n_markets
            copy_source.n_markets .= 0
        end
        push!(a_summaries,MM.residual_summary(copy_source))
    end
end
a_res = vcat(a_summaries...)
CSV.write(joinpath(A_OUT,"line_residuals.csv"),filter(:scope => ==("insample"),a_res))
CSV.write(joinpath(A_OUT,"heldout_line_test.csv"),MM.heldout_comparison(a_held,a_in,a_res))
CSV.write(joinpath(A_OUT,"selection_residuals.csv"),a_all)

# %%
# ===================================================================
# 7. Diagnostic figures; no outcome-based interpretation
# ===================================================================
a_acc=filter(:accepted => identity,a_rates)
# Saving the original log-x histogram first poisoned GR's tick/layout state in
# this headless session. Keep diagnostics ahead of the histogram for isolation.
a_figres=filter(r -> r.tournament==0 && r.n_markets==0,a_res)
a_labels=string.(a_figres.line," / ",a_figres.selection," / ",a_figres.scope)
a_plot=scatter(a_figres.mean,1:nrow(a_figres),xerror=(a_figres.mean.-a_figres.ci_low,a_figres.ci_high.-a_figres.mean),
               yticks=(1:nrow(a_figres),a_labels),legend=false,
               ylabel="selection / fit",xlabel="q - p",size=(1300,1100),left_margin=175Plots.px)
vline!(a_plot,[0.0],color=:black)
savefig(a_plot,joinpath(A_FIG,"A_line_residuals.png"))
a_plot=scatter(log.(a_compare.lambda_h),log.(a_compare.sse_h),label="home",xlabel="KL log rate",
               ylabel="SSE log rate",layout=(1,2),subplot=1)
scatter!(a_plot,log.(a_compare.lambda_a),log.(a_compare.sse_a),label="away",subplot=1)
histogram!(a_plot,a_compare.d_log_h,label="Δ home",subplot=2,xlabel="KL − SSE log rate")
histogram!(a_plot,a_compare.d_log_a,label="Δ away",subplot=2,alpha=0.5)
savefig(a_plot,joinpath(A_FIG,"A_kl_sse.png"))
a_plot=scatter(a_acc.lambda_h,a_acc.lambda_a,group=a_acc.tournament,
               xlabel="lambda home",ylabel="lambda away",title="KL rates by tournament")
savefig(a_plot,joinpath(A_FIG,"A_rates.png"))
# Bin in log10 space (equivalent to a log x axis), including KL values at zero.
# GR's histogram(xscale=:log10) produced an EMPTY picture and "No strict ticks".
a_plot=histogram(log10.(max.(a_acc.kl,1e-16)),group=a_acc.n_selections,
                 xlabel="log10 summed KL (floor 1e-16)",ylabel="fixtures",title="KL by book size")
savefig(a_plot,joinpath(A_FIG,"A_kl_distribution.png"))

# %%
# ===================================================================
# 8. Final checks and elapsed time (all published values stored in CSV)
# ===================================================================
@assert maximum(a_rates.start_spread[isfinite.(a_rates.start_spread)]) <= 1e-6
@assert all(count(==(id), filter(:reason => ==(reason), a_rates).match_id) == 1
            for (id,reason) in A_CONFIG.excluded_matches)
CSV.write(joinpath(A_OUT,"runtime.csv"),DataFrame(wall_seconds=[time()-a_start]))
println("Phase A outputs: ",A_OUT)
println("Wall seconds: ",time()-a_start)
println("R01_DONE")
