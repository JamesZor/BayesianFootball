# r03 — Latent tier-step GLMs and the pyramid-linearity test.
#
#   include("experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r03_econometric_tier_glms.jl")
#
# Long format: every fixture gives two rows, "club i attacking club j".
#
#   log μ_ij = β0 + comp_c + h·Home_ij + h_cup·Home_ij·Cup + g_T(i) + g_T(j) + ½(θ_T(i) − θ_T(j))
#
# g_T is the tier's goal level (lower leagues score more) and θ_T its net strength,
# θ_1 ≡ 0.  This is an exact reparameterisation of separate attack / concession tier
# offsets  A_T = g_T + θ_T/2  (the α tier offset in the L1 models) and
# D_T = g_T − θ_T/2  (the β tier offset: + ⇒ concedes more).  Same-tier league games
# identify g_T; only cross-tier cup ties identify θ.  Tier step k→k+1 is
# τ_k = θ_k − θ_{k+1}  (log goal-rate supremacy of an average tier-k club over an
# average tier-(k+1) club at a neutral venue).
#
# Standard errors: two-way clustered on attacking club-season and defending
# club-season (Cameron–Gelbach–Miller), because the tier-only model leaves every
# club's own strength in the residual.

# ── 1. Setup ────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "_common.jl"))
using GLM, Distributions, Optim

fx_all = load_fixtures(:long)

# ── 2. Long-format design (long_design, design_row) ──────────────────────────
include(joinpath(@__DIR__, "l03_tier_design.jl"))

# ── 3. Estimation: Poisson and NB2 (θ_nb profiled), sandwich covariance ──────
function fit_glm(D; family = :poisson)
    if family === :poisson
        m = glm(D.X, D.y, Poisson(), LogLink())
        r = 0.0
    else
        # Profile the NB2 size r over a 1-D likelihood; GLM.jl then fits β given r.
        prof(lr) = -loglikelihood(glm(D.X, D.y, NegativeBinomial(exp(lr)), LogLink()))
        o = optimize(prof, log(1.0), log(500.0))
        r = exp(Optim.minimizer(o))
        m = glm(D.X, D.y, NegativeBinomial(r), LogLink())
    end
    β = coef(m); μ = exp.(D.X * β)
    w = family === :poisson ? μ : μ ./ (1 .+ μ ./ r)          # Fisher weights (log link)
    s = family === :poisson ? (D.y .- μ) : (D.y .- μ) ./ (1 .+ μ ./ r)   # score multipliers
    B = inv(Symmetric(D.X' * (w .* D.X)))
    S = D.X .* s
    meat(g) = (G = unique(g); idx = Dict(v => i for (i, v) in enumerate(G));
               A = zeros(length(G), size(S, 2)); for i in eachindex(g); A[idx[g[i]], :] .+= S[i, :]; end;
               (length(G) / (length(G) - 1)) .* (A' * A))
    inter = string.(D.att_cs, "|", D.def_cs)
    V2 = B * (meat(D.att_cs) .+ meat(D.def_cs) .- meat(inter)) * B
    # CGM can go non-PSD in finite samples; floor eigenvalues at zero.
    E = eigen(Symmetric(V2)); V2 = E.vectors * Diagonal(max.(E.values, 0)) * E.vectors'
    Vm = B * meat(D.match) * B
    return (; m, β, V = V2, Vmatch = Vm, Vmodel = Matrix(vcov(m)), r, ll = loglikelihood(m),
              names = D.names, n = length(D.y), nmatch = D.nmatch)
end

coef_table(F) = DataFrame(term = F.names, coef = F.β, se_cluster2 = sqrt.(diag(F.V)),
                          se_match = sqrt.(diag(F.Vmatch)), se_model = sqrt.(diag(F.Vmodel)),
                          z = F.β ./ sqrt.(diag(F.V)),
                          p = 2 .* ccdf.(Normal(), abs.(F.β ./ sqrt.(diag(F.V)))))

"Linear combination c'β with its cluster SE."
lincomb(F, c) = (est = dot(c, F.β), se = sqrt(max(c' * F.V * c, 0.0)))
cvec(F, pairs...) = (c = zeros(length(F.β)); for (nm, w) in pairs; i = findfirst(==(nm), F.names); i === nothing || (c[i] += w); end; c)

"Tier steps τ between consecutive levels in `chain` (e.g. [\"T1\",\"T2\",...]); θ_T1 ≡ 0."
function steps(F, chain)
    rows = NamedTuple[]
    for (a, b) in zip(chain[1:end-1], chain[2:end])
        # τ = θ_a − θ_b ; attack share ΔA = A_a − A_b ; concession ΔD = D_a − D_b
        cτ = cvec(F, "θ_" * a => 1.0, "θ_" * b => -1.0)
        cA = cvec(F, "g_" * a => 1.0, "g_" * b => -1.0, "θ_" * a => 0.5, "θ_" * b => -0.5)
        cD = cvec(F, "g_" * a => 1.0, "g_" * b => -1.0, "θ_" * a => -0.5, "θ_" * b => 0.5)
        τ = lincomb(F, cτ); A = lincomb(F, cA); Dd = lincomb(F, cD)
        push!(rows, (step = a * "→" * b, tau = τ.est, se = τ.se, lo95 = τ.est - 1.96τ.se, hi95 = τ.est + 1.96τ.se,
                     rate_ratio = exp(τ.est), d_attack = A.est, se_attack = A.se, d_concede = Dd.est, se_concede = Dd.se))
    end
    DataFrame(rows)
end

"Wald χ² for equal consecutive steps along `chain` (cluster covariance)."
function wald_equal_steps(F, chain)
    cs_ = [cvec(F, "θ_" * a => 1.0, "θ_" * b => -1.0) for (a, b) in zip(chain[1:end-1], chain[2:end])]
    R = reduce(vcat, ((cs_[i] .- cs_[i+1])' for i in 1:length(cs_)-1))
    rb = R * F.β
    W = rb' * pinv(R * F.V * R') * rb
    q = size(R, 1)
    Wm = rb' * pinv(R * F.Vmodel * R') * rb
    (restrictions = q, wald_chi2 = W, p = ccdf(Chisq(q), W), wald_chi2_model_se = Wm, p_model_se = ccdf(Chisq(q), Wm))
end

"LRT: unconstrained θ vs θ linear in tier number along `chain` (other levels free)."
function lrt_linear(D, F, chain; family = :poisson)
    # Replace the θ columns of the chain levels by one column ½(T_j − T_i) (θ_T = −s·(T−1)).
    num = Dict(k => i - 1 for (i, k) in enumerate(chain))           # T1 ↦ 0, T2 ↦ 1, …
    xi = [get(num, a, 0) for a in D.att]; xj = [get(num, b, 0) for b in D.dfn]
    inchain_i = [haskey(num, a) for a in D.att]; inchain_j = [haskey(num, b) for b in D.dfn]
    lin = 0.5 .* (xj .* inchain_j .- xi .* inchain_i)
    drop = Set("θ_" * k for k in chain)
    keepi = [i for (i, nm) in enumerate(D.names) if !(nm in drop)]
    D0 = merge(D, (X = hcat(D.X[:, keepi], lin), names = vcat(D.names[keepi], "θ_linear_step")))
    F0 = fit_glm(D0; family = family)
    LR = 2 * (F.ll - F0.ll); q = length(chain) - 2
    (F0 = F0, lr_chi2 = LR, df = q, p = ccdf(Chisq(q), LR), linear_step = F0.β[end], linear_step_se = sqrt(F0.V[end, end]))
end

# ── 4. Main fits ────────────────────────────────────────────────────────────
spec = [
    ("primary", :primary, false, false),
    ("long", :long, false, false),
    ("long_oldfirm_split", :long, true, false),
    ("primary_oldfirm_split", :primary, true, false),
    ("long_t5split", :long, false, true),
]
fits = Dict{Tuple{String, Symbol}, Any}()
designs = Dict{String, Any}()
coef_rows, step_rows, test_rows = DataFrame[], DataFrame[], NamedTuple[]
for (lab, win, of, t5) in spec
    local fx = win === :primary ? fx_all[fx_all.in_primary_window, :] : fx_all
    D = long_design(fx; split_old_firm = of, t5split = t5); designs[lab] = D
    chain_all = filter(in(D.levels), of ? ["T0", "T1", "T2", "T3", "T4", "T5", "T6"] : ["T1", "T2", "T3", "T4", "T5", "T6"])
    chain_spfl = filter(in(D.levels), ["T1", "T2", "T3", "T4"])
    chain_pyr  = filter(!=("T0"), chain_all)      # T0 (Old Firm) is not a pyramid tier; never in the test
    for fam in (:poisson, :negbin)
        F = fit_glm(D; family = fam); fits[(lab, fam)] = F
        ct = coef_table(F); insertcols!(ct, 1, :fit => lab, :family => string(fam)); push!(coef_rows, ct)
        local st = steps(F, chain_all); insertcols!(st, 1, :fit => lab, :family => string(fam)); push!(step_rows, st)
        for (hname, ch) in (("H0 SPFL: τ12=τ23=τ34", chain_spfl), ("H0 all: equal steps T1..T5+", chain_pyr))
            w = wald_equal_steps(F, ch); l = lrt_linear(D, F, ch; family = fam)
            push!(test_rows, (fit = lab, family = string(fam), hypothesis = hname, n_rows = F.n, n_matches = F.nmatch,
                              nb_size = F.r, w.restrictions, w.wald_chi2, w.p, w.wald_chi2_model_se, w.p_model_se,
                              l.lr_chi2, lr_p = l.p, l.linear_step, l.linear_step_se))
        end
    end
end
coefs = vcat(coef_rows...); stepsdf = vcat(step_rows...); tests = DataFrame(test_rows)
save_csv("r03_glm_coefficients.csv", coefs); save_csv("r03_tier_steps.csv", stepsdf); save_csv("r03_linearity_tests.csv", tests)
save_md("r03_glm_coefficients_primary_long.md",
        coefs[in.(coefs.fit, Ref(("primary", "long"))), [:fit, :family, :term, :coef, :se_cluster2, :se_match, :z, :p]]; digits = 3)
save_md("r03_tier_steps.md", stepsdf[:, [:fit, :family, :step, :tau, :se, :lo95, :hi95, :rate_ratio, :d_attack, :d_concede]]; digits = 3)
save_md("r03_linearity_tests.md", tests[:, [:fit, :family, :hypothesis, :n_matches, :restrictions, :wald_chi2, :p, :lr_chi2, :lr_p, :linear_step, :linear_step_se]]; digits = 3)

# ── 5. Era stability of the steps (long window split in three) ──────────────
eras = [("08/09–13/14", 2008, 2013), ("14/15–19/20", 2014, 2019), ("20/21–26/27", 2020, 2026)]
era_rows = DataFrame[]
for (lab, a, b) in eras, of in (false, true)
    D = long_design(fx_all[(fx_all.fs .>= a) .& (fx_all.fs .<= b), :]; split_old_firm = of)
    F = fit_glm(D; family = :poisson)
    local st = steps(F, filter(in(D.levels), ["T0", "T1", "T2", "T3", "T4", "T5"]))
    insertcols!(st, 1, :era => lab, :old_firm_split => of, :n_matches => F.nmatch); push!(era_rows, st)
end
era = vcat(era_rows...)
save_csv("r03_tier_steps_by_era.csv", era); save_md("r03_tier_steps_by_era.md", era[:, [:era, :old_firm_split, :n_matches, :step, :tau, :se, :d_attack, :d_concede]]; digits = 3)

# ── 6. Goal-level offsets g_T and home advantage (for the L1 intercept design) ─
lvl = coefs[(coefs.family .== "poisson") .& in.(coefs.fit, Ref(("primary", "long"))) .&
            (startswith.(coefs.term, "g_") .| startswith.(coefs.term, "home") .| startswith.(coefs.term, "comp")), :]
save_md("r03_levels_and_home.md", lvl[:, [:fit, :term, :coef, :se_cluster2, :p]]; digits = 3)

# ── 7. Console summary ──────────────────────────────────────────────────────
println("Tier steps (Poisson):")
show(stepsdf[stepsdf.family .== "poisson", [:fit, :step, :tau, :se, :d_attack, :d_concede]]; allrows = true); println()
println("Linearity tests:")
show(tests[:, [:fit, :family, :hypothesis, :restrictions, :wald_chi2, :p, :lr_chi2, :lr_p, :linear_step]]; allrows = true); println()
println("Era steps:"); show(era[:, [:era, :old_firm_split, :n_matches, :step, :tau, :se]]; allrows = true); println()
println("NB sizes: ", [(k, round(v.r; digits = 1)) for (k, v) in fits if k[2] === :negbin])
