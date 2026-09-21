# Deterministic contract tests; no database, data snapshot, or sampling required.
using Test, LinearAlgebra, Statistics, Random
using ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l10_momentum_grw_loader.jl"))
const M = MomentumGRW

function test_design(h, k, n_teams=4)
    acc = M.GPH_PG.grw_accumulators(h, k)
    grw = M.B.GRWDynamicsDesign(CartesianIndex{2}[], CartesianIndex{2}[],
        acc.initial, acc.season, acc.target, Val(k > 0), h, k, h+k)
    return M.momentum_design(grw, n_teams)
end

@testset "Polynomial dynamics versus independent recurrence" begin
    rng = MersenneTwister(22)
    for h in (1, 2, 3), k in (2, 3, 20), φ in (0.0, 0.2, 0.7, 0.999999, 1.0)
        d = test_design(h, k)
        z = randn(rng, 4, k-1)
        original = copy(z)
        expected = zeros(4, h+k)
        velocity = zeros(4)
        position = zeros(4)
        for t in 1:k
            position += velocity
            expected[:, h+t] = position
            t < k && (velocity = φ * velocity + 0.03 * z[:, t])
        end
        @test M.momentum_positions(z, 0.03, φ, d) ≈ expected atol=1e-13
        @test z == original
        @test M.momentum_positions(z, 0.0, φ, d) == zero(expected)
    end
    for k in (0, 1)
        @test test_design(2, k).active === Val(false)
    end
end

@testset "Synthetic chain reconstruction and forecast" begin
    for h in (1, 2), k in (0, 1, 2, 5)
        n = 3
        columns = Symbol[]
        values = Float64[]
        put(name, value) = (push!(columns, Symbol(name)); push!(values, value))
        for side in ("α", "β")
            p = "dyn.$side"
            put("$p.σ₀", 0.2)
            put("$p.σₛ", 0.1)
            k > 0 && put("$p.σₖ", 0.03)
            for t in 1:n
                put("$p.z_init[$t]", t-2.0)
                for j in 1:h-1
                    put("$p.z_season[$t, $j]", t*j/10)
                end
                for j in 1:k
                    put("$p.z_target[$t, $j]", (t-j)/10)
                end
            end
            if k >= 2
                put("$p.φ", 0.7)
                put("$p.σᵥ", 0.02)
                for t in 1:n, j in 1:k-1
                    put("$p.z_velocity[$t, $j]", (t+j)/10)
                end
            end
        end
        # Nonidentical draws across two chains test draw ordering.
        array = repeat(reshape(values, 1, :, 1), 2, 1, 2)
        array[:, findfirst(==(Symbol("dyn.α.z_init[1]")), columns), 2] .+= 0.5
        chain = M.Chains(array, columns)
        draw = M.B._cb_extract_dynamics(chain, M.MomentumMultiScaleGRW(), "dyn", n)
        cfg = M.MomentumMultiScaleGRW()
        design = test_design(h,k,n)
        tm = M.momentum_side(cfg,cfg.attack_velocity,cfg.level.α_σ₀,
            cfg.level.α_σₛ,cfg.level.α_σₖ,design,n,Val(k>0))
        params = (; σ₀=0.2, σₛ=0.1, σₖ=0.03,
            z_init=collect(1:n) .- 2.0,
            z_season=[t*j/10 for t in 1:n,j in 1:h-1],
            z_target=[(t-j)/10 for t in 1:n,j in 1:k],
            φ=0.7, σᵥ=0.02, z_velocity=[(t+j)/10 for t in 1:n,j in 1:max(k-1,0)])
        generated = M.DynamicPPL.returned(tm, params)
        @test generated ≈ draw.α[:,:,1] atol=1e-13
        @test size(draw.α) == (n, h+k, 4)
        @test maximum(abs.(sum(draw.α; dims=1))) < 1e-13
        baseline = M.GPH_PG._grw_reconstruct_trajectory(chain, "dyn.α", n, h, k)
        if k < 2
            @test draw.α == baseline
            @test all(iszero, draw.velocity_α)
        else
            z = [(t+j)/10 for t in 1:n, j in 1:k-1]
            addition = M.momentum_positions(z, 0.02, 0.7, test_design(h, k))
            addition .-= mean(addition; dims=1)
            @test draw.α ≈ baseline .+ reshape(addition, n, h+k, 1) atol=1e-13
            terminal = 0.02 .* z * (0.7 .^ collect(k-1:-1:1))
            terminal .-= mean(terminal)
            @test draw.velocity_α ≈ repeat(terminal, 1, 4) atol=1e-13
        end
        forecast = M.B._cb_oos_dynamics(M.MomentumMultiScaleGRW(), draw, nothing, 1, 1, 2, 4)
        @test forecast.att_h == vec(draw.α[1,end,:]) + vec(draw.velocity_α[1,:])
        unknown = M.B._cb_oos_dynamics(M.MomentumMultiScaleGRW(), draw, nothing, 1, 0, 2, 4)
        @test unknown.att_h == zeros(4)
    end
end

function replay_allocations(tape, gradient, x)
    for _ in 1:30
        M.ReverseDiff.gradient!(gradient, tape, x)
    end
    return @allocated M.ReverseDiff.gradient!(gradient, tape, x)
end

@testset "Compiled kernel gradients, including persistence endpoints" begin
    d = test_design(2, 20)
    f(x) = sum(abs2.(M.momentum_positions(reshape(x[3:end],4,19), x[1], x[2], d)))
    x = vcat([0.02, 0.5], sin.(1:76))
    raw = M.ReverseDiff.GradientTape(f, x)
    tape = M.ReverseDiff.compile(raw)
    g = similar(x)
    for φ in (0.0, 0.01, 0.5, 0.99, 0.999999, 1.0)
        p = vcat([0.03, φ], cos.(1:76))
        M.ReverseDiff.gradient!(g, tape, p)
        @test g ≈ M.ForwardDiff.gradient(f, p) rtol=1e-10 atol=1e-10
    end
    for _ in 1:20
        M.ReverseDiff.gradient!(g, tape, x)
    end
    bytes = replay_allocations(tape, g, x)
    println("MOMENTUM_KERNEL allocated_bytes=", bytes, " instructions=", length(raw.tape))
    @test bytes == 0
end
