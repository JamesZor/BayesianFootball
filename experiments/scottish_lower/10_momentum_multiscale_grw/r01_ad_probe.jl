# Development-only deterministic AD localization; no sampling.
using ThreadPinning, LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l10_momentum_grw_loader.jl"))
const M = MomentumGRW
const RD = M.ReverseDiff
const acc = M.GPH_PG.grw_accumulators(2, 20)
const design = M.momentum_design(M.B.GRWDynamicsDesign(CartesianIndex{2}[],
    CartesianIndex{2}[], acc.initial, acc.season, acc.target, Val(true), 2, 20, 22), 4)
function scalar_path(x)
    z = reshape(x[3:end], 4, 19)
    scaled = z .* x[1]
    kernel = M.velocity_kernel(x[2], design)
    increments = scaled * kernel
    states = increments * design.grw.target_accumulator
    println.((typeof(z), typeof(scaled), typeof(kernel), typeof(increments), typeof(states)))
    return sum(states)
end
function array_path(x)
    z = reshape(x[3:end], 4, 19)
    scaled = z .* reshape(x[1:1],1,1)
    kernel = M.velocity_kernel(reshape(x[2:2],1,1), design)
    return sum((scaled * kernel) * design.grw.target_accumulator)
end
function instruction_bytes(i)
    RD.forward_exec!(i)
    RD.reverse_exec!(i)
    f = @allocated RD.forward_exec!(i)
    r = @allocated RD.reverse_exec!(i)
    return (f,r)
end
function bench(f, x)
    raw = RD.GradientTape(f, x)
    tape = RD.compile(raw)
    g = similar(x)
    for _ in 1:30
        RD.gradient!(g,tape,x)
    end
    bytes = @allocated RD.gradient!(g,tape,x)
    println("BENCH ", f, " instructions=", length(raw.tape), " bytes=", bytes)
    allocations = Dict{String,Int}()
    for instruction in raw.tape
        bytes = instruction_bytes(instruction)
        if sum(bytes) > 0
            key = string(instruction.func)
            allocations[key] = get(allocations,key,0) + sum(bytes)
        end
    end
    println("INSTRUCTION ALLOCATIONS ", allocations)
    return raw
end
function optimized_path(x)
    z = reshape(x[3:end],4,19)
    return sum(abs2.(M.momentum_positions(z,x[1],x[2],design)))
end
x = vcat([0.02, 0.5], sin.(1:76))
bench(scalar_path, x)
bench(array_path, x)
bench(optimized_path, x)

# Real full-tape localization, independent of the isolated dynamics benchmark.
ds = M.gph_load_data()
model = last(M.models())[2]
inputs = M.gph_fold_inputs(ds,M.gph_splitter(["24/25","25/26"]),model)
tm = M.GPH_PG.build_turing_model(model, first(inputs.feature_sets[20]))
vi = M.DynamicPPL.VarInfo(tm)
tm(vi)
ld = M.DynamicPPL.LogDensityFunction(tm)
objective(x) = M.LogDensityProblems.logdensity(ld,x)
bench(objective,copy(vi[:]))
