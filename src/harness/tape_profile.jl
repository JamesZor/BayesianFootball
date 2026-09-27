# src/harness/tape_profile.jl
#
# Bytes per compiled gradient, and where they come from.
#
# Self-contained on purpose: it depends on ReverseDiff alone, so `scripts/tape_allocation_audit.jl`
# and `test/tape_allocation_tests.jl` include this file directly and can therefore be run against
# an OLDER checkout's `src` (`julia --project=<old> <new>/scripts/tape_allocation_audit.jl`) — which
# is how the before-table and the "test fails on the pre-fix code" demonstration are produced.
# The harness module includes the same file for its `tape_allocation` smoke check.
#
# Why this is a metric at all: docs/turing_ad_performance_guide.md §10.5. In short, a tracked
# scalar (or a plain `Real`) inside a fused broadcast sends ReverseDiff to `tracker_∇broadcast`,
# whose reverse pass allocates O(rows) per call. A 0.24 ms gradient looked fine and still halved
# 16-thread NUTS throughput through GC.

"""
    compiled_gradient_bytes(tape, gradient, θ; warmup = 20, reps = 20) -> Int

Heap bytes one `ReverseDiff.gradient!` on a compiled tape allocates: the MINIMUM over `reps`
warmed replays. Behind a function barrier so `@allocated` sees only the replay, not dynamic
dispatch on the caller's abstractly-typed locals.

Minimum rather than a single sample because Julia runtime bookkeeping can land on one otherwise
allocation-free replay (observed once as 800 B, `l11_decompression_loader.jl` §4); a genuinely
allocating instruction allocates on every replay.
"""
function compiled_gradient_bytes(tape, gradient, θ; warmup::Int = 20, reps::Int = 20)
    reps >= 1 || error("compiled_gradient_bytes needs reps ≥ 1; got $reps")
    for _ in 1:warmup
        ReverseDiff.gradient!(gradient, tape, θ)
    end
    best = typemax(Int)
    for _ in 1:reps
        best = min(best, @allocated ReverseDiff.gradient!(gradient, tape, θ))
    end
    return best
end

"Minimum bytes over `reps` isolated forward + reverse replays of ONE tape instruction."
function _instruction_bytes(instruction, reps::Int)
    ReverseDiff.forward_exec!(instruction)
    ReverseDiff.reverse_exec!(instruction)
    best = typemax(Int)
    for _ in 1:reps
        forward = @allocated ReverseDiff.forward_exec!(instruction)
        reverse = @allocated ReverseDiff.reverse_exec!(instruction)
        best = min(best, forward + reverse)
    end
    return best
end

_tape_arg_shape(x::ReverseDiff.TrackedArray) = "T" * join(size(x), "×")
_tape_arg_shape(::ReverseDiff.TrackedReal) = "TReal"
_tape_arg_shape(x::AbstractArray) = "Arr" * join(size(x), "×")
_tape_arg_shape(::Real) = "Real"
_tape_arg_shape(x) = string(nameof(typeof(x)))

function _tape_function_name(f)
    f isa Tuple && return join(map(_tape_function_name, f), ",")
    name = try
        string(nameof(f))
    catch
        string(typeof(f))
    end
    # A fused broadcast's flattened kernel is an anonymous closure whose type name is noise.
    return startswith(name, "#") || occursin("var\"#", name) ? "fused" : first(name, 60)
end

"""
    describe_tape_instruction(instruction) -> String

`func[kernel](input shapes) -> output shape`, e.g.
`tracker_∇broadcast[clamp](T2390, Real, Real) -> T2390`. `T` marks a tracked array, `TReal` a
tracked scalar, `Arr` an untracked array, `Real` an untracked scalar.
"""
function describe_tape_instruction(instruction)
    func = instruction.func
    inputs = instruction.input isa Tuple ? instruction.input : (instruction.input,)
    kernel = ""
    if instruction isa ReverseDiff.SpecialInstruction && func === ReverseDiff.tracker_∇broadcast
        kernel = "[" * _tape_function_name(first(instruction.cache)) * "]"
    end
    return string(_tape_function_name(func), kernel, "(",
                  join(map(_tape_arg_shape, inputs), ", "), ") -> ",
                  _tape_arg_shape(instruction.output))
end

"One allocating tape instruction: its position on the tape, bytes per replay, and shape."
const TapeAllocationRow = NamedTuple{(:index, :bytes, :description),Tuple{Int,Int,String}}

"""
    tape_allocation_profile(raw_tape; reps = 3) -> Vector{TapeAllocationRow}

Every instruction of a recorded `GradientTape` that allocates when replayed on its own, as
`(; index, bytes, description)`. The per-instruction sum is not exactly the compiled total — the
compiled tape does not pay per-instruction dispatch — but a zero-allocation tape lists nothing,
and a non-zero compiled total is always explained by the rows here.

Replaying instructions one at a time scribbles on the tape's value and derivative buffers.
Measure the compiled gradient first; re-run a gradient before trusting its numbers again.
"""
function tape_allocation_profile(raw_tape; reps::Int = 3)
    rows = TapeAllocationRow[]
    for (index, instruction) in enumerate(raw_tape.tape)
        bytes = _instruction_bytes(instruction, reps)
        bytes > 0 && push!(rows, (; index, bytes, description = describe_tape_instruction(instruction)))
    end
    return rows
end
