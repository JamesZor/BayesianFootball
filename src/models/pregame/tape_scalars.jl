# src/models/pregame/tape_scalars.jl
#
# Scalars inside broadcasts, without allocating on every gradient.
#
# ReverseDiff 1.17 (`derivatives/broadcast.jl`, `get_implementation`) differentiates a FUSED
# broadcast in one of two ways:
#
#   * `∇broadcast`         — every argument is an array (tracked or not) or a non-`Real` such as a
#                            `Ref`. Its forward pass stores per-element dual partials in a buffer
#                            allocated once, when the tape is recorded. Zero bytes per replay.
#   * `tracker_∇broadcast` — ANY argument is a `Real`, tracked OR untracked. Its reverse pass
#                            builds fresh O(rows) partial arrays for every argument on every
#                            gradient. On the W2 `td_base` fold that was 432 KB per gradient,
#                            and at 16 threads the GC halved sampler throughput.
#
# An UNFUSED binary `+ - * / ^` of a tracked scalar against an array (`z .* σ`, `η .+ log_κ`) takes
# a third path with its own preallocated kernel and is already free — splitting a fused expression
# into unfused binary steps is therefore one fix. The other is to keep the kernel fused and change
# what the scalars look like to ReverseDiff:
#
#   * a SAMPLED scalar enters a fused broadcast as a one-element tracked vector (`tape_scalar`);
#   * a CONSTANT scalar enters as `Ref(c)` (`tape_scalar` again, by dispatch).
#
# Outside ReverseDiff — `Float64` evaluation, ForwardDiff duals, posterior extraction — both
# spellings broadcast exactly like the bare scalar, so the arithmetic is unchanged to the bit.
#
# Authority and measurement: docs/turing_ad_performance_guide.md §10.5; the audit is
# scripts/tape_allocation_audit.jl and the regression gate test/tape_allocation_tests.jl plus the
# harness `tape_allocation` smoke check. The pattern graduated from the prototype adapters in
# experiments/scottish_lower/10_momentum_multiscale_grw/l10_momentum_grw_loader.jl (commit
# 7bea5069) and experiments/scottish_lower/11_decompression_pxg_covariate/l11_decompression_loader.jl.

"""
    tape_fill(x, n) -> vector

`fill(x, n)`, with a zero-allocation adjoint when `x` is a ReverseDiff tracked scalar.

ReverseDiff's own `fill` rule allocates on replay (48 B for three seasons, 368 B for 43 teams on
the W2 fold). This one records a single instruction that owns its output buffer: forward
`fill!`s it, reverse adds the summed adjoint to `x`. Any other `x` (Float64, a ForwardDiff dual)
goes straight to `fill`.
"""
tape_fill(x::Real, n::Integer) = fill(x, n)

function tape_fill(x::ReverseDiff.TrackedReal{V,D}, n::Integer) where {V,D}
    tape = ReverseDiff.tape(x)
    out = ReverseDiff.track(fill(ReverseDiff.value(x), n), D, tape)
    ReverseDiff.record!(tape, ReverseDiff.SpecialInstruction, tape_fill, x, out)
    return out
end

function ReverseDiff.special_forward_exec!(
        instruction::ReverseDiff.SpecialInstruction{typeof(tape_fill)})
    ReverseDiff.pull_value!(instruction.input)
    fill!(ReverseDiff.value(instruction.output), ReverseDiff.value(instruction.input))
    return nothing
end

function ReverseDiff.special_reverse_exec!(
        instruction::ReverseDiff.SpecialInstruction{typeof(tape_fill)})
    ReverseDiff.increment_deriv!(instruction.input, sum(ReverseDiff.deriv(instruction.output)))
    ReverseDiff.unseed!(instruction.output)
    return nothing
end

"""
    tape_scalar(x)

A scalar spelled so that a FUSED broadcast containing it takes ReverseDiff's preallocated
`∇broadcast` adjoint rather than the allocating `tracker_∇broadcast` one.

  * a ReverseDiff tracked scalar → a one-element tracked vector (`tape_fill(x, 1)`); its gradient
    is the sum over the broadcast, exactly as for the bare scalar;
  * anything else (`Float64`, a ForwardDiff dual, a fixed config value) → `Ref(x)`, which every
    broadcast treats as the bare scalar and ReverseDiff treats as an untracked constant.

Use it only where the other operands are arrays: `tape_scalar(ν) .* x` has the shape of `x`, but a
broadcast of tape scalars alone would come back as a one-element vector under ReverseDiff.
"""
tape_scalar(x::Real) = Ref(x)
tape_scalar(x::ReverseDiff.TrackedReal) = tape_fill(x, 1)
