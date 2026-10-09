# Manager-amended derivative audit. Definitions only; no fitting or threshold tuning.
module MarketRateFDAudit
import CSV, DataFrames, ForwardDiff, LinearAlgebra, Random

"Central difference with relative coordinate step, preserving the input point."
function central(f, point, coordinate, relative_step)
    h = relative_step * max(one(eltype(point)), abs(point[coordinate]))
    plus, minus = copy(point), copy(point)
    plus[coordinate] += h
    minus[coordinate] -= h
    fp, fm = f(plus), f(minus)
    return (; derivative = (fp - fm) / (2h), fp, fm, h)
end

relative_error(a, b) = abs(a - b) / max(abs(a), abs(b), one(a))

"Replay historical failures and audit their union plus 50 fixed-seed other coordinates."
function audit(f, points, gradients, arm, output)
    # This is selection/provenance only, not a reinstated single-step acceptance rule.
    failed = Set{Int}()
    counts = Int[]
    for (point, gradient) in zip(points, gradients)
        count_failed = 0
        for coordinate in eachindex(point)
            plus, minus = copy(point), copy(point)
            plus[coordinate] += 1e-5
            minus[coordinate] -= 1e-5
            fd = (f(plus) - f(minus)) / 2e-5
            if relative_error(gradient[coordinate], fd) > 1e-6
                push!(failed, coordinate)
                count_failed += 1
            end
        end
        push!(counts, count_failed)
    end
    expected = arm == "grw" ? 65 : 305
    sum(counts) == expected || error("historical $arm failure count differs: $counts (expected total $expected)")
    remaining = setdiff(collect(eachindex(first(points))), collect(failed))
    rng = Random.MersenneTwister(arm == "grw" ? 20261020 : 20261021)
    sampled = remaining[Random.randperm(rng, length(remaining))[1:50]]
    selected = sort!(vcat(collect(failed), sampled))
    manifest = DataFrames.DataFrame(arm = fill(arm, length(selected)), coordinate = selected,
        historically_failed = in.(selected, Ref(failed)))
    CSV.write(joinpath(output, "fd_coordinates_$arm.csv"), manifest)
    rows, extrapolations, comparisons = NamedTuple[], NamedTuple[], NamedTuple[]
    setprecision(BigFloat, 128) do
        for (point_number, (point, gradient)) in enumerate(zip(points, gradients))
            bigpoint = BigFloat.(point)
            f(bigpoint) isa BigFloat || error("density did not preserve high precision")
            biggradient = ForwardDiff.gradient(f, bigpoint)
            converted = Float64.(biggradient)
            cross_error = LinearAlgebra.norm(converted - gradient) /
                max(LinearAlgebra.norm(converted), LinearAlgebra.norm(gradient), 1.0)
            push!(comparisons, (; arm, point = point_number, high_precision_ad_relative_error = cross_error))
            for coordinate in selected
                for (precision, x, reference) in (("Float64", point, gradient),
                                                  ("BigFloat128", bigpoint, biggradient))
                    curve = NamedTuple[]
                    for relative_step in (1e-3, 1e-4, 1e-5, 1e-6)
                        step = convert(eltype(x), relative_step)
                        value = central(f, x, coordinate, step)
                        scale = max(abs(reference[coordinate]), abs(value.derivative), one(step))
                        error = relative_error(value.derivative, reference[coordinate])
                        # Conservative arithmetic floor for subtracting two complete log densities.
                        floor = 64eps(eltype(x)) * ((abs(value.fp) + abs(value.fm)) /
                            (2value.h) + max(abs(reference[coordinate]), one(step))) / scale
                        push!(curve, (; relative_step, error, floor))
                    end
                    for k in eachindex(curve)
                        item = curve[k]
                        order = k < length(curve) && item.error > 0 && curve[k+1].error > 0 ?
                            log10(item.error / curve[k+1].error) : NaN
                        at_floor = item.error <= item.floor ||
                            (k < length(curve) && curve[k+1].error <= curve[k+1].floor)
                        trend_ok = k == length(curve) || at_floor || 1.5 <= order <= 2.5
                        push!(rows, (; arm, point = point_number, coordinate, precision,
                            historically_failed = coordinate in failed,
                            h = item.relative_step, relative_error = Float64(item.error),
                            rounding_floor = Float64(item.floor), order_to_next = Float64(order),
                            at_floor, trend_ok))
                    end
                end
                large = central(f, bigpoint, coordinate, BigFloat(1e-4)).derivative
                small = central(f, bigpoint, coordinate, BigFloat(5e-5)).derivative
                richardson = (4small - large) / 3
                error = relative_error(richardson, biggradient[coordinate])
                push!(extrapolations, (; arm, point = point_number, coordinate,
                    precision = "BigFloat128", h_large = 1e-4, h_small = 5e-5,
                    relative_error = Float64(error), passed = error <= BigFloat(1e-8)))
            end
        end
    end
    CSV.write(joinpath(output, "fd_convergence_$arm.csv"), DataFrames.DataFrame(rows))
    CSV.write(joinpath(output, "fd_richardson_$arm.csv"), DataFrames.DataFrame(extrapolations))
    CSV.write(joinpath(output, "fd_precision_comparison_$arm.csv"), DataFrames.DataFrame(comparisons))
    println("MARKET_FD arm=", arm, " historical_failures=", counts,
        " failed_union=", length(failed), " random_others=", length(sampled),
        " selected=", length(selected), " richardson_max=", maximum(r.relative_error for r in extrapolations),
        " high_precision_trend_failures=", count(r -> r.precision == "BigFloat128" && !r.trend_ok, rows))
    return (; rows, extrapolations, comparisons, counts)
end
end
