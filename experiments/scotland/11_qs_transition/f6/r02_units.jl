# Existing suites plus dedicated fold-cadence / same-state assertions.
using Test, CSV, DataFrames
root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
result = @testset "F6 scoped clock and QS regression gates" begin
    include(joinpath(root,"test","scoped_splitter_tests.jl"))
    include(joinpath(root,"test","test_quality_style_grw.jl"))
    include(joinpath(root,"test","test_quality_style_transition.jl"))
end
counts = Test.get_test_counts(result)
CSV.write(joinpath(@__DIR__,"unit_gates.csv"), DataFrame([(; pass=true, counts=repr(counts))]))
println("F6_UNITS_PASS ",repr(counts))
