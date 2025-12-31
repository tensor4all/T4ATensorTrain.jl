using Test
using T4ATensorTrain
using LinearAlgebra
using Random

@testset "T4ATensorTrain.jl" begin
    @testset "Aqua" begin
        using Aqua
        Aqua.test_all(T4ATensorTrain; ambiguities=false, deps_compat=false)
    end

    @testset "JET" begin
        if VERSION >= v"1.9"
            using JET
            JET.test_package(T4ATensorTrain; target_defined_modules=true)
        end
    end

    include("test_tensortrain.jl")
    include("test_cachedtensortrain.jl")
    include("test_contraction.jl")
end
