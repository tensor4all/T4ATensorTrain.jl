import T4ATensorTrain as TT
using Random

function _tomat(tto::TT.TensorTrain{T,4}) where {T}
    sitedims = TT.sitedims(tto)
    localdims1 = [s[1] for s in sitedims]
    localdims2 = [s[2] for s in sitedims]
    mat = Matrix{T}(undef, prod(localdims1), prod(localdims2))
    for (i, inds1) in enumerate(CartesianIndices(Tuple(localdims1)))
        for (j, inds2) in enumerate(CartesianIndices(Tuple(localdims2)))
            mat[i, j] = TT.evaluate(tto, collect(zip(Tuple(inds1), Tuple(inds2))))
        end
    end
    return mat
end

@testset "Contraction" begin
    @testset "_contract" begin
        a = rand(2, 3, 4)
        b = rand(2, 5, 4)
        ab = TT._contract(a, b, (1, 3), (1, 3))
        @test vec(reshape(permutedims(a, (2, 1, 3)), 3, :) * reshape(permutedims(b, (1, 3, 2)), :, 5)) ≈ vec(ab)
    end

    @testset "MPO-MPO contraction (naive)" begin
        Random.seed!(1234)
        N = 4
        bonddims_a = [1, 2, 3, 2, 1]
        bonddims_b = [1, 2, 3, 2, 1]
        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        localdims3 = [2, 2, 2, 2]

        a = TT.TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
            for n = 1:N
        ])
        b = TT.TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
            for n = 1:N
        ])

        ab = TT.contract(a, b; algorithm=:naive)
        @test TT.sitedims(ab) == [[localdims1[i], localdims3[i]] for i = 1:N]
        @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
    end

    @testset "MPO-MPO contraction (zipup)" begin
        Random.seed!(1234)
        N = 4
        bonddims_a = [1, 2, 3, 2, 1]
        bonddims_b = [1, 2, 3, 2, 1]
        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        localdims3 = [2, 2, 2, 2]

        a = TT.TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1])
            for n = 1:N
        ])
        b = TT.TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1])
            for n = 1:N
        ])

        for method in [:SVD, :LU]
            ab = TT.contract(a, b; algorithm=:zipup, method=method)
            @test _tomat(ab) ≈ _tomat(a) * _tomat(b)
        end
    end

    @testset "Algorithm dispatch" begin
        @test TT.Algorithm(:naive) isa TT.Algorithm{:naive}
        @test TT.Algorithm("zipup") isa TT.Algorithm{:zipup}
        @test TT.Algorithm"naive" == TT.Algorithm{:naive}
    end
end
