import T4ATensorTrain as TT
using Random

@testset "CachedTensorTrain" begin
    @testset "TTCache basic" begin
        ValueType = Float64

        N = 4
        bonddims = [1, 2, 3, 2, 1]
        localdims = [2, 3, 3, 2]

        tt = TT.TensorTrain{ValueType,3}([rand(bonddims[n], localdims[n], bonddims[n+1]) for n in 1:N])
        ttc = TT.TTCache(tt)

        @test [tt(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))] ≈
              [ttc(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))]
    end

    @testset "TTCache batchevaluate" begin
        N = 4
        bonddims = [1, 2, 3, 2, 1]
        A = TT.TTCache([rand(bonddims[n], 2, bonddims[n+1]) for n in 1:N])

        leftindexset = [[1], [2]]
        rightindexset = [[1], [2]]

        result = A(leftindexset, rightindexset, Val(2))
        for cindex in [[1, 1], [1, 2]]
            for (il, lindex) in enumerate(leftindexset)
                for (ir, rindex) in enumerate(rightindexset)
                    @test result[il, cindex..., ir] ≈ TT.evaluate(A, vcat(lindex, cindex, rindex))
                end
            end
        end
    end

    @testset "TTCache with multi site indices" begin
        ValueType = Float64

        N = 4
        bonddims = [1, 2, 3, 2, 1]
        localdims = [4, 4, 4, 4]
        sitedims = [[2, 2] for _ in 1:N]

        tt = TT.TensorTrain{ValueType,3}([rand(bonddims[n], localdims[n], bonddims[n+1]) for n in 1:N])
        ttc = TT.TTCache(tt, sitedims)

        ref = [tt(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))]
        rest = [ttc([[i1, j1], [i2, j2], [i3, j3], [i4, j4]])
                for i1 in 1:2, j1 in 1:2, i2 in 1:2, j2 in 1:2, i3 in 1:2, j3 in 1:2, i4 in 1:2, j4 in 1:2]
        @test vec(ref) ≈ vec(rest)
    end
end
