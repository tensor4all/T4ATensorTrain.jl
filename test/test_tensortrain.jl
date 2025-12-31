import T4ATensorTrain as TT
import LinearAlgebra as LA
using Random

@testset "TensorTrain basic" begin
    @testset "TT creation and evaluation" begin
        Random.seed!(1234)
        T = Float64
        localdims = [2, 2, 2]
        linkdims_ = [1, 2, 3, 1]
        L = length(localdims)

        tt = TT.TensorTrain{T,3}([randn(T, linkdims_[n], localdims[n], linkdims_[n+1]) for n in 1:L])

        @test TT.rank(tt) == maximum(linkdims_[2:end-1])
        @test TT.linkdims(tt) == linkdims_[2:end-1]
        @test length(tt) == L

        # Test evaluation
        idx = [1, 2, 1]
        val = TT.evaluate(tt, idx)
        @test tt(idx) == val
    end

    @testset "TT compression" begin
        Random.seed!(1234)
        T = Float64
        N = 5
        localdims = fill(2, N)
        linkdims_ = vcat(1, fill(4, N - 1), 1)

        tt = TT.TensorTrain{T,3}([randn(T, linkdims_[n], localdims[n], linkdims_[n+1]) for n in 1:N])
        tt_orig = deepcopy(tt)

        for method in [:LU, :CI, :SVD]
            tt_compressed = deepcopy(tt)
            TT.compress!(tt_compressed, method; maxbonddim=2)
            @test TT.rank(tt_compressed) <= 2
        end
    end

    @testset "TT addition and multiplication" begin
        Random.seed!(10)
        T = Float64
        localdims = [2, 2, 2]
        linkdims_ = [1, 2, 3, 1]
        L = length(localdims)

        tt1 = TT.TensorTrain{T,3}([randn(T, linkdims_[n], localdims[n], linkdims_[n+1]) for n in 1:L])
        tt2 = TT.TensorTrain{T,3}([randn(T, linkdims_[n], localdims[n], linkdims_[n+1]) for n in 1:L])

        indices = [[i, j, k] for i in 1:2, j in 1:2, k in 1:2]
        ttadd = TT.add(tt1, tt2)
        @test ttadd.(indices) ≈ [tt1(v) + tt2(v) for v in indices]
        ttadd2 = tt1 + tt2
        @test ttadd2.(indices) ≈ [tt1(v) + tt2(v) for v in indices]

        tt1mul = 1.6 * tt1
        @test tt1mul.(indices) ≈ 1.6 .* tt1.(indices)

        tt1div = tt1mul / 3.2
        @test tt1div.(indices) ≈ tt1.(indices) ./ 2.0

        tt1sub = tt1 - tt1div
        @test tt1sub.(indices) ≈ tt1.(indices) ./ 2.0
    end

    @testset "TT norm" begin
        T = Float64
        sitedims_ = [[2], [2], [2]]
        N = length(sitedims_)
        bonddims = [1, 1, 1, 1]

        tt = TT.TensorTrain([
            ones(bonddims[n], sitedims_[n]..., bonddims[n+1]) for n in 1:N
        ])

        @test LA.norm2(tt) ≈ prod(only.(sitedims_))
        @test LA.norm2(2 * tt) ≈ 4 * prod(only.(sitedims_))
        @test LA.norm2(tt) ≈ LA.norm(tt)^2
    end

    @testset "TT fulltensor" begin
        Random.seed!(1234)
        T = Float64
        linkdims_ = [1, 2, 3, 1]
        L = length(linkdims_) - 1
        localdims = fill(4, L)

        tts = TT.TensorTrain{T,3}([randn(T, linkdims_[n], localdims[n], linkdims_[n+1]) for n in 1:L])

        # Reference implementation
        ref = [tts(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))]
        ref = reshape(ref, localdims...)

        @test ref ≈ TT.fulltensor(tts)
    end

    @testset "TT sum" begin
        Random.seed!(1234)
        T = Float64
        localdims = [2, 3, 2]
        linkdims_ = [1, 2, 2, 1]
        L = length(localdims)

        tt = TT.TensorTrain{T,3}([randn(T, linkdims_[n], localdims[n], linkdims_[n+1]) for n in 1:L])

        # Reference: sum all elements
        s = 0.0
        for i in CartesianIndices(Tuple(localdims))
            s += tt(collect(Tuple(i)))
        end

        @test s ≈ TT.sum(tt)
    end

    @testset "TT reverse" begin
        Random.seed!(1234)
        T = Float64
        localdims = [2, 3, 4]
        linkdims_ = [1, 2, 3, 1]
        L = length(localdims)

        tt = TT.TensorTrain{T,3}([randn(T, linkdims_[n], localdims[n], linkdims_[n+1]) for n in 1:L])
        ttr = reverse(tt)

        for i in CartesianIndices(Tuple(localdims))
            idx = collect(Tuple(i))
            @test tt(idx) ≈ ttr(reverse(idx))
        end
    end
end
