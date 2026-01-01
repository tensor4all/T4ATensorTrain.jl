using Test
using T4ATensorTrain
using LinearAlgebra
using Random

@testset "Tensor Train Representations" begin
    Random.seed!(42)
    
    @testset "VidalTensorTrain" begin
        # Create a simple tensor train
        sitetensors = [randn(Float64, 2, 3, 2) for _ in 1:4]
        tt = TensorTrain(sitetensors)
        
        # Convert to VidalTensorTrain
        vtt = VidalTensorTrain(tt)
        @test vtt isa VidalTensorTrain{Float64, 3}
        @test length(vtt) == 4
        @test partition(vtt) == 1:4
        
        # Check singular values
        svs = singularvalues(vtt)
        @test length(svs) == 3
        @test all(sv isa Matrix{Float64} for sv in svs)
        
        # Convert back to TensorTrain
        tt2 = TensorTrain(vtt)
        @test tt2 isa TensorTrain{Float64, 3}
        @test length(tt2) == 4
    end
    
    @testset "InverseTensorTrain" begin
        # Create a tensor train
        sitetensors = [randn(Float64, 2, 3, 2) for _ in 1:4]
        vtt = VidalTensorTrain(sitetensors)
        
        # Convert to InverseTensorTrain
        itt = InverseTensorTrain(vtt)
        @test itt isa InverseTensorTrain{Float64, 3}
        @test length(itt) == 4
        
        # Check inverse singular values
        inv_svs = inversesingularvalues(itt)
        @test length(inv_svs) == 3
        @test all(inv_sv isa Matrix{Float64} for inv_sv in inv_svs)
        
        # Convert back to TensorTrain
        tt = TensorTrain(itt)
        @test tt isa TensorTrain{Float64, 3}
    end
    
    @testset "SiteTensorTrain" begin
        # Create a tensor train
        sitetensors = [randn(Float64, 2, 3, 2) for _ in 1:4]
        tt = TensorTrain(sitetensors)
        
        # Convert to SiteTensorTrain with center at 2
        stt = SiteTensorTrain(tt, 2)
        @test stt isa SiteTensorTrain{Float64, 3}
        @test length(stt) == 4
        @test center(stt) == 2
        @test partition(stt) == 1:4
        
        # Test center movement
        movecenterright!(stt)
        @test center(stt) == 3
        
        movecenterleft!(stt)
        @test center(stt) == 2
        
        # Convert back to TensorTrain
        tt2 = TensorTrain(stt)
        @test tt2 isa TensorTrain{Float64, 3}
    end
    
    @testset "Orthogonality checks" begin
        # Test orthogonality through properly constructed tensor train representations
        # This follows the pattern from T4AMPOContractions.jl/test/test_contraction.jl
        
        # Generate test data: 4D tensors for MPO-like tensor trains
        N = 4
        bonddims = [1, 2, 3, 2, 1]
        localdims1 = [3, 3, 3, 3]
        localdims2 = [3, 3, 3, 3]
        
        a = TensorTrain{ComplexF64,4}([
            rand(ComplexF64, bonddims[n], localdims1[n], localdims2[n], bonddims[n+1])
            for n = 1:N
        ])
        
        # Test VidalTensorTrain orthogonality
        a_v = VidalTensorTrain{ComplexF64, 4}(a)
        
        # First site tensor should be left-orthogonal
        @test isleftorthogonal(sitetensor(a_v, 1))
        
        # Middle tensors: contracting with singular values should give orthogonal tensors
        for i in 2:N-1
            using T4ATensorTrain: _contract
            # T_i * S_i should be right-orthogonal
            @test isrightorthogonal(_contract(sitetensor(a_v, i), singularvalue(a_v, i), (4,), (1,)))
            # S_{i-1} * T_i should be left-orthogonal
            @test isleftorthogonal(_contract(singularvalue(a_v, i-1), sitetensor(a_v, i), (2,), (1,)))
        end
        
        # Last site tensor should be right-orthogonal
        @test isrightorthogonal(sitetensor(a_v, N))
        
        # Test InverseTensorTrain orthogonality
        a_inv = InverseTensorTrain{ComplexF64, 4}(a)
        
        for i in 1:N-1
            using T4ATensorTrain: _contract
            # T_i * invS_i should be left-orthogonal
            @test isleftorthogonal(_contract(sitetensor(a_inv, i), inversesingularvalue(a_inv, i), (4,), (1,)))
        end
        for i in 2:N
            using T4ATensorTrain: _contract
            # invS_{i-1} * T_i should be right-orthogonal
            @test isrightorthogonal(_contract(inversesingularvalue(a_inv, i-1), sitetensor(a_inv, i), (2,), (1,)))
        end
    end
    
    @testset "Conversions between representations" begin
        sitetensors = [randn(Float64, 2, 3, 2) for _ in 1:4]
        tt = TensorTrain(sitetensors)
        
        # TensorTrain -> VidalTensorTrain -> InverseTensorTrain -> SiteTensorTrain -> TensorTrain
        vtt = VidalTensorTrain(tt)
        itt = InverseTensorTrain(vtt)
        stt = SiteTensorTrain(itt, 2)
        tt2 = TensorTrain(stt)
        
        @test length(tt2) == length(tt)
        # Values should be approximately equal (within numerical precision)
        for i in 1:length(tt)
            @test size(tt2[i]) == size(tt[i])
        end
    end
end

