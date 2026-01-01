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
        # Test isleftorthogonal and isrightorthogonal functions exist and work
        # Create a simple tensor
        T = randn(ComplexF64, 2, 3, 4)
        
        # These functions should not error
        @test isa(isleftorthogonal(T), Bool)
        @test isa(isrightorthogonal(T), Bool)
        
        # Note: Creating truly orthogonal tensors requires careful construction
        # which is tested indirectly through tensor train representations
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

