# Tensor train representations: Vidal, Inverse, and Site canonical forms
# These are moved from T4AMPOContractions.jl to T4ATensorTrain.jl

"""
    function isleftorthogonal(T::AbstractArray{ValueType,N}; atol::Float64=1e-7)::Bool where {ValueType, N}

Check if a tensor is left-orthogonal, i.e., contracting the first N-1 dimensions with their conjugates gives the identity.
"""
function isleftorthogonal(T::AbstractArray{ValueType,N}; atol::Float64=1e-7)::Bool where {ValueType, N}
    return isapprox(_contract(conj(T), T, Tuple(1:(N-1)), Tuple(1:(N-1))), LinearAlgebra.I; atol)
end

"""
    function isrightorthogonal(T::AbstractArray{ValueType,N}; atol::Float64=1e-7)::Bool where {ValueType, N}

Check if a tensor is right-orthogonal, i.e., contracting the last N-1 dimensions with their conjugates gives the identity.
"""
function isrightorthogonal(T::AbstractArray{ValueType,N}; atol::Float64=1e-7)::Bool where {ValueType, N}
    return isapprox(_contract(T, conj(T), Tuple(2:N), Tuple(2:N)), LinearAlgebra.I; atol)
end

"""
    function reshapephysicalright(T::AbstractArray{ValueType, N}) where {ValueType, N}

Reshape tensor to matrix with first dimension as rows and all other dimensions as columns.
"""
function reshapephysicalright(T::AbstractArray{ValueType, N}) where {ValueType, N}
    return reshape(T, first(Base.size(T)), :)
end

"""
    function reshapephysicalleft(T::AbstractArray{ValueType, N}) where {ValueType, N}

Reshape tensor to matrix with all dimensions except last as rows and last dimension as columns.
"""
function reshapephysicalleft(T::AbstractArray{ValueType, N}) where {ValueType, N}
    return reshape(T, :, last(Base.size(T)))
end

# VIDAL TENSOR TRAIN

mutable struct VidalTensorTrain{ValueType,N} <: AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}
    singularvalues::Vector{Matrix{Float64}}
    partition::UnitRange{Int}

    function VidalTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, singularvalues::AbstractVector{<:AbstractMatrix{Float64}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
        n = length(sitetensors)
        step(partition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
        first(partition) >= 1 && last(partition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))
        
        for i in first(partition):last(partition)-1
            if (last(Base.size(sitetensors[i])) != Base.size(sitetensors[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        for i in first(partition)+1:last(partition)-1
            if !isrightorthogonal(_contract(sitetensors[i], singularvalues[i], (N,), (1,)))
                throw(ArgumentError(
                    "Error: contracting the tensor at $i with the singular value at $i does not lead to a right-orthogonal tensor."
                ))
            end
            if !isleftorthogonal(_contract(singularvalues[i-1], sitetensors[i], (2,), (1,)))
                throw(ArgumentError(
                    "Error: contracting the singular value at $(i-1) with the tensor at $i does not lead to a left-orthogonal tensor."
                ))
            end
        end
        new{ValueType,N}(sitetensors, singularvalues, partition)
    end
end

function Base.show(io::IO, obj::VidalTensorTrain{ValueType,N}) where {ValueType,N}
    print(
        io,
        "$(typeof(obj)) of rank $(maximum(linkdims(obj)))"
    )
end

function VidalTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, partition::AbstractRange{<:Integer})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    # Minimal constructor: generate identity singular values consistent with adjacent bond dimensions.
    n = length(sitetensors)
    step(partition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
    first(partition) >= 1 && last(partition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))
    
    sitetensors = deepcopy(sitetensors)
    singularvalues = Vector{Matrix{Float64}}(undef, n-1)
    
    for i in first(partition):last(partition)-1
        Q, R = LinearAlgebra.qr(reshapephysicalleft(sitetensors[i]))
        sitetensors[i] = reshape(Matrix(Q), Base.size(sitetensors[i])...)
        sitetensors[i+1] = _contract(Matrix(R), sitetensors[i+1], (2,), (1,))
    end
    
    for i in last(partition):-1:first(partition)+1
        left, diamond, right, _ = _factorize(
            reshapephysicalright(sitetensors[i]),
            :SVD; tolerance=0.0, maxbonddim=first(Base.size(sitetensors[i])), diamond=true
        )
        # store as Float64 dense matrix
        singularvalues[i-1] = Matrix(LinearAlgebra.Diagonal(Float64.(diamond)))
        
        sitetensors[i] = reshape(right, Base.size(sitetensors[i])...)
        sitetensors[i-1] = _contract(sitetensors[i-1], left*singularvalues[i-1], (N,), (1,))
    end

    for i in first(partition):last(partition)-1
        d = LinearAlgebra.diag(singularvalues[i])
        sitetensors[i] = _contract(sitetensors[i], LinearAlgebra.Diagonal(1.0 ./ d), (N,), (1,))
    end
    return VidalTensorTrain{ValueType,N}(sitetensors, singularvalues, partition)
end

function VidalTensorTrain{ValueType,N}(tt::AbstractTensorTrain{ValueType}, partition::AbstractRange{<:Integer})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    return VidalTensorTrain{ValueType, N}(sitetensors(tt), partition)
end

function singularvalues(tt::VidalTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.singularvalues
end

function singularvalue(tt::VidalTensorTrain{ValueType, N}, i::Int) where {ValueType, N}
    return tt.singularvalues[i]
end

function partition(tt::VidalTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.partition
end

function setpartition!(tt::VidalTensorTrain{ValueType,N}, newpartition::AbstractRange{<:Integer}) where {ValueType,N}
    n = length(tt.sitetensors)

    step(newpartition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
    first(newpartition) >= 1 && last(newpartition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))
    for i in first(newpartition):last(newpartition)-1
        if (last(Base.size(tt.sitetensors[i])) != first(Base.size(tt.sitetensors[i+1])))
            throw(ArgumentError(
                "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
            ))
        end
    end

    for i in first(newpartition)+1:last(newpartition)-1
        if !isrightorthogonal(_contract(tt.sitetensors[i], tt.singularvalues[i], (N,), (1,)))
            throw(ArgumentError(
                "Error: contracting the tensor at $i with the singular value at $i does not lead to a right-orthogonal tensor."
            ))
        end
        if !isleftorthogonal(_contract(tt.singularvalues[i-1], tt.sitetensors[i], (2,), (1,)))
            throw(ArgumentError(
                "Error: contracting the singular value at $(i-1) with the tensor at $i does not lead to a left-orthogonal tensor."
            ))
        end
    end
    
    tt.partition = newpartition
end

function VidalTensorTrain{ValueType,N}(tt::AbstractTensorTrain{ValueType})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    return VidalTensorTrain{ValueType, N}(tt, 1:length(sitetensors(tt)))
end

function VidalTensorTrain(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    N = ndims(sitetensors(tt)[1])
    return VidalTensorTrain{ValueType, N}(tt)
end

function VidalTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    return VidalTensorTrain{ValueType, N}(sitetensors, 1:length(sitetensors))
end

function VidalTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, singularvalues::AbstractVector{<:AbstractMatrix{Float64}})::VidalTensorTrain{ValueType,N} where {ValueType, N}
    return VidalTensorTrain{ValueType,N}(sitetensors, singularvalues, 1:length(sitetensors))
end

function VidalTensorTrain{ValueType2,N}(tt::VidalTensorTrain{ValueType1,N})::VidalTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
    return VidalTensorTrain{ValueType2,N}(Array{ValueType2}.(sitetensors(tt)), Array{ValueType2}.(singularvalues(tt)))
end

function VidalTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(sitetensors)
end

function VidalTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(sitetensors, partition)
end

function VidalTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, singularvalues::AbstractArray{<:AbstractMatrix{ValueType}}) where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(sitetensors, singularvalues)
end

function VidalTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, singularvalues::AbstractArray{<:AbstractMatrix{ValueType}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return VidalTensorTrain{ValueType,N}(sitetensors, singularvalues, partition)
end

function VidalTensorTrain{ValueType2,N2}(tt::VidalTensorTrain{ValueType1,N1}, localdims)::VidalTensorTrain{ValueType2,N2} where {ValueType1,ValueType2,N1,N2}
    for d in localdims
        Base.length(d) == N2 - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(Base.size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    return VidalTensorTrain{ValueType2,N2}(
        [reshape(Array{ValueType2}(t), Base.size(t, 1), localdims[n]..., Base.size(t)[end]) for (n, t) in enumerate(sitetensors(tt))], Array{ValueType2}.(singularvalues(tt)), partition(tt)
    )
end

function VidalTensorTrain{N2}(tt::VidalTensorTrain{ValueType,N1}, localdims)::VidalTensorTrain{ValueType,N2} where {ValueType,N1,N2}
    return VidalTensorTrain{ValueType,N2}(tt, localdims)
end

function vidaltensortrain(a)
    return VidalTensorTrain(a)
end

function vidaltensortrain(a, b)
    return VidalTensorTrain(a, b)
end

function vidaltensortrain(a, b, c)
    return VidalTensorTrain(a, b, c)
end

# INVERSE TENSOR TRAIN

mutable struct InverseTensorTrain{ValueType,N} <: AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}
    inversesingularvalues::Vector{Matrix{Float64}}
    partition::UnitRange{Int}

    function InverseTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, inversesingularvalues::AbstractVector{<:AbstractMatrix{Float64}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
        n = length(sitetensors)
        step(partition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
        first(partition) >= 1 && last(partition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))

        for i in first(partition):last(partition)-1
            last(Base.size(sitetensors[i])) == Base.size(sitetensors[i+1], 1) || throw(ArgumentError(
                "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
            ))
        end

        new{ValueType,N}(sitetensors, inversesingularvalues, partition)
    end
end

function Base.show(io::IO, obj::InverseTensorTrain{ValueType,N}) where {ValueType,N}
    print(
        io,
        "$(typeof(obj)) of rank $(maximum(linkdims(obj)))"
    )
end

function InverseTensorTrain{ValueType,N}(tt::AbstractTensorTrain{ValueType}, partition::AbstractRange{<:Integer})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    if !isa(tt, VidalTensorTrain{ValueType,N})
        tt = VidalTensorTrain{ValueType,N}(tt, partition) # Convert with partition
    end
    n = length(tt)
    sitetensors = Vector{Array{ValueType, N}}(undef, n)
    inversesingularvalues = Vector{Matrix{Float64}}(undef, n-1)

    sitetensors[1] = _contract(sitetensor(tt, 1), singularvalue(tt, 1), (N,), (1,))
    for i in 2:n-1
        sitetensors[i] = _contract(singularvalue(tt, i-1), sitetensor(tt, i), (2,), (1,))
        sitetensors[i] = _contract(sitetensors[i], singularvalue(tt, i), (N,), (1,))
    end
    sitetensors[n] = _contract(singularvalue(tt, n-1), sitetensor(tt, n), (2,), (1,))

    for i in 1:n-1
        d = LinearAlgebra.diag(singularvalue(tt, i))
        inversesingularvalues[i] = Matrix(LinearAlgebra.Diagonal(1.0 ./ d))
    end
    return InverseTensorTrain{ValueType,N}(sitetensors, inversesingularvalues, partition)
end

function InverseTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,4}}, partition::AbstractRange{<:Integer})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    return InverseTensorTrain{ValueType,N}(VidalTensorTrain{ValueType,N}(sitetensors), partition)
end

function InverseTensorTrain{ValueType2,N}(tt::InverseTensorTrain{ValueType1,N})::InverseTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
    return InverseTensorTrain{ValueType2,N}(Array{ValueType2}.(sitetensors(tt)), inversesingularvalues(tt), partition(tt))
end

function setpartition!(tt::InverseTensorTrain{ValueType,N}, newpartition::AbstractRange{<:Integer}) where {ValueType,N}
    n = length(tt.sitetensors)
    step(newpartition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
    first(newpartition) >= 1 && last(newpartition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))
    for i in first(newpartition):last(newpartition)-1
        last(Base.size(tt.sitetensors[i])) == first(Base.size(tt.sitetensors[i+1])) || throw(ArgumentError("Bond dimensions between site $i and $(i+1) mismatch."))
    end
    tt.partition = newpartition
end

function inversesingularvalues(tt::InverseTensorTrain{ValueType, N})::AbstractVector{<:AbstractMatrix{Float64}} where {ValueType, N}
    return tt.inversesingularvalues
end

function inversesingularvalue(tt::InverseTensorTrain{ValueType, N}, i::Int)::AbstractMatrix{Float64} where {ValueType, N}
    return tt.inversesingularvalues[i]
end

function partition(tt::InverseTensorTrain{ValueType, N})::AbstractRange{<:Integer} where {ValueType, N}
    return tt.partition
end

function settwositetensors!(tt::InverseTensorTrain{ValueType,N}, i::Int, tensor1::AbstractArray{ValueType,N}, matrix::AbstractMatrix{Float64}, tensor2::AbstractArray{ValueType,N}) where {ValueType,N}
    tt.sitetensors[i] = tensor1
    tt.inversesingularvalues[i] = matrix
    tt.sitetensors[i+1] = tensor2
end

function InverseTensorTrain{ValueType,N}(tt::AbstractTensorTrain{ValueType})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    n = length(tt)
    return InverseTensorTrain{ValueType, N}(tt, 1:n)
end

function InverseTensorTrain(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    N = ndims(sitetensors(tt)[1])
    return InverseTensorTrain{ValueType, N}(tt)
end

function InverseTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    n = length(sitetensors)
    return InverseTensorTrain{ValueType, N}(sitetensors, 1:n)
end

function InverseTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, inversesingularvalues::AbstractVector{<:AbstractMatrix{Float64}})::InverseTensorTrain{ValueType,N} where {ValueType, N}
    n = length(sitetensors)
    return InverseTensorTrain{ValueType,N}(sitetensors, inversesingularvalues, 1:n)
end

function InverseTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(sitetensors)
end

function InverseTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(sitetensors, partition)
end

function InverseTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, inversesingularvalues::AbstractArray{<:AbstractMatrix{ValueType}}) where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(sitetensors, inversesingularvalues)
end

function InverseTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, inversesingularvalues::AbstractArray{<:AbstractMatrix{ValueType}}, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return InverseTensorTrain{ValueType,N}(sitetensors, inversesingularvalues, partition)
end

function InverseTensorTrain{ValueType2,N2}(tt::InverseTensorTrain{ValueType1,N1}, localdims)::InverseTensorTrain{ValueType2,N2} where {ValueType1,ValueType2,N1,N2}
    for d in localdims
        Base.length(d) == N2 - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(Base.size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    return InverseTensorTrain{ValueType2,N2}(
        [reshape(Array{ValueType2}(t), Base.size(t, 1), localdims[n]..., Base.size(t)[end]) for (n, t) in enumerate(sitetensors(tt))], inversesingularvalues(tt), partition(tt)
    )
end

function InverseTensorTrain{N2}(tt::InverseTensorTrain{ValueType,N1}, localdims)::InverseTensorTrain{ValueType,N2} where {ValueType,N1,N2}
    return InverseTensorTrain{ValueType,N2}(tt, localdims)
end

function inversetensortrain(a)
    return InverseTensorTrain(a)
end

function inversetensortrain(a, b)
    return InverseTensorTrain(a, b)
end

function inversetensortrain(a, b, c)
    return InverseTensorTrain(a, b, c)
end

# SITE TENSOR TRAIN

function centercanonicalize(sitetensors::Vector{Array{ValueType, N}}, center::Int) where {ValueType, N}
    sitetensors_copy = deepcopy(sitetensors)
    centercanonicalize!(sitetensors_copy, center)
    return sitetensors_copy
end

function centercanonicalize!(sitetensors::Vector{Array{ValueType, N}}, center::Int) where {ValueType, N}
    # LEFT
    for i in 1:center-1
        Q, R = LinearAlgebra.qr(reshape(sitetensors[i], prod(Base.size(sitetensors[i])[1:end-1]), Base.size(sitetensors[i])[end]))
        Q = Matrix(Q)

        sitetensors[i] = reshape(Q, Base.size(sitetensors[i])[1:end-1]..., Base.size(Q, 2))

        tmptt = reshape(sitetensors[i+1], Base.size(R, 2), :)
        tmptt = Matrix(R) * tmptt
        sitetensors[i+1] = reshape(tmptt, Base.size(sitetensors[i+1])...)
    end
    # RIGHT
    for i in length(sitetensors):-1:center+1
        W = sitetensors[i]
        dims = Base.size(W)
        bonddim_left = dims[1]
        bonddim_right = dims[end]
        W_mat = reshape(W, bonddim_left, prod(dims[2:end-1]) * bonddim_right)

        L, Q = LinearAlgebra.lq(W_mat)
        Q = Matrix(Q)
        
        sitetensors[i] = reshape(Q, Base.size(Q, 1), dims[2:end-1]..., bonddim_right)

        tmptt = reshape(sitetensors[i-1], :, Base.size(L, 1))
        tmptt = tmptt * Matrix(L)
        sitetensors[i-1] = reshape(tmptt, Base.size(sitetensors[i-1], 1), dims[2:end-1]..., Base.size(tmptt, 2))
    end
end

mutable struct SiteTensorTrain{ValueType,N} <: AbstractTensorTrain{ValueType}
    sitetensors::Vector{Array{ValueType,N}}
    center::Int
    partition::UnitRange{Int}

    function SiteTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, center::Int, partition::AbstractRange{<:Integer}) where {ValueType,N}
        n = length(sitetensors)
        step(partition) == 1 || throw(ArgumentError("partition must be a contiguous range (step 1)"))
        first(partition) >= 1 && last(partition) <= n || throw(ArgumentError("All partition indices must be between 1 and $n"))
        center >= first(partition) && center <= last(partition) || throw(ArgumentError("center ($center) must lie within partition $partition"))

        for i in first(partition):last(partition)-1
            if (last(Base.size(sitetensors[i])) != Base.size(sitetensors[i+1], 1))
                throw(ArgumentError(
                    "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
                ))
            end
        end

        all_left_orth = true
        all_right_orth = true
        
        for i in first(partition):center-1
            if !isleftorthogonal(sitetensors[i])
                all_left_orth = false
            end
        end
        
        for i in center+1:last(partition)
            if !isrightorthogonal(sitetensors[i])
                all_right_orth = false
            end
        end
        
        if !(all_left_orth && all_right_orth)
            sitetensors = centercanonicalize(sitetensors, center)
        end

        new{ValueType,N}(sitetensors, center, partition)
    end

    # This is to make JET compile, actually implement this
    function SiteTensorTrain{ValueType,N}(sitetensors, center, partition) where {ValueType,N}
        new{ValueType,N}(sitetensors, center, partition)
    end
end

# Simple partition setter (no re-orthogonalization; assumes tensors already consistent)
function setpartition!(tt::SiteTensorTrain{ValueType,N}, newpartition::AbstractRange{Int}) where {ValueType,N}
    n = length(tt.sitetensors)
    step(newpartition) == 1 || throw(ArgumentError("partition must be contiguous (step=1)"))
    first(newpartition) >= 1 && last(newpartition) <= n || throw(ArgumentError("partition indices must lie within 1:$n"))
    for i in first(newpartition):last(newpartition)-1
        Base.size(tt.sitetensors[i], N) == Base.size(tt.sitetensors[i+1], 1) || throw(ArgumentError("Bond dimension mismatch between sites $i and $(i+1)"))
    end
    tt.partition = newpartition
end

function SiteTensorTrain{ValueType,N}(tt::AbstractTensorTrain{ValueType}, center::Int, partition::AbstractRange{<:Integer})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(sitetensors(tt), center, partition)
end

function Base.show(io::IO, obj::SiteTensorTrain{ValueType,N}) where {ValueType,N}
    print(io, "$(typeof(obj)) of rank $(maximum(linkdims(obj))) centered at $(obj.center)")
end

function center(tt::SiteTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.center
end

function partition(tt::SiteTensorTrain{ValueType, N}) where {ValueType, N}
    return tt.partition
end

function settwositetensors!(tt::SiteTensorTrain{ValueType,N}, i::Int, tensor1::AbstractArray{ValueType,N}, tensor2::AbstractArray{ValueType,N}) where {ValueType,N}
    tt.sitetensors[i] = tensor1
    tt.sitetensors[i+1] = tensor2

    if center(tt) == i || center(tt) == i + 1
        if isleftorthogonal(tensor1) && !isleftorthogonal(tensor2) && !isrightorthogonal(tensor2)
            tt.center = i + 1
        elseif isrightorthogonal(tensor2) && !isleftorthogonal(tensor1) && !isrightorthogonal(tensor1)
            tt.center = i
        else
            throw(ArgumentError("Error inserting at $i,$(i+1). [L, R]: [$(isleftorthogonal(tensor1)), $(isrightorthogonal(tensor1))], [$(isleftorthogonal(tensor2)), $(isrightorthogonal(tensor2))]"))
        end
        return
    end

    if i < center(tt) && i in partition(tt)
        isleftorthogonal(tensor1) || throw(ArgumentError("The tensor at $i must be left-orthogonal."))
    elseif i > center(tt) && i in partition(tt)
        isrightorthogonal(tensor1) || throw(ArgumentError("The tensor at $(i+1) must be right-orthogonal."))
    end

    if i+1 < center(tt) && i+1 in partition(tt)
        isleftorthogonal(tensor2) || throw(ArgumentError("The tensor at $(i+1) must be left-orthogonal."))
    elseif i+1 > center(tt) && i+1 in partition(tt)
        isrightorthogonal(tensor2) || throw(ArgumentError("The tensor at $(i+1) must be right-orthogonal."))
    end

    if i >= first(partition(tt)) + 1 && i <= last(partition(tt))
        last(Base.size(sitetensor(tt, i-1))) == first(Base.size(sitetensor(tt, i))) || throw(ArgumentError(
            "The tensors at $(i-1) and $i must have consistent dimensions for a tensor train."
        ))
    end
    if i >= first(partition(tt)) && i <= last(partition(tt)) - 1
        last(Base.size(sitetensor(tt, i))) == first(Base.size(sitetensor(tt, i+1))) || throw(ArgumentError(
            "The tensors at $i and $(i+1) must have consistent dimensions for a tensor train."
        ))
    end
    if i >= first(partition(tt)) - 1 && i <= last(partition(tt)) - 2
        last(Base.size(sitetensor(tt, i+1))) == first(Base.size(sitetensor(tt, i+2))) || throw(ArgumentError(
            "The tensors at $(i+1) and $(i+2) must have consistent dimensions for a tensor train."
        ))
    end
end

function setcenter!(tt::SiteTensorTrain{ValueType,N}, newcenter::Int) where {ValueType,N}
    if newcenter < first(partition(tt)) || newcenter > last(partition(tt))
        throw(ArgumentError("newcenter ($newcenter) must lie within partition $(partition(tt))"))
    end
    diff = newcenter - center(tt)
    if diff < 0
        for c in (center(tt)-1):-1:newcenter
            movecenterleft!(tt)
        end
    elseif diff > 0
        for c in (center(tt)+1):newcenter
            movecenterright!(tt)
        end
    end
end

function SiteTensorTrain{ValueType,N}(tt::AbstractTensorTrain{ValueType}, center::Int)::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt)
    return SiteTensorTrain{ValueType,N}(tt, center, 1:n)
end

function SiteTensorTrain{ValueType,N}(tt::AbstractTensorTrain{ValueType})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt)
    return SiteTensorTrain{ValueType,N}(tt, 1, 1:n)
end

function SiteTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, center::Int)::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(sitetensors)
    return SiteTensorTrain{ValueType,N}(sitetensors, center, 1:n)
end

function SiteTensorTrain{ValueType,N}(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(sitetensors)
    return SiteTensorTrain{ValueType,N}(sitetensors, 1, 1:n)
end

# Default constructor: center at 1
function SiteTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}) where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(sitetensors)
end

function SiteTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, center::Int) where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(sitetensors, center)
end

function SiteTensorTrain(sitetensors::AbstractVector{<:AbstractArray{ValueType,N}}, center::Int, partition::AbstractRange{<:Integer}) where {ValueType,N}
    return SiteTensorTrain{ValueType,N}(sitetensors, center, partition)
end

# Construct from an AbstractTensorTrain, default center = 1
function SiteTensorTrain(tt::AbstractTensorTrain{ValueType}) where {ValueType}
    return SiteTensorTrain{ValueType, ndims(sitetensors(tt)[1])}(tt)
end

function SiteTensorTrain(tt::AbstractTensorTrain{ValueType}, center::Int) where {ValueType}
    return SiteTensorTrain{ValueType, ndims(sitetensors(tt)[1])}(tt, center)
end

function SiteTensorTrain(tt::AbstractTensorTrain{ValueType}, center::Int, partition::AbstractRange{<:Integer}) where {ValueType}
    return SiteTensorTrain{ValueType, ndims(sitetensors(tt)[1])}(tt, center, partition)
end

# Convert InverseTensorTrain to SiteTensorTrain
function SiteTensorTrain{ValueType,N}(tt::InverseTensorTrain{ValueType,N}, center::Int, partition::AbstractRange{<:Integer})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    # Convert InverseTensorTrain to TensorTrain first, then to SiteTensorTrain
    tt_tensor = TensorTrain{ValueType,N}(tt)
    return SiteTensorTrain{ValueType,N}(tt_tensor, center, partition)
end

function SiteTensorTrain{ValueType,N}(tt::InverseTensorTrain{ValueType,N}, center::Int)::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt)
    return SiteTensorTrain{ValueType,N}(tt, center, 1:n)
end

function SiteTensorTrain{ValueType,N}(tt::InverseTensorTrain{ValueType,N})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt)
    return SiteTensorTrain{ValueType,N}(tt, 1, 1:n)
end

# Base.convert for InverseTensorTrain to SiteTensorTrain
function Base.convert(::Type{SiteTensorTrain{ValueType,N}}, tt::InverseTensorTrain{ValueType,N})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt)
    return SiteTensorTrain{ValueType,N}(tt, 1, 1:n)
end

# Type conversion: change element type of a SiteTensorTrain
function SiteTensorTrain{ValueType2,N}(tt::SiteTensorTrain{ValueType1,N})::SiteTensorTrain{ValueType2,N} where {ValueType1,ValueType2,N}
    return SiteTensorTrain{ValueType2,N}(Array{ValueType2}.(sitetensors(tt)), center(tt), partition(tt))
end

# Construct from an AbstractTensorTrain and reshape according to localdims
function SiteTensorTrain{ValueType2,N2}(tt::SiteTensorTrain{ValueType1,N1}, localdims)::SiteTensorTrain{ValueType2,N2} where {ValueType1,ValueType2,N1,N2}
    for d in localdims
        Base.length(d) == N2 - 2 || error("Each element of localdims be a list of N-2 integers.")
    end
    for n in 1:length(tt)
        prod(Base.size(tt[n])[2:end-1]) == prod(localdims[n]) || error("The local dimensions at n=$n must match the tensor sizes.")
    end
    return SiteTensorTrain{ValueType2,N2}([reshape(Array{ValueType2}(t), Base.size(t, 1), localdims[n]..., Base.size(t)[end]) for (n, t) in enumerate(sitetensors(tt))], center(tt), partition(tt))
end

# Generic wrapper for specifying localdims without explicit type parameter
function SiteTensorTrain{N2}(tt::SiteTensorTrain{ValueType,N1}, localdims)::SiteTensorTrain{ValueType,N2} where {ValueType,N1,N2}
    return SiteTensorTrain{ValueType,N2}(tt, localdims)
end

# Convenience wrapper names
function sitetensortrain(a)
    return SiteTensorTrain(a)
end

function sitetensortrain(a, b)
    return SiteTensorTrain(a, b)
end

function sitetensortrain(a, b, c)
    return SiteTensorTrain(a, b, c)
end

function movecenterright!(tt::SiteTensorTrain{ValueType,N}) where {ValueType,N}
    c = center(tt)
    if c >= last(tt.partition)
        throw(ArgumentError("Cannot move center right: already at the rightmost position of partition"))
    end
    
    # QR decomposition of current center tensor
    T = sitetensor(tt, c)
    Q, R = LinearAlgebra.qr(reshape(T, prod(Base.size(T)[1:end-1]), Base.size(T)[end]))
    Q = Matrix(Q)
    
    # Update current center tensor with Q
    tt.sitetensors[c] = reshape(Q, Base.size(T)[1:end-1]..., Base.size(Q, 2))
    
    # Contract R into the next tensor
    tmptt = reshape(sitetensor(tt, c+1), Base.size(R, 2), :)
    tmptt = Matrix(R) * tmptt
    tt.sitetensors[c+1] = reshape(tmptt, Base.size(sitetensor(tt, c+1))...)
    
    # Move center to the right
    tt.center = c + 1
end

function movecenterleft!(tt::SiteTensorTrain{ValueType,N}) where {ValueType,N}
    c = center(tt)
    if c <= first(tt.partition)
        throw(ArgumentError("Cannot move center left: already at the leftmost position of partition"))
    end
    
    # LQ decomposition of current center tensor
    T = sitetensor(tt, c)
    dims = Base.size(T)
    bonddim_left = dims[1]
    bonddim_right = dims[end]
    physical_dims = dims[2:end-1]
    T_mat = reshape(T, bonddim_left, prod(physical_dims) * bonddim_right)
    
    L, Q = LinearAlgebra.lq(T_mat)
    Q = Matrix(Q)
    
    # Update current center tensor with Q
    tt.sitetensors[c] = reshape(Q, Base.size(Q, 1), physical_dims..., bonddim_right)
    
    # Contract L into the previous tensor
    prev_dims = Base.size(sitetensor(tt, c-1))
    tmptt = reshape(sitetensor(tt, c-1), :, Base.size(L, 1))
    tmptt = tmptt * Matrix(L)
    tt.sitetensors[c-1] = reshape(tmptt, prev_dims[1], physical_dims..., Base.size(tmptt, 2))
    
    # Move center to the left
    tt.center = c - 1
end

function movecenterleft(tt::SiteTensorTrain{ValueType,N})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    tt_copy = deepcopy(tt)
    movecenterleft!(tt_copy)
    return tt_copy
end

function movecenterright(tt::SiteTensorTrain{ValueType,N})::SiteTensorTrain{ValueType,N} where {ValueType,N}
    tt_copy = deepcopy(tt)
    movecenterright!(tt_copy)
    return tt_copy
end

function movecenterto!(tt::SiteTensorTrain{ValueType,N}, newcenter::Int) where {ValueType,N}
    while(center(tt) < newcenter)
         movecenterright!(tt)
    end
    while(center(tt) > newcenter)
         movecenterleft!(tt)
    end
end

function setsitetensor!(tt::SiteTensorTrain{ValueType,N}, i::Int, tensor::AbstractArray{ValueType,N}) where {ValueType,N}
    if i in partition(tt) && i < center(tt)
        if !isleftorthogonal(tensor)
            throw(ArgumentError("The tensor at site $i must be left-orthogonal."))
        end
    end
    if i in partition(tt) && i > center(tt)
        if !isrightorthogonal(tensor)
            throw(ArgumentError("The tensor at site $i must be right-orthogonal."))
        end
    end
    tt.sitetensors[i] = tensor
end

# Conversion functions from tensor train representations to TensorTrain

function TensorTrain{ValueType,N}(tt::InverseTensorTrain{ValueType,N})::TensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt.sitetensors)
    sitetensors = [_contract(sitetensor(tt, i), inversesingularvalue(tt, i), (N,), (1,)) for i in 1:n-1]
    push!(sitetensors, sitetensor(tt, n))
    return TensorTrain{ValueType,N}(sitetensors)
end

function TensorTrain{ValueType,N}(tt::VidalTensorTrain{ValueType,N})::TensorTrain{ValueType,N} where {ValueType,N}
    n = length(tt.sitetensors)
    sitetensors = [_contract(sitetensor(tt, i), singularvalue(tt, i), (N,), (1,)) for i in 1:n-1]
    push!(sitetensors, sitetensor(tt, n))
    return TensorTrain{ValueType,N}(sitetensors)
end

function TensorTrain{ValueType,N}(tt::SiteTensorTrain{ValueType,N})::TensorTrain{ValueType,N} where {ValueType,N}
    return TensorTrain{ValueType,N}(sitetensors(tt))
end

function TensorTrain(tt::SiteTensorTrain{ValueType,N})::TensorTrain{ValueType,N} where {ValueType,N}
    return TensorTrain{ValueType,N}(sitetensors(tt))
end

function TensorTrain(tt::InverseTensorTrain{ValueType,N})::TensorTrain{ValueType,N} where {ValueType,N}
    return TensorTrain{ValueType,N}(tt)
end

