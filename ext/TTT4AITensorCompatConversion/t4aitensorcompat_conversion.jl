"""
    T4AITensorCompat.TensorTrain(tt::TT.TensorTrain{ValueType,N}; sites=nothing)

Convert a `T4ATensorTrain.TensorTrain` to a `T4AITensorCompat.TensorTrain`.

# Arguments
- `tt::TT.TensorTrain{ValueType,N}`: The tensor train to convert
- `sites`: Optional site indices. If `nothing`, default indices will be generated.

# Returns
- `T4AITensorCompat.TensorTrain`: The converted tensor train
"""
function T4AITensorCompat.TensorTrain(tt::TT.TensorTrain{ValueType,N}; sites=nothing) where {ValueType,N}
    N_sites = length(tt)
    localdims = TT.sitedims(tt)

    # Generate site indices if not provided
    if sites === nothing
        if N == 3  # MPS case: one physical index per site
            sites = [Index(localdims[n][1], "n=$n") for n in 1:N_sites]
        else  # MPO case: multiple physical indices per site
            sites = [
                [Index(localdims[n][ell], "ell=$ell, n=$n") for ell in 1:length(localdims[n])]
                for n in 1:N_sites
            ]
        end
    end

    # Generate link indices
    linkdims = [1, TT.linkdims(tt)..., 1]
    links = [Index(linkdims[l + 1], "link,l=$l") for l in 0:N_sites]

    # Convert each tensor to ITensor
    tensors = Vector{ITensor}(undef, N_sites)
    for n in 1:N_sites
        core = tt[n]
        if N == 3  # MPS case
            tensors[n] = ITensor(core, links[n], sites[n], links[n + 1])
        else  # MPO case
            tensors[n] = ITensor(core, links[n], sites[n]..., links[n + 1])
        end
    end

    # Set boundary conditions
    tensors[1] *= onehot(links[1] => 1)
    tensors[end] *= onehot(links[end] => 1)

    return T4AITensorCompat.TensorTrain(tensors)
end

"""
    T4AITensorCompat.TensorTrain(tt::TT.AbstractTensorTrain{ValueType}; sites=nothing)

Convert an `AbstractTensorTrain` to a `T4AITensorCompat.TensorTrain` by first converting to `TensorTrain`.

# Arguments
- `tt::TT.AbstractTensorTrain{ValueType}`: The tensor train to convert
- `sites`: Optional site indices. If `nothing`, default indices will be generated.

# Returns
- `T4AITensorCompat.TensorTrain`: The converted tensor train
"""
function T4AITensorCompat.TensorTrain(tt::TT.AbstractTensorTrain{ValueType}; sites=nothing) where {ValueType}
    return T4AITensorCompat.TensorTrain(TT.tensortrain(tt); sites=sites)
end

"""
    TT.TensorTrain(tt::T4AITensorCompat.TensorTrain; sites=nothing)
    TT.TensorTrain{V, N}(tt::T4AITensorCompat.TensorTrain; sites=nothing)

Convert a `T4AITensorCompat.TensorTrain` to a `T4ATensorTrain.TensorTrain`.

# Arguments
- `tt::T4AITensorCompat.TensorTrain`: The tensor train to convert
- `sites`: Optional site indices. If `nothing`, site indices will be extracted from the tensor train.
- `V`: Optional type parameter for the element type (e.g., Float64, ComplexF64)
- `N`: Optional type parameter for the number of dimensions per tensor

# Returns
- `TT.TensorTrain{ValueType,N}`: The converted tensor train
"""
function TT.TensorTrain(tt::T4AITensorCompat.TensorTrain; sites=nothing)
    N = length(tt)
    N == 0 && return TT.TensorTrain(Vector{Array{Float64,3}}())

    # Extract site indices if not provided
    if sites === nothing
        sites = T4AITensorCompat.siteinds(tt)
    end

    # T4AITensorCompat.siteinds always returns Vector{Vector{Index}}
    # Determine if this is MPS (one index per site) or MPO (multiple indices per site)
    is_mps = all(length(s) == 1 for s in sites)

    if is_mps
        # MPS case: convert to TensorTrain{ValueType,3}
        # Extract link indices (these are internal links between tensors)
        links = T4AITensorCompat.linkinds(tt)

        # Flatten sites to Vector{Index}
        sites_flat = [s[1] for s in sites]

        # Detect scalar type from input tensors by promoting all element types
        # Get element type from first tensor
        T = eltype(Array(tt[1], sites_flat[1], links[1]))
        # Promote with middle tensors
        for i in 2:length(tt)-1
            Ti = eltype(Array(tt[i], links[i-1], sites_flat[i], links[i]))
            T = promote_type(T, Ti)
        end
        # Promote with last tensor
        if length(tt) > 1
            Ti = eltype(Array(tt[end], links[end], sites_flat[end]))
            T = promote_type(T, Ti)
        end

        # Convert each ITensor to Array, following the same logic as ttmpsconversion.jl
        Tfirst = zeros(T, 1, dim(sites_flat[1]), dim(links[1]))
        Tfirst[1, :, :] = Array(tt[1], sites_flat[1], links[1])

        Tlast = zeros(T, dim(links[end]), dim(sites_flat[end]), 1)
        Tlast[:, :, 1] = Array(tt[end], links[end], sites_flat[end])

        return TT.TensorTrain{T,3}(
            vcat(
                [Tfirst],
                [Array(tt[i], links[i-1], sites_flat[i], links[i]) for i in 2:length(tt)-1],
                [Tlast]
            )
        )
    else
        # MPO case: convert to TensorTrain{ValueType,N} where N > 3
        # This is more complex as we need to determine N from the structure
        error("MPO conversion from T4AITensorCompat.TensorTrain to T4ATensorTrain.TensorTrain is not yet implemented. Please convert via ITensorMPS.MPO first.")
    end
end

function TT.TensorTrain{V, N}(tt::T4AITensorCompat.TensorTrain; sites=nothing) where {V, N}
    N_tt = length(tt)
    N_tt == 0 && return TT.TensorTrain{V, N}(Vector{Array{V, N}}())

    # Extract site indices if not provided
    if sites === nothing
        sites = T4AITensorCompat.siteinds(tt)
    end

    # T4AITensorCompat.siteinds always returns Vector{Vector{Index}}
    # Determine if this is MPS (one index per site) or MPO (multiple indices per site)
    is_mps = all(length(s) == 1 for s in sites)

    if is_mps && N == 3
        # MPS case: convert to TensorTrain{V,3}
        # Extract link indices (these are internal links between tensors)
        links = T4AITensorCompat.linkinds(tt)

        # Flatten sites to Vector{Index}
        sites_flat = [s[1] for s in sites]

        # Use the specified type parameter V
        # Conversion errors will be raised when Array() is called if types are incompatible
        Tfirst = zeros(V, 1, dim(sites_flat[1]), dim(links[1]))
        Tfirst[1, :, :] = Array(tt[1], sites_flat[1], links[1])

        Tlast = zeros(V, dim(links[end]), dim(sites_flat[end]), 1)
        Tlast[:, :, 1] = Array(tt[end], links[end], sites_flat[end])

        return TT.TensorTrain{V, N}(
            vcat(
                [Tfirst],
                [Array(tt[i], links[i-1], sites_flat[i], links[i]) for i in 2:length(tt)-1],
                [Tlast]
            )
        )
    elseif !is_mps && N > 3
        # MPO case: convert to TensorTrain{V,N} where N > 3
        # This is more complex as we need to determine N from the structure
        error("MPO conversion from T4AITensorCompat.TensorTrain to T4ATensorTrain.TensorTrain is not yet implemented. Please convert via ITensorMPS.MPO first.")
    else
        error("Type parameter N=$N does not match the tensor train structure. MPS requires N=3, MPO requires N>3.")
    end
end
