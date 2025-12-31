_getindex(x, indices) = ntuple(i -> x[indices[i]], Base.length(indices))

function _contract(
    a::AbstractArray{T1,N1},
    b::AbstractArray{T2,N2},
    idx_a::NTuple{n1,Int},
    idx_b::NTuple{n2,Int}
) where {T1,T2,N1,N2,n1,n2}
    Base.length(idx_a) == Base.length(idx_b) || error("length(idx_a) != length(idx_b)")
    # check if idx_a contains only unique elements
    Base.length(unique(idx_a)) == Base.length(idx_a) || error("idx_a contains duplicate elements")
    # check if idx_b contains only unique elements
    Base.length(unique(idx_b)) == Base.length(idx_b) || error("idx_b contains duplicate elements")
    # check if idx_a and idx_b are subsets of 1:N1 and 1:N2
    all(1 <= idx <= N1 for idx in idx_a) || error("idx_a contains elements out of range")
    all(1 <= idx <= N2 for idx in idx_b) || error("idx_b contains elements out of range")

    rest_idx_a = setdiff(1:N1, idx_a)
    rest_idx_b = setdiff(1:N2, idx_b)

    amat = reshape(permutedims(a, (rest_idx_a..., idx_a...)), prod(_getindex(Base.size(a), rest_idx_a)), prod(_getindex(Base.size(a), idx_a)))
    bmat = reshape(permutedims(b, (idx_b..., rest_idx_b...)), prod(_getindex(Base.size(b), idx_b)), prod(_getindex(Base.size(b), rest_idx_b)))

    return reshape(amat * bmat, _getindex(Base.size(a), rest_idx_a)..., _getindex(Base.size(b), rest_idx_b)...)
end


function _contractsitetensors(a::Array{T,4}, b::Array{T,4})::Array{T,4} where {T}
    # indices: (link_a, s1, s2, link_a') * (link_b, s2, s3, link_b')
    ab::Array{T,6} = _contract(a, b, (3,), (2,))
    # => indices: (link_a, s1, link_a', link_b, s3, link_b')
    abpermuted = permutedims(ab, (1, 4, 2, 5, 3, 6))
    # => indices: (link_a, link_b, s1, s3, link_a', link_b')
    return reshape(abpermuted,
        Base.size(a, 1) * Base.size(b, 1),  # link_a * link_b
        Base.size(a, 2), Base.size(b, 3),  # s1, s3
        Base.size(a, 4) * Base.size(b, 4)   # link_a' * link_b'
    )
end

function contract_naive(
    a::TensorTrain{T,4}, b::TensorTrain{T,4};
    tolerance=0.0, maxbonddim=typemax(Int)
)::TensorTrain{T,4} where {T}
    tt = TensorTrain{T,4}(_contractsitetensors.(sitetensors(a), sitetensors(b)))
    if tolerance > 0 || maxbonddim < typemax(Int)
        compress!(tt, :SVD; tolerance, maxbonddim)
    end
    return tt
end

function _reshape_fusesites(t::AbstractArray{T}) where {T}
    shape = Base.size(t)
    return reshape(t, shape[1], prod(shape[2:end-1]), shape[end]), shape[2:end-1]
end

function _reshape_splitsites(
    t::AbstractArray{T},
    legdims::Union{AbstractVector{Int},Tuple},
) where {T}
    return reshape(t, Base.size(t, 1), legdims..., Base.size(t, ndims(t)))
end


"""
See SVD version:
https://tensornetwork.org/mps/algorithms/zip_up_mpo/
"""
function contract_zipup(
    A::TensorTrain{ValueType,4},
    B::TensorTrain{ValueType,4};
    tolerance::Float64=1e-12,
    method::Symbol=:SVD, # :SVD, :LU
    maxbonddim::Int=typemax(Int)
) where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end
    R::Array{ValueType,3} = ones(ValueType, 1, 1, 1)

    sitetensors_result = Vector{Array{ValueType,4}}(undef, length(A))
    for n in 1:length(A)
        # R:     (link_ab, link_an, link_bn)
        # A[n]:  (link_an, s_n, s_n', link_anp1)
        RA = _contract(R, A[n], (2,), (1,))

        # RA[n]: (link_ab, link_bn, s_n, s_n' link_anp1)
        # B[n]:  (link_bn, s_n', s_n'', link_bnp1)
        # C:     (link_ab, s_n, link_anp1, s_n'', link_bnp1)
        #  =>    (link_ab, s_n, s_n'', link_anp1, link_bnp1)
        C = permutedims(_contract(RA, B[n], (2, 4), (1, 2)), (1, 2, 4, 3, 5))
        if n == length(A)
            sitetensors_result[n] = reshape(C, Base.size(C)[1:3]..., 1)
            break
        end

        # Cmat:  (link_ab * s_n * s_n'', link_anp1 * link_bnp1)

        #lu = rrlu(Cmat; reltol, abstol, leftorthogonal=true)
        left_factor, right_factor, newbonddim = _factorize(
            reshape(C, prod(Base.size(C)[1:3]), prod(Base.size(C)[4:5])),
            method; tolerance, maxbonddim
        )

        # U:     (link_ab, s_n, s_n'', link_ab_new)
        sitetensors_result[n] = reshape(left_factor, Base.size(C)[1:3]..., newbonddim)

        # R:     (link_ab_new, link_an, link_bn)
        R = reshape(right_factor, newbonddim, Base.size(C)[4:5]...)
    end

    return TensorTrain{ValueType,4}(sitetensors_result)
end

"""
    function contract(
        A::TensorTrain{V1,4},
        B::TensorTrain{V2,4};
        algorithm::Symbol=:naive,
        tolerance::Float64=1e-12,
        maxbonddim::Int=typemax(Int),
        kwargs...
    ) where {V1,V2}

Contract two tensor trains `A` and `B`.

Available implementations:
 1. `algorithm=:naive` uses a naive tensor contraction and subsequent SVD recompression of the tensor train.
 2. `algorithm=:zipup` uses a naive tensor contraction with on-the-fly LU decomposition.

For TCI-based contraction (algorithm=:TCI), use T4ATensorCI.jl which extends this function.

Arguments:
- `A` and `B` are the tensor trains to be contracted.
- `algorithm` chooses the algorithm used to evaluate the contraction.
- `tolerance` is the tolerance of the SVD recompression.
- `maxbonddim` sets the maximum bond dimension of the resulting tensor train.
- `method` chooses the method used for the factorization in the `algorithm=:zipup` case (`:SVD` or `:LU`).
"""
function contract(
    A::TensorTrain{V1,4},
    B::TensorTrain{V2,4};
    algorithm::Symbol=:naive,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
    kwargs...
)::TensorTrain{promote_type(V1, V2),4} where {V1,V2}
    Vres = promote_type(V1, V2)
    A_ = TensorTrain{Vres,4}(A)
    B_ = TensorTrain{Vres,4}(B)
    return contract(Algorithm(algorithm), A_, B_; tolerance=tolerance, maxbonddim=maxbonddim, kwargs...)
end

# Algorithm dispatch
function contract(::Algorithm"naive", A::TensorTrain{T,4}, B::TensorTrain{T,4};
    tolerance=1e-12, maxbonddim=typemax(Int), kwargs...) where {T}
    return contract_naive(A, B; tolerance=tolerance, maxbonddim=maxbonddim)
end

function contract(::Algorithm"zipup", A::TensorTrain{T,4}, B::TensorTrain{T,4};
    tolerance=1e-12, maxbonddim=typemax(Int), method=:SVD, kwargs...) where {T}
    return contract_zipup(A, B; tolerance=tolerance, maxbonddim=maxbonddim, method=method)
end
