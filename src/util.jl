function replacenothing(value::Union{T,Nothing}, default::T)::T where {T}
    if isnothing(value)
        return default
    else
        return value
    end
end

"""
Construct slice for the site indices of one tensor core
Returns a slice and the corresponding shape for `resize`
"""
function projector_to_slice(p::AbstractVector{<:Integer})
    return [x == 0 ? Colon() : x for x in p], [x == 0 ? Colon() : 1 for x in p]
end
