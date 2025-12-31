"""
    struct Algorithm{Name} end

A type for dispatching on algorithm names. Used for the `algorithm` keyword in functions like `contract`.

# Example
```julia
contract(A, B; algorithm=:naive)  # Dispatches to contract(Algorithm"naive"(), A, B; ...)
```
"""
struct Algorithm{Name} end

Algorithm(s::Symbol) = Algorithm{s}()
Algorithm(s::AbstractString) = Algorithm{Symbol(s)}()

"""
    @Algorithm_str(s)

A string macro for creating `Algorithm` types.

# Example
```julia
Algorithm"naive"  # Same as Algorithm{:naive}
```
"""
macro Algorithm_str(s)
    return :(Algorithm{$(QuoteNode(Symbol(s)))})
end
