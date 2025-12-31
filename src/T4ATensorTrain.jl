module T4ATensorTrain

using LinearAlgebra
import LinearAlgebra: rank
import LinearAlgebra as LA
import Base: length, sum, size

# Import from T4AMatrixCI
using T4AMatrixCI
import T4AMatrixCI: rrlu, MatrixLUCI, left, right, npivots

include("util.jl")
include("algorithm.jl")
include("abstracttensortrain.jl")
include("tensortrain.jl")
include("cachedtensortrain.jl")
include("contraction.jl")

# Types
export AbstractTensorTrain
export TensorTrain
export TTCache, BatchEvaluator
export TensorTrainFit
export Algorithm, @Algorithm_str

# Type aliases
export LocalIndex, MultiIndex

# Functions from abstracttensortrain.jl
export linkdims, linkdim, sitedims, sitedim, rank
export sitetensors, sitetensor
export evaluate, add, subtract

# Functions from tensortrain.jl
export tensortrain, compress!, fulltensor
export multiply!, multiply, divide!, divide
export flatten, to_tensors

# Functions from cachedtensortrain.jl
export evaluateleft, evaluateright, batchevaluate
export isbatchevaluable, ttcache

# Functions from contraction.jl
export contract, contract_naive, contract_zipup
export _contract, _contractsitetensors

# Functions from util.jl
export projector_to_slice

# Re-export from T4AMatrixCI for convenience
export rrlu, MatrixLUCI

end
