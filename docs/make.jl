using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using T4ATensorTrain

makedocs(
    modules=[T4ATensorTrain],
    sitename="T4ATensorTrain.jl",
)

deploydocs(
    repo="github.com/tensor4all/T4ATensorTrain.jl.git",
    devbranch="main",
)


