# DerivableFunctions

*A Julia package for backend-agnostic differentiation combined with symbolic passthrough.*

| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RafaelArutjunjan.github.io/DerivableFunctions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RafaelArutjunjan.github.io/DerivableFunctions.jl/dev) |
[![Build Status](https://ci.appveyor.com/api/projects/status/github/RafaelArutjunjan/DerivableFunctions.jl?svg=true)](https://ci.appveyor.com/project/RafaelArutjunjan/DerivableFunctions-jl) [![Codecov](https://codecov.io/gh/RafaelArutjunjan/DerivableFunctions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RafaelArutjunjan/DerivableFunctions.jl) |


This package provides a front-end for differentiation operations in Julia that allows for code written by the user to be agnostic with respect to many of the available automatic and symbolic differentiation tools available in Julia. Moreover, the differentiation operators provided by **DerivableFunctions.jl** are overloaded to allow for passthrough of symbolic variables. That is, if symbolic types such as `Symbolics.Num` are detected, the differentiation operators automatically switch to symbolic differentiation.

In addition to these operators, **DerivableFunctions.jl** also provides the `DFunction` type, which stores methods for the first and second derivatives to allow for more convenient and potentially more performant computations if the derivatives are known.

For detailed examples, please see the [**documentation**](https://RafaelArutjunjan.github.io/DerivableFunctions.jl/dev).

```julia
julia> D = DFunction(x->[exp(x[1]^2 - x[2]), log(sin(x[2]))])
(::DerivableFunction) (generic function with 1 method)

julia> EvalF(D,[1,2])
2-element Vector{Float64}:
  0.36787944117144233
 -0.09508303609516061

julia> EvaldF(D,[1,2])
2×2 Matrix{Float64}:
 0.735759  -0.367879
 0.0       -0.457658

julia> EvalddF(D,[1,2])
2×2×2 Array{Float64, 3}:
[:, :, 1] =
 2.20728  -0.735759
 0.0       0.0

[:, :, 2] =
 -0.735759   0.367879
  0.0       -1.20945

julia> using Symbolics; @variables z[1:2]
1-element Vector{Symbolics.Arr{Num, 1}}:
 z[1:2]

julia> EvalddF(D, z)
2×2×2 Array{Num, 3}:
[:, :, 1] =
 2exp(z[1]^2 - z[2]) + 4(z[1]^2)*exp(z[1]^2 - z[2])  -2exp(z[1]^2 - z[2])*z[1]
 0                                                    0

[:, :, 2] =
 -2exp(z[1]^2 - z[2])*z[1]                                                             exp(z[1]^2 - z[2])
  0                         (-(cos(z[2])^2)) / (sin(z[2])^2) + (-sin(z[2])) / sin(z[2])
```
