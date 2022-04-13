
## Differentiation Operators

[**DerivableFunctions.jl**](https://github.com/RafaelArutjunjan/DerivableFunctions.jl) aims to provide a backend-agnostic interface for differentiation and currently allows the user to seamlessly switch between [**ForwardDiff.jl**](https://github.com/JuliaDiff/ForwardDiff.jl), [**ReverseDiff.jl**](https://github.com/JuliaDiff/ReverseDiff.jl), [**Zygote.jl**](https://github.com/FluxML/Zygote.jl), [**FiniteDifferences.jl**](https://github.com/JuliaDiff/FiniteDifferences.jl) and [**Symbolics.jl**](https://github.com/JuliaSymbolics/Symbolics.jl).

The desired backend is optionally specified in the first argument (default is ForwardDiff) via a `Symbol` or `Val`. The available backends can be listed via `diff_backends()`.

Next, the function that is to be differentiated is provided. We will illustrate this syntax using the `GetMatrixJac` method:
```@example 1
using DerivableFunctions
Metric(x) = [exp(x[1]^3) sin(cosh(x[2])); log(sqrt(x[1])) x[1]^2*x[2]^5]
Jac = GetMatrixJac(Val(:ForwardDiff), Metric)
Jac([1,2.])
```

Moreover, these operators are overloaded to allow for passthrough of symbolic variables.
```@example 1
using Symbolics
@variables z[1:2]
J = Jac(z)
J[:,:,1], J[:,:,2]
```

Since the function `Metric` in this example can be represented in terms of analytic expressions, it is also possible to construct its derivative symbolically:
```@example 1
SymJac = GetMatrixJac(Val(:Symbolic), Metric)
SymJac([1,2.])
```
Currently, [**DerivableFunctions.jl**](https://github.com/RafaelArutjunjan/DerivableFunctions.jl) exports `GetDeriv(), GetGrad(), GetHess(), GetJac(), GetDoubleJac()` and `GetMatrixJac()`.


Furthermore, these operators also have in-place versions:
```@example 1
Jac! = GetMatrixJac!(Val(:ForwardDiff), Metric)
Y = Array{Float64}(undef, 2, 2, 2)
Jac!(Y, [1,2.])
```

Just like the out-of-place versions, the in-place operators are overloaded for symbolic passthrough:
```@example 1
Ynum = Array{Num}(undef, 2, 2, 2)
Jac!(Ynum, z)
Ynum[:,:,1], Ynum[:,:,2]
```

The exported in-place operators include `GetGrad!(), GetHess!(), GetJac!()` and `GetMatrixJac!()`.


## Differentiation Backend-Agnostic Programming

Essentially, the abstraction layer provided by **DerivableFunctions.jl** only requires the user to specify the "semantic" meaning of a given differentiation operation while allowing for flexible post hoc choice of backend as well as enabling symbolic pass through for the resulting computation.

For example, when calculating differential-geometric quantities such as the Riemann or Ricci tensors, which depend on complicated combinations of up to second derivatives of the components of the metric tensor, a single implementation simultaneously provides a performant numerical implementation as well as allowing for analytical insight for simple examples.
```julia
using DerivableFunctions, Tullio, LinearAlgebra
MetricPartials(Metric::Function, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff)) = GetMatrixJac(ADmode, Metric)(θ)
function ChristoffelSymbol(Metric::Function, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff))
  PDV = MetricPartials(Metric, θ; ADmode);  InvMetric = inv(Metric(θ))
  @tullio Γ[a,i,j] := ((1/2) * InvMetric)[a,m] * (PDV[j,m,i] + PDV[m,i,j] - PDV[i,j,m])
end
function ChristoffelPartials(Metric::Function, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff))
  GetMatrixJac(ADmode, x->ChristoffelSymbol(Metric, x; ADmode))(θ)
end
function Riemann(Metric::Function, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff))
  Γ = ChristoffelSymbol(Metric, θ; ADmode)
  ∂Γ = ChristoffelPartials(Metric, θ; ADmode)
  @tullio Riem[i,j,k,l] := ∂Γ[i,j,l,k] - ∂Γ[i,j,k,l]
  @tullio Riem[i,j,k,l] += Γ[i,a,k]*Γ[a,j,l] - Γ[i,a,l]*Γ[a,j,k]
end
function Ricci(Metric::Function, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff))
  Riem = Riemann(Metric, θ; ADmode)
  @tullio Ric[a,b] := Riem[s,a,s,b]
end
function RicciScalar(Metric::Function, θ::AbstractVector; ADmode::Val=Val(:ForwardDiff))
  InvMetric = inv(Metric(θ))
  tr(transpose(Ricci(Metric, θ; ADmode)) * InvMetric)
end
```
Clearly, this simplified implementation features some redundant evaluations of the inverse metric and could be made more efficient.
Nevertheless, it nicely illustrates how succinctly complex real-world examples can be formulated.

Given the metric tensor induced by the canonical embedding of ``S^2`` into ``\\mathbb{R}^3`` with spherical coordinates, it can be shown that the Ricci scalar assumes a constant value of ``R=2`` everywhere on ``S^2``.
```julia
S2metric((θ,ϕ)) = [1.0 0; 0 sin(θ)^2]
2 ≈ RicciScalar(S2metric, rand(2); ADmode=Val(:ForwardDiff)) ≈ RicciScalar(S2metric, rand(2); ADmode=Val(:ReverseDiff))
```

(In this particular instance, due to a term in the `ChristoffelSymbol` where the `sin` in the numerator does not cancel with the identical term in the denominator, the symbolic computation does not recognize the fact that the final expression can be simplified to yield exactly ``R=2``.)
```julia
using Symbolics;  @variables p[1:2]
RicciScalar(S2metric, p) |> simplify_fractions
```
