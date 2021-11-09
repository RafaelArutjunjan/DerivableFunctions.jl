
### Differentiation Operators

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
