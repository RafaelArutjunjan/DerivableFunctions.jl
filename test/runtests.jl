
using SafeTestsets

@safetestset "Differentiation Operator Backends (Bare)" begin
    using DerivableFunctions, Test, Symbolics, ForwardDiff
    Metric3(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0; 0 cosh(x[2]) cos(x[2])*x[3]*x[2]; exp(x[2]) cos(x[3])*x[1]*x[2] 1.]

    X = ForwardDiff.gradient(x->x[1]^2 + exp(x[2]), [5,10.])
    Y = ForwardDiff.jacobian(x->[x[1]^2 + exp(x[2])], [5,10.])
    Z = ForwardDiff.hessian(x->x[1]^2 + exp(x[2]) + x[1]*x[2], [5,10.])
    Mat = reshape(ForwardDiff.jacobian(vec∘Metric3, [5,10,15.]), 3, 3, 3)

    function MyTest(ADmode::Symbol; kwargs...)
        Grad, Jac, Hess = GetGrad(ADmode; kwargs...), GetJac(ADmode; kwargs...), GetHess(ADmode; kwargs...)
        MatrixJac = GetMatrixJac(ADmode; order=8, kwargs...)
        @test Grad(x->x[1]^2 + exp(x[2]), [5,10.]) ≈ X
        @test Jac(x->[x[1]^2 + exp(x[2])], [5,10.]) ≈ Y
        @test Hess(x->x[1]^2 + exp(x[2]) + x[1]*x[2], [5,10.]) ≈ Z
        @test maximum(abs.(MatrixJac(Metric3, [5,10,15.]) - Mat)) < 1e-5
    end

    for ADmode ∈ [:ForwardDiff, :Zygote, :ReverseDiff, :FiniteDiff]
        MyTest(ADmode)
    end
end

@safetestset "Symbolic Derivatives" begin
    using DerivableFunctions, Test, RuntimeGeneratedFunctions
   # :Symbolic
end


@safetestset "Symbolic Passthrough" begin
    include("SymbolicTests.jl")
end


@safetestset "DFunctions" begin
    using DerivableFunctions, Test

    # R -> R
    D1 = DFunction(x->x^2)
    @test EvalF(D1, 5) == 25
    @test EvaldF(D1, 5) == 10
    @test EvalddF(D1, 5) == 2

    # # R -> R^n
    # D2 = DFunction(x->[exp(x), log(5sinh(x))])
    # @test EvaldF(D2, rand()) isa AbstractMatrix
    # @test EvalddF(D2, rand()) isa AbstractArray{<:Number,3}
    #
    # # R -> R^(n×m)
    # D3 = DFunction(x->[exp(x) x*log(x) sinh(cosh(x)); tanh(x) x^2 2x])
    # @test EvalF(D3, rand()) isa AbstractMatrix
    # @test EvaldF(D3, rand()) isa AbstractArray{<:Number,3}
    # @test EvalddF(D3, rand()) isa AbstractArray{<:Number,4}

    # R^n -> R
    D4 = DFunction(x->x[1]^2 + x[2]^3)
    @test EvaldF(D4, rand(2)) isa AbstractVector
    @test EvalddF(D4, rand(2)) isa AbstractMatrix

    #R^n -> R^m
    D5 = DFunction(x->[x[1]^2+x[2]^2, exp(x[2]-x[1]), log(x[1] + x[2])])
    @test EvaldF(D5, rand(2)) isa AbstractMatrix
    @test EvalddF(D5, rand(2)) isa AbstractArray{<:Number,3}

    # R^n -> R^(n×m)
    Metric3(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0 x[2]; 0 cosh(x[2]) cos(x[2])*x[3]*x[2] x[3]; exp(x[2]) cos(x[3])*x[1]*x[2] 1.0 x[1]]
    D6 = DFunction(Metric3)
    @test EvaldF(D6, rand(3)) isa AbstractArray{<:Number,3}
    @test EvalddF(D6, rand(3)) isa AbstractArray{<:Number,4}
end
