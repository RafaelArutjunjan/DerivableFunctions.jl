
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

# @safetestset "Symbolic Derivatives" begin
#     using DerivableFunctions, Test, Symbolics
#    # :Symbolic
# end


@safetestset "Symbolic Passthrough" begin
    include("SymbolicTests.jl")
end


@safetestset "DFunctions" begin
    using DerivableFunctions, Test, RuntimeGeneratedFunctions

    D1 = DFunction((x)->[x[1]^2+x[2]^2, exp(x[2]-x[1])])
    @test GetJac(Val(true), D1) === EvaldF(D1) === D1.dF

    # EvalF(D1, [5,3])


    D2 = DFunction(x->x^2)
    @test EvalF(D2, 5) == 25
    @test EvaldF(D2, 5) == 10
    @test EvalddF(D2, 5) == 2

end
