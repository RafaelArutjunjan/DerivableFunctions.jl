
using SafeTestsets

@safetestset "Bare Differentiation Operator Backends (out-of-place)" begin
    using DerivableFunctions, Test, ForwardDiff
    Metric3(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0; 0 cosh(x[2]) cos(x[2])*x[3]*x[2]; exp(x[2]) cos(x[3])*x[1]*x[2] 1.]

    X = ForwardDiff.gradient(x->x[1]^2 + exp(x[2]), [5,10.])
    Y = ForwardDiff.jacobian(x->[x[1]^2, exp(x[2])], [5,10.])
    Z = ForwardDiff.hessian(x->x[1]^2 + exp(x[2]) + x[1]*x[2], [5,10.])
    Mat = reshape(ForwardDiff.jacobian(vec∘Metric3, [5,10,15.]), 3, 3, 3)
    Djac = reshape(ForwardDiff.jacobian(p->vec(ForwardDiff.jacobian(x->[exp(x[1])*sin(x[2]), cosh(x[2])*x[1]*x[2]],p)), [5,10.]), 2,2,2)

    function MyTest(ADmode::Symbol; kwargs...)
        Grad, Jac, Hess = GetGrad(ADmode; kwargs...), GetJac(ADmode; kwargs...), GetHess(ADmode; kwargs...)
        MatrixJac = GetMatrixJac(ADmode; order=8, kwargs...)

        @test Grad(x->x[1]^2 + exp(x[2]), [5,10.]) ≈ X
        @test Jac(x->[x[1]^2, exp(x[2])], [5,10.]) ≈ Y
        @test Hess(x->x[1]^2 + exp(x[2]) + x[1]*x[2], [5,10.]) ≈ Z
        @test maximum(abs.(MatrixJac(Metric3, [5,10,15.]) - Mat)) < 1e-5
    end

    for ADmode ∈ [:Zygote, :ReverseDiff]
        MyTest(ADmode)
    end

    function TestDoubleJac(ADmode::Symbol; kwargs...)
        DoubleJac = GetDoubleJac(ADmode; order=8, kwargs...)
        maximum(abs.(DoubleJac(x->[exp(x[1])*sin(x[2]), cosh(x[2])*x[1]*x[2]], [5,10.]) - Djac)) < 1e-5
    end

    for ADmode ∈ [:ReverseDiff]
        @test TestDoubleJac(ADmode)
    end
    # Zygote does not support mutating arrays
    @test_broken TestDoubleJac(:Zygote)
end

@safetestset "Bare Differentiation Operator Backends (in-place)" begin
    using DerivableFunctions, Test, ForwardDiff
    Metric3(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0; 0 cosh(x[2]) cos(x[2])*x[3]*x[2]; exp(x[2]) cos(x[3])*x[1]*x[2] 1.]

    X = ForwardDiff.gradient(x->x[1]^2 + exp(x[2]), [5,10.])
    Y = ForwardDiff.jacobian(x->[x[1]^2, exp(x[2])], [5,10.])
    Z = ForwardDiff.hessian(x->x[1]^2 + exp(x[2]) + x[1]*x[2], [5,10.])
    Mat = reshape(ForwardDiff.jacobian(vec∘Metric3, [5,10,15.]), 3, 3, 3)

    function MyInplaceTest(ADmode::Symbol; kwargs...)
        Grad! = GetGrad!(ADmode, x->x[1]^2 + exp(x[2]); kwargs...)
        Jac! = GetJac!(ADmode, x->[x[1]^2, exp(x[2])]; kwargs...)
        Hess! = GetHess!(ADmode, x->x[1]^2 + exp(x[2]) + x[1]*x[2]; kwargs...)
        MatrixJac! = GetMatrixJac!(ADmode, Metric3; order=8, kwargs...)

        Xres = similar(X);  Yres = similar(Y);  Zres = similar(Z);  Matres = similar(Mat)

        Grad!(Xres, [5,10.]);   @test Xres ≈ X
        Jac!(Yres, [5,10.]);   @test Yres ≈ Y
        Hess!(Zres, [5,10.]);   @test Zres ≈ Z
        MatrixJac!(Matres, [5,10,15.]);   @test maximum(abs.(Matres - Mat)) < 1e-5
    end

    for ADmode ∈ [:Zygote, :ReverseDiff]
        MyInplaceTest(ADmode)
    end
end
