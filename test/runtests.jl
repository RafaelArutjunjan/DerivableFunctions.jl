
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

    for ADmode ∈ [:ForwardDiff, :Zygote, :ReverseDiff, :FiniteDiff]
        MyTest(ADmode)
    end


    function TestDoubleJac(ADmode::Symbol; kwargs...)
        DoubleJac = GetDoubleJac(ADmode; order=8, kwargs...)
        maximum(abs.(DoubleJac(x->[exp(x[1])*sin(x[2]), cosh(x[2])*x[1]*x[2]], [5,10.]) - Djac)) < 1e-5
    end

    for ADmode ∈ [:ForwardDiff, :ReverseDiff, :FiniteDiff]
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

    for ADmode ∈ [:ForwardDiff, :Zygote, :ReverseDiff, :FiniteDiff]
        MyInplaceTest(ADmode)
    end
end


@safetestset "Symbolic Passthrough" begin
    include("SymbolicTests.jl")
end


@safetestset "DFunctions with symbolic derivatives (out-of-place)" begin
    using DerivableFunctions, Test, RuntimeGeneratedFunctions

    F1(x) = x^2
    F2(x) = [exp(x), log(5sinh(x))]
    F3(x) = [exp(x) x*log(x) sinh(cosh(x)); tanh(x) x^2 2x]
    F4(x) = x[1]^2 + x[2]^3
    F5(x) = [x[1]^2+x[2]^2, exp(x[2]-x[1]), log(x[1] + x[2])]
    F6(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0 x[2]; 0 cosh(x[2]) cos(x[2])*x[3]*x[2] x[3]]

    AllSymbolic(D::DFunction) = (@test D.dF isa RuntimeGeneratedFunction; @test D.ddF isa RuntimeGeneratedFunction)

    # R -> R
    D1 = DFunction(F1; ADmode=Val(:Symbolic))
    @test EvalF(D1, 5) == 25
    @test EvaldF(D1, 5) == 10
    @test EvalddF(D1, 5) == 2
    AllSymbolic(D1)

    # R -> R^n
    D2 = DFunction(F2; ADmode=Val(:Symbolic))
    @test EvaldF(D2, rand()) isa AbstractMatrix
    @test EvalddF(D2, rand()) isa AbstractArray{<:Number,3}
    AllSymbolic(D2)

    # R -> R^(n×m)
    D3 = DFunction(F3; ADmode=Val(:Symbolic))
    @test EvalF(D3, rand()) isa AbstractMatrix
    @test EvaldF(D3, rand()) isa AbstractArray{<:Number,3}
    @test EvalddF(D3, rand()) isa AbstractArray{<:Number,4}
    AllSymbolic(D3)

    # R^n -> R
    D4 = DFunction(F4; ADmode=Val(:Symbolic))
    @test EvaldF(D4, rand(2)) isa AbstractVector
    @test EvalddF(D4, rand(2)) isa AbstractMatrix
    # AllSymbolic(D4)
    # Not using Symbolics.hessian in this case since it returns wrong results
    @test D4.dF isa RuntimeGeneratedFunction


    #R^n -> R^m
    D5 = DFunction(F5; ADmode=Val(:Symbolic))
    @test EvaldF(D5, rand(2)) isa AbstractMatrix
    @test EvalddF(D5, rand(2)) isa AbstractArray{<:Number,3}
    AllSymbolic(D5)

    # R^n -> R^(n×m)
    D6 = DFunction(F6; ADmode=Val(:Symbolic))
    @test EvaldF(D6, rand(3)) isa AbstractArray{<:Number,3}
    @test EvalddF(D6, rand(3)) isa AbstractArray{<:Number,4}
    AllSymbolic(D6)
end

@safetestset "Function Structure Inference" begin
    using DerivableFunctions, Test
    import DerivableFunctions: GetInOut, FindSubHyperCube

    # Out-of-place
    F1(x) = x^2
    F2(x) = [exp(x), log(5sinh(x))]
    F3(x) = [exp(x) x*log(x) sinh(cosh(x)); tanh(x) x^2 2x]
    F4(x) = x[1]^2 + x[2]^3
    F5(x) = [x[1]^2+x[2]^2, exp(x[2]-x[1]), log(x[1] + x[2])]
    F6(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0 x[2]; 0 cosh(x[2]) cos(x[2])*x[3]*x[2] x[3]]

    @test GetInOut(F1) == (-1,-1)
    @test GetInOut(F2) == (-1, 2)
    @test GetInOut(F3) == (-1, (2,3))
    @test GetInOut(F4) == ( 2,-1)
    @test GetInOut(F5) == ( 2, 3)
    @test GetInOut(F6) == ( 3, (2,4))

    # In-place
    F2!(y,x) = copyto!(y,[exp(x), log(5sinh(x))])
    F3!(y,x) = y[1:2, 1:3] .= [exp(x) x*log(x) sinh(cosh(x)); tanh(x) x^2 2x]
    F5!(y,x) = copyto!(y,[x[1]^2+x[2]^2, exp(x[2]-x[1]), log(x[1] + x[2])])
    F6!(y,x) = y[1:2,1:4] .= [sinh(x[3]) exp(x[1])*sin(x[2]) 0 x[2]; 0 cosh(x[2]) cos(x[2])*x[3]*x[2] x[3]]
    # copyto! fills array in column-wise order, i.e. does not respect the shape of the copied array.
    # This means the correct size tuple cannot be inferred when using copyto! for arrays of rank ≥ 2

    @test GetInOut(F2!) == (-1, 2)
    @test GetInOut(F3!) == (-1, (2,3))
    @test GetInOut(F5!) == ( 2, 3)
    @test GetInOut(F6!) == ( 3, (2,4))

    # Test FindSubHyperCube by filling array in random places and check that correct smallest length is determined
    dim = 3;    max = 50;   N = 5
    for _ in 1:3
        locations = [reduce(vcat,[[rand(1:max)] for j in 1:dim]) for i in 1:N]
        Z = zeros([max for i in 1:dim]...);    for loc in locations    Z[loc...] = 0.1 + rand()   end
        @test FindSubHyperCube(Z, x->x!=0) == Tuple([maximum(getindex.(locations,i)) for i in 1:dim])
    end
end
