
using DerivableFunctions, Test, Symbolics
function CreateVariable(i::Int)
    i == -1 && return (@variables z)[1]
    (@variables z[1:i])[1]
end

F1(x) = x^2
F2(x) = [exp(x), log(5sinh(x))]
F3(x) = [exp(x) x*log(x) sinh(cosh(x)); tanh(x) x^2 2x]
F4(x) = x[1]^2 + x[2]^3
F5(x) = [x[1]^2+x[2]^2, exp(x[2]-x[1]), log(x[1] + x[2])]
F6(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0 x[2]; 0 cosh(x[2]) cos(x[2])*x[3]*x[2] x[3]]

# R -> R
D1 = DFunction(F1; ADmode=Val(:ForwardDiff))
@test EvaldF(D1, CreateVariable(-1)) isa Num
@test EvalddF(D1, CreateVariable(-1)) isa Num

# R -> R^n
D2 = DFunction(F2; ADmode=Val(:ForwardDiff))
@test EvaldF(D2, CreateVariable(-1)) isa AbstractMatrix{<:Num}
@test EvalddF(D2, CreateVariable(-1)) isa AbstractArray{<:Num,3}

# R -> R^(n×m)
D3 = DFunction(F3; ADmode=Val(:ForwardDiff))
@test EvaldF(D3, CreateVariable(-1)) isa AbstractArray{<:Num,3}
@test EvalddF(D3, CreateVariable(-1)) isa AbstractArray{<:Num,4}

# R^n -> R
D4 = DFunction(F4; ADmode=Val(:ForwardDiff))
@test EvaldF(D4, CreateVariable(2)) isa AbstractVector{<:Num}
@test EvalddF(D4, CreateVariable(2)) isa AbstractMatrix{<:Num}

#R^n -> R^m
D5 = DFunction(F5; ADmode=Val(:ForwardDiff))
@test EvaldF(D5, CreateVariable(2)) isa AbstractMatrix{<:Num}
@test EvalddF(D5, CreateVariable(2)) isa AbstractArray{<:Num,3}

# R^n -> R^(n×m)
D6 = DFunction(F6; ADmode=Val(:ForwardDiff))
@test EvaldF(D6, CreateVariable(3)) isa AbstractArray{<:Num,3}
@test EvalddF(D6, CreateVariable(3)) isa AbstractArray{<:Num,4}
