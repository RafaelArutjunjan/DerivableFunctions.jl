
using DerivableFunctions, Test, Symbolics
function CreateVariable(i::Int)
    i == -1 && return (@variables z)[1]
    (@variables z[1:i])[1]
end

@test GetDeriv(x->x^2)(CreateVariable(-1)) isa Num

# Metric3(x) = [sinh(x[3]) exp(x[1])*sin(x[2]) 0; 0 cosh(x[2]) cos(x[2])*x[3]*x[2]; exp(x[2]) cos(x[3])*x[1]*x[2] 1.]
# @test GetMatrixJac(Metric3)(CreateVariable(3)) isa AbstractArray
#
# @test GetHess(x->(x[1]-x[2])*exp(x[2]/x[1]))(CreateVariable(2)) isa AbstractMatrix{<:Num}
# @test GetJac(x->[x[1]^2 - x[2]^2, x[2]^7*exp(sum(x))])(CreateVariable(2)) isa AbstractMatrix{<:Num}
