module DerivableFunctions


using ForwardDiff, ReverseDiff, Zygote
using Symbolics, FiniteDifferences


include("Utils.jl")
export GetArgLength


include("DFunctions.jl")
export DFunction, DerivableFunction


include("DifferentiationOperators.jl")
export GetDeriv, GetGrad, GetJac, GetHess, GetMatrixJac, GetDoubleJac
export GetGrad!, GetJac!, GetHess!, GetMatrixJac!


end
