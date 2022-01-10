module DerivableFunctions

using Reexport
using ReverseDiff, Zygote, FiniteDiff, FiniteDifferences

@reexport using DerivableFunctionsBase

# import such that they are available downstream as if defined here
import DerivableFunctionsBase: MaximalNumberOfArguments, KillAfter, Builder
import DerivableFunctionsBase: _GetArgLength, _GetArgLengthOutOfPlace, _GetArgLengthInPlace
import DerivableFunctionsBase: GetSymbolicDerivative, SymbolicPassthrough

import DerivableFunctionsBase: _GetDeriv, _GetGrad, _GetJac, _GetHess, _GetMatrixJac, _GetDoubleJac
import DerivableFunctionsBase: _GetDerivPass, _GetGradPass, _GetJacPass, _GetHessPass, _GetDoubleJacPass, _GetMatrixJacPass
import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!
import DerivableFunctionsBase: _GetGradPass!, _GetJacPass!, _GetHessPass!, _GetMatrixJacPass!


import DerivableFunctionsBase: suff
suff(x::ReverseDiff.TrackedReal) = typeof(x)

## Add new backends to the output of diff_backends()
import DerivableFunctionsBase: AddedBackEnds
AddedBackEnds(::Val{true}) = [:ReverseDiff, :Zygote, :FiniteDifferences, :FiniteDiff]

include("DifferentiationOperators.jl")

end
