module DerivableFunctions

using Reexport
using ReverseDiff, Zygote

@reexport using DerivableFunctionsBase

# import such that they are available downstream as if defined here
import DerivableFunctionsBase: MaximalNumberOfArguments, KillAfter, Builder
import DerivableFunctionsBase: _GetArgLength, _GetArgLengthOutOfPlace, _GetArgLengthInPlace
import DerivableFunctionsBase: GetSymbolicDerivative, SymbolicPassthrough


import DerivableFunctionsBase: suff
suff(x::ReverseDiff.TrackedReal) = typeof(x)

## Add new backends to the output of diff_backends()
import DerivableFunctionsBase: AddedBackEnds
AddedBackEnds(::Val{true}) = [:ReverseDiff, :Zygote]


import DerivableFunctionsBase: _GetDeriv, _GetGrad, _GetJac, _GetHess, _GetMatrixJac, _GetDoubleJac
# Deriv not available for ReverseDiff
_GetGrad(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient
_GetJac(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian
_GetHess(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian
# Deriv not available for Zygote
_GetGrad(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.gradient(Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.jacobian(Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.hessian(Func, p; kwargs...)

import DerivableFunctionsBase: _GetDerivPass, _GetGradPass, _GetJacPass, _GetHessPass, _GetDoubleJacPass, _GetMatrixJacPass



import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!
_GetGrad!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient!
_GetJac!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian!
_GetHess!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian!
_GetMatrixJac!(ADmode::Val{:ReverseDiff}; kwargs...) = _GetJac!(ADmode; kwargs...) # DELIBERATE!!!! _GetJac!() recognizes output format from given Array

import DerivableFunctionsBase: _GetGradPass!, _GetJacPass!, _GetHessPass!, _GetMatrixJacPass!

end
