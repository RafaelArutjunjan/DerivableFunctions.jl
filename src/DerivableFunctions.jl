module DerivableFunctions

using Reexport
using ReverseDiff, Zygote, FiniteDiff, FiniteDifferences

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

_GetDeriv(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_derivative
_GetGrad(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_gradient
_GetJac(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_jacobian
_GetHess(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_hessian


_GetDeriv(ADmode::Val{:FiniteDifferences}; kwargs...) = throw("GetDeriv not available for FiniteDifferences.jl")
_GetGrad(ADmode::Val{:FiniteDifferences}; order::Int=3, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.grad(central_fdm(order,1), Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:FiniteDifferences}; order::Int=3, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:FiniteDifferences}; order::Int=5, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), z->FiniteDifferences.grad(central_fdm(order,1), Func, z)[1], p)[1]




import DerivableFunctionsBase: _GetDerivPass, _GetGradPass, _GetJacPass, _GetHessPass, _GetDoubleJacPass, _GetMatrixJacPass



import DerivableFunctionsBase: _GetGrad!, _GetJac!, _GetHess!, _GetMatrixJac!
_GetGrad!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient!
_GetJac!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian!
_GetHess!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian!
_GetMatrixJac!(ADmode::Val{:ReverseDiff}; kwargs...) = _GetJac!(ADmode; kwargs...) # DELIBERATE!!!! _GetJac!() recognizes output format from given Array


#_GetDeriv!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_derivative!
_GetGrad!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_gradient!
_GetJac!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_jacobian!
_GetHess!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_hessian!
_GetMatrixJac!(ADmode::Val{:FiniteDiff}; kwargs...) = _GetJac!(ADmode; kwargs...)




# Fake in-place
function _GetGrad!(ADmode::Union{<:Val{:Zygote},<:Val{:FiniteDifferences}}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetGrad!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceGrad!(Y::AbstractVector,F::Function,X::AbstractVector) = copyto!(Y, _GetGrad(ADmode; kwargs...)(F, X))
end
function _GetJac!(ADmode::Union{Val{:Zygote},<:Val{:FiniteDifferences}}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetJac!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceJac!(Y::AbstractMatrix,F::Function,X::AbstractVector) = copyto!(Y, _GetJac(ADmode; kwargs...)(F, X))
end
function _GetHess!(ADmode::Union{Val{:Zygote},<:Val{:FiniteDifferences}}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetHess!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceHess!(Y::AbstractMatrix,F::Function,X::AbstractVector) = copyto!(Y, _GetHess(ADmode; kwargs...)(F, X))
end
function _GetMatrixJac!(ADmode::Union{Val{:Zygote},<:Val{:FiniteDifferences}}; verbose::Bool=false, kwargs...)
    verbose && (@warn "Using fake in-place differentiation operator GetMatrixJac!() for ADmode=$ADmode because backend does not supply appropriate method.")
    FakeInPlaceMatrixJac!(Y::AbstractArray,F::Function,X::AbstractVector) = (Y[:] .= vec(_GetJac(ADmode; kwargs...)(F, X)))
end



import DerivableFunctionsBase: _GetGradPass!, _GetJacPass!, _GetHessPass!, _GetMatrixJacPass!

end