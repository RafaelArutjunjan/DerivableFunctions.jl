


## out-of-place operator backends
_GetDeriv(ADmode::Val{:ReverseDiff}; kwargs...) = throw("GetDeriv() not available for ReverseDiff.jl")
_GetGrad(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient
_GetJac(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian
_GetHess(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian

_GetDeriv(ADmode::Val{:Zygote}; kwargs...) = throw("GetDeriv() not available for Zygote.jl")
_GetGrad(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.gradient(Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.jacobian(Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:Zygote}; order::Int=-1, kwargs...) = (Func::Function,p;Kwargs...) -> Zygote.hessian(Func, p; kwargs...)
_GetDoubleJac(ADmode::Val{:Zygote}; kwargs...) = throw("GetDoubleJac() not available for Zygote.jl") # Zygote does not support mutating arrays

_GetDeriv(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_derivative
_GetGrad(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_gradient
_GetJac(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_jacobian
_GetHess(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_hessian
_GetDoubleJac(ADmode::Val{:FiniteDiff}; kwargs...) = throw("GetDoubleJac() not available for FiniteDiff.jl")

_GetDeriv(ADmode::Val{:FiniteDifferences}; kwargs...) = throw("GetDeriv() not available for FiniteDifferences.jl")
_GetGrad(ADmode::Val{:FiniteDifferences}; order::Int=3, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.grad(central_fdm(order,1), Func, p; kwargs...)[1]
_GetJac(ADmode::Val{:FiniteDifferences}; order::Int=3, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), Func, p; kwargs...)[1]
_GetHess(ADmode::Val{:FiniteDifferences}; order::Int=5, kwargs...) = (Func::Function,p;Kwargs...) -> FiniteDifferences.jacobian(central_fdm(order,1), z->FiniteDifferences.grad(central_fdm(order,1), Func, z)[1], p)[1]



## in-place operator backends
_GetGrad!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.gradient!
_GetJac!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.jacobian!
_GetHess!(ADmode::Val{:ReverseDiff}; kwargs...) = ReverseDiff.hessian!
_GetMatrixJac!(ADmode::Val{:ReverseDiff}; kwargs...) = _GetJac!(ADmode; kwargs...) # DELIBERATE!!!! _GetJac!() recognizes output format from given Array


#_GetDeriv!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_derivative!
_GetGrad!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_gradient!
function _GetJac!(ADmode::Val{:FiniteDiff}; kwargs...)
    function FiniteDiff__finite_difference_jacobian!(Y::AbstractArray{<:Number}, F::Function, X, args...; kwargs...)
        # in-place FiniteDiff operators assume that function itself is also in-place
        if MaximalNumberOfArguments(F) > 1
            FiniteDiff.finite_difference_jacobian!(Y, F, X, args...; kwargs...)
        else
            # Use fake method
            (Y[:] .= vec(_GetJac(ADmode; kwargs...)(F, X, args...)))
            # FiniteDiff.finite_difference_jacobian!(Y, (Res,x)->copyto!(Res,F(x)), args...; kwargs...)
        end
    end
end
_GetHess!(ADmode::Val{:FiniteDiff}; kwargs...) = FiniteDiff.finite_difference_hessian!
_GetMatrixJac!(ADmode::Val{:FiniteDiff}; kwargs...) = _GetJac!(ADmode; kwargs...)


# Fake in-place methods
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
