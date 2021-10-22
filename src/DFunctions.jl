

function GetInLength(F::Function; kwargs...)
    In = GetArgLength(F; kwargs...)
    if In > 1
        return In
    elseif In == 1 # return -1 for functions that take Number instead of a Vector of length 1
        try (F(rand());  return -1) catch; (F(rand(1)); return 1) end
    end
end

GetOutLength(F::Function, In::Int=GetInLength(F)) = GetOutLength(F, (In == -1 ? rand() : rand(In)))
function GetOutLength(F::Function, input::Union{Number,AbstractVector{<:Number}})
    output = F(input)
    output isa Number ? -1 : (output isa AbstractVector ? length(output) : size(output))
end
GetInOut(F::Function) = (In = GetInLength(F);    Out = GetOutLength(F,In);    (In,Out))
GetInOut(F::Function, input::Union{Number, AbstractVector{<:Number}}) = (length(input), GetOutLength(F,input))


function _GetFirstDeriv(F::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:Symbolic))
    # function GetDerivSymb(InOut::Tuple{Int,Union{Int,Tuple}})
    #     if !(Out(InOut) isa Number)
    #         :matrixjacobian
    #     elseif In(InOut) == -1
    #         Out(InOut) == -1 ? :derivative : :jacobian
    #     else
    #         Out(InOut) == -1 ? :gradient : :jacobian
    #     end
    # end
    # SymbolicDeriv = try GetSymbolicDerivative(F, abs(In(InOut)), GetDerivSymb(InOut))   catch;  nothing end
    # if !isnothing(SymbolicDeriv)
    #     SymbolicDeriv
    # else
        if !(Out(InOut) isa Number)
            GetMatrixJac(ADmode,F,abs(In(InOut)),_MakeTuple(Out(InOut)))
        elseif In(InOut) == -1
            # Out is an Int but must be Tuple
            Out(InOut) == -1 ? GetDeriv(ADmode,F) : GetMatrixJac(ADmode,F,abs(In(InOut)),_MakeTuple(Out(InOut)))
        else
            Out(InOut) == -1 ? GetGrad(ADmode,F,abs(In(InOut))) : GetJac(ADmode,F,abs(In(InOut)))
        end
    # end
end
function _GetSecondDeriv(F::Function, dF::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:Symbolic))
    if Out(InOut) isa Number && Out(InOut) == -1
        ## For scalar functions, Symbolics often produces false results.
        # try     GetSymbolicDerivative(F, 1, :hessian)   catch;  x->GetHess(ADmode)(F,x)     end
        In(InOut) == -1 ? GetDeriv(ADmode, dF) : GetHess(Val(:ForwardDiff), F, abs(In(InOut)))
    else
        ## Already know info about dimensions, do not call GetDoubleJac.
        # SymbDeriv = try  GetSymbolicDerivative(dF, abs(In(InOut)), :matrixjacobian)  catch;  nothing end
        # !isnothing(SymbDeriv) ? SymbDeriv : GetMatrixJac(ADmode,dF,abs(In(InOut)),(Out(InOut)...,abs(In(InOut))))
        GetMatrixJac(ADmode,dF,abs(In(InOut)),(Out(InOut)...,abs(In(InOut))))
    end
end
function _GetSecondDeriv(F::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:Symbolic))
    _GetSecondDeriv(F, _GetFirstDeriv(F,InOut;ADmode=ADmode), InOut; ADmode=ADmode)
end

# In = -1 for scalar input and 1 for vector input of length 1

"""
Stores input-output dimensions as well as derivatives of a given function for potentially faster computations when derivatives are known.
"""
mutable struct DerivableFunction <: Function
    F::Function
    dF::Function
    ddF::Function
    InOut::Tuple{Val,Val}
    function DerivableFunction(F::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:Symbolic))
        DerivableFunction(F, _GetFirstDeriv(F,InOut;ADmode=ADmode), InOut; ADmode=ADmode)
    end
    function DerivableFunction(F::Function, dF::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:Symbolic))
        DerivableFunction(F, dF, _GetSecondDeriv(F,dF,InOut;ADmode=ADmode), InOut)
    end
    function DerivableFunction(F::Function, dF::Function, ddF::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F))
        DerivableFunction(F, dF, ddF, (Val(InOut[1]), Val(InOut[2])))
    end
    function DerivableFunction(F::Function, dF::Function, ddF::Function, InOut::Tuple{Val,Val})
        @assert In(InOut) ≥ -1 && (Out(InOut) isa Number ? (Out(InOut) ≥ -1) : all(x->x≥1, Out(InOut)))
        new(F, dF, ddF, InOut)
    end
end
DFunction = DerivableFunction

(D::DFunction)(x) = EvalF(D, x)

InOut(D::DFunction) = (In(D), Out(D))
In(D::DFunction) = In(D.InOut);      In(InOut::Tuple) = _content(InOut[1])
Out(D::DFunction) = Out(D.InOut);    Out(InOut::Tuple) = _content(InOut[2])
_content(::Val{N}) where N = _content(N)
_content(N::Int) = N;   _content(tup::Tuple) = tup
_content(N) = throw("Unexpected content.")


EvalF(D::DFunction, x; kwargs...) = D.F(x; kwargs...)
EvalF(D::DFunction) = D.F
EvaldF(D::DFunction, x; kwargs...) = D.dF(x; kwargs...)
EvaldF(D::DFunction) = D.dF
EvalddF(D::DFunction, x; kwargs...) = D.ddF(x; kwargs...)
EvalddF(D::DFunction) = D.ddF

EvalF(F::Function, x; kwargs...) = F(x; kwargs...)
EvalF(F::Function) = F
EvaldF(F::Function, x; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = GetMatrixJac(ADmode, F)(x)
EvaldF(F::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = GetMatrixJac(ADmode, F)
EvalddF(F::Function, x; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = GetMatrixJac(ADmode, GetMatrixJac(ADmode, F))(x)
EvalddF(F::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = GetMatrixJac(ADmode, GetMatrixJac(ADmode, F))
