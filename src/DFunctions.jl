

function GetInLength(F::Function; kwargs...)
    In = GetArgLength(F; kwargs...)
    if In > 1
        return In
    elseif In == 1 # return -1 for functions that take Number instead of a Vector of length 1
        try
            F(1.);  return -1
        catch;
            F(ones(1)); return 1
        end
    end
end

GetOutLength(F::Function, In::Int=GetInLength(F)) = GetOutLength(F, (In == -1 ? 1. : ones(In)))
function GetOutLength(F::Function, input::Union{Number,AbstractVector{<:Number}})
    output = F(input)
    output isa Number ? -1 : (output isa AbstractVector ? length(output) : size(output))
end

GetInOut(F::Function) = (In = GetInLength(F);    Out = GetOutLength(F,In);    (In,Out))
GetInOut(F::Function, input::Union{Number, AbstractVector{<:Number}}) = (length(input), GetOutLength(F,input))

function _GetFirstDeriv(F::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    function GetDerivSymb(InOut::Tuple{Int,Union{Int,Tuple}})
        # if Out(InOut) isa Tuple :matrixjacobian end
        if In(InOut) == -1
            if Out(InOut) isa Number
                Out(InOut) == -1 ? :derivative : :jacobian
            else
                :matrixjacobian
            end
        else
            if Out(InOut) isa Number
                Out(InOut) == -1 ? :gradient : :jacobian
            else
                :matrixjacobian
            end
        end
    end
    try
        GetSymbolicDerivative(F, abs(In(InOut)), GetDerivSymb(InOut))
    catch;
        if Out(InOut) isa Number
            Out(InOut) ≥ 1 ? GetJac(ADmode,F) : (Out(InOut) == -1 ? GetDeriv(ADmode,F) : GetGrad(ADmode,F))
        else
            GetMatrixJac(ADmode,F)
        end
    end
end
function _GetSecondDeriv(F::Function, dF::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
    if !(Out(InOut) isa Number)
        GetMatricJac(ADmode, dF)
    elseif Out(InOut) == -1
        ## For scalar functions, Symbolics often produces false results.
        # try     GetSymbolicDerivative(F, 1, :hessian)   catch;  x->GetHess(ADmode)(F,x)     end
        if In(InOut) == -1
            GetDeriv(ADmode, dF)
        else
            GetHess(ADmode, F)
        end
    else
        # Already know info about dimensions, do not call GetDoubeJac.
        try
            GetSymbolicDerivative(dF, In(InOut), :matrixjacobian)
        catch;
            GetMatrixJac(ADmode, dF)
            # if Out(InOut) == 1
            #     x->reshape(GetJac(ADmode)(vec∘dF, x), In(InOut), In(InOut))
            # else
            #     x->reshape(GetJac(ADmode)(vec∘dF, x), Out(InOut), In(InOut), In(InOut))
            # end
        end
    end
end
function _GetSecondDeriv(F::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
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
    function DerivableFunction(F::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
        DerivableFunction(F, _GetFirstDeriv(F,InOut;ADmode=ADmode), InOut; ADmode=ADmode)
    end
    function DerivableFunction(F::Function, dF::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:ForwardDiff))
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
EvalF(D::DFunction, x) = D.F(x)
EvalF(D::DFunction) = D.F
EvaldF(D::DFunction, x) = D.dF(x)
EvaldF(D::DFunction) = D.dF
EvalddF(D::DFunction, x) = D.ddF(x)
EvalddF(D::DFunction) = D.ddF
In(D::DFunction) = In(D.InOut);      In(InOut::Tuple) = _content(InOut[1])
Out(D::DFunction) = Out(D.InOut);    Out(InOut::Tuple) = _content(InOut[2])
_content(::Val{N}) where N = _content(N)
_content(N::Int) = N;   _content(tup::Tuple) = tup
_content(N) = throw("Unexpected content.")
