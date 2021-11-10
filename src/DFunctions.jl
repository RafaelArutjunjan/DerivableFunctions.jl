

ProduceTestQuantity(n::Int) = n == -1 ? rand() : (n > 0 ? rand(n) : throw("Got invalid length value $n."))
ProduceTestQuantity(Tup::Tuple{Vararg{Int}}) = ((@assert all(x->x > 0, Tup) "Got invalid Tuple $Tup."); rand(Tup...))


GetOutLength(F::Function, In::Int=_GetArgLength(F)) = GetOutLength(F, ProduceTestQuantity(In))
"""
    GetOutLength(F::Function, input::Union{Number,AbstractVector{<:Number}})
Returns output dimensions of given `F`. If it outputs arrays of more than one dimension, a tuple is returned.
This can also be used to determine the approximate size of the input for mutating `F` which accept 2 arguments.

!!! note
    Discriminates between `Real` and `Vector{Real}` of length one, i.e.:
    `Real`↦`-1` and `x::AbstractVector{<:Real}`↦`length(x)`.
"""
function GetOutLength(F::Function, testinput::Union{Number,AbstractVector{<:Number}})
    if MaximalNumberOfArguments(F) == 1
        output = F(testinput)
        output isa Number ? -1 : (output isa AbstractVector ? length(output) : size(output))
    else
        _GetArgLengthInPlace(F)[1]
    end
end

function GetInOut(F::Function)
    if MaximalNumberOfArguments(F) == 1
        In = _GetArgLengthOutOfPlace(F);    Out = GetOutLength(F,In);    (In,Out)
    else
        # _GetArgLengthInPlace uses opposite order
        _GetArgLengthInPlace(F) |> reverse
    end
end

GetInOut(F::Function, input::Union{Number, AbstractVector{<:Number}}) = (In = (input isa Number ? -1 : length(input)); (In,GetOutLength(F,input)))


function _GetFirstDeriv(F::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:Symbolic))
    if !(Out(InOut) isa Number)
        GetMatrixJac(ADmode,F,abs(In(InOut)),_MakeTuple(Out(InOut)))
    elseif In(InOut) == -1
        # Out is an Int but must be Tuple
        Out(InOut) == -1 ? GetDeriv(ADmode,F) : GetMatrixJac(ADmode,F,abs(In(InOut)),_MakeTuple(Out(InOut)))
    else
        Out(InOut) == -1 ? GetGrad(ADmode,F,abs(In(InOut))) : GetJac(ADmode,F,abs(In(InOut)))
    end
end
function _GetSecondDeriv(F::Function, dF::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:Symbolic))
    if Out(InOut) isa Number && Out(InOut) == -1
        ## For scalar functions, Symbolics often produces false results.
        In(InOut) == -1 ? GetDeriv(ADmode, dF) : GetHess(Val(:ForwardDiff), F, abs(In(InOut)))
    else
        ## Already know info about dimensions, do not call GetDoubleJac.
        GetMatrixJac(ADmode,dF,abs(In(InOut)),(Out(InOut)...,abs(In(InOut))))
    end
end
function _GetSecondDeriv(F::Function, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F); ADmode::Union{Val,Symbol}=Val(:Symbolic))
    _GetSecondDeriv(F, _GetFirstDeriv(F,InOut;ADmode=ADmode), InOut; ADmode=ADmode)
end


"""
    DerivableFunction(F::Function; ADmode::Union{Val,Symbol}=Val(:Symbolic))
    DerivableFunction(F::Function, testinput::Union{Number,AbstractVector{<:Number}}; ADmode::Union{Val,Symbol}=Val(:Symbolic))
    DerivableFunction(F::Function, dF::Function; ADmode::Union{Val,Symbol}=Val(:Symbolic))
    DerivableFunction(F::Function, dF::Function, ddF::Function)
Stores derivatives of a given function (as well as input-output dimensions) for potentially faster computations when derivatives are known.
"""
mutable struct DerivableFunction <: Function
    F::Function
    dF::Function
    ddF::Function
    InOut::Tuple{Val,Val}
    function DerivableFunction(F::Function, testinput::Union{Number,AbstractVector{<:Number}}, InOut::Tuple{Int,Union{Int,Tuple}}=GetInOut(F,testinput); ADmode::Union{Val,Symbol}=Val(:Symbolic))
        DerivableFunction(F, _GetFirstDeriv(F,InOut;ADmode=ADmode), InOut; ADmode=ADmode)
    end
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
    DerivableFunction(D::DerivableFunction, args...; kwargs...) = D
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

# No inference is performed on the input function here, it is assumed to not output scalar values
EvaldF(F::Function, x; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = GetMatrixJac(ADmode, F)(x)
EvaldF(F::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = GetMatrixJac(ADmode, F)
EvalddF(F::Function, x; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = GetMatrixJac(ADmode, GetMatrixJac(ADmode, F))(x)
EvalddF(F::Function; ADmode::Union{Symbol,Val}=Val(:ForwardDiff)) = GetMatrixJac(ADmode, GetMatrixJac(ADmode, F))


_InputSpace(D::DFunction) = _SpaceWord(In(D))
_OutputSpace(D::DFunction) = _SpaceWord(Out(D))
_SpaceWord(n::Int) = n == -1 ? "R" : "R^$n"
_SpaceWord(Tup::Tuple) = "R^(" * prod([string(n)*"×" for n in Tup])[1:end-1] * ")"
Base.show(io::IO, D::DFunction) = println(io, "DerivableFunction : " * _InputSpace(D) * " ⟶  " * _OutputSpace(D))
