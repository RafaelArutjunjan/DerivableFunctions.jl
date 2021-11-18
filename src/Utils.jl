
"""
    suff(x) -> Type
If `x` stores BigFloats, `suff` returns BigFloat, else `suff` returns `Float64`.
"""
suff(x::BigFloat) = BigFloat
suff(x::Float32) = Float32
suff(x::Float16) = Float16
suff(x::Real) = Float64
suff(x::Complex) = suff(real(x))
suff(x::AbstractArray) = suff(x[1])
suff(x::Tuple) = suff(x...)
suff(x::DataFrame) = suff(x[1,1])
suff(x::Union{Nothing, Missing}) = Float64
suff(args...) = try suff(promote(args...)[1]) catch;  suff(args[1]) end

suff(x::Num) = Num
# Allow for differentiation through suff arrays.
suff(x::ForwardDiff.Dual) = typeof(x)
suff(x::ReverseDiff.TrackedReal) = typeof(x)


"""
    MaximalNumberOfArguments(F::Function) -> Int
Infers argument structure of given function, i.e. whether it is of the form `F(x)` or `F(x,y)` or `F(x,y,z)` etc. and returns maximal number of accepted arguments of all overloads of `F` as integer.
"""
MaximalNumberOfArguments(F::Function) = maximum([length(Base.unwrap_unionall(m.sig).parameters)-1 for m in methods(F)])


"""
    GetArgLength(F::Function; max::Int=100) -> Int
Attempts to determine input structure of `F`, i.e. whether it accepts `Number`s or `AbstractVector`s and of what length.
This is achieved by successively evaluating the function on `rand(i)` until the evaluation no longer throws errors.
As a result, `GetArgLength` will be unable to determine the correct input structure if `F` errors on `rand(i)`.
!!! note
    Does NOT discriminate between `Real` and `Vector{Real}` of length one, i.e. `Real`↦`+1`.
    To disciminate between these two options, use `DerivableFunctions._GetArgLength` instead.
"""
GetArgLength(F::Function; max::Int=100) = _GetArgLength(F; max=max) |> abs

"""
    _GetArgLength(F::Function; max::Int=100) -> Int
!!! note
    Discriminates between `Real` and `Vector{Real}` of length one, i.e.:
    `Real`↦`-1` and `x::AbstractVector{<:Real}`↦`length(x)`.
"""
function _GetArgLength(F::Function; max::Int=100)
    num = MaximalNumberOfArguments(F)
    if num == 1
         _GetArgLengthOutOfPlace(F; max=max)
    elseif num == 2
        _GetArgLengthInPlace(F; max=max)[2]
    else
        throw("Given function $F appears to take $num number of arguments. It should either accept only 1 argument or 2 for functions which mutate their first argument.")
    end
end
function _GetArgLengthOutOfPlace(F::Function; max::Int=100)
    @assert max > 1
    try     F(rand());  return -1    catch; end
    for i in 1:(max+1)
        try
            res = F(rand(i))
            isnothing(res) && throw("Function returned Nothing for i=$i.")
        catch y
            (isa(y, BoundsError) || isa(y, MethodError) || isa(y, DimensionMismatch) || isa(y, ArgumentError) || isa(y, AssertionError)) && continue
            @warn "Encountered apparent error in specification of function."
            rethrow()
        end
        i < (max + 1) ? (return i) : throw(ArgumentError("Function input appears to have >$max components, aborting. Either increase keyword max or ensure function errors on rand(i) for i larger than true component length."))
    end
end
function _GetArgLengthInPlace(F::Function; max::Int=100)
    @assert max > 1
    res = 0.;     Res = zeros(max);     RES = zeros(max,max);   RESS = zeros(max,max,max)
    function _TryOn(output, input)
        try
            F(output, input)
        catch y
            if !(isa(y, BoundsError) || isa(y, MethodError) || isa(y, DimensionMismatch) || isa(y, ArgumentError) || isa(y, AssertionError))
                @warn "Encountered apparent error in specification of function."
                rethrow()
            end
            return nothing
        end;    (FindSubHyperCube(output), (input isa Number ? -1 : length(input)))
    end
    function TryAll(input)
        for AnOut in (res, Res, RES, RESS)
            Z = _TryOn(AnOut, input)
            !isnothing(Z) && return Z
        end
        nothing
    end
    X = TryAll(rand());    !isnothing(X) && return X
    for i in 1:max
        X = TryAll(rand(i))
        !isnothing(X) && return X
    end
    throw(ArgumentError("Function input appears to have >$max components, aborting. Either increase keyword max or ensure function errors on rand(i) for i larger than true component length."))
end

FindSubHyperCube(N::Number, tester::Function=(x->x!=0.0)) = -1
FindSubHyperCube(Vec::AbstractVector{<:Number}, tester::Function=(x->x!=0.0)) = findlast(tester, Vec)

# Still highly inefficient but apparently works without bugs
function FindSubHyperCube(Mat::AbstractArray{<:Number, dim}, tester::Function=(x->x!=0.0)) where dim
    Size = [size(Mat)...];    Biggest = copy(Size)
    for ind in 1:dim
        for val in 1:Size[ind]
            if all(!tester, view(Mat, [Colon() for j in 1:ind-1]..., val, [Colon() for j in ind+1:dim]...))
                if all(!tester, view(Mat, [1:Biggest[i] for i in 1:ind-1]..., val+1:Size[ind], [Colon() for j in ind+1:dim]...))
                    Biggest[ind] = val-1;   break
                end
            end
        end
    end;    Tuple(Biggest)
end


"""
    KillAfter(F::Function, args...; timeout::Real=5, verbose::Bool=false, kwargs...)
Tries to evaluate a given function `F` before a set `timeout` limit is reached and interrupts the evaluation and returns `nothing` if necessary.
NOTE: The given function is evaluated via F(args...; kwargs...).
"""
function KillAfter(F::Function, args...; timeout::Real=5, verbose::Bool=false, kwargs...)
    Res = nothing
    G() = try F(args...; kwargs...) catch Err
        if verbose
            if Err isa DivideError
                @warn "KillAfter: Could not evaluate given Function $(nameof(F)) before timeout limit of $timeout seconds was reached."
            else
                @warn "KillAfter: Could not evaluate given Function $(nameof(F)) because error was thrown: $Err."
            end
        end
    end
    task = @async(G())
    if timedwait(()->istaskdone(task), timeout) == :timed_out
        @async(Base.throwto(task, DivideError())) # kill task
    else
        Res = fetch(task)
    end;    Res
end

"""
    Builder(Fexpr::Union{<:AbstractVector{<:Num},<:Num}, args...; inplace::Bool=false, parallel::Bool=false, kwargs...)
Builds `RuntimeGeneratedFunctions` from expressions via build_function().
"""
function Builder(Fexpr::Union{<:AbstractArray{<:Num},<:Num}, args...; inplace::Bool=false, parallel::Bool=false, kwargs...)
    parallelization = parallel ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    Res = if (Fexpr isa Num && args[1] isa Num)
        # build_function throws error when using parallel keyword for R⟶R functions
        Symbolics.build_function(Fexpr, args...; expression=Val{false}, kwargs...)
    else
        Symbolics.build_function(Fexpr, args...; expression=Val{false}, parallel=parallelization, kwargs...)
    end
    try
        Res[inplace ? 2 : 1]
    catch;
        Res
    end
end
