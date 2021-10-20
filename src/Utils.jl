
"""
    suff(x) -> Type
If `x` stores BigFloats, `suff` returns BigFloat, else `suff` returns `Float64`.
"""
suff(x::BigFloat) = BigFloat
suff(x::Float32) = Float32
suff(x::Float16) = Float16
suff(x::Real) = Float64
suff(x::Num) = Num
suff(x::Complex) = suff(real(x))
suff(x::AbstractArray) = suff(x[1])
suff(x::Tuple) = suff(x...)
suff(args...) = try suff(promote(args...)[1]) catch;  suff(args[1]) end
# Allow for differentiation through suff arrays.
suff(x::ForwardDiff.Dual) = typeof(x)
suff(x::ReverseDiff.TrackedReal) = typeof(x)

"""
    MaximalNumberOfArguments(F::Function) -> Int
Infers argument structure of given function, i.e. whether it is of the form `F(x)` or `F(x,y)` or `F(x,y,z)` etc. and returns maximal number of accepted arguments of all overloads of `F` as integer.
"""
MaximalNumberOfArguments(F::Function) = maximum([length(Base.unwrap_unionall(m.sig).parameters)-1 for m in methods(F)])


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

"""
    GetArgLength(F::Function; max::Int=100) -> Int
Attempts to determine input structure of `F`, i.e. whether it accepts `Number`s or `AbstractVector`s and of what length.
This is achieved by successively evaluating the function on `ones(i)` until the evaluation no longer throws errors.
As a result, `GetArgLength` will be unable to determine the correct input structure if `F` errors on `ones(i)`.
"""
function GetArgLength(F::Function; max::Int=100)
    num = MaximalNumberOfArguments(F)
    if num == 1
         _GetArgLengthOutOfPlace(F; max=max)
    elseif num == 2
        _GetArgLengthInPlace(F; max=max)
    else
        throw("Given function $F appears to take $num number of arguments.")
    end
end
function _GetArgLengthOutOfPlace(F::Function; max::Int=100)
    @assert max > 1
    try     F(1.);  return 1    catch; end
    for i in 1:(max+1)
        try
            res = F(ones(i))
            isnothing(res) ? throw("pdim: Function returned Nothing for i=$i.") : res
        catch y
            (isa(y, BoundsError) || isa(y, MethodError) || isa(y, DimensionMismatch) || isa(y, ArgumentError) || isa(y, AssertionError)) && continue
            println("pdim: Encountered error in specification of model function.");     rethrow()
        end
        i == (max + 1) ? throw(ArgumentError("pdim: Parameter space appears to have >$max dims. Aborting. Maybe wrong type of x was inserted?")) : return i
    end
end
function _GetArgLengthInPlace(F::Function; max::Int=100)
    @assert max > 1
    res = 1.;     Res = zeros(max);     RES = zeros(max,max);   RESS = zeros(max,max,max)
    function _TryOn(output, input)
        try
            F(output, input)
            return length(input)
        catch y
            if !(isa(y, BoundsError) || isa(y, MethodError) || isa(y, DimensionMismatch) || isa(y, ArgumentError) || isa(y, AssertionError))
                println("pdim: Encountered error in specification of model function.");     rethrow()
            end
            nothing
        end
    end
    function TryAll(input)
        !isnothing(_TryOn(res, input)) && return length(input)
        !isnothing(_TryOn(Res, input)) && return length(input)
        !isnothing(_TryOn(RES, input)) && return length(input)
        !isnothing(_TryOn(RESS, input)) && return length(input)
        nothing
    end
    X = TryAll(1.);    !isnothing(X) && return 1
    i = 1
    while i < max+1
        X = TryAll(ones(i))
        !isnothing(X) && return i
        i += 1
    end
    throw(ArgumentError("pdim: Parameter space appears to have >$max dims. Aborting. Maybe wrong type of x was inserted?"))
end
