
### DFunctions

The `DFunction` type stores the first and second derivatives of a given input function which is not only convenient but can enhance performance significantly. At this point, the `DFunction` type requires the given function to be out-of-place, however, this will likely be extended to in-place functions in the future.

Once constructed, a `DFunction` object `D` can be evaluated at `x` via the syntax `EvalF(D,x)`, `EvaldF(D,x)` and `EvalddF(D,x)`.

In order to construct the appropriate derivatives, the input and output dimensions of a given function `F` are assessed and the appropriate operators (`GetGrad(), GetJac()` and so on) called.

By default, `DFunction()` attempts to construct the derivatives symbolically, however, this can be specified via the `ADmode` keyword:
```@example 2
using DerivableFunctions

D = DFunction(x->[x^7 - sin(x), tanh(x)]; ADmode=Val(:ReverseDiff))
EvalF(D, 5.), EvaldF(D, 5.), EvalddF(D, 5.)

using Symbolics;  @variables y
EvalF(D, y), EvaldF(D, y), EvalddF(D, y)
```
