# DerivableFunctions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RafaelArutjunjan.github.io/DerivableFunctions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RafaelArutjunjan.github.io/DerivableFunctions.jl/dev)
[![Build Status](https://github.com/RafaelArutjunjan/DerivableFunctions.jl/workflows/CI/badge.svg)](https://github.com/RafaelArutjunjan/DerivableFunctions.jl/actions)


This package provides a front-end for differentiation operations which allows for code that is agnostic with respect to many of the available automatic and symbolic differentiation tools available in Julia. Moreover, the functors provided by **DerivableFunctions.jl** are also overloaded to allow for passthrough of symbolic variables. That is, if symbolic types such as `Symbolics.Num` are detected, the differentiation functors automatically switch to symbolic differentiation.
