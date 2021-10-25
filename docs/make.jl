using DerivableFunctions
using Documenter

DocMeta.setdocmeta!(DerivableFunctions, :DocTestSetup, :(using DerivableFunctions); recursive=true)

makedocs(;
    modules=[DerivableFunctions],
    authors="Rafael Arutjunjan",
    repo="https://github.com/RafaelArutjunjan/DerivableFunctions.jl/blob/{commit}{path}#{line}",
    sitename="DerivableFunctions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RafaelArutjunjan.github.io/DerivableFunctions.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Differentiation Operators" => "Operators.md",
        "DFunctions" => "DFunctions.md"
    ],
)

deploydocs(;
    repo="github.com/RafaelArutjunjan/DerivableFunctions.jl",
    devbranch="master",
)
