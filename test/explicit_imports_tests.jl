@testitem "ExplicitImports" begin
    using ExplicitImports

    # explicit imports only, and only of public names (runs a test set of its own)
    test_explicit_imports(HubbardAtoms)
end
