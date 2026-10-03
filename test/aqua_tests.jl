@testitem "Aqua" begin
    using Aqua

    @testset Aqua.test_all(HubbardAtoms; ambiguities=(recursive=false,))
end
