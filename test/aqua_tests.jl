@testitem "Aqua" begin
    using Aqua

    # julia-downgrade-compat moves the test-only dependencies into [deps] in CI
    @testset Aqua.test_all(HubbardAtoms; ambiguities=(recursive=false,), stale_deps=(ignore=[:Aqua, :ReTestItems],))
end
