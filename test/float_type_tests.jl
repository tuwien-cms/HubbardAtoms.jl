@testsetup module AllFunctions
using HubbardAtoms, SparseIR

export results

"All public functions of channel `r` at one generic frequency point."
function results(r, at; ν=FermionicFreq(3), ν´=FermionicFreq(-1), ω=BosonicFreq(2))
    w = (ν, ν´, ω)
    [bare_vertex(r, at), gf(at, ν), chi(r, at, w), chi(r, at, ω), chi0(r, at, w),
     chi0(r, at, (ν, ω)), full_vertex(r, at, w), gamma(r, at, w), irreducible_vertex(r, at, w),
     channel_reducible_vertex(r, at, w), g3(r, at, (ν, ω)), hedin(r, at, (ν, ω))]
end
end

@testitem "Generic floating-point types" setup = [AllFunctions] begin
    U, β = 1.3, 2.1
    for r in CHANNELS
        ref = results(r, HubbardAtom(U, β))

        res32 = results(r, HubbardAtom(Float32(U), Float32(β)))
        @test all(x -> real(typeof(x)) === Float32, res32)
        @test all(isapprox.(res32, ref; rtol=1e-5, atol=1e-6))

        # No Float64 constant may limit the precision: 256 and 512 bits must agree far beyond eps(Float64).
        res256 = setprecision(() -> results(r, HubbardAtom(big(U), big(β))), BigFloat, 256)
        res512 = setprecision(() -> results(r, HubbardAtom(big(U), big(β))), BigFloat, 512)
        @test all(x -> real(typeof(x)) === BigFloat, res256)
        @test all(isapprox.(res256, ref; rtol=1e-13, atol=1e-14))
        @test all(isapprox.(res256, res512; rtol=1e-70, atol=1e-70))
    end

    @test HubbardAtom(2, 10) isa HubbardAtom{Float64}
    @test HubbardAtom(2.0f0, 10) isa HubbardAtom{Float32}
end
