@testitem "Constructor validation" begin
    for (U, β) in ((1.0, 0.0), (1.0, -1.0), (1.0, Inf), (1.0, NaN), (Inf, 1.0), (-Inf, 1.0), (NaN, 1.0))
        @test_throws DomainError HubbardAtom(U, β)
    end
end

@testitem "Extreme βU" setup = [AllFunctions] begin
    using SparseIR

    # exp(±βU/2) over- or underflows in Float64 for all of these; BigFloat does not.
    for (U, β) in ((2.0, 800.0), (-2.0, 800.0), (5.0, 1e4), (-5.0, 1e4))
        at = HubbardAtom(U, β)
        for r in CHANNELS
            res = results(r, at)
            ref = results(r, HubbardAtom(big(U), big(β)))
            @test all(isfinite, res)
            # Entry 9 is `irreducible_vertex`: for β = 1e4, Λ_t ≈ 1e-3 is a sum of Γ's of size up to
            # 45, each of which is only determined to ≈ 6e-12 by a 1-ulp change of U.
            @test all(isapprox(res[i], ref[i]; rtol=i == 9 ? 1e-8 : 1e-10, atol=1e-300) for i in eachindex(res))
        end

        # The atom is empty or doubly occupied with probability p = 1/(1 + exp(βU/2)),
        # which is 1 for βU → -∞ and 0 for βU → +∞.
        p = U < 0 ? 1 : 0
        @test chi(DensityChannel(), at, BosonicFreq(0)) ≈ -β * p
        @test chi(MagneticChannel(), at, BosonicFreq(0)) ≈ -β * (1 - p)
        @test chi(SingletChannel(), at, BosonicFreq(0)) ≈ -β * p / 2
    end
end
