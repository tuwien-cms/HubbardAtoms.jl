@testitem "G₃ from the generalized susceptibility" begin
    using SparseIR

    # G₃(r) = -1/β Σ_ν´ χ(r, (ν, ν´, ω)); the truncated sum converges like 1/N.
    N = 20_000
    for (U, β) in ((0.0, 1.0), (1.3, 2.1), (-0.7, 3.0), (5.0, 2.0)), r in CHANNELS[1:3], (n, k) in ((0, 0), (0, 1), (-2, 2), (2, -1))
        at = HubbardAtom(U, β)
        ν, ω = FermionicFreq(2n + 1), BosonicFreq(2k)
        S = sum(chi(r, at, (ν, FermionicFreq(2j + 1), ω)) for j in -N:N-1)
        @test g3(r, at, (ν, ω)) ≈ -S / β rtol = 1e-4
    end
    @test iszero(g3(TripletChannel(), HubbardAtom(1.3, 2.1), (FermionicFreq(1), BosonicFreq(2))))
end

@testitem "Integrated susceptibilities" begin
    using SparseIR

    for (U, β) in ((0.0, 1.0), (1.3, 2.1), (-0.7, 3.0), (5.0, 2.0))
        at = HubbardAtom(U, β)

        # Exact ensemble of the four atomic states |0⟩, |↑⟩, |↓⟩, |↑↓⟩ with H = U(n↑ - 1/2)(n↓ - 1/2)
        w₀, w₁ = exp(-β * U / 4), exp(β * U / 4)
        Z = 2w₀ + 2w₁
        @test chi(DensityChannel(), at, BosonicFreq(0)) ≈ -β * 2w₀ / Z    # -β⟨(n - 1)²⟩
        @test chi(MagneticChannel(), at, BosonicFreq(0)) ≈ -β * 2w₁ / Z   # -β⟨(n↑ - n↓)²⟩
        @test chi(SingletChannel(), at, BosonicFreq(0)) ≈ -β * w₀ / Z     # -β⟨c↓c↑ c↑⁺c↓⁺⟩

        for r in CHANNELS
            @test iszero(chi(r, at, BosonicFreq(2)))
            @test iszero(chi(TripletChannel(), at, BosonicFreq(0)))
        end

        # chi(r, ω) = -2/β² Σ_νν´ χ(r, (ν, ν´, ω)) = 2/β Σ_ν G₃(r, (ν, ω))
        for r in CHANNELS[1:3], k in (0, 1)
            S = sum(g3(r, at, (FermionicFreq(2j + 1), BosonicFreq(2k))) for j in -200_000:199_999)
            @test chi(r, at, BosonicFreq(2k)) ≈ 2S / β atol = 1e-4
        end
    end
end

@testitem "Hedin vertex" begin
    using SparseIR

    (d, m, s, t) = CHANNELS
    sign = Dict(d => 1, m => 1, s => -1)
    for (U, β) in ((0.0, 1.0), (1.3, 2.1), (-0.7, 3.0), (5.0, 2.0)), n in -3:2, k in -2:2
        at = HubbardAtom(U, β)
        w = (FermionicFreq(2n + 1), BosonicFreq(2k))
        U == 0 && @test all(r -> hedin(r, at, w) ≈ sign[r], (d, m, s))
        @test hedin(s, at, w) ≈ -hedin(d, at, w)
        # Asymptotics for |ν| → ∞
        for r in (d, m, s)
            @test hedin(r, at, (FermionicFreq(2 * 10^6 + 1), BosonicFreq(2k))) ≈ sign[r] rtol = 1e-6
        end
    end
    @test_throws ArgumentError hedin(t, HubbardAtom(1.3, 2.1), (FermionicFreq(1), BosonicFreq(0)))
end

@testitem "Hedin equation for the self-energy" begin
    using SparseIR

    # Krien and Valli, Phys. Rev. B 100, 245147 (2019), Eq. (B2) with r = 1/2: apart from the Hartree
    # term, Σ(ν) = -1/(2β) Σ_ω G(ν + ω) (w_ch λ_ch + w_sp λ_sp) with w = U + U χ U / 2. For ω ≠ 0,
    # χ = 0 and λ_ch = λ_sp, so the terms cancel and only ω = 0 remains. For the atom Σ(ν) = U²/(4iν).
    (d, m) = CHANNELS
    for (U, β) in ((1.3, 2.1), (-0.7, 3.0), (5.0, 2.0), (0.4, 20.0))
        at = HubbardAtom(U, β)
        ω = BosonicFreq(0)
        w_ch = U * (1 + U * chi(d, at, ω) / 2)
        w_sp = -U * (1 - U * chi(m, at, ω) / 2)
        for n in -3:2
            ν = FermionicFreq(2n + 1)
            Σ = -gf(at, ν) * (w_ch * hedin(d, at, (ν, ω)) + w_sp * hedin(m, at, (ν, ω))) / 2β
            @test Σ ≈ U^2 / (4 * SparseIR.valueim(ν, β))
            @test hedin(d, at, (ν, BosonicFreq(2))) == hedin(m, at, (ν, BosonicFreq(2)))
        end
    end
end
