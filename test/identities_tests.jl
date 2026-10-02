@testitem "χ = ±(χ₀ - χ₀ F χ₀)" begin
    using SparseIR

    # From the Bethe-Salpeter equations (Eq. 7) with F = Γ ∓ Γχ₀F/β²; the minus sign is for r = s.
    for (U, β) in ((1.3, 2.1), (-0.7, 3.0), (5.0, 2.0), (2.0, 400.0)), r in CHANNELS
        at = HubbardAtom(U, β)
        κ = r isa SingletChannel ? -1 : 1
        for n in -3:2, n´ in -3:2, k in -2:2
            ν, ν´, ω = FermionicFreq(2n + 1), FermionicFreq(2n´ + 1), BosonicFreq(2k)
            χ₀χ₀F = chi0(r, at, (ν, ω)) * full_vertex(r, at, (ν, ν´, ω)) * chi0(r, at, (ν´, ω))
            @test chi(r, at, (ν, ν´, ω)) ≈ κ * chi0(r, at, (ν, ν´, ω)) - χ₀χ₀F rtol = 1e-12 atol = 1e-12 * abs(χ₀χ₀F)
        end
    end
end

@testitem "Λ: crossing symmetry and weak coupling" begin
    using SparseIR

    (d, m, s, t) = CHANNELS
    crossing(Λ, a, Λd, b, Λm) = isapprox(Λ, a * Λd + b * Λm; atol=1e-12 * (abs(Λd) + abs(Λm)))
    for (U, β) in ((1.3, 2.1), (-0.7, 3.0), (5.0, 2.0))
        at = HubbardAtom(U, β)
        L(r, w) = irreducible_vertex(r, at, w)
        for n in -2:1, n´ in -2:1, k in -2:2
            ν, ν´, ω = FermionicFreq(2n + 1), FermionicFreq(2n´ + 1), BosonicFreq(2k)
            w = (ν, ν´, ω)
            pp = (ν, ν´, -ν - ν´ - ω)
            @test crossing(L(s, w), 1 / 2, L(d, pp), -3 / 2, L(m, pp))
            @test crossing(L(t, w), 1 / 2, L(d, pp), 1 / 2, L(m, pp))
            ph = (ν, ν + ω, ν´ - ν)
            @test crossing(L(d, w), -1 / 2, L(d, ph), -3 / 2, L(m, ph))
            @test crossing(L(m, w), -1 / 2, L(d, ph), 1 / 2, L(m, ph))
            ph = (ν, -ν´ - ω, ν´ - ν)
            @test crossing(L(s, w), 1 / 2, L(d, ph), -3 / 2, L(m, ph))
            @test crossing(L(t, w), -1 / 2, L(d, ph), -1 / 2, L(m, ph))
        end
    end

    # To lowest order, Λ is the bare interaction of the channel.
    at = HubbardAtom(1e-4, 2.1)
    w = (FermionicFreq(3), FermionicFreq(-3), BosonicFreq(2))
    for r in (d, m, s)
        @test irreducible_vertex(r, at, w) ≈ bare_vertex(r, at) rtol = 1e-6
    end
    @test abs(irreducible_vertex(t, at, w)) < 1e-10
end

@testitem "Removable singularity in other floating-point types" setup = [Eq19] begin
    using SparseIR

    U₀ = 3.961870127108698  # B_d² = π² for β = 1
    w = (FermionicFreq(1), FermionicFreq(1), BosonicFreq(0))
    for U in (Float32(U₀), nextfloat(Float32(U₀)), prevfloat(Float32(U₀)))
        ref = setprecision(() -> Γref(DensityChannel(), big(U), big(1.0), w), BigFloat, 512)
        @test gamma(DensityChannel(), HubbardAtom(U, 1.0f0), w) ≈ ref rtol = 1e-5
    end
    for prec in (128, 256)
        ref = setprecision(() -> Γref(DensityChannel(), big(U₀), big(1.0), w), BigFloat, 2prec)
        val = setprecision(() -> gamma(DensityChannel(), HubbardAtom(big(U₀), big(1.0)), w), BigFloat, prec)
        @test val ≈ ref rtol = 100 * 2.0^-prec
    end
end

@testitem "ψ(t) = (√t cot √t - 1)/t" begin
    using HubbardAtoms: _ψ

    ψref(t) = (x = big(t); r = sqrt(abs(x)); (x > 0 ? r * cot(r) : r * coth(r)) - 1) / x
    setprecision(BigFloat, 512) do
        for T in (Float64, Float32), t in (-1, -0.5, -1e-3, -1e-9, 1e-9, 1e-3, 0.5, 1)
            @test _ψ(T(t)) ≈ ψref(T(t)) rtol = 4eps(T)
        end
    end
    @test _ψ(0.0) ≈ -1 / 3
    @test _ψ(1e-300) ≈ -1 / 3
    @test _ψ(-1e-300) ≈ -1 / 3
end

@testitem "Return types, type stability and allocations" setup = [AllFunctions] begin
    using SparseIR

    for T in (Float64, Float32, BigFloat), r in CHANNELS
        at = HubbardAtom(T(1.3), T(2.1))
        ν, ν´, ω = FermionicFreq(3), FermionicFreq(-3), BosonicFreq(2)
        w = (ν, ν´, ω)
        # all quantities except the Green's function are real
        @test (@inferred gf(at, ν)) isa Complex{T}
        @test (@inferred bare_vertex(r, at)) isa T
        @test (@inferred chi(r, at, w)) isa T
        @test (@inferred chi(r, at, ω)) isa T
        @test (@inferred chi0(r, at, w)) isa T
        @test (@inferred chi0(r, at, (ν, ω))) isa T
        @test (@inferred full_vertex(r, at, w)) isa T
        @test (@inferred gamma(r, at, w)) isa T
        @test (@inferred irreducible_vertex(r, at, w)) isa T
        @test (@inferred channel_reducible_vertex(r, at, w)) isa T
        @test (@inferred g3(r, at, (ν, ω))) isa T
        r isa TripletChannel || @test (@inferred hedin(r, at, (ν, ω))) isa T
    end

    # Behind function barriers with concrete argument types, so that the test item's globals do not count.
    alloc_gamma(r, at, w) = (gamma(r, at, w); @allocated gamma(r, at, w))
    alloc_F(r, at, w) = (full_vertex(r, at, w); @allocated full_vertex(r, at, w))
    at = HubbardAtom(1.3, 2.1)
    for r in CHANNELS, w in ((FermionicFreq(3), FermionicFreq(-3), BosonicFreq(2)), (FermionicFreq(1), FermionicFreq(1), BosonicFreq(0)))
        @test alloc_gamma(r, at, w) == 0
        @test alloc_F(r, at, w) == 0
    end
end

@testitem "README example" begin
    using SparseIR

    U = 2.0
    beta = 10.0
    at = HubbardAtom(U, beta)
    w = (FermionicFreq(11), FermionicFreq(-3), BosonicFreq(8))
    @test full_vertex(MagneticChannel(), at, w) isa Real
end
