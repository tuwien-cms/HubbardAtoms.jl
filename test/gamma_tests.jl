@testsetup module Eq19
using HubbardAtoms, SparseIR

export Γref, B²ref, bisect

# Table I, with the original exp(βU/2) form
function B²ref(r, U, β)
    U²₄ = U^2 / 4
    e = exp(β * U / 2)
    r isa MagneticChannel ? U²₄ * (3 - e) / (1 + e) : U²₄ * (3e - 1) / (1 + e)
end

"""
Literal transcription of Eq. 19, independent of the package implementation. It cancels
catastrophically near Eq. 21, so it is meant to be evaluated in high-precision `BigFloat`.
"""
function Γref(r, U, β, (n, n´, m))
    ν, ν´, ω = SparseIR.value(n, β), SparseIR.value(n´, β), SparseIR.value(m, β)
    U²₄ = U^2 / 4
    i = findfirst(==(r), CHANNELS)
    A² = (3U²₄, -U²₄, zero(U), -U²₄)[i]
    B² = i == 4 ? zero(U) : B²ref(r, U, β)
    𝒜₀ = ℬ₀ = (1, 1, 1 // 2, -1 // 2)[i]
    absℬ₂² = (1, 1, 1 // 2, 0)[i]
    ℬ₁² = (-1, 1, -1 // 2, 0)[i]
    δ(a, b) = a == b ? 1 : 0
    P = (ν^2 + U²₄) * ((ν + ω)^2 + U²₄)

    Γ = β * A² / 2𝒜₀ * P / ((ν * (ν + ω) - A²) * ν * (ν + ω)) * (δ(n, n´) - δ(n, -n´ - m))
    i == 4 && return Γ
    Γ += β * B² / 2ℬ₀ * P / ((ν * (ν + ω) - B²) * ν * (ν + ω)) * (δ(n, n´) + δ(n, -n´ - m))
    sq = √(Complex(4B² + ω^2))
    σ = i == 2 ? -1 : 1
    Γ -= U * absℬ₂² / ℬ₀^2 * U²₄ * (U²₄ * (B² / U²₄ + 1)^2 + ω^2) /
         ((U * tan(β / 4 * (sq + ω)) / sq + σ) * (ν * (ν + ω) - B²) * (ν´ * (ν´ + ω) - B²))
    real(Γ - U * ℬ₁² / ℬ₀^2)
end

function bisect(f, lo, hi)
    flo = f(lo)
    sign(flo) != sign(f(hi)) || error("no sign change")
    for _ in 1:2precision(lo)
        mid = (lo + hi) / 2
        fmid = f(mid)
        sign(fmid) == sign(flo) ? (lo = mid; flo = fmid) : (hi = mid)
    end
    (lo + hi) / 2
end
end

@testitem "Γ against literal Eq. 19" setup = [Eq19] begin
    using SparseIR

    setprecision(BigFloat, 512) do
        for U in (-5.0, -1.3, 0.01, 1.3, 5.0), β in (0.5, 2.1, 10.0), r in CHANNELS
            at = HubbardAtom(U, β)
            for n in -2:1, n´ in -2:1, k in -1:1
                w = (FermionicFreq(2n + 1), FermionicFreq(2n´ + 1), BosonicFreq(2k))
                @test gamma(r, at, w) ≈ Γref(r, big(U), big(β), w) rtol = 1e-11
            end
        end
    end
    @test gamma(DensityChannel(), HubbardAtom(1.0, 1.0), (FermionicFreq(1), FermionicFreq(1), BosonicFreq(0))) isa Float64
end

@testitem "Γ at the removable singularities (Eq. 21)" setup = [Eq19] begin
    using SparseIR

    # Regression for the reported failure (previously ≈ 7.4e15)
    w = (FermionicFreq(1), FermionicFreq(1), BosonicFreq(0))
    @test gamma(DensityChannel(), HubbardAtom(3.961870127108698, 1.0), w) ≈ -37.10545999730081 rtol = 1e-13

    setprecision(BigFloat, 512) do
        # (channel, β, ν = (2n+1)π/β, ω = 2kπ/β, bracket for U), with ν(ν + ω) of either sign. The
        # last three have 2ν + ω = 0, where both Kronecker deltas of Eq. 19 hold and B² = -ω²/4.
        for (r, β, n, k, Ulo, Uhi) in ((DensityChannel(), 1.0, 0, 0, 0.1, 50.0), (SingletChannel(), 1.0, 0, 0, 0.1, 50.0),
                                       (DensityChannel(), 3.0, 1, 1, 0.1, 50.0), (DensityChannel(), 1.0, -2, 1, 0.1, 50.0),
                                       (MagneticChannel(), 1.0, 0, 0, -50.0, -0.1), (MagneticChannel(), 2.0, 0, -2, 0.1, 50.0),
                                       (MagneticChannel(), 1.0, -1, 1, 0.5, 50.0), (DensityChannel(), 1.0, -1, 1, -50.0, -0.5),
                                       (MagneticChannel(), 2.0, 1, -3, 0.5, 50.0))
            βb = big(β)
            ν = SparseIR.value(FermionicFreq(2n + 1), βb)
            ω = SparseIR.value(BosonicFreq(2k), βb)
            U₀ = Float64(bisect(U -> B²ref(r, U, βb) - ν * (ν + ω), big(Ulo), big(Uhi)))
            Us = vcat(U₀, nextfloat(U₀), prevfloat(U₀), [U₀ * (1 + d) for d in (1e-15, 1e-12, 1e-9, 1e-6, 1e-3) for d in (d, -d)])
            # ν´ = ν and ν´ = -ν - ω combine both singular terms, the others only the third
            for U in Us, n´ in (n, -n - k - 1, n + 1, n - 2)
                w = (FermionicFreq(2n + 1), FermionicFreq(2n´ + 1), BosonicFreq(2k))
                @test gamma(r, HubbardAtom(U, β), w) ≈ Γref(r, big(U), big(β), w) rtol = 1e-11
            end
        end
    end
end

@testitem "Γ at U = 0" begin
    using SparseIR

    at = HubbardAtom(0.0, 1.0)
    tiny = HubbardAtom(1e-200, 1.0)
    for r in CHANNELS, n in -2:1, n´ in -2:1, k in -1:1
        w = (FermionicFreq(2n + 1), FermionicFreq(2n´ + 1), BosonicFreq(2k))
        @test iszero(gamma(r, at, w))
        @test iszero(irreducible_vertex(r, at, w))
        @test iszero(channel_reducible_vertex(r, at, w))
        @test abs(gamma(r, tiny, w)) < 1e-190
    end
end

@testitem "Bethe-Salpeter equation at a removable singularity" begin
    using SparseIR

    # Independent of Eq. 19: F - Γ = -1/β Σ_ν₁ Γ(ν, ν₁, ω) χ₀(ν₁, ω) F(ν₁, ν´, ω) at the point of the
    # reported failure; the truncated sum converges like 1/N.
    d = DensityChannel()
    at = HubbardAtom(3.961870127108698, 1.0)
    ν, ω = FermionicFreq(1), BosonicFreq(0)
    N = 10^5
    for ν´ in (FermionicFreq(1), FermionicFreq(3))
        Φ = -sum(j -> gamma(d, at, (ν, FermionicFreq(2j + 1), ω)) * chi0(d, at, (FermionicFreq(2j + 1), ω)) *
                      full_vertex(d, at, (FermionicFreq(2j + 1), ν´, ω)), -N:N-1)
        @test Φ ≈ full_vertex(d, at, (ν, ν´, ω)) - gamma(d, at, (ν, ν´, ω)) rtol = 1e-5
    end
end

@testitem "Γ at low temperatures, at the accuracy of its inputs" setup = [Eq19] begin
    using SparseIR

    # For β ≫ 1 the first two terms of Eq. 19 are ∝ β/ν² and nearly cancel at ν´ = -ν - ω for
    # A² ≈ B² (r = d, m and βU ≫ 1). For ω = 0 and |βU| ≳ 700 the third term is a ratio of two
    # numbers ∝ exp(-|βU|) that underflow in Float64. Γ must be as accurate as a 1-ulp change of U
    # or β allows. The literal Eq. 19 loses ~|βU|/ln(10) digits there, hence the precision.
    for (U, β) in ((3.0, 50.0), (-3.0, 50.0), (3.0, 1e3), (-3.0, 1e3), (20.0, 50.0), (-20.0, 50.0), (20.0, 200.0), (-20.0, 200.0))
        setprecision(BigFloat, 256 + ceil(Int, 3abs(β * U))) do
            for r in CHANNELS, n in (-4, -1, 0), n´ in (-1, 0, 3), k in (-3, 0, 2)
                w = (FermionicFreq(2n + 1), FermionicFreq(2n´ + 1), BosonicFreq(2k))
                ref = Γref(r, big(U), big(β), w)
                sensitivity = max(abs(Γref(r, big(nextfloat(U)), big(β), w) - ref), abs(Γref(r, big(U), big(nextfloat(β)), w) - ref))
                @test abs(gamma(r, HubbardAtom(U, β), w) - ref) <= 10sensitivity + 1e-13 * abs(ref)
            end
        end
    end
end

@testitem "Γ near genuine divergences" setup = [Eq19] begin
    using SparseIR

    # Unlike Eq. 21, these divergences of Eq. 19 are physical and must survive: a local one at
    # ν(ν + ω) = A_d² = 3U²/4 and a global one where U tan(β(s + ω)/4)/s + 1 = 0 (d, β = 1, ν = ν´ = π, ω = 0).
    d = DensityChannel()
    β = 1.0
    w = (FermionicFreq(1), FermionicFreq(1), BosonicFreq(0))
    setprecision(BigFloat, 512) do
        T(U) = (s = 2sqrt(B²ref(d, U, big(β))); U * tan(β * s / 4) / s + 1)
        for U₀ in (2π / sqrt(3), Float64(bisect(T, big(4.0), big(8.0))))
            for δ in (1e-3, 1e-6, 1e-9)
                Γ₊, Γ₋ = (gamma(d, HubbardAtom(U₀ * (1 + x), β), w) for x in (δ, -δ))
                for (U, Γ) in ((U₀ * (1 + δ), Γ₊), (U₀ * (1 - δ), Γ₋))
                    ref = Γref(d, big(U), big(β), w)
                    @test abs(Γ - ref) <= 10abs(Γref(d, big(nextfloat(U)), big(β), w) - ref) + 1e-13 * abs(ref)
                end
                @test sign(Γ₊) == -sign(Γ₋)                   # pole of first order:
                @test abs(Γ₊ * δ) ≈ abs(gamma(d, HubbardAtom(U₀ * (1 + 1e-3), β), w) * 1e-3) rtol = 1e-2
            end
        end
    end
end
