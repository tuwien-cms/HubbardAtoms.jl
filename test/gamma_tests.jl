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
