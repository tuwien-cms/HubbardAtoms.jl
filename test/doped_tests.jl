@testsetup module Doped
using HubbardAtoms, SparseIR, LinearAlgebra

export F, B, boxgamma, bisect

F(n) = FermionicFreq(2n + 1)
B(k) = BosonicFreq(2k)

"""
Γ from the numerical inverse of χ on the box |2n + 1 + k| ≤ 2M + 1, Richardson-extrapolated
(quadratically in 1/M) from M, 2M and 4M (independent of the closed forms for Γ).
"""
function boxgamma(r, at, k, pts; M=64)
    η = r isa SingletChannel ? -1 : 1
    vals = map((M, 2M, 4M)) do M
        idx = [(j - 1 - k) ÷ 2 for j in (-(2M + 1)):(2M + 1) if iseven(j - 1 - k)]
        X = [chi(r, at, (F(a), F(b), B(k))) for a in idx, b in idx]
        Y = inv(X)
        pos(x) = findfirst(==(x), idx)
        [at.beta^2 * (Y[pos(n), pos(m)] - (n == m) * η / chi0(r, at, (F(n), F(n), B(k))))
         for (n, m) in pts]
    end
    (8vals[3] - 6vals[2] + vals[1]) / 3
end

"Bisection for a sign change of the real function `f` in (lo, hi), in BigFloat."
function bisect(f, lo, hi)
    lo, hi = big(lo), big(hi)
    flo = sign(f(lo))
    for _ in 1:300
        mid = (lo + hi) / 2
        sign(f(mid)) == flo ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end
end

@testitem "Doped atom: constructor" begin
    for (U, β, μ) in ((1.0, 0.0, 0.0), (1.0, -1.0, 0.0), (1.0, Inf, 0.0), (Inf, 1.0, 0.0),
                      (NaN, 1.0, 0.0),
                      (1.0, 1.0, Inf), (1.0, 1.0, NaN))
        @test_throws DomainError DopedHubbardAtom(U, β, μ)
    end
    @test DopedHubbardAtom(2, 10, 1) isa DopedHubbardAtom{Float64}
    @test DopedHubbardAtom(2.0f0, 10, 0.5f0) isa DopedHubbardAtom{Float32}
    @test DopedHubbardAtom(2.0, 3.0, 0.3) == DopedHubbardAtom(2.0, 3.0, 0.3)
    @test hash(DopedHubbardAtom(2.0, 3.0, 0.3)) == hash(DopedHubbardAtom(2.0, 3.0, 0.3))
    @test DopedHubbardAtom(2.0, 3.0, 0.3) != DopedHubbardAtom(2.0, 3.0, -0.3)
    @test sprint(show, DopedHubbardAtom(2.0, 3.0, 0.3)) == "DopedHubbardAtom(2.0, 3.0, 0.3)"
end

@testitem "Doped atom: half filling" setup = [AllFunctions] begin
    using SparseIR

    # At μ = 0 every function agrees with the half-filled atom.
    for (U, β) in ((1.3, 2.1), (-0.7, 3.0), (5.0, 2.0), (4.0, 20.0), (-6.0, 30.0)),
        r in CHANNELS
        for (ν, ν′, ω) in ((FermionicFreq(3), FermionicFreq(-3), BosonicFreq(2)),
                           (FermionicFreq(1), FermionicFreq(-1), BosonicFreq(0)),
                           (FermionicFreq(-1), FermionicFreq(5), BosonicFreq(-4)))
            ref = results(r, HubbardAtom(U, β); ν, ν′, ω)
            res = results(r, DopedHubbardAtom(U, β, 0.0); ν, ν′, ω)
            # entry 9 (Λ) is a sum of much larger terms
            scale = maximum(abs, ref)
            @test all(isapprox(res[i], ref[i]; rtol=1e-12,
                               atol=(i == 9 ? 1e-12 : 1e-13) * scale)
                      for i in eachindex(ref))
            # and continuously so (at low temperature, the dependence on μ is genuinely strong)
            β <= 3 || continue
            res = results(r, DopedHubbardAtom(U, β, 1e-12); ν, ν′, ω)
            @test all(isapprox(res[i], ref[i]; rtol=1e-10, atol=1e-11 * scale)
                      for i in eachindex(ref))
        end
    end
end

@testitem "Doped atom: exact diagonalization" setup = [ExactDiagonalization, Doped] begin
    using SparseIR, LinearAlgebra

    (d, m, s, t) = CHANNELS
    for (U, β, μ) in
        ((2.0, 3.0, 0.3), (-1.5, 2.0, -0.4), (3.0, 5.0, 0.7), (1.0, 0.5, -2.0),
         (0.0, 1.0, 0.5))
        at = DopedHubbardAtom(U, β, μ)
        for n in -3:2
            @test gf(at, F(n)) ≈ G_ed(U, β, n; μ) rtol = 1e-12
        end
        for k in -2:2
            @test chi(d, at, B(k)) ≈ χ_ed(U, β, nup + ndn - I, k; μ) atol = 1e-12
            @test chi(m, at, B(k)) ≈ χ_ed(U, β, nup - ndn, k; μ) atol = 1e-12
            # -⟨c↓c↑; c↑⁺c↓⁺⟩ at -ω (the two agree at half filling)
            @test chi(s, at, B(k)) ≈ χpair_ed(U, β, -k; μ) atol = 1e-12
        end
        U == 0 && continue  # the connected parts vanish
        for n in -2:1, n′ in -2:1, k in -1:1
            χ_uu, χ_ud = χph_ed(U, β, true, n, n′, k; μ), χph_ed(U, β, false, n, n′, k; μ)
            kpp = -k - (n + n′ + 1)
            χpp_uu, χpp_ud = χph_ed(U, β, true, n, n′, kpp; μ),
                             χph_ed(U, β, false, n, n′, kpp; μ)
            χ0pp = n == n′ ? -β / 2 * G_ed(U, β, n; μ) * G_ed(U, β, -n - k - 1; μ) : 0
            w = (F(n), F(n′), B(k))
            scale = max(abs(χ_uu), abs(χ_ud), abs(χpp_uu), abs(χpp_ud), abs(χ0pp))
            @test chi(d, at, w) ≈ χ_uu + χ_ud atol = 1e-12 * scale
            @test chi(m, at, w) ≈ χ_uu - χ_ud atol = 1e-12 * scale
            @test chi(s, at, w) ≈ (-χpp_uu + 2χpp_ud - 2χ0pp) / 4 atol = 1e-12 * scale
            @test chi(t, at, w) ≈ (χpp_uu + 2χ0pp) / 4 atol = 1e-12 * scale
        end
    end
end

@testitem "Doped atom: Γ against finite-box inversion of χ" setup = [Doped] begin
    for (U, β, μ) in ((2.0, 3.0, 0.3), (-1.5, 2.0, -0.4), (4.0, 8.0, 0.9)), k in (0, 1),
        r in CHANNELS
        at = DopedHubbardAtom(U, β, μ)
        pts = ((0, 0), (0, -k - 1), (2, -1), (-3, 1))
        ref = boxgamma(r, at, k, pts)
        val = [gamma(r, at, (F(n), F(m), B(k))) for (n, m) in pts]
        @test val ≈ ref rtol = 1e-5 atol = 1e-5 * maximum(abs, val)
    end
end

@testitem "Doped atom: generic floating-point types" setup = [AllFunctions] begin
    U, β, μ = 1.3, 2.1, 0.4
    for r in CHANNELS
        ref = results(r, DopedHubbardAtom(U, β, μ))
        @test all(x -> real(typeof(x)) === Float64, ref)

        res32 = results(r, DopedHubbardAtom(Float32(U), Float32(β), Float32(μ)))
        @test all(x -> real(typeof(x)) === Float32, res32)
        @test all(isapprox.(res32, ref; rtol=1e-4, atol=1e-5))

        res256 = setprecision(() -> results(r, DopedHubbardAtom(big(U), big(β), big(μ))),
                              BigFloat, 256)
        res512 = setprecision(() -> results(r, DopedHubbardAtom(big(U), big(β), big(μ))),
                              BigFloat, 512)
        @test all(x -> real(typeof(x)) === BigFloat, res256)
        @test all(isapprox.(res256, ref; rtol=1e-12, atol=1e-13))
        @test all(isapprox.(res256, res512; rtol=1e-70, atol=1e-70))
    end
end

@testitem "Doped atom: range of Float32" setup = [AllFunctions] begin
    using SparseIR

    # No intermediate product may over- or underflow: for energy units far from 1, for large frequencies,
    # and for distinct frequencies that round to the same Float32 value. Reference: Float64.
    for (c, ν, ν′, ω) in ((1e-6, FermionicFreq(1), FermionicFreq(3), BosonicFreq(0)),
                          (1e6, FermionicFreq(1), FermionicFreq(-5), BosonicFreq(2)),
                          (1.0, FermionicFreq(2 * 10^8 + 1), FermionicFreq(2 * 10^8 + 3), BosonicFreq(2)),
                          (1.0, FermionicFreq(33554435), FermionicFreq(33554437), BosonicFreq(0)))
        U, β, μ = 2c, 3 / c, 0.3c
        for r in CHANNELS
            res = results(r, DopedHubbardAtom(Float32(U), Float32(β), Float32(μ)); ν, ν′, ω)
            ref = results(r, DopedHubbardAtom(U, β, μ); ν, ν′, ω)
            # entries 9 (Λ) and 10 (Φ = F - Γ) are sums of terms of the size of U
            @test all(isapprox(res[i], ref[i]; rtol=1e-3,
                               atol=i in (9, 10) ? 1e-3 * abs(U) : 0)
                      for i in eachindex(ref))
        end
    end
end

@testitem "Doped atom: accuracy against BigFloat" setup = [Doped] begin
    # Including low temperatures, attractive U, μ near 0 and in and outside the gap, and large frequencies.
    setprecision(BigFloat, 320) do
        for U in (-6.0, -0.5, 2.0, 6.0), β in (0.5, 8.0, 30.0),
            μ in (0.0, 1e-9, 0.3, -0.9, 4.0), k in (0, 1, -3, 50)
            at = DopedHubbardAtom(U, β, μ)
            bt = DopedHubbardAtom(big(U), big(β), big(μ))
            for r in CHANNELS,
                (n, m) in ((0, 0), (0, -k - 1), (2, -3), (40, 7), (1000, 1001))
                w = (F(n), F(m), B(k))
                for f in (gamma, full_vertex, chi)
                    @test f(r, at, w) ≈ f(r, bt, w) rtol = 1e-11
                end
            end
        end
    end
end

@testitem "Doped atom: Float64 setups" setup = [Doped] begin
    using HubbardAtoms: _ph_setup_raw, _pp_setup_raw, _floatok

    (d, m, s, t) = CHANNELS
    # Where the plain closed forms cancel in Float64: low temperature (U < 0 with k = 0, near half
    # filling, U > 0 away from it), large transfers, a root of e near the lattice. The setups need
    # no BigFloat (no tiny weight, estimated loss ≤ 6 bits), and Γ agrees with BigFloat.
    cases = ((d, -2.0, 30.0, 0.01, 0, (0, 1)), (d, -6.0, 100.0, 1e-6, 1, (0, 1)),
             (m, -1.0883204881841495, 29.24155260783198, 0.0039439465086138145, 5000,
              (0, 1)),
             (m, 13.79515781611038, 35.133852121266685, -0.0875681483096054, -30, (14, 17)),
             (d, 7.145931675912827, 10.1102742644105, 0.0041009900553070034, -200, (-1, 2)),
             (s, -0.6595262732322414, 997.5299430008005, -3.7701189911842553e-8, -5000,
              (40, 7)))
    # and z real next to a root of e, with v far away (Q in Float64 for a setup in BigFloat), and a
    # tiny weight p₀ ≈ 1e-175 (setup in BigFloat)
    extra = ((d, -0.09211723649223727, 177.72502978267823, -0.0063991951577591875, -1,
              (0, 42)), (m, 6.0, 100.0, 1.0, 1, (0, 1)))
    setprecision(BigFloat, 256) do
        for (r, U, β, μ, k, (n, n′)) in (cases..., extra...)
            at = DopedHubbardAtom(U, β, μ)
            if (r, U, β, μ, k, (n, n′)) in cases
                raw = r == s ? _pp_setup_raw(at, B(k)) :
                      _ph_setup_raw(at, B(k), r == d ? 1 : -1)
                @test _floatok(at) && raw.loss <= 6
            end
            w = (F(n), F(n′), B(k))
            ref = gamma(r, DopedHubbardAtom(big(U), big(β), big(μ)), w)
            @test gamma(r, at, w) ≈ ref rtol = 1e-13 atol = 1e-13 * abs(U)
        end
    end
end

@testitem "Doped atom: extreme βU" setup = [AllFunctions] begin
    using SparseIR

    # The thermal weights over- or underflow in Float64 without care; at (-2, 800, 0), p₁ = 0
    for (U, β, μ) in
        ((2.0, 800.0, 0.3), (-2.0, 800.0, 0.3), (5.0, 1e4, -1.0), (2.0, 800.0, 3.0),
         (-2.0, 800.0, 0.0))
        at = DopedHubbardAtom(U, β, μ)
        for r in CHANNELS,
            (ν′, ω) in
            ((FermionicFreq(-3), BosonicFreq(2)), (FermionicFreq(-3), BosonicFreq(0)))

            res = results(r, at; ν′, ω)
            ref = setprecision(() -> results(r, DopedHubbardAtom(big(U), big(β), big(μ));
                                             ν′, ω), BigFloat, 256)
            @test all(isfinite, res)
            # entries 9 (Λ) and 10 (Φ = F - Γ) are sums of much larger terms (Λ contains Γ of all channels)
            scale = maximum(abs, ref)
            @test all(isapprox(res[i], ref[i]; rtol=1e-10,
                               atol=i in (9, 10) ? 1e-12 * scale : 1e-300)
                      for i in eachindex(res))
        end
    end
    # in Float32, p₁ underflows to 0 already at β = 110
    s, w = SingletChannel(), (FermionicFreq(1), FermionicFreq(3), BosonicFreq(0))
    ref = setprecision(() -> gamma(s, DopedHubbardAtom(big(-2.0), big(110.0), big(0.0)), w),
                       BigFloat, 256)
    @test gamma(s, DopedHubbardAtom(-2.0f0, 110.0f0, 0.0f0), w) ≈ ref rtol = 1e-5
end

@testitem "Doped atom: removable singularities" setup = [Doped] begin
    using HubbardAtoms: _ph_setup_raw

    (d, m, s, t) = CHANNELS
    # Parameters (rounded to Float64) where a denominator of the closed form vanishes but Γ is finite;
    # see the note. Float64 must agree with BigFloat at the same parameters.
    function check(r, U, β, μ, k, pts)
        at = DopedHubbardAtom(Float64(U), Float64(β), Float64(μ))
        bt = DopedHubbardAtom(big(Float64(U)), big(Float64(β)), big(Float64(μ)))
        for (n, m) in pts
            w = (F(n), F(m), B(k))
            @test gamma(r, at, w) ≈ gamma(r, bt, w) rtol = 1e-12
        end
    end
    p(U, β, μ) = (a=DopedHubbardAtom(U, β, μ); (a._p0, a._p1, a._p2))
    setprecision(BigFloat, 512) do
        # coinciding roots of e, at s ≠ 0 and at s = 0
        μs = bisect(μ -> real(_ph_setup_raw(DopedHubbardAtom(big(2.0), big(2.0), μ), B(0),
                                            1).Δs^2), 0.2, 0.4)
        check(d, 2, 2, μs, 0, ((0, 0), (0, 1), (2, -3)))
        check(m, 2, 6.5237953660, 0.6092416812, 2, ((0, 0), (0, 1), (2, -3)))
        # half filling, magnetic, βa = ln 3: û = 0
        check(m, 2, log(big(3)), 0, 0, ((0, 0), (0, 1), (2, -3)))
        # singlet, t₀ + Aₛ = 0
        βs = bisect(β -> (q=p(big(2.0), β, big(0.5)); x=big(0.5) + im * big(π) / β;
                          real(x * (1 - x) + (q[3] - q[1]) + (1 - 8q[2]))), 2.21, 2.22)
        check(s, 2, βs, 0.5, 0, ((0, 0), (0, -1), (0, 3), (2, 2)))
        # lattice zero of e at a mirror fixed point, and on a mirror orbit {-1, -2}
        e(D, z) = (z^2 - D.s1) * (z^2 - D.s2)
        μe = bisect(μ -> real(e(_ph_setup_raw(DopedHubbardAtom(big(-9.0), big(1.0), μ),
                                              B(1), 1), μ)), 0.7, 0.8)
        check(d, -9, 1, μe, 1, ((-1, -1), (-1, 0), (0, 2), (2, -3)))
        # (β, μ) of the orbit zero, from a two-dimensional Newton iteration, rounded to 12 digits
        βp = 9.795466760219
        check(d, -2.5, βp, 0.105710538361, 2,
              ((-1, -1), (-1, -2), (-2, -2), (-1, 0), (2, -5)))
        # half filling: βH(z₀) = +λh (removable) at k = 0, and both kinds at the fixed point z = 0 (odd k)
        βh = bisect(β -> (q=p(big(2.0), β, big(0));
                          (big(π) / β)^2 - (1 + 4 * (q[2] - q[1]))), 1.9, 2.1)
        check(d, 2, βh, 0, 0, ((0, 0), (0, -1), (0, 2), (1, 1)))
        βa = bisect(β -> (q=p(big(2.0), β, big(0)); 4 * (q[2] - q[1]) - 1 - (big(π) / β)^2),
                    3.3, 3.45)
        check(m, 2, βa, 0, 1, ((-1, -1), (-1, 0), (0, 0), (2, -3)))
        check(m, 2, big(π), 0, 1, ((-1, -1), (-1, 0), (0, 0), (2, -3)))
    end
    # and against finite-box inversion, at the coinciding roots
    at = DopedHubbardAtom(2.0, 2.0, 0.2870446966721)
    pts = ((0, 0), (0, 1), (2, -3))
    @test [gamma(d, at, (F(n), F(m), B(0))) for (n, m) in pts] ≈ boxgamma(d, at, 0, pts) rtol = 1e-5
end

@testitem "Doped atom: crossing relations and Λ" setup = [Doped] begin
    (d, m, s, t) = CHANNELS
    crossing(X, a, Xd, b, Xm) = isapprox(X, a * Xd + b * Xm;
                                         atol=1e-12 * (abs(Xd) + abs(Xm)))
    for (U, β, μ) in ((1.3, 2.1, 0.4), (-0.7, 3.0, -0.2), (5.0, 2.0, 1.5)),
        f in (full_vertex, irreducible_vertex)
        at = DopedHubbardAtom(U, β, μ)
        X(r, w) = f(r, at, w)
        for n in -2:1, n′ in -2:1, k in -2:2
            ν, ν′, ω = F(n), F(n′), B(k)
            w = (ν, ν′, ω)
            pp = (ν, ν′, -ν - ν′ - ω)
            @test crossing(X(s, w), 1 / 2, X(d, pp), -3 / 2, X(m, pp))
            @test crossing(X(t, w), 1 / 2, X(d, pp), 1 / 2, X(m, pp))
            ph = (ν, ν + ω, ν′ - ν)
            @test crossing(X(d, w), -1 / 2, X(d, ph), -3 / 2, X(m, ph))
            @test crossing(X(m, w), -1 / 2, X(d, ph), 1 / 2, X(m, ph))
            ph = (ν, -ν′ - ω, ν′ - ν)
            @test crossing(X(s, w), 1 / 2, X(d, ph), -3 / 2, X(m, ph))
            @test crossing(X(t, w), -1 / 2, X(d, ph), -1 / 2, X(m, ph))
        end
    end
    # Λ is fully irreducible: Λ - U_r = O(U⁴), while Γ - U_r = O(U²)
    w = (F(0), F(1), B(1))
    for r in CHANNELS
        dev(U) = abs(irreducible_vertex(r, DopedHubbardAtom(U, 2.0, 0.3), w) -
                     bare_vertex(r, DopedHubbardAtom(U, 2.0, 0.3)))
        @test dev(0.1) / dev(0.05) ≈ 16 rtol = 0.05
    end
end

@testitem "Doped atom: χ = ηχ₀ - χ₀ F χ₀ and the BSE" setup = [Doped] begin
    for (U, β, μ) in ((1.3, 2.1, 0.4), (-0.7, 3.0, -0.2)), r in CHANNELS
        at = DopedHubbardAtom(U, β, μ)
        κ = r isa SingletChannel ? -1 : 1
        for n in -3:2, n′ in -3:2, k in -2:2
            ν, ν′, ω = F(n), F(n′), B(k)
            χ₀χ₀F = chi0(r, at, (ν, ω)) * full_vertex(r, at, (ν, ν′, ω)) *
                    chi0(r, at, (ν′, ω))
            tol = 1e-12 * abs(χ₀χ₀F)
            @test chi(r, at, (ν, ν′, ω)) ≈ κ * chi0(r, at, (ν, ν′, ω)) - χ₀χ₀F rtol = 1e-12 atol = tol
        end
        # Φ = F - Γ = ∓(1/β) Σ Γ χ₀ F (BSE); the truncated sum converges like 1/N, so extrapolate
        for (n, n′, k) in ((0, 1, 0), (-2, 1, 1))
            ν, ν′, ω = F(n), F(n′), B(k)
            S(N) = sum(gamma(r, at, (ν, F(j), ω)) * chi0(r, at, (F(j), ω)) *
                       full_vertex(r, at, (F(j), ν′, ω)) for j in (-N - k):(N - 1))
            @test channel_reducible_vertex(r, at, (ν, ν′, ω)) ≈
                  -κ * (2S(4000) - S(2000)) / β rtol = 1e-6
        end
    end
end

@testitem "Doped atom: G₃ and χ(ω) from sums of χ" setup = [Doped] begin
    # G₃(r) = -1/β Σ_ν′ χ(r, (ν, ν′, ω)) and χ(r, ω) = 2/β Σ_ν G₃(r, (ν, ω)) for r = d, m, s;
    # the truncated sums over symmetric boxes converge like 1/N.
    N = 20_000
    for (U, β, μ) in ((1.3, 2.1, 0.4), (-0.7, 3.0, -0.2), (0.0, 1.0, 0.5)),
        r in CHANNELS[1:3], (n, k) in ((0, 0), (0, 1), (-2, 2), (2, -1))
        at = DopedHubbardAtom(U, β, μ)
        S = sum(chi(r, at, (F(n), F(j), B(k))) for j in (-N - k):(N - 1))
        @test g3(r, at, (F(n), B(k))) ≈ -S / β rtol = 1e-4 atol = 1e-6
    end
    for (U, β, μ) in ((1.3, 2.1, 0.4), (-0.7, 3.0, -0.2)), r in CHANNELS[1:3],
        k in (0, 1, -2)
        at = DopedHubbardAtom(U, β, μ)
        S = sum(g3(r, at, (F(j), B(k))) for j in (-200_000 - k):199_999)
        @test chi(r, at, B(k)) ≈ 2S / at.beta atol = 1e-4
    end
    @test iszero(g3(TripletChannel(), DopedHubbardAtom(1.3, 2.1, 0.4), (F(0), B(1))))
    # triplet: the bare pp bubble
    at = DopedHubbardAtom(1.3, 2.1, 0.4)
    for k in (0, 1, -2)
        bubble = sum(gf(at, F(j)) * gf(at, F(-j - 1 - k)) for j in (-10 ^ 6):(10 ^ 6 - 1)) /
                 at.beta
        @test chi(TripletChannel(), at, B(k)) ≈ bubble rtol = 1e-5 atol = 1e-6
    end
end

@testitem "Doped atom: Hedin vertex" setup = [Doped] begin
    (d, m, s, t) = CHANNELS
    sign = Dict(d => 1, m => 1, s => -1)
    for (U, β, μ) in ((0.0, 1.0, 0.3), (1.3, 2.1, 0.4), (-0.7, 3.0, -0.2)), n in -3:2,
        k in -2:2
        at = DopedHubbardAtom(U, β, μ)
        w = (F(n), B(k))
        U == 0 && @test all(r -> hedin(r, at, w) ≈ sign[r], (d, m, s))
        for r in (d, m, s)
            @test hedin(r, at, (F(10^6), B(k))) ≈ sign[r] rtol = 1e-5
        end
    end
    @test_throws ArgumentError hedin(t, DopedHubbardAtom(1.3, 2.1, 0.4), (F(0), B(0)))
end

@testitem "Doped atom: type stability, allocations and threads" setup = [Doped] begin
    using Test: @inferred

    at = DopedHubbardAtom(2.0, 3.0, 0.3)
    w = (F(1), F(-2), B(1))
    # Behind a function barrier, so that the test item's globals do not count.
    allocs(f, r, at, w) = (f(r, at, w); @allocated f(r, at, w))
    for r in CHANNELS
        for f in
            (chi, chi0, full_vertex, gamma, irreducible_vertex, channel_reducible_vertex)
            @test @inferred(f(r, at, w)) isa ComplexF64
        end
        @test @inferred(chi(r, at, B(1))) isa ComplexF64
        @test @inferred(g3(r, at, (F(1), B(1)))) isa ComplexF64
        @test allocs(gamma, r, at, w) == 0  # after the first call, which caches the setup
        @test allocs(full_vertex, r, at, w) == 0
    end
    @test @inferred(gf(at, F(1))) isa ComplexF64

    # concurrent evaluation, including the first (setup) calls
    pts = [(r, (F(n), F(m), B(k))) for r in CHANNELS for n in -3:2 for m in -3:2
           for k in -2:2]
    fresh = DopedHubbardAtom(-6.0, 8.0, -4.0)
    res = Vector{ComplexF64}(undef, length(pts))
    Threads.@threads for i in eachindex(pts)
        res[i] = gamma(pts[i][1], fresh, pts[i][2])
    end
    serial = DopedHubbardAtom(-6.0, 8.0, -4.0)
    @test res == [gamma(r, serial, w) for (r, w) in pts]
end
