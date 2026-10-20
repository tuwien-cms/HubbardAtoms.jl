@testsetup module FieldTests
using HubbardAtoms, LinearAlgebra
export F, B, sectors, boxvertices
F(n) = FermionicFreq(2n + 1)
B(k) = BosonicFreq(2k)
sectors() = (LongitudinalSector(1, 1), LongitudinalSector(1, -1),
             LongitudinalSector(-1, 1), LongitudinalSector(-1, -1),
             TransverseSector(1), TransverseSector(-1), OppositeSpinPairSector(1),
             OppositeSpinPairSector(-1), EqualSpinPairSector(1), EqualSpinPairSector(-1))

# Independent inversion of the package's Lehmann χ, not of the closed-form Γ.
function boxvertices(at, k, pts; M=12)
    rs = sectors()[1:8]
    vals = map((M, 2M, 4M)) do N
        idx = [n for n in (-N - k - 1):N if abs(2n + 1 + k) <= 2N + 1]
        labels = [(σ, n) for σ in (1, -1) for n in idx]
        X = [chi(LongitudinalSector(σ, τ), at, (F(n), F(m), B(k)))
             for (σ, n) in labels, (τ, m) in labels]
        Y = inv(X)
        result = ComplexF64[]
        for r in rs
            if r isa LongitudinalSector
                for (n, m) in pts
                    val = Y[findfirst(==((r.sigma, n)), labels),
                            findfirst(==((r.tau, m)), labels)]
                    if r.sigma == r.tau && n == m
                        val -= 1 / chi0(r, at, (F(n), F(n), B(k)))
                    end
                    push!(result, at.beta^2 * val)
                end
            else
                Xr = [chi(r, at, (F(n), F(m), B(k))) for n in idx, m in idx]
                Yr = inv(Xr)
                for (n, m) in pts
                    val = Yr[findfirst(==(n), idx), findfirst(==(m), idx)]
                    n == m && (val -= 1 / chi0(r, at, (F(n), F(n), B(k))))
                    push!(result, at.beta^2 * val)
                end
            end
        end
        result
    end
    (8vals[3] - 6vals[2] + vals[1]) / 3
end
end

@testitem "Field atom: constructor and spin validation" setup = [FieldTests] begin
    for args in ((1, 0, 0, 0), (1, -1, 0, 0), (1, Inf, 0, 0), (Inf, 1, 0, 0),
                 (NaN, 1, 0, 0), (1, 1, Inf, 0), (1, 1, 0, NaN), (1, 1, 0, Inf))
        @test_throws DomainError FieldHubbardAtom(args...)
    end
    @test FieldHubbardAtom(2, 3, 1, 1) isa FieldHubbardAtom{Float64}
    @test FieldHubbardAtom(2.0f0, 3, 1.0f0, 1) isa FieldHubbardAtom{Float32}
    at = FieldHubbardAtom(2.0, 3.0, 0.3, 0.2)
    @test at == FieldHubbardAtom(2.0, 3.0, 0.3, 0.2)
    @test at != FieldHubbardAtom(2.0, 3.0, 0.3, -0.2)
    @test hash(at) == hash(FieldHubbardAtom(2.0, 3.0, 0.3, 0.2))
    @test sprint(show, at) == "FieldHubbardAtom(2.0, 3.0, 0.3, 0.2)"
    @test_throws ArgumentError gf(at, 0, F(0))
    for σ in (0, 2), R in (TransverseSector, OppositeSpinPairSector, EqualSpinPairSector)
        @test_throws ArgumentError R(σ)
        @test_throws ArgumentError LongitudinalSector(σ, 1)
        @test_throws ArgumentError LongitudinalSector(1, σ)
    end
    @test size(gf.(at, 1, [F(0), F(1)])) == (2,)
    @test size(chi.(TransverseSector(1), at, [(F(0), F(0), B(0))])) == (1,)
end

@testitem "Field atom: independent Lehmann susceptibilities" setup = [ExactDiagonalization,
                                                                      FieldTests] begin
    using .ExactDiagonalization: correlator
    worst = 0.0
    for (U, β, μ, h) in ((1.3, 1.1, 0.3, 0.2), (-1.7, 0.9, -0.31, 0.17),
                         (1.6, 1.4, 0.0, 0.3), (1.5, 1.6, 0.3, 0.0), (1.5, 1.6, 0.3, 0.3),
                         (-1.1, 2.2, 0.4, -0.4), (0.0, 1.1, 0.3, 0.2))
        at = FieldHubbardAtom(U, β, μ, h)
        c(σ) = σ == 1 ? cup : cdn
        cd(σ) = σ == 1 ? cup⁺ : cdn⁺
        Ged(σ, n) = -correlator(U, β, (c(σ), cd(σ)), (ν(n, β),), (true,); μ, h)
        for σ in (1, -1), n in -2:1
            @test gf(at, σ, F(n)) ≈ Ged(σ, n) rtol = 2e-13
        end
        for r in sectors(), k in (-1, 0, 1), (n, m) in ((0, 0), (0, -k - 1), (1, -2))
            σ = r.sigma
            τ = r isa LongitudinalSector ? r.tau : σ
            kp = r isa Union{OppositeSpinPairSector,EqualSpinPairSector} ? -k - n - m - 1 :
                 k
            ops = r isa Union{TransverseSector,OppositeSpinPairSector} ?
                  (cd(σ), c(-σ), cd(-σ), c(σ)) : (cd(σ), c(σ), cd(τ), c(τ))
            v, vp, om = ν(n, β), ν(m, β), ω(kp, β)
            ref = correlator(U, β, ops, (-v, v + om, -vp - om), (true, true, true); μ, h)
            if r isa LongitudinalSector
                k == 0 && (ref -= β * Ged(σ, n) * Ged(τ, m))
            elseif r isa OppositeSpinPairSector
                ref = -ref
            elseif r isa EqualSpinPairSector
                ref = -ref / 2
            end
            val = chi(r, at, (F(n), F(m), B(k)))
            worst = max(worst, abs(val - ref))
            @test val ≈ ref atol = 3e-13 rtol = 3e-12
        end
    end
    @info "Field χ against independent ED: worst absolute error" worst
end

@testitem "Field atom: zero-field and Shiba limits" setup = [FieldTests] begin
    (d, mag, s, t) = CHANNELS
    worst = Ref(0.0)
    setprecision(BigFloat, 256) do
        for (U, β, μ) in
            ((big"1.3", big"1.1", big"0.3"), (big"-1.7", big"0.9", big"-0.31"),
             (big"1.5", big"1.6", big"0")), axis in (:h, :mu)

            at = axis == :h ? FieldHubbardAtom(U, β, μ, 0) : FieldHubbardAtom(U, β, 0, μ)
            near = axis == :h ? FieldHubbardAtom(U, β, μ, big"1e-25") :
                   FieldHubbardAtom(U, β, big"1e-25", μ)
            for r in sectors(), k in (-1, 0, 1), (n, m) in ((0, 0), (0, -k - 1), (2, -1))
                w = (F(n), F(m), B(k))
                σ = r.sigma
                source = axis == :h ? DopedHubbardAtom(U, β, μ) :
                         DopedHubbardAtom(-U, β, r isa LongitudinalSector ? μ : σ * μ)
                if r isa LongitudinalSector
                    τ = r.tau
                    v = axis == :h ? w :
                        (σ == 1 ? w[1] : -w[1] - w[3], τ == 1 ? w[2] : -w[2] - w[3], w[3])
                    sign = axis == :h ? 1 : σ * τ
                    ref = sign * (gamma(d, source, v) + σ * τ * gamma(mag, source, v)) / 2
                elseif (axis == :h && r isa TransverseSector) ||
                       (axis == :mu && r isa OppositeSpinPairSector)
                    ref = gamma(mag, source, w)
                elseif r isa EqualSpinPairSector
                    ref = -gamma(t, source, w) / 2
                else
                    ref = (gamma(s, source, w) - gamma(t, source, w)) / 2
                end
                @test gamma(r, at, w) ≈ ref atol = big"1e-65" rtol = big"1e-65"
                error = abs(gamma(r, near, w) - ref) / max(1, abs(ref))
                worst[] = max(worst[], Float64(error))
                @test error < big"1e-20"
            end
        end
    end
    @info "Field axis continuity: worst scaled error at offset 1e-25" worst=worst[]
end

@testitem "Field atom: Γ against extrapolated inversion of own χ" setup = [FieldTests] begin
    pts = ((0, 0), (0, -1), (2, -1))
    worst = 0.0
    for (U, β, μ, h) in ((1.3, 1.1, 0.3, 0.2), (-1.7, 0.9, -0.31, 0.17),
                         (1.5, 1.6, 0.3, 0.3), (-1.1, 2.2, 0.4, -0.4)), k in (-1, 0, 1)

        at = FieldHubbardAtom(U, β, μ, h)
        ref = boxvertices(at, k, pts)
        val = [gamma(r, at, (F(n), F(m), B(k))) for r in sectors()[1:8] for (n, m) in pts]
        error = maximum(abs.(val - ref) ./ max.(1, abs.(val)))
        worst = max(worst, error)
        @test error < 2e-5
    end
    @info "Field Γ vs χ box inversion M=12,24,48: worst scaled error" worst
end

@testitem "Field atom: symmetries, bubbles and equal-spin BSE" setup = [FieldTests] begin
    worst = Ref(BigFloat(0))
    setprecision(BigFloat, 256) do
        for (U, β, μ, h) in ((big"1.3", big"1.1", big"0.3", big"0.2"),
                             (big"-1.7", big"0.9", big"-0.31", big"0.17"))
            at = FieldHubbardAtom(U, β, μ, h)
            flipped = FieldHubbardAtom(U, β, μ, -h)
            shiba = FieldHubbardAtom(-U, β, h, μ)
            for r in sectors(), k in (-1, 0, 1), (n, m) in ((0, 0), (0, -k - 1), (2, -1))
                w = (F(n), F(m), B(k))
                rf = r isa LongitudinalSector ? LongitudinalSector(-r.sigma, -r.tau) :
                     typeof(r)(-r.sigma)
                for f in (chi, chi0, gamma)
                    @test f(r, at, w) ≈ f(rf, flipped, w) atol = big"1e-60" rtol = big"1e-60"
                    @test conj(f(r, at, w)) ≈ f(r, at, (-w[1], -w[2], -w[3])) atol = big"1e-60" rtol = big"1e-60"
                    if r isa LongitudinalSector
                        v = (r.sigma == 1 ? w[1] : -w[1] - w[3],
                             r.tau == 1 ? w[2] : -w[2] - w[3], w[3])
                        @test f(r, at, w) ≈ r.sigma * r.tau * f(r, shiba, v) atol = big"1e-60" rtol = big"1e-60"
                        @test f(r, at, w) ≈
                              f(LongitudinalSector(r.tau, r.sigma), at, (w[2], w[1], w[3])) atol = big"1e-60" rtol = big"1e-60"
                    elseif r isa OppositeSpinPairSector
                        @test f(r, at, w) ≈ f(TransverseSector(1),
                                              FieldHubbardAtom(-U, β, r.sigma * h, μ), w) atol = big"1e-60" rtol = big"1e-60"
                    end
                end
                if r isa EqualSpinPairSector
                    # χ(Γ/β² + B⁻¹) = M₋; only the diagonal and mirror contribute.
                    v(j) = (F(j), F(m), B(k))
                    nb = -n - k - 1
                    residual = sum(chi(r, at, (F(n), F(j), B(k))) * gamma(r, at, v(j)) /
                                   β^2 for j in unique((n, nb)))
                    if m == n || m == nb
                        residual += chi(r, at, w) / (2chi0(r, at, (F(m), B(k))) * β)
                    end
                    residual -= ((n == m) - (nb == m)) / 2
                    worst[] = max(worst[], abs(residual))
                    @test abs(residual) < big"1e-65"
                end
            end
        end
    end
    @info "Field equal-spin restricted BSE: worst residual" worst=worst[]
end

@testitem "Field atom: floating-point types, free limit and threads" setup = [FieldTests] begin
    using Test: @inferred
    w = (F(0), F(0), B(1))
    at = FieldHubbardAtom(1.3, 1.1, 0.3, 0.2)
    for r in sectors(), f in (chi, chi0, gamma)
        @test @inferred(f(r, at, w)) isa ComplexF64
        ref = f(r, at, w)
        a32 = FieldHubbardAtom(1.3f0, 1.1f0, 0.3f0, 0.2f0)
        @test f(r, a32, w) isa ComplexF32
        @test f(r, a32, w) ≈ ref rtol = 2e-3 atol = 2e-4
        res256 = setprecision(() -> f(r,
                                      FieldHubbardAtom(big(1.3), big(1.1), big(0.3),
                                                       big(0.2)), w), BigFloat, 256)
        res512 = setprecision(() -> f(r,
                                      FieldHubbardAtom(big(1.3), big(1.1), big(0.3),
                                                       big(0.2)), w), BigFloat, 512)
        @test res256 isa Complex{BigFloat}
        @test res256 ≈ ref rtol = 2e-11 atol = 1e-12
        @test res256 ≈ res512 rtol = big"1e-65" atol = big"1e-65"
        free = FieldHubbardAtom(0, 1.1, 0.3, 0.2)
        @test iszero(gamma(r, free, w))
        @test chi(r, free, w) ≈ chi0(r, free, w) atol = 1e-14
    end
    pts = [(r, (F(n), F(m), B(k))) for r in sectors() for n in -2:1 for m in -2:1
           for k in -1:1]
    fresh = FieldHubbardAtom(1.3, 1.1, 0.3, 0.2)
    res = Vector{ComplexF64}(undef, length(pts))
    Threads.@threads for i in eachindex(pts)
        res[i] = gamma(pts[i][1], fresh, pts[i][2])
    end
    @test res == [gamma(r, at, w) for (r, w) in pts]
end

@testitem "Field atom: near-axis precision and longitudinal confluence" setup = [FieldTests] begin
    using HubbardAtoms: _field_l_needs_limit
    setprecision(BigFloat, 256) do
        for (μ, h) in ((1e-9, 0.2), (0.3, 1e-9)), r in sectors(), k in (-1, 0, 1)
            at = FieldHubbardAtom(1.3, 1.1, μ, h)
            bt = FieldHubbardAtom(big(1.3), big(1.1), big(μ), big(h))
            w = (F(0), F(-1), B(k))
            @test gamma(r, at, w) ≈ gamma(r, bt, w) rtol = 1e-9 atol = 1e-10
        end
        # Rounded static confluence from the reference's check_exceptional.jl, off every axis.
        at = FieldHubbardAtom(1.3, 1.1, -0.8, 0.7999929392816533)
        @test _field_l_needs_limit(at, 1, 1, 0, 0, 0)
        ref = boxvertices(at, 0, ((0, 0),))
        for (i, r) in enumerate(sectors()[1:4])
            val = gamma(r, at, (F(0), F(0), B(0)))
            @test val ≈ ref[i] rtol = 2e-5 atol = 2e-5
        end
    end
    # A mirror fixed point has no Pauli-odd component, even in the free bubble.
    for σ in (1, -1), f in (chi, chi0, gamma), m in -2:1
        @test iszero(f(EqualSpinPairSector(σ), FieldHubbardAtom(1.3, 1.1, 0.3, 0.2),
                       (F(-1), F(m), B(1))))
    end
end
