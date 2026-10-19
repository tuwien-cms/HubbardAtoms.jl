# Appendix "The vertices written out", transcribed without the main text's coefficients.
# Run: julia --project=../.. -t 1 check_explicit.jl
using HubbardAtoms, Printf

function explicit_inputs(U, β, μ, k)
    a, ω = U / 2, 2k * (π / β)
    p0 = exp(-β * μ) / (2cosh(β * μ) + 2exp(β * a))
    p1 = exp(β * a) / (2cosh(β * μ) + 2exp(β * a))
    p2 = exp(β * μ) / (2cosh(β * μ) + 2exp(β * a))
    ρ = sinh(β * μ) / (cosh(β * μ) + exp(β * a))
    T(y) = iseven(k) ? tanh(β * y / 2) : coth(β * y / 2)
    x(n) = μ + im * (2n + 1) * (π / β)
    xb(n) = μ - im * (2n + 1 + 2k) * (π / β)
    z(n) = μ + im * (2n + 1 + k) * (π / β)
    (; a, ω, β, μ, k, p0, p1, p2, ρ, T, x, xb, z, b=2μ - im * ω)
end

function explicit_t(c, n, m)
    (; a, ρ, β, k, x, xb) = c
    n == -n - k - 1 && return zero(x(n))
    β * a^2 * (1 - ρ^2) * (x(n)^2 - a^2) * (xb(n)^2 - a^2) /
    ((x(n) + a * ρ) * (xb(n) + a * ρ) *
     ((x(n) + a * ρ) * (xb(n) + a * ρ) + a^2 * (1 - ρ^2))) *
    ((n == m) - (m == -n - k - 1))
end

function explicit_s(c, n, m)
    (; a, ρ, β, k, p0, p1, p2, T, x, xb, b) = c
    ξ = sqrt(b^2 / 4 + a * ρ * b + a^2 * (1 - 8p1))
    thermal = iszero(ξ) ? β / 2 : T(ξ) / ξ
    β * a^2 * (ρ^2 - 1 + 8p1) * (x(n)^2 - a^2) * (xb(n)^2 - a^2) /
    ((x(n) + a * ρ) * (xb(n) + a * ρ) *
     ((x(n) + a * ρ) * (xb(n) + a * ρ) + a^2 * (1 - ρ^2 - 8p1))) *
    ((n == m) + (m == -n - k - 1)) + 4a -
    16a^3 * (4a * p1 - (p1 + p2) * b) * (4a * p1 + (p0 + p1) * b) /
    ((1 + a * thermal) *
     ((x(n) + a * ρ) * (xb(n) + a * ρ) + a^2 * (1 - ρ^2 - 8p1)) *
     ((x(m) + a * ρ) * (xb(m) + a * ρ) + a^2 * (1 - ρ^2 - 8p1)))
end

function explicit_ph(c, ε, n, m; sign1=1, sign2=1)
    (; a, ρ, β, μ, ω, k, p0, p1, p2, T, z) = c
    r = 2a * sqrt(complex(a^2 * ρ^4 -
                          ρ^2 * (a^2 + ω^2 / 4 + 4ε * a^2 * p1) + 4a^2 * p0 * p2))
    s = (2a^2 * ρ^2 - a^2 - ω^2 / 4 - 4ε * a^2 * p1 + r,
         2a^2 * ρ^2 - a^2 - ω^2 / 4 - 4ε * a^2 * p1 - r)
    u = (sign1 * sqrt(s[1]), sign2 * sqrt(s[2]))
    zn, zm = z(n), z(m)
    # The printed diagonal replacement, not differentiation of a shared helper.
    dense = if n == m
        ((zn + a)^2 + ω^2 / 4) * ((zn - a)^2 + ω^2 / 4) /
        (2zn * (zn^2 - s[1]) * (zn^2 - s[2])) +
        2zn * (zn^2 - a^2 + ω^2 / 4) / ((zn^2 - s[1]) * (zn^2 - s[2])) -
        zn * ((zn + a)^2 + ω^2 / 4) * ((zn - a)^2 + ω^2 / 4) *
        (2zn^2 - s[1] - s[2]) / ((zn^2 - s[1])^2 * (zn^2 - s[2])^2)
    else
        (zn * ((zn + a)^2 + ω^2 / 4) * ((zn - a)^2 + ω^2 / 4) /
         ((zn^2 - s[1]) * (zn^2 - s[2])) -
         zm * ((zm + a)^2 + ω^2 / 4) * ((zm - a)^2 + ω^2 / 4) /
         ((zm^2 - s[1]) * (zm^2 - s[2]))) / (zn^2 - zm^2)
    end
    sep = zero(zn)
    for j in 1:2
        # The static magnetic atomic root is omitted analytically, not divided by zero.
        if ε == -1 && k == 0 && j == argmin(abs.(s .- a^2))
            @assert abs(s[j] - a^2) < big"1e-60"
            continue
        end
        sep += 4ε * a^2 * ρ * u[j] * ((u[j] + a)^2 + ω^2 / 4) *
               ((u[j] - a)^2 + ω^2 / 4) /
               ((s[j] - s[3-j]) * (zn^2 - s[j]) * (zm^2 - s[j])) *
               (s[j] - 2a * ρ * u[j] + a^2 + ω^2 / 4 + 4ε * a^2 * p1 +
                2ε * a^2 * (ρ * T(u[j] - μ) + p0 + p2)) /
               (s[j] - 2a * ρ * u[j] + a^2 + ω^2 / 4 + 4ε * a^2 * p1 -
                2ε * a^2 * (ρ * T(u[j] - μ) + p0 + p2))
    end
    β * ((zn + a)^2 + ω^2 / 4) * ((zn - a)^2 + ω^2 / 4) *
    (1 / ((zn + a * ρ)^2 + ω^2 / 4) -
     (zn^2 - 2a * ρ * zn + a^2 + ω^2 / 4 + 4ε * a^2 * p1) /
     ((zn^2 - s[1]) * (zn^2 - s[2]))) * (n == m) + 2ε * a - 4ε * a^2 * ρ * dense + sep
end

function explicit_hf(c, ε, n, m)
    (; a, β, ω, k, p0, p1, T, z) = c
    ŝ = 4ε * a^2 * (p0 - p1) - a^2 - ω^2 / 4
    ŝ′ = -4ε * a^2 * (p0 + p1) - a^2 - ω^2 / 4
    û = sqrt(complex(ŝ))
    thermal = iszero(û) ? β / 2 : T(û) / û
    zn, zm = z(n), z(m)
    β * ((zn + a)^2 + ω^2 / 4) * ((zn - a)^2 + ω^2 / 4) *
    (1 / (zn^2 + ω^2 / 4) -
     (zn^2 + a^2 + ω^2 / 4 + 4ε * a^2 * p1) /
     ((zn^2 - ŝ) * (zn^2 - ŝ′))) * (n == m) + 2ε * a -
    4ε * a^2 * β * p0 * ((zn + a)^2 + ω^2 / 4) * ((zn - a)^2 + ω^2 / 4) /
    ((zn^2 - ŝ) * (zn^2 - ŝ′)) * (m == -n - k - 1) -
    2ε * a * ((ŝ + a^2 + ω^2 / 4)^2 - 4a^2 * ŝ) /
    ((1 + ε * a * thermal) * (zn^2 - ŝ) * (zm^2 - ŝ))
end

function check_explicit()
    worst = Dict{String,BigFloat}()
    counts = Dict{String,Int}()
    for U in (big"1.3", big"-2.1", big"3.7"), β in (big"0.8", big"2.5", big"5.0"),
        μ in (big"0", big"0.37", big"-0.9", big"1.6"), k in -3:3
        c = explicit_inputs(U, β, μ, k)
        at = DopedHubbardAtom(U, β, μ)
        @assert isapprox(c.p0 + 2c.p1 + c.p2, 1; atol=big"1e-70")
        @assert isapprox(c.ρ, c.p2 - c.p0; atol=big"1e-70")
        for (n, m) in ((0, 0), (0, -k - 1), (2, 2), (-3, -2), (3, -1), (-2, -2))
            w = (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(2k))
            for r in
                (TripletChannel(), SingletChannel(), DensityChannel(), MagneticChannel())
                name, value = if r isa TripletChannel
                    "t", explicit_t(c, n, m)
                elseif r isa SingletChannel
                    "s", explicit_s(c, n, m)
                else
                    ε = r isa DensityChannel ? 1 : -1
                    if iszero(μ)
                        "hf-$(ε)", explicit_hf(c, ε, n, m)
                    else
                        val = explicit_ph(c, ε, n, m)
                        # Both independent root signs must give the same result.
                        for (sg1, sg2) in ((-1, 1), (1, -1), (-1, -1))
                            other = explicit_ph(c, ε, n, m; sign1=sg1, sign2=sg2)
                            @assert isapprox(val, other; rtol=big"1e-40", atol=big"1e-40")
                        end
                        "ph-$(ε)", val
                    end
                end
                ref = gamma(r, at, w)
                err = abs(value - ref) / max(1, abs(ref))
                @assert err < big"1e-40" (U, β, μ, k, n, m, name, err)
                worst[name] = max(get(worst, name, zero(β)), err)
                counts[name] = get(counts, name, 0) + 1
            end
        end
    end
    # At βa = -log(3), μ = k = 0, both ξ and the density root û are exactly zero.
    let U=-log(big"3"), β=big"2", μ=big"0", k=0
        c = merge(explicit_inputs(U, β, μ, k),
                  (; p0=big"3" / 8, p1=big"1" / 8, p2=big"3" / 8, ρ=big"0"))
        at = DopedHubbardAtom(U, β, μ)
        for (n, m) in ((0, 0), (0, -1), (2, -3), (3, -1))
            w = (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(0))
            @assert isapprox(explicit_s(c, n, m), gamma(SingletChannel(), at, w);
                             rtol=big"1e-40", atol=big"1e-40")
            @assert isapprox(explicit_hf(c, 1, n, m), gamma(DensityChannel(), at, w);
                             rtol=big"1e-40", atol=big"1e-40")
        end
    end
    # Package-only sanity checks of the printed U = 0 shortcut, not formula comparisons.
    for μ in (big"0", big"0.4"), k in -1:1, r in CHANNELS
        at = DopedHubbardAtom(big"0", big"2", μ)
        @assert iszero(gamma(r, at, (FermionicFreq(1), FermionicFreq(-1), BosonicFreq(2k))))
    end
    for name in sort!(collect(keys(worst)))
        @printf("%-6s %5d entries, worst normalized error %.2e\n", name, counts[name],
                worst[name])
    end
    println("PASS: expanded zero-root formulas (8 entries)")
    println("PASS: package-only U = 0 sanity checks (24 entries)")
end

setprecision(check_explicit, BigFloat, 256)
