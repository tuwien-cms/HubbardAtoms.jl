# Field parts of Appendices F–H, evaluated from the printed definitions, not package internals.
# From the repository root:
# OPENBLAS_NUM_THREADS=2 timeout 1200 julia -t 2 --project=. derivation/doped-gamma/check_field.jl
using HubbardAtoms, LinearAlgebra, Printf

F(n) = FermionicFreq(2n + 1)
freqs(n, m, k) = (F(n), F(m), BosonicFreq(2k))
function field_data(U, β, μ, h, k)
    Z = 2cosh(β * μ) + 2exp(β * U / 2) * cosh(β * h)
    p0, pu, pd, p2 = (exp(-β * μ), exp(β * (U / 2 + h)),
                      exp(β * (U / 2 - h)), exp(β * μ)) ./ Z
    ρ, mag, a = p2 - p0, pu - pd, U / 2
    ν(n) = (2n + 1) * (π / β)
    ω = 2k * (π / β)
    μs(σ) = μ + σ * h
    r(σ) = ρ - σ * mag
    z(c, n) = c + im * (ν(n) + ω / 2)
    S(c, x) = -β / 2 * (iseven(k) ? tanh(β * (x - c) / 2) : coth(β * (x - c) / 2))
    G(σ, n) = (im * ν(n) + μs(σ) + a * r(σ)) / ((im * ν(n) + μs(σ))^2 - a^2)
    (; U, β, μ, h, k, a, p0, pu, pd, p2, ρ, mag, ν, ω, μs, r, z, S, G)
end

function transverse_data(c)
    (; a, ρ, mag, p0, pu, pd, p2, ω, h, β, μ, S) = c
    W = im * ω - 2h
    q1 = a^2 - W^2 / 4 - 2a^2 * (pu + pd) - a * W * mag
    H(x) = x^2 + 2a * ρ * x + q1
    P(x) = ((x + a)^2 - W^2 / 4) * ((x - a)^2 - W^2 / 4)
    e(x) = (x^2 + q1)^2 - 4a^2 * ρ^2 * x^2 - 16a^4 * p0 * p2
    d = sqrt(complex(4a^4 * ρ^4 - 4a^2 * ρ^2 * q1 + 16a^4 * p0 * p2))
    s = (2a^2 * ρ^2 - q1 + d, 2a^2 * ρ^2 - q1 - d)
    K(u) = β * H(-u) - 4a^2 * (ρ * S(μ, u) - β * (p0 + p2) / 2)
    (; W, q1, H, P, e, s, K)
end

function printed_T(c, n, m; branch=1)
    (; U, β, μ, h, a, ρ, mag, p0, p2, k, z, S) = c
    (; W, q1, H, P, e, s) = transverse_data(c)
    x, y = z(μ, n), z(μ, m)
    out = -U +
          β * P(x) * (1 / ((x + a * ρ)^2 - (W / 2 + a * mag)^2) - H(-x) / e(x)) * (n == m)
    if iszero(μ)
        sh = -4a^2 * p0 - q1
        uh = branch * sqrt(complex(sh))
        Dh = β + 2a * (iszero(uh) && iseven(k) ? -β^2 / 4 : S(0, uh) / uh)
        return out + 4a^2 * β * p0 * P(x) / e(x) * (m == -n - k - 1) +
               2a * β * P(uh) / (Dh * (x^2 - sh) * (y^2 - sh))
    end
    # Exact derivative of the printed diagonal quotient.
    dP(x) = 4x * (x^2 + a^2 - W^2 / 4) - 8a^2 * x
    de(x) = 4x * (x^2 + q1) - 8a^2 * ρ^2 * x
    Ft(x) = P(x) / e(x)
    dd = n == m ? (Ft(x) + x * (dP(x) * e(x) - P(x) * de(x)) / e(x)^2) / (2x) :
         (x * Ft(x) - y * Ft(y)) / (x^2 - y^2)
    out += 4a^2 * ρ * dd
    for j in 1:2
        k == 0 && iszero(h) && abs(s[j] - a^2) < big"1e-55" && continue
        u = branch * sqrt(complex(s[j]))
        th = 4a^2 * (ρ * S(μ, u) - β * (p0 + p2) / 2)
        out -= 4a^2 * ρ * u * P(u) / ((s[j] - s[3-j]) * (x^2 - s[j]) * (y^2 - s[j])) *
               (β * H(-u) + th) / (β * H(-u) - th)
    end
    out
end
function printed_E(c, σ, n, m)
    (; a, β, μs, r, ν, ω, k) = c
    n == -n - k - 1 && return zero(complex(a))
    b = 2μs(σ) - im * ω
    x = μs(σ) + im * ν(n)
    t = x * (b - x)
    -β * a^2 * (1 - r(σ)^2) * ((t + a^2)^2 - a^2 * b^2) /
    (2 * (t + a * r(σ) * b + a^2 * r(σ)^2) * (t + a * r(σ) * b + a^2)) *
    ((n == m) - (m == -n - k - 1))
end

# Recover the quartic coefficients from the displayed E8, then solve independently by simultaneous
# polynomial iteration. Only this elementary root finder is shared between the formula checks.
function four_roots(E)
    nodes = BigFloat.(0:4)
    coeff = [x^j for x in nodes, j in 0:4] \ [E(sqrt(complex(x))) for x in nodes]
    coeff ./= coeff[5]
    p(s) = evalpoly(s, coeff)
    dp(s) = evalpoly(s, [j * coeff[j+1] for j in 1:4])
    R = 1 + maximum(abs, coeff[1:4])
    roots = [R * cis(big(π) * (2j + big"0.3") / 4) for j in 0:3]
    for iteration in 1:10000
        corrections = [p(roots[j]) / prod(roots[j] - roots[i] for i in 1:4 if i != j)
                       for j in 1:4]
        roots -= corrections
        maximum(abs, corrections) < big"1e-65" * (1 + maximum(abs, roots)) && break
        iteration == 10000 && error("quartic root iteration did not converge")
    end
    for j in 1:4, _ in 1:5
        roots[j] -= p(roots[j]) / dp(roots[j])
    end
    @assert maximum(abs(p(s)) / (1 + abs(s)^4) for s in roots) < big"1e-60"
    sort(roots; by=s -> (real(s), imag(s)))
end

function longitudinal_data(c; branch=1)
    (; U, a, β, ω, p0, pu, pd, p2, ρ, mag, μs, r, S, k) = c
    λ, q0, πP, πS = U^2, a^2 + ω^2 / 4, p0 * p2, pu * pd
    dπ = πP - πS
    pw(t) = t == 1 ? pu : pd
    D(t) = p2 * pw(t) - p0 * pw(-t)
    ct(t) = p2 * pw(t) + p0 * pw(-t)
    H(t, x) = x^2 + 2a * r(t) * x + q0
    P(x) = (x^2 + q0)^2 - 4a^2 * x^2
    Q(t, x) = H(t, -x) * H(-t, x) * H(-t, -x) - λ^2 * (πP * H(-t, -x) + πS * H(-t, x))
    E(x) = H(1, x) * H(1, -x) * H(-1, x) * H(-1, -x) -
           λ^2 *
           (πS * (H(1, x) * H(-1, x) + H(1, -x) * H(-1, -x)) +
            πP * (H(1, x) * H(-1, -x) + H(1, -x) * H(-1, x))) + λ^4 * dπ^2
    α(t, x) = D(t) * S(μs(t), x) - β * ct(t) / 2
    αP(t, x) = ρ * S(μs(-t), x) - β * (p0 + p2) / 2
    αS(t, x) = -t * mag * S(μs(-t), x) - β * (pu + pd) / 2
    mode(u) = [-β * Q(1, u), 2λ^2 * (u^2 + q0) * α(1, u),
               -λ * (H(1, -u) * H(-1, -u) + λ^2 * dπ) * αS(-1, u),
               -λ * (H(1, -u) * H(-1, u) - λ^2 * dπ) * αP(-1, u)]
    s = four_roots(E)
    modes = map(eachindex(s)) do j
        sj = s[j]
        u = branch * sqrt(sj)
        f = mode(u)
        K(t) = t == 1 ? f[1] + f[2] : f[3] + f[4]
        keep = !(k == 0 && abs(sj - a^2) < big"1e-55")
        Ep = prod(sj - s[i] for i in eachindex(s) if i != j)
        Ξ = (sj + q0)^2 - 4a^2 * r(1) * r(-1) * sj - λ^2 * dπ
        Π = (sj + q0)^2 + 4a^2 * r(1) * r(-1) * sj + λ^2 * dπ
        function cc(t)
            A = 2λ^2 * D(t) * P(u) * (sj + q0) / (β^2 * Ep)
            W = -λ * P(u) * (ρ * Ξ + t * mag * Π) / (β^2 * Ep)
            abs(K(t)) >= abs(K(-t)) ? A / K(t) : W / K(-t)
        end
        (; s=sj, u, f, K, cc, keep)
    end
    (; λ, q0, πP, πS, dπ, D, H, P, Q, E, α, mode, modes)
end
function printed_L(c, l, σ, t, n, m)
    (; β, U, a, ρ, mag, μ, h, μs, r, ω, z, k, p0, pd) = c
    (; λ, q0, dπ, D, H, P, Q, E, modes) = l
    x, v = z(μs(σ), n), z(μs(t), m)
    mirror = m == -n - k - 1
    # The displayed exact kernel limits, not a small offset of the formula.
    samekernel = iszero(μs(t)) && mirror ? β * p0 * (t == 1 ? pd : c.pu) : D(t) / (x + v)
    pairkernel = iszero(μ) && mirror ? β * p0 : ρ / (x + v)
    spinkernel = iszero(h) && n == m ? σ * β * pd : mag / (x - v)
    out = if σ == t
        2λ^2 * (v^2 + q0) * P(v) / E(v) * samekernel
    else
        U -
        λ * P(v) / E(v) *
        (t * (H(t, -v) * H(-t, -v) + λ^2 * dπ) * spinkernel +
         (H(t, -v) * H(-t, v) - λ^2 * dπ) * pairkernel)
    end
    for j in modes
        j.keep || continue
        i = σ == 1 ? 1 : 3
        out -= β^2 * j.cc(t) / (v^2 - j.s) * (j.f[i] / (x - j.u) + j.f[i+1] / (x + j.u))
    end
    if σ == t && n == m
        out += β * P(x) * (1 / ((x + a * r(t))^2 + ω^2 / 4) - Q(t, x) / E(x))
    end
    out
end

function check_formulas()
    worst = Dict(name => big"0.0"
                 for name in ("L", "T", "T-axis", "A", "E", "bubble", "branches", "ratio"))
    counts = Dict(name => 0 for name in keys(worst))
    function compare(name, value, ref; tol=big"1e-40")
        err = abs(value - ref) / max(1, abs(ref))
        @assert isfinite(err) && err < tol (name, value, ref, err)
        worst[name] = max(worst[name], err)
        counts[name] += 1
    end
    # Both signs, both orientations, all spin entries, static/nonstatic/odd fixed-point transfers,
    # all four axes (μ=0, h=0, μ=±h) at two temperatures, and the double zero of P_T at
    # W = ±2a (k = 0, |h| = |U|/2: h = 0.65 for U = 1.3, h = -1.05 for U = -2.1).
    for U in (big"1.3", big"-2.1"), β in (big"0.8", big"2.5"),
        (μ, h) in ((big"0.37", big"0.21"), (big"-0.9", big"0.4"),
                   (big"0.0", big"0.4"), (big"0.37", big"0.0"),
                   (big"0.3", big"0.3"), (big"0.3", big"-0.3"),
                   (big"0.37", big"0.65"), (big"0.0", big"0.65"),
                   (big"0.3", big"-1.05")), k in (-1, 0, 2)

        c = field_data(U, β, μ, h, k)
        l, lm = longitudinal_data(c), longitudinal_data(c; branch=-1)
        at = FieldHubbardAtom(U, β, μ, h)
        for j in l.modes
            j.keep || continue
            fm = l.mode(-j.u)
            compare("ratio", (j.f[2] / j.f[1]) * (fm[2] / fm[1]), 1)
        end
        for (n, m) in ((0, 0), (0, -k - 1), (-1, -1), (2, -3)), σ in (1, -1)
            w = freqs(n, m, k)
            tdata = field_data(U, β, μ, σ * h, k)
            adata = field_data(-U, β, σ * h, μ, k)
            tv = printed_T(tdata, n, m)
            compare(iszero(μ) ? "T-axis" : "T", tv, gamma(TransverseSector(σ), at, w))
            compare("branches", printed_T(tdata, n, m; branch=-1), tv)
            compare("A", printed_T(adata, n, m), gamma(OppositeSpinPairSector(σ), at, w))
            compare("E", printed_E(c, σ, n, m), gamma(EqualSpinPairSector(σ), at, w))
            compare("bubble", -β * c.G(σ, n) * c.G(-σ, n + k) * (n == m),
                    chi0(TransverseSector(σ), at, w))
            compare("bubble", β * c.G(σ, n) * c.G(-σ, -n - k - 1) * (n == m),
                    chi0(OppositeSpinPairSector(σ), at, w))
            compare("bubble",
                    β / 2 * c.G(σ, n) * c.G(σ, -n - k - 1) * ((n == m) - (m == -n - k - 1)),
                    chi0(EqualSpinPairSector(σ), at, w))
            for t in (1, -1)
                lv = printed_L(c, l, σ, t, n, m)
                compare("L", lv, gamma(LongitudinalSector(σ, t), at, w))
                compare("branches", printed_L(c, lm, σ, t, n, m), lv)
                compare("bubble", -β * c.G(σ, n) * c.G(σ, n + k) * (σ == t && n == m),
                        chi0(LongitudinalSector(σ, t), at, w))
            end
        end
    end
    # Orientation is flipped before Shiba, not after it: an explicit negative control for an
    # incorrect all-spin identity at unchanged fermionic frequencies.
    let U=big"1.3", β=big"2.5", μ=big"0.37", h=big"0.21", k=2, n=0, m=-2
        at = FieldHubbardAtom(U, β, μ, h)
        down = printed_T(field_data(-U, β, -h, μ, k), n, m)
        mirrored = printed_T(field_data(-U, β, h, μ, k), -n-k-1, -m-k-1)
        compare("A", down, mirrored)
        compare("A", down, gamma(OppositeSpinPairSector(-1), at, freqs(n, m, k)))
        wrong = printed_T(field_data(-U, β, h, -μ, k), n, m)
        @assert abs(down - wrong) / max(1, abs(down)) > big"1e-3"
    end
    for name in sort(collect(keys(worst)))
        @printf("%-9s %5d comparisons; worst normalized error %.3e\n", name, counts[name],
                worst[name])
    end
end

function bisect(f, lo, hi)
    flo = f(lo)
    @assert isfinite(flo) && flo * f(hi) < 0 (lo, hi, flo, f(hi))
    for _ in 1:240
        mid = (lo + hi) / 2
        fm = f(mid)
        if flo * fm <= 0
            hi = mid
        else
            lo, flo = mid, fm
        end
    end
    (lo + hi) / 2
end
function spectral(c, s)
    (; a, β, μ, k, p0, p2) = c
    u = sqrt(complex(s))
    D = cosh(β * u) + (-1)^k * cosh(β * μ)
    R = (cosh(β * μ) * cosh(β * u) + (-1)^k) / D
    Φ = D - a * (iszero(u) ? β : sinh(β * u) / u)
    d = transverse_data(c)
    Ke = β * (s + d.q1) + 4a^2 * β * sqrt(p0 * p2) * R
    (; Φ, Ke, ell=(-Ke / d.P(u)))
end
function tpoint(U, μ, h, k, j=1)
    c = field_data(U, one(U), μ, h, k)
    d = transverse_data(c)
    s = sort(collect(d.s); by=s -> (real(s), imag(s)))[j]
    (; c, d, s, spectral(c, s)...)
end
function lpoint(U, μ, h, k, j)
    c = field_data(U, one(U), μ, h, k)
    l = longitudinal_data(c)
    mode = filter(x -> x.keep, l.modes)[j]
    fm = l.mode(-mode.u)
    ratio = mode.f[2] / mode.f[1]
    phi = (ratio - fm[2] / fm[1]) / (2mode.u)
    (; c, l, mode, ratio, phi)
end

function box_check(label, sector, pars, k, gfun; coupled=false, equalspin=false,
                   cancelled=false)
    U, μ, h = Float64.(pars)
    at = FieldHubbardAtom(U, 1.0, μ, h)
    values, deficits = Float64[], Float64[]
    for N in (12, 24, 48)
        ns = collect((-N - fld(k, 2)):(N - 1 - fld(k, 2)))
        X = if coupled
            inds = [(σ, n) for σ in (1, -1) for n in ns]
            [chi(LongitudinalSector(σ, t), at, freqs(n, m, k))
             for (σ, n) in inds, (t, m) in inds]
        else
            [chi(sector, at, freqs(n, m, k)) for n in ns, m in ns]
        end
        g = ComplexF64.(gfun(ns))
        if equalspin
            # Remove the structural mirror-even kernel before diagnosing a physical divergence.
            orbits = [(i, findfirst(==(-n - k - 1), ns))
                      for (i, n) in enumerate(ns) if n < -n - k - 1]
            V = zeros(ComplexF64, length(ns), length(orbits))
            for (j, (i, l)) in enumerate(orbits)
                V[i, j], V[l, j] = 1 / sqrt(2), -1 / sqrt(2)
            end
            X, g = V' * X * V, V' * g
        end
        ev = eigen(X)
        overlaps = [abs(dot(g, v)) / (norm(g) * norm(v)) for v in eachcol(ev.vectors)]
        j = argmax(overlaps)
        push!(values, abs(ev.values[j]))
        push!(deficits, max(0, 1 - overlaps[j]))
        @printf("%-16s N=%2d min|λ|=%.3e aligned|λ|=%.3e overlap deficit=%.3e\n", label, N,
                minimum(abs, ev.values), values[end], deficits[end])
        if !cancelled
            @assert j == argmin(abs.(ev.values)) ||
                    isapprox(values[end], minimum(abs, ev.values); atol=1e-14)
        end
    end
    if cancelled
        @assert minimum(values) > 1e-4
    elseif equalspin
        @assert maximum(values) < 1e-13 && maximum(deficits) < 1e-12
    else
        @assert values[3] < values[2] < values[1] < 1e-6
        @assert values[3] < 0.1values[1] && maximum(deficits) < 1e-7
    end
end

function check_divergences()
    μ, h = big"0.3", big"0.2"
    U = bisect(U -> real(tpoint(U, μ, h, 0).Φ), big"-5.5", big"-5.2")
    t = tpoint(U, μ, h, 0)
    @assert abs(t.Ke) < big"1e-55" && abs(t.d.K(sqrt(t.s))) < big"1e-55"
    for σ in (1, -1)
        g(ns) = [1 / (t.c.z(μ, n)^2 - (σ == 1 ? t.s : conj(t.s))) for n in ns]
        box_check("T static σ=$σ", TransverseSector(σ), (U, μ, h), 0, g)
        box_check("A static σ=$σ", OppositeSpinPairSector(σ), (-U, h, μ), 0,
                  ns -> [1 / (t.c.z(μ, σ == 1 ? n : -n - 1)^2 - t.s) for n in ns])
    end
    # Cancelled Φ=0 branch: the mode-aligned eigenvalue must stay away from zero.
    Uc = bisect(U -> real(tpoint(U, μ, h, 0).Φ), big"-4.3", big"-4.2")
    tc = tpoint(Uc, μ, h, 0)
    @assert abs(tc.Φ) < big"1e-55" && abs(tc.Ke) > big"1e-3"
    box_check("T cancelled", TransverseSector(1), (Uc, μ, h), 0,
              ns -> [1 / (tc.c.z(μ, n)^2 - tc.s) for n in ns]; cancelled=true)
    for (k, j, lo, hi, cancelled) in
        ((0, 1, big"3.65", big"3.75", false), (0, 2, big"5.05", big"5.2", false),
         (0, 1, big"1.26", big"1.28", true), (2, 1, big"-8.42", big"-8.39", false))
        U = bisect(U -> real(lpoint(U, big"0.2", big"0.1", k, j).phi), lo, hi)
        t = lpoint(U, big"0.2", big"0.1", k, j)
        @assert abs(t.ratio - (cancelled ? 1 : -1)) < big"1e-50"
        g(ns) = [t.mode.f[σ == 1 ? 1 : 3] / (t.c.z(t.c.μs(σ), n)^2 - t.mode.s)
                 for σ in (1, -1) for n in ns]
        box_check("L k=$k root $j", nothing, (U, big"0.2", big"0.1"), k, g; coupled=true,
                  cancelled)
    end
    # Nonstatic complex-root transverse point: solve the two real printed equations at 256 bits.
    x = BigFloat[big"-16.69512285", big"0.75904959"]
    f(x) = (t=tpoint(x[1], x[2], big"1.0", 2); [real(t.Φ), imag(t.Φ)])
    δ = big"1e-25"
    for _ in 1:10
        J = hcat([(f(x + δ * e) - f(x - δ * e)) / (2δ) for e in ([1, 0], [0, 1])]...)
        x -= J \ f(x)
    end
    t = tpoint(x[1], x[2], big"1.0", 2)
    @assert abs(t.Ke) < big"1e-45"
    box_check("T nonstatic", TransverseSector(1), (x[1], x[2], big"1.0"), 2,
              ns -> [1 / (t.c.z(x[2], n)^2 - t.s) for n in ns])
    box_check("A nonstatic", OppositeSpinPairSector(1), (-x[1], big"1.0", x[2]), 2,
              ns -> [1 / (t.c.z(x[2], n)^2 - t.s) for n in ns])
    # Equal-spin construction: both signs of U and both spin orientations.
    for signU in (1, -1), σ in (1, -1)
        d = big"0.7"
        a = signU * sqrt(d^2 + 3big(π)^2)
        h = bisect(h -> d + a * field_data(2a, big"1.0", d - σ * h, h, 2).r(σ), big"-30.0",
                   big"30.0")
        μ = d - σ * h
        c = field_data(2a, big"1.0", μ, h, 2)
        @assert abs(d + a * c.r(σ)) < big"1e-55"
        @assert abs(a^2 * (1 - c.r(σ)^2) - 3big(π)^2) < big"1e-55"
        box_check("E U-sign=$signU σ=$σ", EqualSpinPairSector(σ), (2a, μ, h), 2,
                  ns -> [(n == -1) - (n == -2) for n in ns]; equalspin=true)
    end
    println("PASS: printed field divergence conditions and null vectors")
end

# Analytic derivative in s of the displayed lattice sum (β=1 in divergence checks).
function Tprime(c, s, k)
    u = sqrt(complex(s))
    D = cosh(u) + (-1)^k * cosh(c)
    -((1 + (-1)^k * cosh(c) * cosh(u)) - sinh(u) / u * D) / (4s * D^2)
end
function newton2(f, x)
    δ = big"1e-24"
    for _ in 1:12
        J = hcat([(f(x + δ * e) - f(x - δ * e)) / (2δ) for e in ([1, 0], [0, 1])]...)
        x -= J \ f(x)
    end
    @assert norm(f(x)) < big"1e-40"
    x
end
function check_spectrum_and_eps()
    # A nonzero eigenvalue at two fields and both transfer parities, not just at divergences.
    worst = 0.0
    for k in (0, 1), h in (big"0.2", big"0.7")
        c = field_data(big"-8.0", big"1.0", big"0.3", h, k)
        lo, hi = iseven(k) ? (big(π), 5big(π) / 4) : (big(π) / 2, big(π))
        y = bisect(y -> real(spectral(c, -y^2).Φ), lo, hi)
        s = -y^2
        pred = spectral(c, s).ell
        ns = -48:47
        X = [chi(TransverseSector(1), FieldHubbardAtom(-8.0, 1.0, 0.3, Float64(h)),
                 freqs(n, m, k)) for n in ns, m in ns]
        g = ComplexF64[1 / (c.z(c.μ, n)^2 - s) for n in ns]
        ev = eigen(X)
        ov = [abs(dot(g, v)) / (norm(g) * norm(v)) for v in eachcol(ev.vectors)]
        j = argmax(ov)
        err = abs(ev.values[j] - pred)
        @assert err < 1e-7 && ov[j] > 1 - 1e-7
        worst = max(worst, err)
    end
    @printf("PASS transverse nonzero spectrum: worst box eigenvalue error %.3e\n", worst)
    for coupled in (false, true)
        function conditions(x)
            U, μ = x
            if coupled
                t = lpoint(U, μ, big"0.5", 0, 1)
                ratio = real(t.mode.f[3] / t.mode.f[1])
                normsum = (Tprime(μ + big"0.5", t.mode.s, 0) +
                           ratio^2 * Tprime(μ - big"0.5", t.mode.s, 0)) / (1 + ratio^2)
                [real(t.phi), real(normsum)]
            else
                t = tpoint(U, μ, big"0.5", 0)
                [real(t.Φ), real(Tprime(μ, t.s, 0))]
            end
        end
        guess = coupled ? BigFloat[big"4.78028", big"0.64162"] :
                BigFloat[big"-8.96512", big"0.97773"]
        U, μ = newton2(conditions, guess)
        t = coupled ? lpoint(U, μ, big"0.5", 0, 1) : tpoint(U, μ, big"0.5", 0)
        @assert coupled ? abs(t.ratio + 1) < big"1e-40" : abs(t.Ke) < big"1e-40"
        at = FieldHubbardAtom(Float64(U), 1.0, Float64(μ), 0.5)
        nearpairs = Float64[]
        for N in (24, 48, 96)
            ns = (-N):(N - 1)
            if coupled
                inds = [(σ, n) for σ in (1, -1) for n in ns]
                X = [chi(LongitudinalSector(σ, τ), at, freqs(n, m, 0))
                     for (σ, n) in inds, (τ, m) in inds]
                g = ComplexF64[t.mode.f[σ == 1 ? 1 : 3] /
                               (t.c.z(t.c.μs(σ), n)^2 - t.mode.s) for (σ, n) in inds]
            else
                X = [chi(TransverseSector(1), at, freqs(n, m, 0)) for n in ns, m in ns]
                g = ComplexF64[1 / (t.c.z(μ, n)^2 - t.s) for n in ns]
            end
            ev = eigen(X)
            ov = [abs(dot(g, v)) / (norm(g) * norm(v)) for v in eachcol(ev.vectors)]
            js = sortperm(ov; rev=true)[1:2]
            pair = abs.(ev.values[js])
            @assert minimum(ov[js]) > 0.999 && maximum(pair) < 1.1minimum(pair)
            push!(nearpairs, maximum(pair))
            @printf("%s EP N=%d pair max|λ|=%.3e overlap deficit=%.3e\n",
                    coupled ? "L" : "T", N, maximum(pair), maximum(1 .- ov[js]))
        end
        @assert nearpairs[3] < 0.1nearpairs[1] < 1e-5
        @printf("PASS %s EP βμ=%.8f βU=%.8f self-overlap residual %.3e\n",
                coupled ? "L" : "T", μ, U, norm(conditions([U, μ])))
    end
end

function main()
    @assert Threads.nthreads() <= 2
    BLAS.set_num_threads(2)
    println("Julia ", VERSION, "; precision ", precision(BigFloat), "; Julia/BLAS threads ",
            Threads.nthreads(), "/", BLAS.get_num_threads())
    check_formulas()
    check_divergences()
    check_spectrum_and_eps()
    println("ALL FIELD CHECKS PASS")
end
abspath(PROGRAM_FILE) == (@__FILE__) && setprecision(main, BigFloat, 256)
