# Appendix "Vertex divergences" of doped-gamma-derivation.typ at ω = 0 and β = 1: the divergence lines of
# Γ_d, Γ_m, Γ_s against the real eigenvalues of χ_r on finite boxes, their null vectors, and the
# exceptional points. Run: julia --project=../.. -t 4 check_divergences.jl
using HubbardAtoms, LinearAlgebra, Printf

wts(U, μ) = (e=(exp(-μ), exp(U / 2), exp(μ)); e ./ (e[1] + 2 * e[2] + e[3]))
cu(s) = s >= 0 ? cosh(sqrt(s)) : cos(sqrt(-s))                                  # cosh u, u² = s
su(s) = s > 0 ? sinh(sqrt(s)) / sqrt(s) : s < 0 ? sin(sqrt(-s)) / sqrt(-s) : 1.0  # sinh(u)/u

# Density (ε = 1) and magnetic (ε = -1): the real roots s_j of e, Φ(s_j) and σ, with σ ≥ 0 on the
# divergence arcs of Φ(s_j) = 0 and σ ≤ 0 on the others (σ = 0 where two such arcs cross)
function ph_line(U, μ, ε, j)
    a = U / 2
    p0, p1, p2 = wts(U, μ)
    ρ = p2 - p0
    q1 = a^2 + 4ε * a^2 * p1
    Δ = a^2 * ρ^4 - ρ^2 * q1 + 4a^2 * p0 * p2
    Δ < 0 && return (NaN, NaN, NaN)
    s = 2a^2 * ρ^2 - q1 + (j == 1 ? 2 : -2) * a * sqrt(Δ)
    Φ = cosh(μ) + cu(s) + ε * a * su(s)
    (Φ, (s + q1) * ε * (1 + cosh(μ) * cu(s)) / (cu(s) + cosh(μ)), s)
end
# Singlet: K_s cosh(ξ/2) and d₋ = t₋ + A_s up to the factor -2a (K_s = 0 = d₋ is no divergence)
function s_line(U, μ)
    a = U / 2
    p0, p1, p2 = wts(U, μ)
    ρ = p2 - p0
    ξ2 = μ^2 + 2a * ρ * μ + a^2 * (1 - 8p1)
    (cu(ξ2 / 4) + a * su(ξ2 / 4) / 2, 4a * p1 + 2μ * (p0 + p1), ξ2 - μ^2)
end
function bisect0(f, l, h)
    fl = f(l)
    for _ in 1:80
        m = (l + h) / 2
        fm = f(m)
        sign(fm) == sign(fl) ? (l=m; fl=fm) : (h = m)
    end
    (l + h) / 2
end
genuine(Φ, σ, s) = σ >= -1e-10 * (1 + abs(s))
# βU of the divergences at fixed βμ, with the root j (0 for the singlet)
function divergences(ch, μ, Us)
    out = Tuple{Float64,Int}[]
    for j in (ch == :s ? (0,) : (1, 2))
        f = ch == :s ? (U -> s_line(U, μ)[1]) :
            (U -> ph_line(U, μ, ch == :d ? 1 : -1, j)[1])
        v = f.(Us)
        for i in 1:(length(Us) - 1)
            (isnan(v[i]) || isnan(v[i+1]) || sign(v[i]) == sign(v[i+1])) && continue
            U = bisect0(f, Us[i], Us[i+1])
            keep = ch == :s ? abs(s_line(U, μ)[2]) > 1e-8 * (1 + abs(U)) :
                   genuine(ph_line(U, μ, ch == :d ? 1 : -1, j)...)
            keep && push!(out, (U, j))
        end
    end
    sort(out)
end

const CH = Dict(:d => DensityChannel(), :m => MagneticChannel(), :s => SingletChannel())
F(n) = FermionicFreq(2n + 1)
function boxchi(ch, U, μ, M)
    [chi(CH[ch], DopedHubbardAtom(U, 1.0, μ), (F(n), F(m), BosonicFreq(0)))
     for n in (-M - 1):M, m in (-M - 1):M]
end
# (negative real eigenvalues, conjugate pairs with negative real part) on the box |2n + 1| ≤ 2M + 1
function counts(ch, U, μ; M=24)
    ev = eigvals(boxchi(ch, U, μ, M))
    tol = 1e-9 * maximum(abs, ev)
    count(e -> abs(imag(e)) <= tol && real(e) < 0, ev),
    count(e -> imag(e) > tol && real(e) < 0, ev)
end

ok = true
# 1. Lines against sign changes of real box eigenvalues (the conjugate-pair count unchanged)
Ufine = range(-25, 25; length=20001);
Ubox = range(-24.99, 24.99; length=1000)
let nlines = 0, ncuts = 0, npair = 0
    for ch in (:d, :m, :s), μ in (0.05, 0.3, 0.57, 0.8, 1.1, 1.3, 1.6, 2.2, 3.0, 3.9)
        pred = [U for (U, _) in divergences(ch, μ, Ufine) if abs(U) > 0.06]
        c = Vector{Tuple{Int,Int}}(undef, length(Ubox))
        Threads.@threads for i in eachindex(Ubox)
            c[i] = counts(ch, Ubox[i], μ)
        end
        box = [(Ubox[i] + Ubox[i+1]) / 2
               for i in 1:(length(Ubox) - 1)
               if c[i][2] == c[i+1][2] && abs(c[i][1] - c[i+1][1]) == 1]
        # unresolved: the real count changes by more than one at an unchanged pair count
        unresolved = count(i -> c[i][2] == c[i+1][2] && abs(c[i][1] - c[i+1][1]) > 1,
                           1:(length(Ubox) - 1))
        npair += count(i -> c[i][2] != c[i+1][2], 1:(length(Ubox) - 1))
        match = length(pred) == length(box) && all(abs.(pred .- box) .< 0.15) &&
                unresolved == 0
        match || @printf("FAIL %s βμ = %.2f: closed form %s, box %s\n", ch, μ,
                         string(round.(pred, digits=3)),
                         string(round.(box, digits=3)))
        global ok &= match
        nlines += length(pred)
        ncuts += 1
    end
    @printf("%-4s %d lines on %d cuts (βμ in [0.05, 3.9], |βU| ≤ 25) match the changes by one of the number of\n     negative real box eigenvalues at an unchanged number of conjugate pairs (step 0.05, tolerance 0.15);\n     no other such changes, none by more than one; %d intervals where the pair number changes, not compared\n",
            ok ? "PASS" : "FAIL", nlines, ncuts, npair)
end

# 2. Null vectors: g_n = 1/(z_n² - s_j) (d, m) and 1/(t_n + A_s) (s), against the box eigenvector of the
# eigenvalue closest to zero, at points on the lines
let worst = 0.0, M = 60, idx = (-M - 1):M
    for (ch, μ, lo, hi) in
        ((:d, 0.3, 3.7, 3.8), (:d, 0.3, 5.0, 5.1), (:d, 0.0, 5.0, 5.3),
         (:d, 1.0, -4.95, -4.85),
         (:d, 1.5, -7.4, -7.2),
         (:m, 0.3, -5.5, -5.2), (:s, 0.3, 5.0, 5.3), (:s, 1.6, 5.8, 6.1))
        U, j = only(divergences(ch, μ, range(lo, hi; length=201)))
        g = if ch == :s
            a = U / 2
            p0, p1, p2 = wts(U, μ)
            As = a * (p2 - p0) * 2μ + a^2 * (1 - 8p1)
            [1 / (abs2(μ + im * (2n + 1) * π) + As) for n in idx]
        else
            s = ph_line(U, μ, ch == :d ? 1 : -1, j)[3]
            [1 / ((μ + im * (2n + 1) * π)^2 - s) for n in idx]
        end
        E = eigen(boxchi(ch, U, μ, M))
        v = E.vectors[:, argmin(abs.(E.values))]
        worst = max(worst, 1 - abs(dot(g, v)) / (norm(g) * norm(v)))
    end
    @printf("%-4s null vectors at 8 points (d, m, s; both signs of U; μ = 0): overlap deficit 1 - |cos∠(v, g)| ≤ %.1e\n",
            worst < 1e-8 ? "PASS" : "FAIL", worst)
    global ok &= worst < 1e-8
end

# 3. Exceptional points: Σ g_n² = T'(s_j) = 0 on a line, with T(s) = Σ 1/(z_n² - s) = -su(s)/(2(cu(s) + cosh μ));
# there the box has two eigenvalues near zero
T(s, μ) = -su(s) / (2 * (cu(s) + cosh(μ)))
Tp(s, μ) = (h=1e-6 * (1 + abs(s)); (T(s + h, μ) - T(s - h, μ)) / (2h))
function ep(ch, j, μs, lo, hi)
    ε = ch == :d ? 1 : -1
    line(μ) = (r=[U for (U, i) in divergences(ch, μ, range(lo, hi; length=400)) if i == j];
               isempty(r) ? NaN : r[1])
    tp(μ) = (U=line(μ); Tp(ph_line(U, μ, ε, j)[3], μ))
    for k in 1:(length(μs) - 1)
        a, b = tp(μs[k]), tp(μs[k+1])
        if !isnan(a) && !isnan(b) && sign(a) != sign(b)
            μ = bisect0(tp, μs[k], μs[k+1])
            return μ, line(μ)
        end
    end
    NaN, NaN
end
function gvec(U, μ, ε, j, M)
    (s=ph_line(U, μ, ε, j)[3]; [1 / ((μ + im * (2n + 1) * π)^2 - s) for n in (-M - 1):M])
end
# eigenvalues whose eigenvectors are aligned with g (overlap > 0.99), and the smallest |λ| of the others
function aligned(ch, U, μ, j, M)
    E = eigen(boxchi(ch, U, μ, M))
    g = gvec(U, μ, ch == :d ? 1 : -1, j, M)
    ov = [abs(dot(g, E.vectors[:, i])) / (norm(g) * norm(E.vectors[:, i]))
          for i in eachindex(E.values)]
    abs.(E.values[ov .> 0.99]), minimum(abs.(E.values[ov .<= 0.99]))
end
# a near-EP pair: exactly two eigenvalues aligned with g (overlap > 0.99), equal in size within 10%, below a
# tenth of all other eigenvalues, at M = 60 and M = 120, and shrinking by more than half between them
function isep(ch, U, μ, j)
    (l60, b60), (l120, b120) = aligned(ch, U, μ, j, 60), aligned(ch, U, μ, j, 120)
    good(l, b) = length(l) == 2 && maximum(l) < 0.1b && maximum(l) < 1.1minimum(l)
    good(l60, b60) && good(l120, b120) && maximum(l120) < 0.5minimum(l60), l120
end
for (label, ch, j, μs, lo, hi, essl) in
    (("charge, U > 0, first loop", :d, 2, 0.5:0.002:0.6, 3.8, 4.6, "0.6"),
     ("charge, U < 0, first line", :d, 1, 0.9:0.004:1.3, -4.5, -6.0, "1.12"),
     ("magnetic, U < 0, first line", :m, 1, 0.8:0.004:1.2, -7.0, -11.0, "0.975"))
    μ, U = ep(ch, j, μs, lo, hi)
    yes, l120 = isep(ch, U, μ, j)
    p = wts(U, μ)
    @printf("%-4s EP %-28s βμ = %.4f, βU = %.4f, ρ = %.4f (Eßl's thesis: βδμ = %s); aligned pair |λ| (M = 120): %s\n",
            yes ? "PASS" : "FAIL", label, μ, U, p[3] - p[1], essl,
            join([@sprintf("%.1e", l) for l in l120], ", "))
    global ok &= yes
end
let U1 = only([U for (U, j) in divergences(:d, 0.3, range(3.7, 3.8; length=101)) if j == 2])
    no = !isep(:d, U1, 0.3, 2)[1] && !isep(:d, 4.4, 0.3, 2)[1]
    @printf("%-4s EP test negative on an ordinary divergence (βμ = 0.3, βU = %.4f) and off the lines (βU = 4.4)\n",
            no ? "PASS" : "FAIL", U1)
    global ok &= no
end

# 3b. A divergence where α_e = 0 and s_j = -q₁ (two arcs of Φ(s_j) = 0 cross; σ = 0): y = π + atan(sinh μ),
# a = y sinh μ, and μ from (sinh⁻²μ - 1)/2 = eᵃ/(cosh μ + eᵃ); then s_j = -y²
let f = μ -> (S=sinh(μ); a=(π + atan(S)) * S; (S^-2 - 1) / 2 - exp(a) / (cosh(μ) + exp(a)))
    μ = bisect0(f, big"0.55", big"0.88")
    y = π + atan(sinh(μ))
    μf, Uf = Float64(μ), Float64(2y * sinh(μ))
    Φ, σ, s = ph_line(Uf, μf, 1, 2)
    λ = [minimum(abs.(eigvals(boxchi(:d, Uf, μf, M)))) for M in (24, 60, 120)]
    good = abs(s + Float64(y)^2) < 1e-10 && abs(Φ) < 1e-12 && genuine(Φ, σ, s) &&
           λ[3] < λ[2] < λ[1] < 1e-9
    @printf("%-4s α_e = 0 on a line (βμ = %.5f, βU = %.5f): Φ = %.0e, σ = %.0e; smallest box |λ| for M = 24, 60, 120: %s\n",
            good ? "PASS" : "FAIL", μf, Uf, Φ, σ,
            join([@sprintf("%.0e", l) for l in λ], ", "))
    global ok &= good
end

# 3b'. The singlet curve where K_s = 0 = d₋ (μ > 0) or d₊ (μ < 0): no box eigenvalue with eigenvector aligned
# with g = 1/(t_n + A_s) is small, and Γ_s has a finite limit approaching the curve
let good = true, worst = Inf
    for μ in (0.3, 1.0, 3.0, -0.3, -1.0, -3.0)
        dfac(U) = (p=wts(U, abs(μ)); 4 * (U / 2) * p[2] + 2 * abs(μ) * (p[1] + p[2]))  # d∓ up to -2a, by μ → -μ
        # the other zero of d∓ is U = -2|μ| (b = ∓2a), below the curve for |βμ| ≲ 2 and above it beyond
        U = abs(μ) < 2 ? bisect0(dfac, -30.0, -2abs(μ) - 0.3) :
            bisect0(dfac, -2abs(μ) + 0.3, -0.5)
        Ks = s_line(U, μ)[1]
        M = 60
        idx = (-M - 1):M
        a = U / 2
        p0, p1, p2 = wts(U, μ)
        As = a * (p2 - p0) * 2μ + a^2 * (1 - 8p1)
        g = [1 / (abs2(μ + im * (2n + 1) * π) + As) for n in idx]
        E = eigen(boxchi(:s, U, μ, M))
        ov = [abs(dot(g, E.vectors[:, i])) / (norm(g) * norm(E.vectors[:, i]))
              for i in eachindex(E.values)]
        λg = abs(E.values[argmax(ov)])                       # the eigenvalue most aligned with g
        w = (F(0), F(-1), BosonicFreq(0))
        Γ = [gamma(SingletChannel(), DopedHubbardAtom(U + δ, 1.0, μ), w)
             for δ in (1e-2, 1e-4, 1e-6, -1e-6)]
        lim = abs(Γ[3] - Γ[4]) < 1e-4 * (1 + abs(Γ[3])) &&
              abs(Γ[2] - Γ[3]) < 1e-2 * (1 + abs(Γ[3]))
        good &= abs(Ks) < 1e-10 && λg > 1e-3 && lim
        worst = min(worst, λg)
    end
    @printf("%-4s singlet K_s = 0 = d∓ at βμ = ±0.3, ±1, ±3 (U < 0): smallest |λ| aligned with g ≥ %.1e; Γ_s finite on both sides\n",
            good ? "PASS" : "FAIL", worst)
    global ok &= good
end

# 3c. Where s_1 = s_2 on a line (the line passes from one root to the other): Δ = 0, Φ = 0 and σ > 0
# (the cancelling arcs meet at Δ = 0 too, with σ < 0)
function switchpoint(ε, x)
    function eqs(x)
        μ, a = x[1], x[2] / 2
        p0, p1, p2 = wts(x[2], μ)
        ρ, q1 = p2 - p0, a^2 + 4ε * a^2 * p1
        s = 2a^2 * ρ^2 - q1
        [(a^2 * ρ^4 - ρ^2 * q1 + 4a^2 * p0 * p2) / a^2, cosh(μ) + cu(s) + ε * a * su(s)]
    end
    for _ in 1:50
        F = eqs(x)
        h = 1e-7
        x = x - hcat([(eqs(x + h * e) - F) / h for e in ([1.0, 0], [0, 1.0])]...) \ F
    end
    μ, a = x[1], x[2] / 2
    p0, p1, p2 = wts(x[2], μ)
    q1 = a^2 + 4ε * a^2 * p1
    s = 2a^2 * (p2 - p0)^2 - q1
    x, maximum(abs, eqs(x)), (s + q1) * ε * (1 + cosh(μ) * cu(s)) / (cu(s) + cosh(μ))
end
for (label, ε, guess) in (("end of the first loop, U > 0", 1, [0.5693, 4.44]),
                          ("first density line, U < 0", 1, [1.45, -6.97]))
    x, r, σ = switchpoint(ε, guess)
    @printf("%-4s s₁ = s₂ on the %s: βμ = %.5f, βU = %.5f, σ = %+.1e\n",
            r < 1e-10 && σ > 0 ? "PASS" : "FAIL", label, x[1], x[2], σ)
    global ok &= r < 1e-10 && σ > 0
end

# 4. Half filling: the lines start at the localized divergences βU = 2π(2n+1)/√3 (d) and at the zeros of D̂
let μ = 1e-6, Us = range(0.06, 25; length=20000)
    loc = [U for (U, j) in divergences(:d, μ, Us)]
    exact = [2π * (2n + 1) / sqrt(3) for n in 0:2]
    good = all(any(abs.(loc .- x) .< 1e-4) for x in exact)
    @printf("%-4s βμ → 0: density lines include 2π(2n+1)/√3 = %s; also %s\n",
            good ? "PASS" : "FAIL",
            join([@sprintf("%.4f", x) for x in exact], ", "),
            join([@sprintf("%.4f", U) for U in loc if !any(abs.(U .- exact) .< 1e-4)],
                 ", "))
    global ok &= good
end
ok || error("SOME CHECKS FAILED")
println("ALL PASS")
