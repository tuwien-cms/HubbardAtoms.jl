# Exceptional points of doped-gamma-derivation.typ (Theorems "singlet", "density and magnetic", Corollary "half filling",
# Divergences): the closed forms at removable points against finite-box inversion, the confluent and D-hat forms, and a
# genuine divergence. Run: julia --project=../.. check_exceptional.jl   (about two minutes)
include("doped_gamma.jl"); const DA = DopedAtom
using HubbardAtoms, Printf, LinearAlgebra
setprecision(BigFloat, 512)
const T = BigFloat
report(name, err, tol) = (@printf("%-4s %-78s %.2e\n", err <= tol ? "PASS" : "FAIL", name, err); err <= tol)
weights(U, β, μ) = (E = (U/4 + μ, -U/4, U/4 - μ); w = exp.(-β .* (E .- minimum(E))); Z = w[1] + 2w[2] + w[3]; (w[1]/Z, w[2]/Z, w[3]/Z))
S(k, β, μ, x) = iseven(k) ? -β/2*tanh(β*(x - μ)/2) : -β/2*coth(β*(x - μ)/2)
"Finite-box inverse (boxes of Proposition 'finite boxes'), Richardson-extrapolated in 1/M from M = 128, 256, 512."
function boxed(U, β, μ, k, r)
    Bs = map((128, 256, 512)) do M
        at = DA.Atom(U, β, μ); idx = [(j - 1 - k) ÷ 2 for j in -(2M+1):(2M+1) if iseven(j - 1 - k)]
        Y = inv([DA.chi(at, r, a, b, k) for a in idx, b in idx]); pos(x) = findfirst(==(x), idx); η = r == :s ? -1 : 1
        (n, m) -> β^2*(Y[pos(n), pos(m)] - (n == m)*η/DA.bubble(at, r, n, k))
    end
    (n, m) -> (8Bs[3](n, m) - 6Bs[2](n, m) + Bs[1](n, m))/3
end
"Common d/m data; s = (s1, s2) are the roots of e in z^2."
function dm(U, β, μ, k, ε)
    a = U/2; p0, p1, p2 = weights(U, β, μ); ρ = p2 - p0; cb = β*(p0 + p2)/2; w = 2T(π)*im*k/β; q0 = a^2 - w^2/4
    q1 = q0 + 4ε*a^2*p1; λ = 4ε*a^2; c2 = 2(q1 - 2a^2*ρ^2); c0 = q1^2 - 16a^4*p0*p2; d = sqrt(complex(c2^2 - 4c0))
    H(z) = z^2 + 2a*ρ*z + q1; P(z) = ((z + a)^2 - w^2/4)*((z - a)^2 - w^2/4); G2(z) = (z + a*ρ)^2 - w^2/4
    α(x) = ρ*S(k, β, μ, x) - cb; Zu(u) = u*P(u)*(β*H(-u) - λ*α(u))/(β*H(-u) + λ*α(u))
    Z(sv) = (u = sqrt(complex(sv)); (Zu(u) + Zu(-u))/2)      # even part: analytic in s, = Z_j at the roots
    (; a, ρ, λ, H, P, G2, Z, qw = -w^2/4, s = ((-c2 + d)/2, (-c2 - d)/2), z = n -> μ + im*T(π)*(2n + 1 + k)/β)
end
"Theorem (density and magnetic); with conf = true the sum over j is the confluent derivative at the double root."
function gamma_dm(U, β, μ, k, ε, n, m; conf = false)
    D = dm(T(U), T(β), T(μ), k, ε); zn, zm = D.z(n), D.z(m); a, ρ, λ = D.a, D.ρ, D.λ
    s1, s2 = conf ? ((D.s[1] + D.s[2])/2, (D.s[1] + D.s[2])/2) : D.s
    e(z) = (z^2 - s1)*(z^2 - s2); F(z) = D.P(z)/e(z)
    Qp(z) = (z + a)^2 + D.qw; Qm(z) = (z - a)^2 + D.qw     # D.qw = -w^2/4
    dF(z) = ((2(z + a)*Qm(z) + 2(z - a)*Qp(z))*e(z) - D.P(z)*2z*(2z^2 - s1 - s2))/e(z)^2
    DD = n == m ? (F(zn) + zn*dF(zn))/(2zn) : (zn*F(zn) - zm*F(zm))/(zn^2 - zm^2)
    f(sv) = D.Z(sv)/((zn^2 - sv)*(zm^2 - sv))
    sep = conf ? (h = T(10)^-60; λ*ρ*(f(s1 + h) - f(s1 - h))/(2h)) : λ*ρ*(f(s1) - f(s2))/(s1 - s2)
    (n == m ? T(β)*D.P(zn)*(1/D.G2(zn) - D.H(-zn)/e(zn)) : 0) + 2ε*a - λ*ρ*DD + sep
end
"Theorem (singlet), with T_k(t) = -S_k(mu + xi)/xi."
function gamma_s(U, β, μ, k, n, m)
    U, β, μ = T(U), T(β), T(μ); a = U/2; p0, p1, p2 = weights(U, β, μ); ρ = p2 - p0; w = 2T(π)*im*k/β; b = 2μ - w
    t(n) = (x = μ + im*T(π)*(2n + 1)/β; x*(b - x)); 𝒫(tt) = (tt + a^2)^2 - a^2*b^2; 𝒢(tt) = tt + a*ρ*b + a^2*ρ^2
    As = a*ρ*b + a^2*(1 - 8p1); ξ = sqrt(complex(b^2/4 + As)); Ks = 1 + 2a/β*(-S(k, β, μ, μ + ξ)/ξ); tn, tm = t(n), t(m)
    β*a^2*(ρ^2 - 1 + 8p1)*𝒫(tn)/(𝒢(tn)*(tn + As))*((n == m) + (m == -n - k - 1)) + 4a - 4a*𝒫(-As)/(Ks*(tn + As)*(tm + As))
end
"Corollary (half filling), with the normalized denominator D-hat."
function gamma_hf(U, β, k, ε, n, m)
    U, β = T(U), T(β); a = U/2; p0, p1, _ = weights(U, β, T(0)); h = β*p0; w = 2T(π)*im*k/β; q0 = a^2 - w^2/4; q1 = q0 + 4ε*a^2*p1; λ = 4ε*a^2
    H(z) = z^2 + q1; P(z) = ((z + a)^2 - w^2/4)*((z - a)^2 - w^2/4); e(z) = H(z)^2 - (λ*h/β)^2; F(z) = P(z)/e(z)
    ŝ = 4ε*a^2*(p0 - p1) - q0; û = sqrt(complex(ŝ))
    D̂ = abs(û) < T(10)^-100 ? (iseven(k) ? β*(1 + ε*a*β/2) : T(Inf)) : β - 2ε*a*S(k, β, T(0), û)/û
    z(n) = im*T(π)*(2n + 1 + k)/β; zn, zm = z(n), z(m)
    (n == m ? β*P(zn)*(1/(zn^2 - w^2/4) - H(zn)/e(zn)) : 0) + 2ε*a - (m == -n - k - 1 ? λ*h*F(zn) : 0) - 2ε*a*β*P(û)/D̂/((zn^2 - ŝ)*(zm^2 - ŝ))
end
function bisect(f, lo, hi)
    lo, hi = T(lo), T(hi)
    for _ in 1:1600
        mid = (lo + hi)/2; sign(f(mid)) == sign(f(lo)) ? (lo = mid) : (hi = mid)
    end
    (lo + hi)/2
end
rel(x, y) = Float64(abs(x - y)/(1 + abs(y)))
pts = ((0, 0), (0, 1), (2, -3))
ok = true
# 1. confluent roots, U = 2, beta = 2, r = d, k = 0
μs = bisect(μ -> (D = dm(T(2), T(2), μ, 0, 1); real((D.s[1] - D.s[2])^2)), 0.2, 0.4)
B = boxed(2.0, 2.0, Float64(μs), 0, :d)
ok &= report(@sprintf("s1 = s2 at mu = %.6f: confluent formula vs Theorem at mu*(1 +- 1e-40)", Float64(μs)),
    maximum(rel(gamma_dm(2, 2, μs*(1 + σ*T(10)^-40), 0, 1, n, m), gamma_dm(2, 2, μs, 0, 1, n, m; conf = true)) for (n, m) in pts, σ in (-1, 1)), 1e-30)
ok &= report("s1 = s2: confluent formula vs finite boxes", maximum(rel(gamma_dm(2, 2, μs, 0, 1, n, m; conf = true), B(n, m)) for (n, m) in pts), 1e-5)
# 2. half filling, r = m, k = 0, beta a = ln 3: s-hat = 0 = K-hat, D-hat form vs the package
ok &= report("u-hat = 0 (beta a = ln 3, m, k = 0): D-hat form vs HubbardAtoms.jl",
    maximum(rel(gamma_hf(2, log(T(3)), 0, -1, n, m), gamma(MagneticChannel(), HubbardAtom(2.0, log(3.0)), (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(0)))) for (n, m) in pts), 1e-13)
# 3. singlet, t_0 + A_s = 0 at mu = 0.5, k = 0
βs = bisect(β -> (p = weights(T(2), β, T(0.5)); x = T(0.5) + im*T(π)/β; real(x*(1 - x) + (p[3] - p[1]) + (1 - 8p[2]))), 2.21, 2.22)
B = boxed(2.0, Float64(βs), 0.5, 0, :s)
ok &= report(@sprintf("singlet t_0 + A_s = 0 (beta = %.6f, mu = 0.5): limit vs finite boxes", Float64(βs)),
    maximum(rel(gamma_s(2, βs*(1 + σ*T(10)^-40), 0.5, 0, n, m), B(n, m)) for (n, m) in ((0, 0), (0, -1), (0, 3), (2, 2)), σ in (-1, 1)), 1e-5)
# 4. d channel, e(z_{-1}) = 0 at mu != 0 (U = -9, beta = 1, k = 1: z_{-1} = mu is real)
μe = bisect(μ -> (D = dm(T(-9), T(1), μ, 1, 1); real((μ^2 - D.s[1])*(μ^2 - D.s[2]))), 0.7, 0.8)
B = boxed(-9.0, 1.0, Float64(μe), 1, :d)
ok &= report(@sprintf("e(z_-1) = 0 (U = -9, beta = 1, k = 1, mu = %.6f): limit vs finite boxes", Float64(μe)),
    maximum(rel(gamma_dm(-9, 1, μe*(1 + σ*T(10)^-40), 1, 1, n, m), B(n, m)) for (n, m) in ((-1, -1), (-1, 0), (0, 2), (2, -3)), σ in (-1, 1)), 1e-5)
# 5. half filling, d, k = 0, n = 0: beta H(z_0) = +lambda h is removable, beta H(z_0) = -lambda h (beta U = 2 pi/sqrt 3) is not
βh = bisect(β -> (p = weights(T(2), β, T(0)); (T(π)/β)^2 - (1 + 4*(p[2] - p[1]))), 1.9, 2.1)
B = boxed(2.0, Float64(βh), 0.0, 0, :d)
ok &= report(@sprintf("beta H(z_0) = +lambda h (mu = 0, beta = %.6f): limit vs finite boxes", Float64(βh)),
    maximum(rel(gamma_hf(2, βh*(1 + σ*T(10)^-40), 0, 1, n, m), B(n, m)) for (n, m) in ((0, 0), (0, -1), (0, 2), (1, 1)), σ in (-1, 1)), 1e-5)
B = boxed(2.0, π/sqrt(3), 0.0, 0, :d)
ok &= report("beta H(z_0) = -lambda h (beta U = 2 pi/sqrt 3): 1/|box Gamma_00 - Gamma_0,-1| (divergence)", 1/abs(B(0, 0) - B(0, -1)), 1e-8)
# 6. half filling, odd k, mirror fixed point n = -1 (z = 0), r = m, U = 2, k = 1: both signs of beta H(0) = +-lambda h removable
p1p0(β) = (p = weights(T(2), β, T(0)); p[2] - p[1])
βa = bisect(β -> 4*p1p0(β) - 1 - (T(π)/β)^2, 3.3, 3.45)          # s-hat = 0: beta H(0) = +lambda h
for (β0, name) in ((βa, @sprintf("s-hat = 0 (beta = %.6f)", Float64(βa))), (T(π), "beta H(0) = -lambda h (beta = pi)"))
    Bf = boxed(2.0, Float64(β0), 0.0, 1, :m)
    global ok &= report("fixed point z = 0, m, k = 1, " * name * ": limit vs finite boxes",
        maximum(rel(gamma_hf(2, β0*(1 + σ*T(10)^-40), 1, -1, n, m), Bf(n, m)) for (n, m) in ((-1, -1), (-1, 0), (0, 0), (2, -3)), σ in (-1, 1)), 1e-5)
end
# 7. double root at s = 0 (m, U = 2, k = 2): the even-part confluent form, with Z not analytic in s for a fixed branch
function newton2T(f, x)
    for _ in 1:80
        J = hcat([(f(x .+ h .* e) .- f(x .- h .* e)) ./ 2h for (h, e) in ((T(10)^-40, T[1, 0]), (T(10)^-40, T[0, 1]))]...)
        x = x .- J \ f(x)
    end
    x
end
βd, μd = newton2T(p -> (D = dm(T(2), p[1], p[2], 2, -1); [real(D.s[1] + D.s[2]), real(D.s[1]*D.s[2])]), T[6.52, 0.61])
B = boxed(2.0, Float64(βd), Float64(μd), 2, :m)
ok &= report(@sprintf("s1 = s2 = 0 (m, k = 2, beta = %.6f, mu = %.6f): confluent form vs boxes", Float64(βd), Float64(μd)),
    maximum(rel(gamma_dm(2, βd, μd, 2, -1, n, m; conf = true), B(n, m)) for (n, m) in pts), 1e-5)
# 8. non-fixed lattice-root pair {-1, -2} at mu != 0 (d, U = -2.5, k = 2)
βp, μp = newton2T(p -> (D = dm(T(-2.5), p[1], p[2], 2, 1); v = (D.z(-1)^2 - D.s[1])*(D.z(-1)^2 - D.s[2]); [real(v), imag(v)]), T[9.795467, 0.105711])
B = boxed(-2.5, Float64(βp), Float64(μp), 2, :d)
ok &= report(@sprintf("e(z_-1) = e(z_-2) = 0 (d, U = -2.5, k = 2, beta = %.6f, mu = %.6f): limit vs boxes", Float64(βp), Float64(μp)),
    maximum(rel(gamma_dm(-2.5, βp, μp*(1 + σ*T(10)^-40), 2, 1, n, m), B(n, m)) for (n, m) in ((-1, -1), (-1, -2), (-2, -2), (-1, 0), (2, -5)), σ in (-1, 1)), 1e-5)
println(ok ? "ALL PASS" : "SOME FAILED")
