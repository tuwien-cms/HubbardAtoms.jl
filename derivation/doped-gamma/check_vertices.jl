# Numerical checks of the theorems of doped-gamma-derivation.typ (Appendix D): Eq. (Gamma-ph) at 256 bits and the
# singlet and triplet theorems against the residue-based evaluator doped_gamma.jl, which computes Gamma from the input
# independently of the closed forms; at mu = 0 (and mu -> 0) against the half-filled package HubbardAtoms.jl.
# Run: julia --project=../.. check_vertices.jl
include("doped_gamma.jl"); const DA = DopedAtom
using HubbardAtoms, Printf
setprecision(BigFloat, 256)

"Γ_r(ν_n, ν_m, ω_k) from Eq. (Gamma-ph); ε = +1 (d) or -1 (m)."
function gamma_dm(U, β, μ, k, ε, n, m)
    T = BigFloat; U, β, μ = T(U), T(β), T(μ); a = U/2
    E = (U/4 + μ, -U/4, U/4 - μ); wt = exp.(-β .* (E .- minimum(E))); Z = wt[1] + 2wt[2] + wt[3]
    p0, p1, p2 = wt[1]/Z, wt[2]/Z, wt[3]/Z; ρ = p2 - p0; cb = β*(p0 + p2)/2
    w = 2*T(π)*im*k/β; q0 = a^2 - w^2/4; q1 = q0 + 4ε*a^2*p1; λ = 4ε*a^2
    S(r) = iseven(k) ? -β/2*tanh(β*(r - μ)/2) : -β/2*coth(β*(r - μ)/2)
    α(r) = ρ*S(r) - cb
    H(z) = z^2 + 2a*ρ*z + q1
    Qp(z) = (z + a)^2 - w^2/4; Qm(z) = (z - a)^2 - w^2/4; P(z) = Qp(z)*Qm(z)
    dP(z) = 2(z + a)*Qm(z) + 2(z - a)*Qp(z)
    B = 2*(q1 - 2a^2*ρ^2); C = q1^2 - 16a^4*p0*p2; d = sqrt(complex(B^2 - 4C)); s = ((-B + d)/2, (-B - d)/2)
    e(z) = (z^2 - s[1])*(z^2 - s[2]); de(z) = 2z*((z^2 - s[1]) + (z^2 - s[2]))
    F(z) = P(z)/e(z); dF(z) = (dP(z)*e(z) - P(z)*de(z))/e(z)^2
    G2(z) = (z + a*ρ)^2 - w^2/4
    z(n) = μ + im*T(π)*(2n + 1 + k)/β
    zn, zm = z(n), z(m)
    DD = n == m ? (F(zn) + zn*dF(zn))/(2zn) : (zn*F(zn) - zm*F(zm))/(zn^2 - zm^2)
    function γ(j)
        r = sqrt(s[j]); abs(P(r)) < 1e-60*(1 + abs(r))^4 && return zero(r)   # atomic pair (r = m, k = 0)
        λ*ρ*r*P(r)/(s[j] - s[3 - j])*(β*H(-r) - λ*α(r))/(β*H(-r) + λ*α(r))
    end
    diag = n == m ? β*P(zn)*(1/G2(zn) - H(-zn)/e(zn)) : 0
    diag + 2ε*a - λ*ρ*DD + sum(γ(j)/((zn^2 - s[j])*(zm^2 - s[j])) for j in 1:2)
end

ok = true
for ks in ((1, 2, -3, 5), (0,))
    worst = 0.0; count = 0
    for (U, β, μ) in ((2.0, 3.0, 0.3), (-1.5, 2.0, -0.4), (3.0, 5.0, 0.7), (1.0, 0.5, -2.0), (4.0, 8.0, 0.9)), k in ks, (ε, r) in ((1, :d), (-1, :m))
        g = DA.GammaPH(DA.Atom(U, β, μ), r, k)
        for (n, m) in ((0, 0), (0, -k - 1), (3, -2), (-5, 1), (40, 7), (2, 2))
            ref = DA.gamma(g, n, m; raw=true)
            worst = max(worst, Float64(abs(gamma_dm(U, β, μ, k, ε, n, m) - ref)/(1 + abs(ref)))); count += 1
        end
    end
    @printf("%-4s Eq. (Gamma-ph) vs reference evaluator, k in %s: %d values, worst normalized error %.2e\n", worst < 1e-40 ? "PASS" : "FAIL", ks, count, worst)
    global ok &= worst < 1e-40
end
let worst = 0.0
    for (U, β) in ((2.0, 3.0), (-1.5, 2.0), (5.0, 1.0)), k in (0, 1, -2), (ε, r) in ((1, DensityChannel()), (-1, MagneticChannel())), (n, m) in ((0, 0), (0, -k - 1), (2, -3), (1, 1))
        ref = gamma(r, HubbardAtom(U, β), (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(2k)))
        worst = max(worst, Float64(abs(gamma_dm(U, β, 1e-30, k, ε, n, m) - ref)/(1 + abs(ref))))
    end
    @printf("%-4s mu -> 0 (mu = 1e-30) vs half-filled HubbardAtoms.jl: worst normalized error %.2e\n", worst < 1e-13 ? "PASS" : "FAIL", worst)
    global ok &= worst < 1e-13
end
# Corollary (half filling): Eq. (Gamma-hf) vs the half-filled package HubbardAtoms.jl (Eq. 19 of Thunstroem et al.)
function gamma_hf(U, β, k, ε, n, m)
    a = U/2; E = (U/4, -U/4, U/4); wt = exp.(-β .* (E .- minimum(E))); Z = wt[1] + 2wt[2] + wt[3]
    p0, p1 = wt[1]/Z, wt[2]/Z; h = β*p0
    w = 2π*im*k/β; q0 = a^2 - w^2/4; q1 = q0 + 4ε*a^2*p1; λ = 4ε*a^2
    H(z) = z^2 + q1; P(z) = ((z + a)^2 - w^2/4)*((z - a)^2 - w^2/4); G2(z) = z^2 - w^2/4
    e(z) = H(z)^2 - (λ*h/β)^2; F(z) = P(z)/e(z)
    S(x) = iseven(k) ? -β/2*tanh(β*x/2) : -β/2*coth(β*x/2)
    ŝ = 4ε*a^2*(p0 - p1) - q0; û = sqrt(complex(ŝ)); K̂ = β*û - 2ε*a*S(û)
    z(n) = im*π*(2n + 1 + k)/β; zn, zm = z(n), z(m)
    (n == m ? β*P(zn)*(1/G2(zn) - H(zn)/e(zn)) : 0) + 2ε*a - (m == -n - k - 1 ? λ*h*F(zn) : 0) -
        2ε*a*β*û*P(û)/K̂/((zn^2 - ŝ)*(zm^2 - ŝ))
end
let worst = 0.0
    for (U, β) in ((2.0, 3.0), (-1.5, 2.0), (5.0, 1.0), (1.0, 7.0)), k in (0, 1, -2, 3), (ε, r) in ((1, DensityChannel()), (-1, MagneticChannel())),
        (n, m) in ((0, 0), (0, -k - 1), (2, -3), (1, 1), (-3, 5), (4, -k - 5))
        ref = gamma(r, HubbardAtom(U, β), (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(2k)))
        worst = max(worst, abs(gamma_hf(U, β, k, ε, n, m) - ref)/(1 + abs(ref)))
    end
    @printf("%-4s Corollary (half filling) vs HubbardAtoms.jl: worst normalized error %.2e\n", worst < 1e-12 ? "PASS" : "FAIL", worst)
    global ok &= worst < 1e-12
end
# Theorems (triplet, singlet) with T_k(t) = -S_k(mu + xi)/xi: vs the reference evaluator (mu != 0) and HubbardAtoms.jl (mu = 0)
function gamma_pp(U, β, μ, k, ch, n, m)
    a = U/2; E = (U/4 + μ, -U/4, U/4 - μ); wt = exp.(-β .* (E .- minimum(E))); Z = wt[1] + 2wt[2] + wt[3]
    p0, p1, p2 = wt[1]/Z, wt[2]/Z, wt[3]/Z; ρ = p2 - p0
    w = 2π*im*k/β; b = 2μ - w; x(n) = μ + im*π*(2n + 1)/β; t(n) = x(n)*(b - x(n))
    S(y) = iseven(k) ? -β/2*tanh(β*(y - μ)/2) : -β/2*coth(β*(y - μ)/2)
    T(tt) = (ξ = sqrt(complex(b^2/4 - tt)); -S(μ + ξ)/ξ)
    𝒫(tt) = (tt + a^2)^2 - a^2*b^2; 𝒢(tt) = tt + a*ρ*b + a^2*ρ^2; As = a*ρ*b + a^2*(1 - 8p1)
    tn, tm = t(n), t(m); mir = m == -n - k - 1
    ch == :t && return n == -n - k - 1 ? zero(tn) : β*a^2*(1 - ρ^2)*𝒫(tn)/(𝒢(tn)*(tn + a*ρ*b + a^2))*((n == m) - mir)
    Ks = 1 + 2a/β*T(-As)
    β*a^2*(ρ^2 - 1 + 8p1)*𝒫(tn)/(𝒢(tn)*(tn + As))*((n == m) + mir) + 4a - 4a*𝒫(-As)/(Ks*(tn + As)*(tm + As))
end
let worst_ref = 0.0, worst_pkg = 0.0
    for (U, β) in ((2.0, 3.0), (-1.5, 2.0), (5.0, 1.0)), k in (0, 1, -2, 3), (ch, R) in ((:s, SingletChannel()), (:t, TripletChannel())),
        (n, m) in ((0, 0), (0, -k - 1), (2, -3), (1, 1), (-3, 5))
        for μ in (0.3, -0.7)
            ref = DA.gamma(DA.GammaPP(DA.Atom(U, β, μ), k), ch, n, m)
            worst_ref = max(worst_ref, abs(gamma_pp(U, β, μ, k, ch, n, m) - ref)/(1 + abs(ref)))
        end
        ref = gamma(R, HubbardAtom(U, β), (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(2k)))
        worst_pkg = max(worst_pkg, abs(gamma_pp(U, β, 0.0, k, ch, n, m) - ref)/(1 + abs(ref)))
    end
    @printf("%-4s Theorems (triplet, singlet) vs reference evaluator: %.2e; at mu = 0 vs HubbardAtoms.jl: %.2e\n", max(worst_ref, worst_pkg) < 1e-12 ? "PASS" : "FAIL", worst_ref, worst_pkg)
    global ok &= max(worst_ref, worst_pkg) < 1e-12
end
println(ok ? "ALL PASS" : "SOME FAILED")
