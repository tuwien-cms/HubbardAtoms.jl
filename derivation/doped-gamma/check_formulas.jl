# Numerical checks of the input and the Matsubara sums of doped-gamma-derivation.typ (Appendix "Numerical checks"),
# against exact diagonalization (ed_doped.jl), direct lattice sums and the reference evaluator (doped_gamma.jl).
# Run: julia --project=../.. check_formulas.jl
include("doped_gamma.jl"); const DA = DopedAtom
include("ed_doped.jl")
using LinearAlgebra, Printf
report(name, err, tol) = (@printf("%-4s %-62s %.2e\n", err <= tol ? "PASS" : "FAIL", name, err); err <= tol)

struct Par; U::Float64; β::Float64; μ::Float64; a::Float64; p0::Float64; p1::Float64; p2::Float64; ρ::Float64; end
function par(U, β, μ)
    E = (U/4 + μ, -U/4, U/4 - μ); w = exp.(-β .* (E .- minimum(E))); Z = w[1] + 2w[2] + w[3]
    Par(U, β, μ, U/2, w[1]/Z, w[2]/Z, w[3]/Z, (w[3] - w[1])/Z)
end
S(p, k, r) = iseven(k) ? -p.β/2*tanh(p.β*(r - p.μ)/2) : -p.β/2*coth(p.β*(r - p.μ)/2)
zl(p, k, n) = p.μ + im*π*(2n + 1 + k)/p.β
cbar(p) = p.β*(p.p0 + p.p2)/2
hh(p) = p.β*sqrt(p.p0*p.p2)
α(p, k, r) = p.ρ*S(p, k, r) - cbar(p)
box(k, N) = [(j - 1 - k) ÷ 2 for j in -(2N+1):(2N+1) if iseven(j - 1 - k)]   # |2n+1+k| <= 2N+1
psum(f, k, N) = sum(f, box(k, N))
Gx(p, x) = (x + p.a*p.ρ)/(x^2 - p.a^2)

ok = true
# ---------- N1: compact input (Lemma "compact input") against ED
for (U, β, μ) in ((2.0, 3.0, 0.3), (-1.5, 2.0, -0.4)), k in (0, 1, -2), ε in (1, -1)
    p = par(U, β, μ); a, ρ = p.a, p.ρ; w = 2π*im*k/β; q0 = a^2 - w^2/4; q = (1 + ρ)/2
    Qp(z) = (z + a)^2 - w^2/4; Qm(z) = (z - a)^2 - w^2/4
    H(z) = z^2 + 2a*ρ*z + q0 + 4ε*a^2*p.p1
    σr = 4β*a^2*(q*(1 - q) + ε*(p.p0*p.p2 - p.p1^2))
    C(n, m) = ρ/(zl(p, k, n) + zl(p, k, m))
    function Ncal(n, m)
        z, v = zl(p, k, n), zl(p, k, m)
        (4a^2*C(n, m) - 2a)/(Qm(z)*Qm(v)) + 4a^2*(1 - ρ)*(z + v + 2a)*(z*v + q0)/(Qp(z)*Qm(z)*Qp(v)*Qm(v))
    end
    function χc(n, m)
        z = zl(p, k, n); x = z - w/2; y = zl(p, k, m) - w/2
        -β*H(z)/(Qp(z)*Qm(z))*(n == m) + ε*Ncal(n, m) + (k == 0 ? σr/((x^2 - a^2)*(y^2 - a^2)) : 0)
    end
    ED.δμ[] = μ
    err = maximum(abs(χc(n, m) - (ED.χph(U, β, (:up, :up), n, m, k) + ε*ED.χph(U, β, (:up, :dn), n, m, k))) for n in -3:2 for m in (-3, 0, 2, -n - k - 1))
    global ok &= report("N1 compact ph input vs ED (U=$U, k=$k, ε=$ε)", err, 1e-12)
end
# ---------- N2: thermal identities for α
for (U, β, μ) in ((2.0, 3.0, 0.3), (-1.5, 2.0, -0.4)), k in (0, 1), r in (0.37 + 0.21im, -1.3 + 0.8im)
    p = par(U, β, μ); h = hh(p); y1, y2 = p.β*(r + p.μ)/2, p.β*(r - p.μ)/2
    αc = iseven(k) ? -h*cosh(y1)/cosh(y2) : -h*sinh(y1)/sinh(y2)
    e = max(abs(α(p, k, r) - αc), abs(α(p, k, r)*α(p, k, -r) - h^2), abs(α(p, k, r)*S(p, k, -r) - (cbar(p)*S(p, k, r) - p.β^2*p.ρ/4)))
    global ok &= report("N2 α closed form, α(r)α(-r)=h², α(r)S(-r) identity (k=$k)", e, 1e-12)
end
# ---------- N3/N4: resolvent rule, constant rule, reciprocity (direct lattice sums, error ~1/N)
p = par(2.0, 3.0, 0.3); Nb = 400_000
for k in (0, 1)
    r = 0.4 + 0.7im; m = 2; zm = zl(p, k, m)
    d = psum(n -> p.ρ/(zm + zl(p, k, n))/(zl(p, k, n) - r), k, Nb)
    global ok &= report("N3 resolvent rule C[(t-r)^-1](z_m) = α(r)/(z_m+r) (k=$k)", abs(d - α(p, k, r)/(zm + r)), 1e-5)
    c1 = psum(n -> p.ρ/(zm + zl(p, k, n)), k, Nb)
    global ok &= report("N3 constant rule C1 = c̄ (k=$k)", abs(c1 - cbar(p)), 1e-5)
    r1, s1, r2, r3 = 0.3 - 1.1im, 0.9 + 0.2im, -0.6 + 0.5im, 1.2 - 0.4im
    ψ(t) = (t - s1)/(t - r1); φ(t) = 1/((t - r2)*(t - r3))
    Cφ(z) = (α(p, k, r2)/(z + r2) - α(p, k, r3)/(z + r3))/(r2 - r3)
    Cψ(z) = cbar(p) + (r1 - s1)*α(p, k, r1)/(z + r1)
    lhs = psum(n -> ψ(zl(p, k, n))*Cφ(zl(p, k, n)), k, Nb)
    rhs = psum(n -> Cψ(zl(p, k, n))*φ(zl(p, k, n)), k, Nb)
    global ok &= report("N4 reciprocity PV Σ ψ Cφ = Σ (Cψ) φ (k=$k)", abs(lhs - rhs), 1e-5)
end
# ---------- N5: Cauchy square on constants: entrywise C² 1 vs C(C 1)
global ok &= report("N5 c̄² - h² = β²ρ²/4", abs(cbar(p)^2 - hh(p)^2 - p.β^2*p.ρ^2/4), 1e-13)

# ---------- N6: symmetry and boundedness of Γ_ph (reference evaluator, raw BigFloat)
for (U, β, μ, k, r) in ((2.0, 3.0, 0.3, 1, :d), (-1.5, 2.0, -0.4, 0, :m), (3.0, 5.0, 0.7, 2, :m))
    g = DA.GammaPH(DA.Atom(U, β, μ), r, k)
    es = maximum(Float64(abs(DA.gamma(g, n, m; raw=true) - DA.gamma(g, m, n; raw=true))/(1 + abs(DA.gamma(g, n, m; raw=true)))) for n in (-4, 0, 3, 100), m in (-2, 1, 7))
    global ok &= report("N6 Γ_$r symmetric (U=$U,k=$k)", es, 1e-30)
    vals = [Float64(abs(DA.gamma(g, n, 1; raw=true))) for n in (10, 10^3, 10^5)]
    global ok &= report("N6 Γ_$r(n,m) bounded in n: |Γ(1e5)|/|Γ(1e3)| - 1 (U=$U,k=$k)", abs(vals[3]/vals[2] - 1), 1e-2)
end
# ---------- N7: pp sums 𝒯_k and its values at the atomic poles
for (U, β, μ) in ((2.0, 3.0, 0.3), (-1.5, 2.0, -0.4)), k in (0, 1, 2)
    p = par(U, β, μ); a = p.a; b = 2p.μ - 2π*im*k/p.β
    tn(n) = (x = p.μ + im*(2n + 1)*π/p.β; x*(b - x))
    𝒯(r) = (e = sqrt(complex(b^2/4 - r)); iseven(k) ? p.β/(2e)*tanh(p.β*e/2) : p.β/(2e)*coth(p.β*e/2))
    rr = 0.3 - 0.8im
    e1 = abs(𝒯(rr) - sum(n -> 1/(tn(n) - rr), -400_000:399_999))
    rp, rm = -a^2 + a*b, -a^2 - a*b
    e2 = max(abs(𝒯(rp) - p.β/(2a - b)*(p.p1 - p.p2)/(p.p1 + p.p2)), abs(𝒯(rm) - p.β/(2a + b)*(p.p1 - p.p0)/(p.p1 + p.p0)))
    global ok &= report("N7 𝒯_k closed form vs sum (U=$U,k=$k)", e1, 1e-5)
    global ok &= report("N7 𝒯_k at r± from atomic weights (U=$U,k=$k)", e2, 1e-12)
end
# ---------- N8: triplet and singlet closed forms vs reference evaluator
for (U, β, μ) in ((2.0, 3.0, 0.3), (-1.5, 2.0, -0.4), (1.0, 0.5, -2.0)), k in (0, 1, -2)
    p = par(U, β, μ); a, ρ = p.a, p.ρ; b = 2μ - 2π*im*k/β
    tn(n) = (x = μ + im*(2n + 1)*π/β; x*(b - x))
    P(t) = (t + a^2)^2 - a^2*b^2; Nn(t) = t + a*ρ*b + a^2*ρ^2; A = a*ρ*b + a^2*(1 - 8p.p1)
    e = sqrt(complex(b^2/4 + A)); 𝒯 = abs(e) < 1e-8 ? β^2/4 : (iseven(k) ? β/(2e)*tanh(β*e/2) : β/(2e)*coth(β*e/2))
    Ds = 1 + 2a/β*𝒯
    Γt(n, m) = (t = tn(n); β*a^2*(1 - ρ^2)*P(t)/(Nn(t)*(t + a*ρ*b + a^2))*((n == m) - (m == -n - k - 1)))
    Γs(n, m) = (t = tn(n); β*a^2*(ρ^2 - 1 + 8p.p1)*P(t)/(Nn(t)*(t + A))*((n == m) + (m == -n - k - 1)) + 4a - 4a*P(-A)/(Ds*(t + A)*(tn(m) + A)))
    g = DA.GammaPP(DA.Atom(U, β, μ), k)
    pts = [(n, m) for n in (-3, 0, 2, 50) for m in (-2, 0, 4, -n - k - 1) if !(isodd(k) && n == -(k + 1) ÷ 2)]
    et = maximum(abs(Γt(n, m) - ComplexF64(DA.gamma(g, :t, n, m; raw=true)))/(1 + abs(Γt(n, m))) for (n, m) in pts)
    es = maximum(abs(Γs(n, m) - ComplexF64(DA.gamma(g, :s, n, m; raw=true)))/(1 + abs(Γs(n, m))) for (n, m) in pts)
    global ok &= report("N8 triplet closed form (U=$U,k=$k)", et, 1e-12)
    global ok &= report("N8 singlet rank-one form (U=$U,k=$k)", es, 1e-12)
end
println(ok ? "ALL PASS" : "SOME FAILED")
