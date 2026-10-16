# Data for the figures and the example table of doped-gamma-derivation.typ.
# Run: julia --project=../../.. figure_data.jl   (writes *.csv next to this file)
include("../doped_gamma.jl");
const DA = DopedAtom
using LinearAlgebra, Printf

function weights(U, β, μ)
    E = (U/4 + μ, -U/4, U/4 - μ)
    w = exp.(-β .* (E .- minimum(E)))
    Z = w[1] + 2w[2] + w[3]
    (w[1]/Z, w[2]/Z, w[3]/Z)
end
# susceptibility from Eqs. (chi-uu)-(pair) with Lemma (compact input) of the note: returns (total, diagonal, Cauchy, rest)
function chi_parts(U, β, μ, k, ε, n, m)
    a = U/2
    p0, p1, p2 = weights(U, β, μ)
    ρ = p2 - p0
    w = 2π*im*k/β
    q0 = a^2 - w^2/4
    z(n) = μ + im*π*(2n + 1 + k)/β
    Qp(z) = (z + a)^2 - w^2/4
    Qm(z) = (z - a)^2 - w^2/4
    H(z) = z^2 + 2a*ρ*z + q0 + 4ε*a^2*p1
    C = μ == 0 ? β*p0*(m == -n - k - 1) : ρ/(z(n) + z(m))
    zn, zm = z(n), z(m)
    x, y = zn - w/2, zm - w/2
    dg = n == m ? -β*H(zn)/(Qp(zn)*Qm(zn)) : 0.0im
    cau = ε*4a^2*C/(Qm(zn)*Qm(zm))
    reg = ε*(-2a + 4a^2*(1 - ρ)*(zn + zm + 2a)*(zn*zm + q0)/(Qp(zn)*Qp(zm)))/(Qm(zn)*Qm(zm))
    ζ = 4β*a^2*((1 - ρ^2)/4 + ε*(p0*p2 - p1^2))
    st = k == 0 ? ζ/((x^2 - a^2)*(y^2 - a^2)) : 0.0im
    (dg + cau + reg + st, dg, cau, reg + st)
end

# Figure 2: matrix structure of chi_d at k = 0, n, m in -20:19
open(joinpath(@__DIR__, "chi_structure.csv"), "w") do io
    println(io, "mu,n,m,total,diagonal,cauchy,rest")
    for μ in (0.0, 0.3), n in -20:19, m in -20:19
        t, d, c, r = chi_parts(2.0, 3.0, μ, 0, 1, n, m)
        @printf(io, "%g,%d,%d,%.6e,%.6e,%.6e,%.6e\n", μ, n, m, abs(t), abs(d), abs(c),
                abs(r))
    end
end

# Figure 3: Gamma_r(nu_n, nu_0, omega_0) for several dopings
open(joinpath(@__DIR__, "gamma_cuts.csv"), "w") do io
    println(io, "channel,mu,n,re,im")
    U, β = 2.0, 4.0
    for μ in (0.0, 0.4, 1.0)
        at = DA.Atom(U, β, μ)
        gd, gm, gpp = DA.GammaPH(at, :d, 0), DA.GammaPH(at, :m, 0), DA.GammaPP(at, 0)
        for n in -12:11
            for (r, v) in ((:d, DA.gamma(gd, n, 0)), (:m, DA.gamma(gm, n, 0)),
                           (:s, DA.gamma(gpp, :s, n, 0)), (:t, DA.gamma(gpp, :t, n, 0)))
                @printf(io, "%s,%g,%d,%.10e,%.10e\n", r, μ, n, real(v), imag(v))
            end
        end
    end
end

println("done")
