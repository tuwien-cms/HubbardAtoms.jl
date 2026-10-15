# Exact (SymPy) checks of identities (A6), (A7) of doped-gamma-derivation.typ (density/magnetic channels), plus the
# hand computation of Step 3 of the proof of the d/m theorem.
#   (A6) image of the constant: -beta H(z) + lam cbar + PV sum_l N(z, z_l)/P(z_l) = beta N2(z)/(2 eps a);
#   (A7) Eq. (source): lam rho (z - v + 2 a rho) P(v) - H(-v) N(z, v) - (lam rho/beta) P(v) Ibar(z; -v)
#                      = -e(v) N2(z) + lam rho z [P(v) - e(v)] + lam W(z; v^2);
#        for r = m, k = 0 also e(z) = (z^2 - a^2)(z^2 - s) and W(z; a^2) = 0.
# Free symbols: a, beta, rho, c = p0 + p2 (p1 = (1 - c)/2), z, v, and om = w/2 (k != 0); for k = 0, w = 0.
# The lattice sums are evaluated by the residue rule (Lemma "residue rule") with S = S_+ at the zeros of Q_+,
# S = S_- at the zeros of Q_-, and S' = S^2 - beta^2/4 at the double zeros for k = 0 (Lemma "rank-two sums").
# Usage: python verify_dm.py d|m nonstatic|static   (each run takes a few seconds)
import sys
import sympy as sp

eps = {"d": 1, "m": -1}[sys.argv[1]]
static = sys.argv[2] == "static"
t, z, v, x, a, beta, rho, c, om = sp.symbols("t z v x a beta rho c om")
failed = []

def check(label, expr):
    ok = sp.simplify(sp.together(expr)) == 0
    print(f"{'PASS' if ok else 'FAIL'}  {label}", flush=True)
    if not ok:
        failed.append(label)

dk = 1 if static else 0                                  # delta_{k0}
p1 = (1 - c)/2; p0, p2 = (c - rho)/2, (c + rho)/2
w = 0 if static else 2*om
q0 = a**2 - w**2/4
lam = 4*eps*a**2; cb = beta*c/2; h2 = beta**2*p0*p2
zeta = beta*a**2*(1 - rho**2 + 4*eps*(p0*p2 - p1**2))
q1 = q0 + 4*eps*a**2*p1
Qp = lambda s: (s + a)**2 - w**2/4
Qm = lambda s: (s - a)**2 - w**2/4
P = lambda s: Qp(s)*Qm(s)
H = lambda s: s**2 + 2*a*rho*s + q1
H0 = lambda s: s**2 + 2*a*rho*s + q0
e = lambda s: sp.expand(H(s)*H(-s) - lam**2*h2/beta**2)
N2 = lambda s: -2*eps*a*H0(s) + dk*zeta*(s**2 - a**2)
N0 = lambda s: (s**2 + 2*q0 - 4*a**2)*N2(s) + (2*eps*a - dk*zeta)*P(s)
N = lambda z, s: N0(s) + z**2*N2(s)
# W0, W2 as printed in Appendix A (varpi = 4(p0 p2 - p1^2))
varpi = 4*(p0*p2 - p1**2)
psi = lambda s: -4*eps*p1*(p0 + p2 - rho**2)*(s + (1 + 2*eps)*a**2) + (1 + eps)*rho**2*varpi*(s - a**2)
W0 = lambda s: eps*a*q0*(varpi - 2*eps*(1 - rho**2))*(s + q0 + 2*eps*a**2) + dk*beta*a**4*psi(s)
W2 = lambda s: eps*a*varpi*(s + q0 + 2*eps*a**2) - dk*beta*a**2*psi(s)
W = lambda z, s: W0(s) + z**2*W2(s)
Sp = beta*(p1 - p0)/(2*(p1 + p0)); Sm = beta*(p2 - p1)/(2*(p1 + p2))   # S_k at the zeros of Q_+ / Q_-

def atomic_sum(G):
    """sum over the lattice of G restricted to the zeros of P: sum of Res[G(t) S_k(t)] there (residue rule)."""
    out = 0
    if static:
        for y, S in ((-a, Sp), (a, Sm)):
            g = sp.cancel((t - y)**2*G)
            out += sp.diff(g, t).subs(t, y)*S + g.subs(t, y)*(S**2 - beta**2/4)
    else:
        for y, S in ((-a + om, Sp), (-a - om, Sp), (a + om, Sm), (a - om, Sm)):
            out += sp.cancel((t - y)*G).subs(t, y)*S
    return out

# ---- (A6) image of the constant (C[1] = cbar as a PV sum; the residue rule applies as N0/P = O(1/t))
check("(A6) -beta H(z) + lam cbar + PV sum N(z, t)/P(t) = beta N2(z)/(2 eps a)",
      -beta*H(z) + lam*cb + atomic_sum(N(z, t)/P(t)) - beta*N2(z)/(2*eps*a))
# ---- (A7) the source identity, with Ibar(z; x) = cbar N(z, x)/(rho P(x)) + atomic residues
Ibar = lambda xx: (cb*N(z, x)/(rho*P(x)) + atomic_sum(N(z, t)/(P(t)*(t - x)))).subs(x, xx)
lhs = lam*rho*(z - v + 2*a*rho)*P(v) - H(-v)*N(z, v) - lam*rho/beta*P(v)*Ibar(-v)
rhs = -e(v)*N2(z) + lam*rho*z*(P(v) - e(v)) + lam*W(z, v**2)
check("(A7) Eq. (source)", lhs - rhs)
if static and eps == -1:
    s_ = a**2*(1 + 4*rho**2 - 4*c)
    check("(A7) r = m, k = 0: e(z) = (z^2 - a^2)(z^2 - s)", e(z) - (z**2 - a**2)*(z**2 - s_))
    check("(A7) r = m, k = 0: W(z; a^2) = 0", W(z, a**2))
# ---- Step 3 of the proof as an abstract identity: insert Ibar(z; +-r) from (A7) at v = -+r into
#      P(z) (chi phi_j)(z) = -beta [L (z + 2 a rho) + r (beta H(-r) - lam alpha)] + beta H(-r) I(z; r) + lam alpha I(z; -r),
#      I(z; x) = Ibar(z; x) + alpha(x) N(z, x)/(rho P(x)); free symbols, with alpha(-r) = h^2/alpha(r) and
#      beta^2 H(r) H(-r) = lam^2 h^2 (i.e. e(r) = 0).
Hp, Hm, al, bt, la, rh, PP, Np, Nm, WW, rr, zz, aa, hh = sp.symbols("Hp Hm alpha beta_ lambda_ rho_ P Np Nm B r z_ a_ h2_")
Hp_ = la**2*hh/(bt**2*Hm)                       # beta^2 H(r) H(-r) = lambda^2 h^2
L = bt*Hm + la*al
Ib_p = bt/(la*rh*PP)*(la*rh*(rr + 2*aa*rh)*PP - Hp_*Nm - la*WW)    # Ibar(z; r),  from (A7) at v = -r
Ib_m = bt/(la*rh*PP)*(la*rh*(-rr + 2*aa*rh)*PP - Hm*Np - la*WW)    # Ibar(z; -r), from (A7) at v = r
I_p = Ib_p + al*Np/(rh*PP); I_m = Ib_m + (hh/al)*Nm/(rh*PP)
image = -bt*(L*(zz + 2*aa*rh) + rr*(bt*Hm - la*al)) + bt*Hm*I_p + la*al*I_m
check("Step 3: P(z) (chi phi_j)(z) = -beta K_j [z + W(z; s_j)/(rho P(u_j))]", image + bt*L*(zz + WW/(rh*PP)))
print("ALL PASS" if not failed else f"{len(failed)} FAILED")
sys.exit(1 if failed else 0)
