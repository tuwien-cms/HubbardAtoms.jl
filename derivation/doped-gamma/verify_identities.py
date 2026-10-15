# Exact (SymPy) checks of the rational identities (A1)-(A5) of doped-gamma-derivation.typ; (A6), (A7) are in verify_dm.py.
# Run: python verify_identities.py   (needs SymPy)
import sys
import sympy as s

a, rho, w, z, v, b, p1, beta, eps, t = s.symbols('a rho w z v b p1 beta epsilon t')
x, y, u = s.symbols('x y u')
T3 = s.symbols('T3')  # stands for T_k(-A_s)
failed = []

def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        failed.append(label)

zero = lambda e: s.simplify(s.together(e)) == 0

def R12(x, y, w):  # Essl et al. Eq. (D3), regular part at zero field, q = (1 + rho)/2 (Appendix C)
    q = (1 + rho) / 2
    return ((q-1)/((y+w+a)*(y+a)*(x-a)) + (1-q)/((x+w-a)*(x-a)*(y+a))
          + (1-q)/((x+w+a)*(y-a)*(x-a)) + (q-1)/((x+w+a)*(x-a)*(y+w+a))
          + (1-q)/((y+a)*(x+w+a)*(y+w+a)) + (1-q)/((y+a)*(x+w+a)*(x+a))
          + (1-q)/((y-a)*(x+a)*(y+w-a)) + (q-1)/((y-a)*(x+w+a)*(x+a))
          + (1-q)/((x+w-a)*(x+a)*(y+w-a)) + (q-1)/((y+a)*(x+w-a)*(x+a))
          - q/((y-a)*(x+w-a)*(y+w-a)) - q/((y-a)*(x+w-a)*(x-a)))

Pi = lambda x, y, w: (1/(x+w-a) + 1/(y-a)) * (1/(y+w-a) + 1/(x-a))
Qp = lambda z: (z + a)**2 - w**2/4
Qm = lambda z: (z - a)**2 - w**2/4
q0 = a**2 - w**2/4
X, Y = z - w/2, v - w/2

# (A1) twelve terms -> two terms
two = (-2*a - rho*(z+v-4*a))/(Qm(z)*Qm(v)) + 4*a**2*(1-rho)*(z+v+2*a)*(z*v+q0)/(Qp(z)*Qm(z)*Qp(v)*Qm(v))
check("(A1) R(z-w/2, v-w/2) = two-term form", zero(R12(X, Y, w) - two))
C = rho/(z+v)
N = (4*a**2*C - 2*a)/(Qm(z)*Qm(v)) + 4*a**2*(1-rho)*(z+v+2*a)*(z*v+q0)/(Qp(z)*Qm(z)*Qp(v)*Qm(v))
check("(A1) pair term: Pi = (z+v-2a)^2/(Q-(z)Q-(v))", zero(Pi(X, Y, w) - (z+v-2*a)**2/(Qm(z)*Qm(v))))
check("(A1) R + C Pi = Eq. (input)", zero(R12(X, Y, w) + C*Pi(X, Y, w) - N))

# (A2) the regular bracket in the proof of the d/m structure proposition
R = lambda z: z**2 + 2*a*rho*z + q0
P = lambda z: Qp(z)*Qm(z)
check("(A2) Q+(z) - Q-(v) = (z+v)(z-v+2a)", s.expand(Qp(z) - Qm(v) - (z+v)*(z-v+2*a)) == 0)
bracket = 4*a**2*rho*(z-v+2*a)*Qp(v) - 2*a*Qp(z)*Qp(v) + 4*a**2*(1-rho)*(z+v+2*a)*(z*v+q0)
check("(A2) bracket = -2a R z^2 - 2a (v^2 + 2q0 - 4a^2) R + 2a P(v), R = v^2 + 2 a rho v + q0",
      s.expand(bracket - (-2*a*R(v)*z**2 - 2*a*(v**2 + 2*q0 - 4*a**2)*R(v) + 2*a*P(v))) == 0)
check("(A2) P(z) N = 4a^2 C + bracket/P(v)", zero(P(z)*N - 4*a**2*C - bracket/P(v)))
H = lambda z: z**2 + 2*a*rho*z + q0 + 4*eps*a**2*p1

# (A3) pp identities and the singlet non-delta part
Pt = lambda t: (t + a**2)**2 - a**2*b**2
tx, ty = x*(b-x), y*(b-y)
rp, rm = -a**2 + a*b, -a**2 - a*b
check("(A3) (x^2-a^2)((b-x)^2-a^2) = P(t)", s.expand((x**2-a**2)*((b-x)**2-a**2) - Pt(tx)) == 0)
check("(A3) (x+a rho)(b-x+a rho) = G(t)", s.expand((x+a*rho)*(b-x+a*rho) - (tx + a*rho*b + a**2*rho**2)) == 0)
check("(A3) (x-a)(b-x-a) = t-r+, (x+a)(b-x+a) = t-r-",
      s.expand((x-a)*(b-x-a) - (tx-rp)) == 0 and s.expand((x+a)*(b-x+a) - (tx-rm)) == 0)
c = rho/b
fin = (c*Pi(x, y, b-x-y) + R12(x, y, b-x-y))/2
num = a*(2*a*c-1)*(tx-rm)*(ty-rm) + a**2*(1-rho)*(b+2*a)*(tx+ty+2*a**2)
check("(A3) singlet non-delta part, numerator form (Eq. chi-s-num)", zero(fin - num/(Pt(tx)*Pt(ty))))
alp, alm = a*(1+rho)*(2*a-b)/(2*b), a*(rho-1)*(2*a+b)/(2*b)
check("(A3) singlet non-delta part, pole form (Eq. chi-s-reg)",
      zero(fin - (alp/((tx-rp)*(ty-rp)) + alm/((tx-rm)*(ty-rm)))))
wp = b - x - y
zp, vp = x + wp/2, y + wp/2
check("(A3) z v + q0 = a^2 + (t+u)/2 at w' = b-x-y", s.expand(zp*vp + a**2 - wp**2/4 - (a**2 + (tx+ty)/2)) == 0)

# (A4) singlet: Xi is diagonal in the basis (1, 1/(t+A_s)); A, T3 stand for A_s, T_k(-A_s)
p2s, p0s = (1 + rho)/2 - p1, (1 - rho)/2 - p1
A = a*rho*b + a**2*(1 - 8*p1)
dp, dm = A + rp, A + rm
Tp = beta/(2*a - b)*(p1 - p2s)/(p1 + p2s)
Tm = beta/(2*a + b)*(p1 - p0s)/(p1 + p0s)
S_tA = (dp*Tp - dm*Tm)/(rp - rm)
S_1 = (Tp - Tm)/(rp - rm)
S_inv = Tp/((rp - rm)*dp) - Tm/((rp - rm)*dm) + T3/(dp*dm)
Tb = s.Matrix([[1, 1], [-dm, -dp]])
Xi = Tb.inv().T*s.diag(1/alp, 1/alm)*Tb.inv() + (2/beta)*s.Matrix([[S_tA, S_1], [S_1, S_inv]])
claimO = s.diag(-1/a, (1 + a*2*T3/beta)/(a*dp*dm))
check("(A4) Xi = diag(-1/a, K_s/(a d+ d-))", all(zero(e) for e in Xi - claimO))
check("(A4) d+ d- = P(-A_s)", zero(dp*dm - Pt(-A)))

# (A5) diagonal and static identities in the proof of the d/m structure proposition; H(+-a) for r = m, k = 0
q = (1 + rho)/2
G = lambda x: (x + a*rho)/(x**2 - a**2)
J = lambda x: 1/(x**2 - a**2)
diagterm = G(X)*G(X + w) + 4*a**2*(q*(1-q) + eps*p1)*J(X)*J(X + w)
check("(A5) diagonal: G G + 4a^2(q(1-q) + eps p1) J J = H/P", zero(diagterm - H(z)/P(z)))
check("(A5) static: P(z) J(z) J(v) = (z^2-a^2)(v^2-a^2)/P(v) at w = 0",
      zero((P(z)*J(z)*J(v) - (z**2-a**2)*(v**2-a**2)/P(v)).subs(w, 0)))
p0, p2 = (1 - rho)/2 - p1, (1 + rho)/2 - p1
Hm0 = lambda z: H(z).subs({eps: -1, w: 0})
check("(A5) r = m, k = 0: H(a) = 4a^2 p2, H(-a) = 4a^2 p0",
      s.expand(Hm0(a) - 4*a**2*p2) == 0 and s.expand(Hm0(-a) - 4*a**2*p0) == 0)

print("ALL PASS" if not failed else f"{len(failed)} FAILED")
sys.exit(1 if failed else 0)
