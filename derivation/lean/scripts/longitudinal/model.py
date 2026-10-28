"""Exact model of the Lean formalization of the longitudinal inverse (column spin t, rows t and n = -t).

Mirrors the Lean definitions: block images of `FieldLSums.lean` (times P(x)), the column ansatz with
hats (coefficients divided by P(v)), the t-normalized mode, and the completion. All in python-flint
over Q, variables u, v, s, a, w, b, P0, rh, M with p0 = (P0 - rh)/2, p2 = (P0 + rh)/2,
pt = (1 - P0 + M)/2, pn = (1 - P0 - M)/2 (M = mL t).
"""
import flint

NAMES = ('u', 'v', 's', 'a', 'w', 'b', 'P0', 'rh', 'M')
ctx = flint.fmpq_mpoly_ctx.get(NAMES, 'lex')
GENS = ctx.gens()
u, v, s, a, wsym, b, P0, rh, M = GENS
ONE = ctx.from_dict({(0,) * len(NAMES): 1})
ZERO = ONE - ONE


def deg(p, var):
    return p.degrees()[NAMES.index(var)] if not p.is_zero() else -1


def subs(p, var, x):
    i = NAMES.index(var)
    return p.compose(*[x if j == i else g for j, g in enumerate(GENS)])


class F:
    __slots__ = ('n', 'd')

    def __init__(self, n, d=None):
        if not isinstance(n, flint.fmpq_mpoly):
            n = ONE * n
        if d is None:
            d = ONE
        if not isinstance(d, flint.fmpq_mpoly):
            d = ONE * d
        if n.is_zero():
            self.n, self.d = ZERO, ONE
            return
        g = n.gcd(d)
        if not g.is_one():
            n, d = n / g, d / g
        c = d.leading_coefficient()
        self.n, self.d = n / c, d / c

    @staticmethod
    def of(x):
        return x if isinstance(x, F) else F(x)

    def __add__(self, o):
        o = F.of(o)
        if self.d == o.d:
            return F(self.n + o.n, self.d)
        g = self.d.gcd(o.d)
        return F(self.n * (o.d / g) + o.n * (self.d / g), self.d * (o.d / g))
    __radd__ = __add__

    def __neg__(self):
        return F(-self.n, self.d)

    def __sub__(self, o):
        return self + (-F.of(o))

    def __rsub__(self, o):
        return F.of(o) - self

    def __mul__(self, o):
        o = F.of(o)
        g1, g2 = self.n.gcd(o.d), o.n.gcd(self.d)
        return F((self.n / g1) * (o.n / g2), (self.d / g2) * (o.d / g1))
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = F.of(o)
        return self * F(o.d, o.n)

    def __rtruediv__(self, o):
        return F.of(o) / self

    def __pow__(self, e):
        r = F(ONE)
        for _ in range(e):
            r = r * self
        return r

    def is_zero(self):
        return self.n.is_zero()


class Model:
    def __init__(self, k0):
        self.k0 = k0
        self.w = ZERO if k0 else wsym
        self.dl = 1 if k0 else 0          # kron k 0
        self.p0 = (P0 - rh) / 2
        self.p2 = (P0 + rh) / 2
        self.pt = (1 - P0 + M) / 2
        self.pn = (1 - P0 - M) / 2
        self.rt = rh - M
        self.rn = rh + M
        self.piP = self.p0 * self.p2
        self.piS = self.pt * self.pn
        self.dpi = self.piP - self.piS
        self.Dt = self.p2 * self.pt - self.p0 * self.pn
        self.ct = self.p2 * self.pt + self.p0 * self.pn
        self.Dn = self.p2 * self.pn - self.p0 * self.pt
        self.cn = self.p2 * self.pn + self.p0 * self.pt
        self.cP = b * (self.p0 + self.p2) / 2
        self.cS = b * (self.pt + self.pn) / 2
        self.lam = 4 * a ** 2
        self.q0 = a ** 2 - self.w ** 2 / 4
        self.kap = 2 * self.q0 - 4 * a ** 2
        self.zeta = 4 * b * a ** 2 * self.dpi
        self.zt = 4 * b * a ** 2 * (self.p0 + self.pt) * (self.pn + self.p2)
        self.zn = 4 * b * a ** 2 * (self.p0 + self.pn) * (self.pt + self.p2)
        # lattice constants (the zeros of P), lattice 't' and 'n'
        self.lat = {}
        for tau, ptau, pother in (('t', self.pt, self.pn), ('n', self.pn, self.pt)):
            Sp = F(b * (ptau - self.p0), 2 * (self.p0 + ptau))
            Sm = F(b * (self.p2 - pother), 2 * (self.p2 + pother))
            Tp = self.dl * (Sp * Sp - F(b ** 2, 4))
            Tm = self.dl * (Sm * Sm - F(b ** 2, 4))
            q0 = self.q0
            l0 = (Sp - Sm) / F(4 * a * q0) + (Tp + Tm) / F(4 * q0)
            l1 = (Tm - Tp) / F(4 * a)
            l2 = (Sm - Sp) / F(4 * a) + (Tp + Tm) / 4
            l3 = (Sp + Sm) / 2 + F(q0 - 2 * a ** 2) * (Tp - Tm) / F(4 * a)
            self.lat[tau] = (l0, l1, l2, l3)

    def H(self, r, x):
        return x ** 2 + 2 * a * r * x + self.q0

    def P(self, x):
        return (x ** 2 + self.q0) ** 2 - 4 * a ** 2 * x ** 2

    def E8(self, x):
        Ht, Htm, Hn, Hnm = self.H(self.rt, x), self.H(self.rt, -x), self.H(self.rn, x), self.H(self.rn, -x)
        lam = self.lam
        return (Ht * Htm * Hn * Hnm - lam ** 2 * (self.piS * (Ht * Hn + Htm * Hnm) +
                self.piP * (Ht * Hnm + Htm * Hn)) + lam ** 4 * self.dpi ** 2)

    def Q(self, x):  # Q_t
        lam = self.lam
        return (self.H(self.rt, -x) * self.H(self.rn, x) * self.H(self.rn, -x) -
                lam ** 2 * (self.piP * self.H(self.rn, -x) + self.piS * self.H(self.rn, x)))

    def RhatC(self, tau, m0, m1, m2, m3, x):
        """Params.RhatC on lattice tau; m's polynomials, x polynomial; returns F."""
        l0, l1, l2, l3 = self.lat[tau]
        Px = self.P(x)
        Mx = m0 + m1 * x + m2 * x ** 2 + m3 * x ** 3
        return (F(Px * (m1 + m2 * x + m3 * x ** 2) - Mx * (x ** 3 + self.kap * x)) * l0 +
                F(Px * (m2 + m3 * x) - Mx * (x ** 2 + self.kap)) * l1 +
                F(Px * m3 - Mx * x) * l2 - F(Mx) * l3)

    # N(z, x) for row sigma ('t' or 'n'): N_0^{-sigma}(x) + z^2 N_2^{-sigma}(x), split into z^0 and z^2 parts
    def Ncoef(self, row):
        r = self.rn if row == 't' else self.rt          # r of the other spin
        dz = self.dl * self.zeta
        q0, a2 = self.q0, a ** 2
        # cN0 = A0 + A2 z^2, cN1 = B0 + B2 z^2, cN2 = C0 + C2 z^2, cN3 = D
        A0 = 4 * a2 ** 2 * dz + 8 * a ** 3 * q0 - 2 * a2 * q0 * dz - 2 * a * q0 ** 2 - q0 ** 2 * dz
        A2 = -a2 * dz - 2 * a * q0
        B0 = -4 * a2 * r * (-4 * a2 + 2 * q0)
        B2 = -4 * a2 * r
        C0 = -(a2 * dz) - 2 * a * q0
        C2 = -2 * a + dz
        D = -4 * a2 * r
        return (A0, B0, C0, D), (A2, B2, C2, ZERO)

    def Nz(self, row, x):
        (A0, B0, C0, D), (A2, B2, C2, _) = self.Ncoef(row)
        return (A0 + B0 * x + C0 * x ** 2 + D * x ** 3, A2 + B2 * x + C2 * x ** 2)

    def RhatL(self, row, x):
        other = 'n' if row == 't' else 't'
        c0, c2 = self.Ncoef(row)
        return self.RhatC(other, *c0, x), self.RhatC(other, *c2, x)

    def RhatSL(self, row, x):
        return self.RhatC(row, -a ** 2, ZERO, ONE, ZERO, x)


if __name__ == '__main__':
    for k0 in (False, True):
        m = Model(k0)
        print('k0' if k0 else 'kw', 'E8 terms', len(m.E8(u)))
