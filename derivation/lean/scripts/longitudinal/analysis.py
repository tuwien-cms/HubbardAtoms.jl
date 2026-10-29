"""Check the longitudinal completion with the Lean-mirroring model; report sizes."""
import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from model import *


def column(m):
    """hats: coefficients divided by P(v)."""
    E8v = m.E8(v)
    lam, dpi = m.lam, m.dpi
    Ht, Hn = (lambda x: m.H(m.rt, x)), (lambda x: m.H(m.rn, x))
    dh = F(-m.Q(v), b * E8v)
    Ah = F(2 * lam ** 2 * m.Dt * (v ** 2 + m.q0), b ** 2 * E8v)
    Bh = F(-M * lam * (Ht(-v) * Hn(-v) + lam ** 2 * dpi), b ** 2 * E8v)
    Ch = F(-rh * lam * (Ht(-v) * Hn(v) - lam ** 2 * dpi), b ** 2 * E8v)
    return dh, Ah, Bh, Ch


def dense(m):
    """row images of the dense column: (identities that must vanish, polynomial coefficients per row)."""
    dh, Ah, Bh, Ch = column(m)
    lam, Pv = m.lam, m.P(v)
    cS, cP = F(m.cS), F(m.cP)
    sigt = F(b * m.ct, 2 * m.Dt)          # S_t(-v)
    sh = -cS / F(M)                       # S_n(v)
    sm = cP / F(rh)                       # S_n(-v)
    dz = m.dl
    Ht, Hn = (lambda x: m.H(m.rt, x)), (lambda x: m.H(m.rn, x))
    zt, zn, zeta = F(m.zt), F(m.zn), F(m.zeta)
    checks = {}
    # row t
    checks['t_delta'] = (-b * Ht(v) * Pv * dh + lam * Pv * (Bh * F(M) * (sh * sh - F(b ** 2, 4)) +
                         Ch * F(rh) * (sm * sm - F(b ** 2, 4))) - F(Pv))
    checks['t_pole_mv'] = (-b * Ht(-v) * Pv * Ah + lam * Pv * (Bh * (-F(rh) * cS / F(M) - cP) +
                           Ch * F(M) * (-cS / F(M) - cP / F(rh))))
    Wt = F(v ** 2 - a ** 2) * sigt + m.RhatSL('t', -v)
    N0p, N2p = m.Nz('t', v)
    N0m, N2m = m.Nz('t', -v)
    R0p, R2p = m.RhatL('t', v)
    R0m, R2m = m.RhatL('t', -v)
    t0 = (-a ** 2 * dz * zt * dh * (v ** 2 - a ** 2) - b * Ah * Pv * (-v + 2 * a * m.rt) -
          a ** 2 * dz * zt * Ah * Wt + Bh * (F(N0p) * sh + R0p) + Ch * (F(N0m) * sm + R0m) -
          F(a ** 2 * dz) * zt / b)
    t1 = -b * Ah * Pv
    t2 = (dz * zt * dh * (v ** 2 - a ** 2) + dz * zt * Ah * Wt + Bh * (F(N2p) * sh + R2p) +
          Ch * (F(N2m) * sm + R2m) + F(dz) * zt / b)
    # row n
    checks['n_pole_v'] = -b * Hn(v) * Pv * Bh + lam * Pv * (dh * F(M) + Ah * (F(rh) * sigt - cP))
    checks['n_pole_mv'] = -b * Hn(-v) * Pv * Ch + lam * Pv * (dh * F(rh) + Ah * (F(M) * sigt - cS))
    Bb = F(v ** 2 - a ** 2) * sh + m.RhatSL('n', v)
    Cb = F(v ** 2 - a ** 2) * sm + m.RhatSL('n', -v)
    n0p, n2p = m.Nz('n', v)
    n0m, n2m = m.Nz('n', -v)
    r0m, r2m = m.RhatL('n', -v)
    n0 = (-b * Pv * (Bh * (v + 2 * a * m.rn) + Ch * (-v + 2 * a * m.rn)) - a ** 2 * dz * zn * (Bh * Bb + Ch * Cb) +
          dh * F(n0p) + Ah * (F(n0m) * sigt + r0m) + F(-2 * a * m.q0 - a ** 2 * dz * m.zeta, b))
    n1 = -b * Pv * (Bh + Ch) + F(-4 * a ** 2 * m.rn, b)
    n2 = (dz * zn * (Bh * Bb + Ch * Cb) + dh * F(n2p) + Ah * (F(n2m) * sigt + r2m) +
          F(-2 * a + dz * m.zeta, b))
    return checks, {'t': (t0, t1, t2), 'n': (n0, n1, n2)}


def chain(m):
    """S_n(u), S_t(-u), S_n(-u) from s = S_t(u) by the relations P1, P3, P2."""
    cS, cP = F(m.cS), F(m.cP)
    aSn = F(M * s - m.cS)                  # alpha^S_n(u)
    aSt = F(b ** 2 * m.piS) / aSn          # alpha^S_t(u)
    s2 = -(aSt + cS) / F(M)                # S_n(u)
    aPn = F(rh * s - m.cP)                 # alpha^P_n(u)
    aPtm = F(b ** 2 * m.piP) / aPn         # alpha^P_t(-u)
    s4 = (aPtm + cP) / F(rh)               # S_n(-u)
    at = F(m.Dt * s - b * m.ct / 2)        # alpha_t(u)
    atm = F(b ** 2 * m.piP * m.piS) / at   # alpha_t(-u)
    s3 = (atm + F(b * m.ct, 2)) / F(m.Dt)  # S_t(-u)
    return s2, s3, s4


def mode(m):
    lam, dpi = m.lam, m.dpi
    Ht, Hn = (lambda x: m.H(m.rt, x)), (lambda x: m.H(m.rn, x))
    ftp = F(-b * m.Q(u))
    ftm = F(2 * lam ** 2 * (u ** 2 + m.q0) * (m.Dt * s - b * m.ct / 2))
    fnp = F(-lam * (Ht(-u) * Hn(-u) + lam ** 2 * dpi) * (M * s - m.cS))
    fnm = F(-lam * (Ht(-u) * Hn(u) - lam ** 2 * dpi) * (rh * s - m.cP))
    return ftp, ftm, fnp, fnm


def mode_image(m):
    s2, s3, s4 = chain(m)
    ftp, ftm, fnp, fnm = mode(m)
    lam, Pu, dz = m.lam, F(m.P(u)), m.dl
    Ht, Hn = (lambda x: m.H(m.rt, x)), (lambda x: m.H(m.rn, x))
    zt, zn = F(m.zt), F(m.zn)
    cS, cP = F(m.cS), F(m.cP)
    S = F(s)
    res = {}
    # row t: same block on L_t (f_t), opp block on L_n (f_n)
    aPt_u, aSt_u = F(rh) * s2 - cP, -F(M) * s2 - cS
    aPt_mu, aSt_mu = F(rh) * s4 - cP, -F(M) * s4 - cS
    res['t_u'] = -b * F(Ht(u)) * Pu * ftp + lam * Pu * (fnp * aSt_u + fnm * aPt_mu)
    res['t_mu'] = -b * F(Ht(-u)) * Pu * ftm + lam * Pu * (fnp * aPt_u + fnm * aSt_mu)
    Wp = F(u ** 2 - a ** 2) * S + m.RhatSL('t', u)
    Wm = F(u ** 2 - a ** 2) * s3 + m.RhatSL('t', -u)
    N0p, N2p = m.Nz('t', u)
    N0m, N2m = m.Nz('t', -u)
    R0p, R2p = m.RhatL('t', u)
    R0m, R2m = m.RhatL('t', -u)
    t0 = (-b * Pu * (ftp * F(u + 2 * a * m.rt) + ftm * F(-u + 2 * a * m.rt)) - a ** 2 * dz * zt * (ftp * Wp + ftm * Wm) +
          fnp * (F(N0p) * s2 + R0p) + fnm * (F(N0m) * s4 + R0m))
    t1 = -b * Pu * (ftp + ftm)
    t2 = dz * zt * (ftp * Wp + ftm * Wm) + fnp * (F(N2p) * s2 + R2p) + fnm * (F(N2m) * s4 + R2m)
    # row n: same block on L_n (f_n), opp block on L_t (f_t)
    aPn_u, aSn_u = F(rh) * S - cP, F(M) * S - cS
    aPn_mu, aSn_mu = F(rh) * s3 - cP, F(M) * s3 - cS
    res['n_u'] = -b * F(Hn(u)) * Pu * fnp + lam * Pu * (ftp * aSn_u + ftm * aPn_mu)
    res['n_mu'] = -b * F(Hn(-u)) * Pu * fnm + lam * Pu * (ftp * aPn_u + ftm * aSn_mu)
    wp = F(u ** 2 - a ** 2) * s2 + m.RhatSL('n', u)
    wm = F(u ** 2 - a ** 2) * s4 + m.RhatSL('n', -u)
    n0p, n2p = m.Nz('n', u)
    n0m, n2m = m.Nz('n', -u)
    r0p, r2p = m.RhatL('n', u)
    r0m, r2m = m.RhatL('n', -u)
    n0 = (-b * Pu * (fnp * F(u + 2 * a * m.rn) + fnm * F(-u + 2 * a * m.rn)) - a ** 2 * dz * zn * (fnp * wp + fnm * wm) +
          ftp * (F(n0p) * S + r0p) + ftm * (F(n0m) * s3 + r0m))
    n1 = -b * Pu * (fnp + fnm)
    n2 = dz * zn * (fnp * wp + fnm * wm) + ftp * (F(n2p) * S + r2p) + ftm * (F(n2m) * s3 + r2m)
    return res, {'t': (t0, t1, t2), 'n': (n0, n1, n2)}


def report(k0):
    m = Model(k0)
    label = 'k0' if k0 else 'kw'
    E8u, E8v = m.E8(u), m.E8(v)
    Eu = E8u / (u ** 2 - a ** 2) if k0 else E8u
    checks, D = dense(m)
    for name, c in checks.items():
        print(label, 'dense', name, 'OK' if c.is_zero() else 'FAIL')
    res, Mi = mode_image(m)
    for name, r in res.items():
        q, rr = divmod(r.n, Eu)
        print(label, 'residue', name, 'OK' if rr.is_zero() else 'FAIL', 'num terms', len(r.n), 'den', len(r.d))
    ftp, ftm, fnp, fnm = mode(m)
    K = {'t': ftp + ftm, 'n': fnp + fnm}
    out = {}
    for row in ('t', 'n'):
        for c in range(3):
            tot = D[row][c]
            n_, dn = tot.n, tot.d
            g = dn.gcd(E8v)
            cj = dn / g
            assert deg(cj, 'v') <= 0
            N = n_ * (E8v / g)
            assert N == subs(N, 'v', -v), (label, row, c, 'odd')
            assert deg(N, 'v') <= 6
            if k0:
                assert subs(N, 'v', a).is_zero(), (label, row, c, 'atomic')
            Mc = Mi[row][c]
            assert deg(Mc.n, 's') <= 1 and deg(Mc.d, 's') <= 0, (label, row, c, 'not affine')
            Nu = subs(N, 'v', u)
            ident = F(b ** 2 * Nu) * K['t'] - F(cj * 2 * m.lam ** 2 * m.Dt * (u ** 2 + m.q0)) * Mc
            q, rr = divmod(ident.n, Eu)
            print(label, 'row', row, 'coef', c, 'N terms', len(N), 'cj', cj, 'M num', len(Mc.n), 'M den', Mc.d,
                  'E2', 'OK' if rr.is_zero() else 'FAIL', len(ident.n), len(q))
            out[(row, c)] = (N, cj, Mc)
    return m, out


if __name__ == '__main__':
    for k0 in (False, True):
        report(k0)
