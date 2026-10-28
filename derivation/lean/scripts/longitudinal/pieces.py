import sys
sys.path.insert(0, '.')
from analysis import *

def mode_pieces(m):
    s2, s3, s4 = chain(m)
    ftp, ftm, fnp, fnm = mode(m)
    Pu, dz = F(m.P(u)), m.dl
    zt, zn = F(m.zt), F(m.zn)
    S = F(s)
    out = {}
    Wp = F(u ** 2 - a ** 2) * S + m.RhatSL('t', u)
    Wm = F(u ** 2 - a ** 2) * s3 + m.RhatSL('t', -u)
    N0p, N2p = m.Nz('t', u); N0m, N2m = m.Nz('t', -u)
    R0p, R2p = m.RhatL('t', u); R0m, R2m = m.RhatL('t', -u)
    out['t'] = {
        0: [-b * Pu * (ftp * F(u + 2 * a * m.rt) + ftm * F(-u + 2 * a * m.rt)),
            -a ** 2 * dz * zt * (ftp * Wp + ftm * Wm),
            fnp * (F(N0p) * s2 + R0p), fnm * (F(N0m) * s4 + R0m)],
        2: [dz * zt * (ftp * Wp + ftm * Wm), fnp * (F(N2p) * s2 + R2p), fnm * (F(N2m) * s4 + R2m)]}
    wp = F(u ** 2 - a ** 2) * s2 + m.RhatSL('n', u)
    wm = F(u ** 2 - a ** 2) * s4 + m.RhatSL('n', -u)
    n0p, n2p = m.Nz('n', u); n0m, n2m = m.Nz('n', -u)
    r0p, r2p = m.RhatL('n', u); r0m, r2m = m.RhatL('n', -u)
    out['n'] = {
        0: [-b * Pu * (fnp * F(u + 2 * a * m.rn) + fnm * F(-u + 2 * a * m.rn)),
            -a ** 2 * dz * zn * (fnp * wp + fnm * wm),
            ftp * (F(n0p) * S + r0p), ftm * (F(n0m) * s3 + r0m)],
        2: [dz * zn * (fnp * wp + fnm * wm), ftp * (F(n2p) * S + r2p), ftm * (F(n2m) * s3 + r2m)]}
    return out

if __name__ == '__main__':
    for k0 in (True, False):
        m = Model(k0)
        P = mode_pieces(m)
        for row in ('t', 'n'):
            for c in (0, 2):
                print('k0' if k0 else 'kw', row, c, [(len(x.n), str(x.d)[:60]) for x in P[row][c]])
