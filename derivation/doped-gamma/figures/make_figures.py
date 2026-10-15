# Figures of doped-gamma-derivation.typ. Run figure_data.jl first, then
# python make_figures.py   (needs NumPy and Matplotlib)
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

here = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
                     "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9, "svg.fonttype": "path"})

# ---------- Figure 1: the complex plane --------------------------------------------------------
U, beta, mu, k, eps = 2.4, 2.0, 0.4, 1, 1
a = U/2; w = 2j*np.pi*k/beta
E = np.array([U/4 + mu, -U/4, U/4 - mu]); wt = np.exp(-beta*(E - E.min())); wt /= wt[0] + 2*wt[1] + wt[2]
p0, p1, p2 = wt; rho = p2 - p0; q0 = a**2 - w**2/4; q1 = q0 + 4*eps*a**2*p1
s = np.roots([1, 2*(q1 - 2*a**2*rho**2), q1**2 - 16*a**4*p0*p2]); zDelta = np.concatenate([np.sqrt(s), -np.sqrt(s)])
n = np.arange(-3, 3); zn = mu + 1j*np.pi*(2*n + 1 + k)/beta
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.axvline(mu, color="0.85", lw=0.8); ax.axvline(-mu, color="0.85", lw=0.8); ax.axhline(0, color="0.92", lw=0.6)
ax.plot(zn.real, zn.imag, "o", ms=4.5, color="k", label=r"lattice $z_n$ (Re $z = \mu$)")
ax.plot(-zn.real, -zn.imag, "o", ms=4.5, mfc="none", color="k", label=r"reflected lattice $-z_n$")
ax.plot([-a, -a], [(w/2).imag, -(w/2).imag], "x", color="C3", ms=6, mew=1.5, label=r"zeros of $Q_+$")
ax.plot([a, a], [(w/2).imag, -(w/2).imag], "+", color="C0", ms=8, mew=1.5, label=r"zeros of $Q_-$")
ax.plot(zDelta.real, zDelta.imag, "D", color="C2", ms=4, label=r"zeros of $\Delta$ ($d$ channel)")
zi, zj = mu + 1j*np.pi*(2*0 + 1 + k)/beta, mu + 1j*np.pi*(2*(-0 - k - 1) + 1 + k)/beta   # n = 0 and its mirror
ax.annotate("", xy=(zj.real + 0.12, zj.imag), xytext=(zi.real + 0.12, zi.imag),
            arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.8, connectionstyle="arc3,rad=-0.4"))
ax.text(zi.real + 1.0, 0.0, r"$n \leftrightarrow \bar n$", fontsize=8, color="0.3", va="center")
ax.set_xlabel(r"Re $z$"); ax.set_ylabel(r"Im $z$")
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-8.5, 8.5)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.5, frameon=False, handletextpad=0.3)
fig.tight_layout(); fig.savefig(os.path.join(here, "complex_plane.svg")); plt.close(fig)

# ---------- Figure 2: structure of chi_d --------------------------------------------------------
rows = list(csv.DictReader(open(os.path.join(here, "chi_structure.csv"))))
def mat(mu, col):
    M = np.zeros((40, 40))
    for r in rows:
        if float(r["mu"]) == mu:
            M[int(r["n"]) + 20, int(r["m"]) + 20] = float(r[col])
    return M
panels = [(0.0, "cauchy", r"(a) Cauchy part, $\mu = 0$"), (0.3, "cauchy", r"(b) Cauchy part, $\mu = 0.3$"),
          (0.3, "rest", r"(c) finite-rank part, $\mu = 0.3$"), (0.3, "total", r"(d) $\chi_\mathrm{d}$, $\mu = 0.3$")]
fig, axs = plt.subplots(2, 2, figsize=(4.4, 4.5), constrained_layout=True); axs = axs.ravel()
norm = LogNorm(vmin=1e-7, vmax=1e0)
for ax, (m_, col, title) in zip(axs, panels):
    M = mat(m_, col); M[M < 1e-12] = 1e-12
    im = ax.imshow(M, origin="upper", extent=(-20.5, 19.5, 19.5, -20.5), norm=norm, cmap="magma_r")
    ax.set_title(title, pad=3); ax.set_xticks([-20, 0, 19]); ax.set_yticks([-20, 0, 19])
    ax.set_xlabel(r"$m$", labelpad=0); ax.set_ylabel(r"$n$", labelpad=0)
cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02); cb.set_label("modulus")
fig.savefig(os.path.join(here, "chi_structure.svg"), bbox_inches="tight"); plt.close(fig)

# ---------- Figure 3: Gamma_r cuts ---------------------------------------------------------------
rows = list(csv.DictReader(open(os.path.join(here, "gamma_cuts.csv"))))
fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.2), sharex=True)
for ax, ch in zip(axs, "dmst"):
    for c, mu in zip(("k", "C0", "C3"), (0.0, 0.4, 1.0)):
        rr = [r for r in rows if r["channel"] == ch and float(r["mu"]) == mu]
        nu = np.array([(2*int(r["n"]) + 1)*np.pi/4.0 for r in rr])
        re = np.array([float(r["re"]) for r in rr]); imv = np.array([float(r["im"]) for r in rr])
        ax.plot(nu, re, "-", color=c, lw=1, marker=".", ms=3, label=rf"Re, $\mu={mu:g}$")
        if mu != 0.0:
            ax.plot(nu, imv, "--", color=c, lw=1, label=rf"Im, $\mu={mu:g}$")
    ax.set_title(rf"$\Gamma_{{\mathrm{{{ch}}}}}(\nu, \pi/\beta, 0)$", pad=3); ax.set_xlabel(r"$\nu$", labelpad=0)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(here, "gamma_cuts.svg")); plt.close(fig)
print("figures written")
