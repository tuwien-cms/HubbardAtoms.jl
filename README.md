# HubbardAtoms.jl

[![CI](https://github.com/tuwien-cms/HubbardAtoms.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tuwien-cms/HubbardAtoms.jl/actions/workflows/CI.yml)

HubbardAtoms evaluates the one- and two-particle functions of the Hubbard atom

```math
H = U\left(n_\uparrow - \tfrac{1}{2}\right)\left(n_\downarrow - \tfrac{1}{2}\right)
  - \mu\,(n_\uparrow + n_\downarrow - 1) - h\,(n_\uparrow - n_\downarrow)
```

in Matsubara frequencies, from exact closed forms: the Green's function, the generalized
susceptibilities, and the full, channel-irreducible and fully irreducible vertices. The chemical
potential $`\mu`$ is measured from half filling, and $`2h`$ is the Zeeman splitting.

| Atom | Parameters |
|---|---|
| `HubbardAtom(U, beta)` | half filling, $`\mu = h = 0`$ |
| `DopedHubbardAtom(U, beta, mu)` | any filling, $`h = 0`$ |
| `FieldHubbardAtom(U, beta, mu, h)` | any filling and field |

The parameters are promoted to one floating-point type, in which all results are computed:
`HubbardAtom(big"2.0", big"10.0")` works in `BigFloat`.

## Installation

```julia
pkg> add HubbardAtoms
```

HubbardAtoms requires Julia 1.11 or later.

## Usage

```julia
using HubbardAtoms

atom = DopedHubbardAtom(2.0, 10.0, 0.3)       # U, β, μ
ν, ν′, ω = FermionicFreq(1), FermionicFreq(-3), BosonicFreq(2)

gf(atom, ν)                                   # G(ν)
chi(DensityChannel(), atom, (ν, ν′, ω))       # χ_d(ν, ν′, ω)
gamma(MagneticChannel(), atom, (ν, ν′, ω))    # Γ_m(ν, ν′, ω)
full_vertex(SingletChannel(), atom, (ν, ν′, ω))
```

`FermionicFreq(n)` with odd `n` is the frequency $`\nu = n\pi/\beta`$, and `BosonicFreq(k)` with
even `k` is $`\omega = k\pi/\beta`$; both come from [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl)
and are re-exported. Four-point functions take `(ν, ν′, ω)`, three-point functions `(ν, ω)`.
The results are complex, except for `HubbardAtom`, whose functions other than `gf` are real.
A vertex on a frequency box is a comprehension:

```julia
νs = FermionicFreq.(-39:2:39)
Γd = [gamma(DensityChannel(), atom, (ν, ν′, BosonicFreq(0))) for ν in νs, ν′ in νs]
```

### Functions

For `HubbardAtom` and `DopedHubbardAtom`, the functions take a channel `r`: `DensityChannel()`,
`MagneticChannel()`, `SingletChannel()` or `TripletChannel()`, all four in `CHANNELS`.

| Function | Quantity |
|---|---|
| `gf(atom, ν)` | Green's function $`G(\nu)`$ |
| `chi(r, atom, (ν, ν′, ω))` | generalized susceptibility $`\chi_r(\nu, \nu', \omega)`$ |
| `chi(r, atom, ω)` | susceptibility $`\chi_r(\omega)`$ |
| `chi0(r, atom, (ν, ν′, ω))`, `chi0(r, atom, (ν, ω))` | bubble $`\chi_{0,r}`$ |
| `full_vertex(r, atom, (ν, ν′, ω))` | full vertex $`F_r`$ |
| `gamma(r, atom, (ν, ν′, ω))` | channel-irreducible vertex $`\Gamma_r`$ |
| `irreducible_vertex(r, atom, (ν, ν′, ω))` | fully irreducible vertex $`\Lambda_r`$ |
| `channel_reducible_vertex(r, atom, (ν, ν′, ω))` | channel-reducible vertex $`\Phi_r = F_r - \Gamma_r`$ |
| `g3(r, atom, (ν, ω))` | three-point Green's function $`G_{3,r}`$ |
| `hedin(r, atom, (ν, ω))` | Hedin vertex $`\lambda_r`$ |
| `bare_vertex(r, atom)` | bare vertex $`U_r`$: $`U`$, $`-U`$, $`2U`$, $`0`$ for d, m, s, t |

`G`, `χ`, `χ₀`, `F`, `Γ`, `Λ`, `Φ` and `G₃` are public Unicode aliases, e.g. `HubbardAtoms.Γ`.
The docstrings give the details and the accuracy of each function.

### Definitions

The Green's function and the generalized susceptibility are

```math
G(\nu) = -\int_0^\beta d\tau\, e^{i\nu\tau}\,\langle T c_\uparrow(\tau)\, c_\uparrow^\dagger\rangle,
```

```math
\chi_{\sigma\sigma'}(\nu, \nu', \omega) = \int_0^\beta d\tau_1\, d\tau_2\, d\tau_3\;
e^{-i\nu\tau_1 + i(\nu+\omega)\tau_2 - i(\nu'+\omega)\tau_3}
\Big[\langle T c_\sigma^\dagger(\tau_1)\, c_\sigma(\tau_2)\, c_{\sigma'}^\dagger(\tau_3)\, c_{\sigma'}\rangle
- \langle T c_\sigma^\dagger(\tau_1)\, c_\sigma(\tau_2)\rangle\,\langle T c_{\sigma'}^\dagger(\tau_3)\, c_{\sigma'}\rangle\Big],
```

where operators without a time argument stand at $`\tau = 0`$. The particle-hole channels are
$`\chi_{d,m} = \chi_{\uparrow\uparrow} \pm \chi_{\uparrow\downarrow}`$. The particle-particle channels
are written in particle-particle notation, with
$`\chi^{pp}_{\sigma\sigma'}(\nu, \nu', \omega) = \chi_{\sigma\sigma'}(\nu, \nu', -\omega-\nu-\nu')`$:

```math
\chi_s = \tfrac{1}{4}\left(-\chi^{pp}_{\uparrow\uparrow} + 2\chi^{pp}_{\uparrow\downarrow} - 2\chi_{0,s}\right),
\qquad
\chi_t = \tfrac{1}{4}\left(\chi^{pp}_{\uparrow\uparrow} + 2\chi_{0,t}\right).
```

The bubbles are

```math
\chi_{0,r}(\nu, \nu', \omega) = \beta\,\delta_{\nu\nu'}\,\chi_{0,r}(\nu, \omega),
\qquad
\chi_{0,r}(\nu, \omega) =
\begin{cases}
-G(\nu)\,G(\nu+\omega) & (r = d, m),\\
-\tfrac{1}{2}\,G(\nu)\,G(-\nu-\omega) & (r = s, t).
\end{cases}
```

With $`\eta_s = -1`$ and $`\eta_d = \eta_m = \eta_t = 1`$, the full and the channel-irreducible vertex
are defined by

```math
\chi_r(\nu, \nu', \omega) = \eta_r\,\chi_{0,r}(\nu, \nu', \omega)
- \chi_{0,r}(\nu, \omega)\,F_r(\nu, \nu', \omega)\,\chi_{0,r}(\nu', \omega),
\qquad
\Gamma_r = \beta^2\left(\chi_r^{-1} - \eta_r\,\chi_{0,r}^{-1}\right),
```

where the inverses are of matrices in $`\nu`$ and $`\nu'`$ at fixed $`\omega`$. The fully irreducible
vertex $`\Lambda_r`$ follows from the parquet equation, Eq. (26) of Thunström et al.,
[Phys. Rev. B 98, 235107 (2018)](https://doi.org/10.1103/PhysRevB.98.235107). The susceptibility,
the three-point Green's function and the Hedin vertex are

```math
\chi_r(\omega) = -\frac{2}{\beta^2}\sum_{\nu\nu'}\chi_r(\nu, \nu', \omega),
\qquad
G_{3,r}(\nu, \omega) = -\frac{1}{\beta}\sum_{\nu'}\chi_r(\nu, \nu', \omega),
\qquad
\lambda_r(\nu, \omega) = -\frac{G_{3,r}(\nu, \omega)}{\chi_{0,r}(\nu, \omega)\,\big[1 + U_r\,\chi_r(\omega)/2\big]}.
```

For $`r = d, m`$, $`\chi_r(\omega)`$ is the charge and spin susceptibility
$`-\langle\rho;\rho\rangle(\omega)`$ with $`\rho = n_\uparrow \pm n_\downarrow`$. The Hedin vertex follows
Krien, Valli and Capone, [Phys. Rev. B 100, 155149 (2019)](https://doi.org/10.1103/PhysRevB.100.155149):
at large frequencies it tends to $`1`$ for d and m and to $`-1`$ for s; the triplet channel has none. Codes with the
opposite sign in the Fourier exponents obtain the same quantities at $`(-\nu, -\nu', -\omega)`$.

## Doping and Zeeman field

A field breaks spin rotation, so the four channels no longer decouple. `FieldHubbardAtom` instead
provides `gf`, `chi`, `chi0` and `gamma` in four spin-resolved sectors; spins are `+1` (↑) and `-1` (↓).

```julia
atom = FieldHubbardAtom(2.0, 10.0, 0.3, 0.2)        # U, β, μ, h
w = (FermionicFreq(1), FermionicFreq(-1), BosonicFreq(0))

gf(atom, +1, w[1])                                  # G↑(ν)
gamma(LongitudinalSector(+1, -1), atom, w)          # Γ_L↑↓
gamma(TransverseSector(+1), atom, w)                # Γ_T↑
chi(OppositeSpinPairSector(-1), atom, w)            # χ_A↓
chi0(EqualSpinPairSector(+1), atom, (w[1], w[3]))   # bubble of E↑
```

| Sector | Susceptibility |
|---|---|
| `LongitudinalSector(σ, τ)` | $`\chi_{\sigma\tau}(\nu, \nu', \omega)`$ as above, a matrix in spin and frequency |
| `TransverseSector(σ)` | the same transform of $`\langle T c_\sigma^\dagger(\tau_1)\, c_{-\sigma}(\tau_2)\, c_{-\sigma}^\dagger(\tau_3)\, c_\sigma\rangle`$, which has no disconnected part |
| `OppositeSpinPairSector(σ)` | $`\chi_{A\sigma}(\nu, \nu', \omega) = -\chi_{T\sigma}(\nu, \nu', -\omega-\nu-\nu')`$ |
| `EqualSpinPairSector(σ)` | $`-\tfrac{1}{2}`$ times the same-spin correlator (without subtraction) at transfer $`-\omega-\nu-\nu'`$ |

The bubbles `chi0(sector, atom, (ν, ω))` are

```math
L_{\sigma\tau}:\ -\delta_{\sigma\tau}\,G_\sigma(\nu)\,G_\sigma(\nu+\omega),
\qquad
T_\sigma:\ -G_\sigma(\nu)\,G_{-\sigma}(\nu+\omega),
\qquad
A_\sigma:\ G_\sigma(\nu)\,G_{-\sigma}(-\nu-\omega),
\qquad
E_\sigma:\ \tfrac{1}{2}\,G_\sigma(\nu)\,G_\sigma(-\nu-\omega),
```

and `chi0(sector, atom, (ν, ν′, ω))` multiplies them by $`\beta\,\delta_{\nu\nu'}`$, for $`E_\sigma`$ by
$`\beta\,(\delta_{\nu\nu'} - \delta_{\nu',-\nu-\omega})`$. In every sector
$`\Gamma = \beta^2(\chi^{-1} - \chi_0^{-1})`$. The longitudinal inverse is taken in spin and
frequency together, not separately in the charge and spin blocks; the equal-spin inverse is taken on
the functions odd under $`\nu \to -\nu-\omega`$, and the vertex vanishes on the even ones.

At $`h = 0`$ the sectors reduce to the channels of `DopedHubbardAtom(U, beta, mu)`,

```math
\Gamma_{L,\sigma\sigma} = \tfrac{1}{2}(\Gamma_d + \Gamma_m),
\qquad
\Gamma_{L,\sigma,-\sigma} = \tfrac{1}{2}(\Gamma_d - \Gamma_m),
\qquad
\Gamma_{T\sigma} = \Gamma_m,
\qquad
\Gamma_{A\sigma} = \tfrac{1}{2}(\Gamma_s - \Gamma_t),
\qquad
\Gamma_{E\sigma} = -\tfrac{1}{2}\Gamma_t,
```

and the particle-hole transformation of the down spin maps $`(U, \mu, h)`$ to $`(-U, h, \mu)`$,
exchanging the transverse and the opposite-spin pair sector.

The longitudinal closed form holds for $`\mu h (\mu^2 - h^2) \neq 0`$. At $`h = 0`$ and $`\mu = 0`$
the package uses the exact zero-field and particle-hole identities above; on the lines
$`\mu = \pm h`$ it uses the continuous extension of the closed form, which agrees with numerical
inversion but is not derived there. At removable singularities of the longitudinal closed form, where two
of its roots coincide or a root lies on the Matsubara lattice, the vertex is evaluated as the
symmetric limit in $`\mu`$ in extended precision. Coincident transverse roots are evaluated by divided
differences; transverse lattice roots and genuine divergences are not regularized.

## Accuracy

The zero-field atoms are accurate at low temperatures, at large frequencies and at the removable
singularities of the closed forms, in any floating-point type. `FieldHubbardAtom` computes the coefficients of the
longitudinal closed form in 256-bit arithmetic. The transverse and opposite-spin pair sectors combine
the root pair by divided differences and use adaptive extended precision for remaining thermal
cancellations, with precision chosen from their measured conditioning. For extreme parameters,
check results with `BigFloat` atoms at higher precision.

## Derivation

[`derivation/`](derivation/README.md) derives the closed forms of `DopedHubbardAtom` and checks
them symbolically, numerically and, for the central theorems, in the Lean 4 proof assistant.

## Citing

For the half-filled atom, please cite Thunström et al. (above). For the atom away from half filling,
please cite the derivation [`derivation/doped-gamma-derivation.pdf`](derivation/doped-gamma-derivation.pdf)
and this package, and for the atom with a field this package (see [`CITATION.cff`](CITATION.cff),
or "Cite this repository" on GitHub).

## License

The package is MIT-licensed ([`LICENSE`](LICENSE)). The derivation note and its figures are
licensed under CC BY 4.0 ([`derivation/LICENSE`](derivation/LICENSE)).
