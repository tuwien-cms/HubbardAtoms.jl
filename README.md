# HubbardAtoms.jl

[![CI](https://github.com/tuwien-cms/HubbardAtoms.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tuwien-cms/HubbardAtoms.jl/actions/workflows/CI.yml)

Exact one- and two-particle quantities of the Hubbard atom in Matsubara frequencies, in closed
form:

- `HubbardAtom(U, beta)`: the half-filled atom, following Thunström et al.,
  [Phys. Rev. B 98, 235107 (2018)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235107).
- `DopedHubbardAtom(U, beta, mu)`: the atom at arbitrary filling, with the closed forms derived in
  [`derivation/`](derivation/README.md).
- `FieldHubbardAtom(U, beta, mu, h)`: the atom with doping and Zeeman field, with spin-resolved
  Green's functions, sector susceptibilities, bubbles and irreducible vertices (see below).

The two zero-field atoms are evaluated accurately at low temperatures, at large frequencies, and at the removable
singularities of the closed forms, in any floating-point type (e.g. `BigFloat`).

## Installation

```julia
pkg> add HubbardAtoms
```

HubbardAtoms requires Julia 1.11 or later.

## Example

```julia
using HubbardAtoms

U, beta = 2.0, 10.0
w = (FermionicFreq(11), FermionicFreq(-3), BosonicFreq(8))   # (ν, ν′, ω)

full_vertex(MagneticChannel(), HubbardAtom(U, beta), w)
gamma(DensityChannel(), DopedHubbardAtom(U, beta, 0.3), w)
```

## Functions

For the zero-field atoms, all functions take a channel (`DensityChannel()`, `MagneticChannel()`, `SingletChannel()`,
`TripletChannel()`, or one of `CHANNELS`) and an atom, except `gf`. The names in parentheses are
the unexported Unicode aliases used in the paper.

| Function | Quantity |
|---|---|
| `gf(atom, ν)` (`G`) | Green's function |
| `chi(r, atom, (ν, ν′, ω))` (`χ`) | generalized susceptibility |
| `chi(r, atom, ω)` | susceptibility |
| `chi0(r, atom, (ν, ν′, ω))`, `chi0(r, atom, (ν, ω))` (`χ₀`) | bare susceptibility |
| `full_vertex` (`F`) | full vertex |
| `gamma` (`Γ`) | channel-irreducible vertex |
| `irreducible_vertex` (`Λ`) | fully irreducible vertex |
| `channel_reducible_vertex` (`Φ`) | channel-reducible vertex |
| `g3(r, atom, (ν, ω))` (`G₃`) | three-point Green's function |
| `hedin(r, atom, (ν, ω))` | Hedin (three-point) vertex |
| `bare_vertex(r, atom)` | bare vertex of the channel |

See the docstrings for definitions and accuracy.

## Conventions

- Frequencies are `FermionicFreq(n)` (odd `n`) and `BosonicFreq(k)` (even `k`), i.e. `ν = nπ/β`
  and `ω = kπ/β`, from [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl); HubbardAtoms
  re-exports both. For `SingletChannel` and `TripletChannel`, `(ν, ν′, ω)` is the
  particle-particle notation of Eq. 4 of Thunström et al.
- Fourier transforms follow Thunström et al. Codes with the opposite sign convention in the
  exponent obtain their vertices at `(-ν, -ν′, -ω)`; away from half filling these differ (by
  complex conjugation).
- `DopedHubbardAtom` has `H = U(n↑ - 1/2)(n↓ - 1/2) - mu(n↑ + n↓ - 1)`: `mu` is measured from half
  filling, and `DopedHubbardAtom(U, beta, 0)` reproduces `HubbardAtom(U, beta)`.
- For `HubbardAtom`, all quantities except `gf` are real; for `DopedHubbardAtom` they are complex.
- `chi(r, atom, ω)` is `-2/β² Σ_νν′ chi(r, atom, (ν, ν′, ω))`, the physical susceptibility in the
  density and magnetic channels.
- `hedin` follows Krien, Valli and Capone,
  [Phys. Rev. B 100, 155149 (2019)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.100.155149):
  it tends to 1 (density, magnetic) and -1 (singlet). There is none in the triplet channel.

## Doping and Zeeman field

```julia
atom = FieldHubbardAtom(1.3, 1.1, 0.3, 0.2)
w = (FermionicFreq(1), FermionicFreq(-1), BosonicFreq(0))
gf(atom, +1, w[1])                         # G↑; spins are +1 (↑), -1 (↓)
gamma(LongitudinalSector(+1, -1), atom, w)  # Γ_L[↑,↓]
gamma(TransverseSector(+1), atom, w)        # Γ_T↑
gamma(OppositeSpinPairSector(-1), atom, w)  # Γ_A↓
gamma(EqualSpinPairSector(+1), atom, w)     # Γ_E↑
chi(TransverseSector(+1), atom, w)
chi0(TransverseSector(+1), atom, (w[1], w[3])) # bubble without β or frequency delta
```

The Hamiltonian is `U(n↑-1/2)(n↓-1/2) - mu(n↑+n↓-1) - h(n↑-n↓)`; the Zeeman splitting
is **2h**. A separate atom and spin-labelled sector types preserve the existing d/m/s/t API
and avoid applying its SU(2) combinations to a spin-asymmetric problem. They reuse the frequency
types and `gf`/`chi`/`chi0`/`gamma` names; other quantities in the table above are not implemented
for the field atom. All four functions support `BigFloat` and return complex results.

The sectors follow the companion's
[`field.tex`, `fieldT.tex`, `fieldL.tex`](https://github.com/SamuelBadr/doped-atom-vertex-paper/tree/main/companion/sections)
(the implementation ports its `code/checks/fieldT/FieldT.jl` and `fieldL/FieldL.jl`):

- **L** is the connected ph matrix in the spin basis. It couples both spins and frequencies:
  **do not invert charge and longitudinal-spin diagonal blocks separately**.
- **Tσ** is the full ph Fourier transform of `⟨T cσ† c₋σ c₋σ† cσ⟩`.
- **Aσ** is the oriented opposite-spin pair, `χ_Aσ(ν,ν′,ω) = -χ_Tσ(ν,ν′,-ω-ν-ν′)`.
  Its orientations are related by reflecting both fermionic frequencies, `ν → -ν-ω`.
- **Eσ** is -1/2 times the full (not connected) same-spin ph correlator in pp notation.
  The inverse is restricted to the mirror-odd Pauli subspace, then extended by zero.

In every field sector, `Γ = β²(χ⁻¹ - χ₀⁻¹)` with **η = +1**. For a frequency box form
`X[i,j] = chi(sector, atom, (ν[i],ν[j],ω))` and likewise `B` from `chi0`, then
`beta^2 * (inv(X) - inv(B))` approaches Γ as the box grows. For L, include spin in both
matrix indices; for E, project both matrices onto the mirror-odd subspace before inversion.
The two-frequency `chi0` omits β; the three-frequency version includes β and the frequency
delta (delta minus mirror for E). Use reflection-invariant boxes for pairing sectors.

At **h = 0**, with all right-hand sides evaluated at `DopedHubbardAtom(U,beta,mu)`,

```text
Γ_Lσσ = (Γ_d + Γ_m)/2       Γ_Lσ,-σ = (Γ_d - Γ_m)/2
Γ_Tσ  = Γ_m                Γ_Aσ = (Γ_s - Γ_t)/2       Γ_Eσ = -Γ_t/2.
```

The susceptibility relations are `χ_Lσσ = (χ_d+χ_m)/2`, `χ_Lσ,-σ = (χ_d-χ_m)/2`,
`χ_Tσ = χ_m`, `χ_Aσ = 2(χ_s-χ_t+B_pp)`, and `χ_Eσ = -2χ_t+2B_pp M₊`, where
`B_pp` is the zero-field pp bubble matrix and `M₊` the mirror-even projector.
At **mu = 0**, the source is `DopedHubbardAtom(-U,beta,h)` for ↑ (replace h by -h for
↓ in T/A/E): `Γ_T = (Γ_s-Γ_t)/2`, `Γ_A = Γ_m`, `Γ_E = -Γ_t/2`. For L, reflect each
down-spin fermionic frequency and multiply the spin-basis source vertex by `sigma*tau`.
This is the general Shiba map `(U,mu,h) → (-U,h,mu)`.

### Validity and numerical limits

The longitudinal theorem is established for **μh(μ²-h²) ≠ 0**, simple nonzero non-atomic
roots off the Matsubara lattice, and nonzero thermal denominators. At h = 0 and μ = 0 the
implementation uses the established zero-field and Shiba identities. On **μ = ±h**, it uses
the reference's exact Cauchy-kernel limits: this is a continuous extension, checked numerically
but not proved on those lines. The static atomic root has zero coefficient and is omitted.

As in the reference, near longitudinal confluences or lattice roots the code averages μ±δ in
extended precision. This prescription is meaningful only for removable singularities, **not
at genuine divergences**; do not interpret such an average as a regularized vertex. Vanishing
thermal denominators and equal-spin susceptibility zeros are not regularized. The generic
transverse/pair formula does not continue confluent or lattice roots. Near its small-density
axis it switches to 256-bit arithmetic; longitudinal setups use 256 bits before rounding to
the requested type. For difficult conditioning, low temperatures or extreme frequencies,
check convergence with increased `BigFloat` precision; the field formulas do not inherit the
zero-field implementation's full range/accuracy guarantees.

## Citing

For the half-filled atom, please cite Thunström et al. (above). For the atom away from half
filling, please cite the derivation in
[`derivation/doped-gamma-derivation.pdf`](derivation/doped-gamma-derivation.pdf) and this package
(see [`CITATION.cff`](CITATION.cff), or "Cite this repository" on GitHub).

## License

The package is MIT-licensed (`LICENSE`). The derivation note and its figures are licensed under
CC BY 4.0 (`derivation/LICENSE`).
