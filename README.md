# HubbardAtoms.jl

[![CI](https://github.com/tuwien-cms/HubbardAtoms.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tuwien-cms/HubbardAtoms.jl/actions/workflows/CI.yml)

Exact one- and two-particle quantities of the Hubbard atom in Matsubara frequencies, in closed
form:

- `HubbardAtom(U, beta)`: the half-filled atom, following Thunström et al.,
  [Phys. Rev. B 98, 235107 (2018)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235107).
- `DopedHubbardAtom(U, beta, mu)`: the atom at arbitrary filling, with the closed forms derived in
  [`derivation/`](derivation/README.md).

Both are evaluated accurately at low temperatures, at large frequencies, and at the removable
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

All functions take a channel (`DensityChannel()`, `MagneticChannel()`, `SingletChannel()`,
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

## Citing

For the half-filled atom, please cite Thunström et al. (above). For the atom away from half
filling, please cite the derivation in
[`derivation/doped-gamma-derivation.pdf`](derivation/doped-gamma-derivation.pdf) and this package
(see [`CITATION.cff`](CITATION.cff), or "Cite this repository" on GitHub).

## License

The package is MIT-licensed (`LICENSE`). The derivation note and its figures are licensed under
CC BY 4.0 (`derivation/LICENSE`).
