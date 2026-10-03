# HubbardAtoms.jl

[![CI](https://github.com/tuwien-cms/HubbardAtoms.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tuwien-cms/HubbardAtoms.jl/actions/workflows/CI.yml)

Analytic one- and two-particle vertices for the Hubbard atom, taken from Phys. Rev. B 98, 235107 (2018) by Thunström et al.:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235107.

Available are the functions `bare_vertex`, `gf`, `chi`, `chi0`, `full_vertex`, `gamma`, `irreducible_vertex`, `channel_reducible_vertex`, `hedin`, `g3`.

Install it with
```julia
pkg> add HubbardAtoms
```

## Conventions
- Frequencies are `FermionicFreq(n)` (odd `n`) and `BosonicFreq(k)` (even `k`), i.e. `ν = nπ/β` and `ω = kπ/β`, from [SparseIR.jl](https://github.com/SpM-lab/SparseIR.jl); HubbardAtoms re-exports both types. Four-point quantities take `(ν, ν´, ω)`; for `SingletChannel` and `TripletChannel` this is the particle-particle notation of Eq. 4 of the paper.
- All quantities except the Green's function `gf` are real.
- `chi(r, atom, ω)` is `-2/β² Σ_νν´ chi(r, atom, (ν, ν´, ω))`, the physical susceptibility for the density, magnetic and singlet channels.
- `hedin` follows Krien, Valli, Capone, [Phys. Rev. B 100, 155149 (2019)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.100.155149): it tends to 1 (density, magnetic) and -1 (singlet). There is none in the triplet channel.

## Example
```julia
using HubbardAtoms

U = 2.0
beta = 10.0
at = HubbardAtom(U, beta)

w = (FermionicFreq(11), FermionicFreq(-3), BosonicFreq(8))

full_vertex(MagneticChannel(), at, w)
```
