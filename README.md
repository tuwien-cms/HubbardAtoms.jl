# HubbardAtoms.jl
Analytic one- and two-particle vertices for the Hubbard atom, taken from Phys. Rev. B 98, 235107 (2018) by Thunström et al.:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235107.

Available are the functions `bare_vertex`, `gf`, `chi`, `chi0`, `full_vertex`, `gamma`, `irreducible_vertex`, `channel_reducible_vertex`, `hedin`.

To use these, you need to import `SparseIR.jl` to be able to create `MatsubaraFreq` objects.

## Example
```julia
using HubbardAtoms, SparseIR

U = 2.0
beta = 10.0
at = HubbardAtom(U, beta)

w = (FermionicFreq(11), FermionicFreq(-3), BosonicFreq(8))

full_vertex(MagneticChannel(), at, w)
```
