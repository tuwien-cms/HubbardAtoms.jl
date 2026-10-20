"""
Exact one- and two-particle quantities of the Hubbard atom in Matsubara frequencies: at half
filling (`HubbardAtom`), following Thunström et al., Phys. Rev. B 98, 235107 (2018),
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235107, whose equation numbers the
docstrings refer to, and at arbitrary filling (`DopedHubbardAtom`), with the closed forms
derived in `derivation/doped-gamma-derivation.pdf` of the package repository. With doping
and Zeeman field (`FieldHubbardAtom`), spin-resolved Green's functions and sector χ, χ₀, Γ
follow the companion's field, fieldT and fieldL sections (see the README for source links).
"""
module HubbardAtoms

using SparseIR: FermionicFreq, BosonicFreq

# re-exported from SparseIR to construct frequency arguments
export FermionicFreq, BosonicFreq
export FieldHubbardAtom, LongitudinalSector, TransverseSector, OppositeSpinPairSector,
       EqualSpinPairSector
export HubbardAtom, DopedHubbardAtom, CHANNELS, FermiBose, FermiFermiBose,
       DensityChannel, MagneticChannel, SingletChannel, TripletChannel,
       bare_vertex, gf, chi, chi0, full_vertex, gamma, irreducible_vertex,
       channel_reducible_vertex, hedin, g3
public FieldSector, AbstractHubbardAtom, SpinChannel, PHChannel, PPChannel, G, χ, χ₀, F, Γ,
       Λ, Φ, G₃

include("channels.jl")
include("interface.jl")
include("half_filling.jl")
include("doped.jl")
include("field.jl")

end
