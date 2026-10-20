"""
Supertype of `HubbardAtom` (half filling), `DopedHubbardAtom` (arbitrary filling), and
`FieldHubbardAtom` (doping and Zeeman field). For `HubbardAtom{T}` all quantities except
`gf` are real (of type `T`); for the other atoms they are `Complex{T}`. The field atom
supports only spin-resolved `gf` and sector `chi`, `chi0`, `gamma`, not the SU(2) channel API.
"""
abstract type AbstractHubbardAtom end

Base.Broadcast.broadcastable(atom::AbstractHubbardAtom) = Ref(atom)

"Frequency argument `(ν, ω)` of three-point quantities."
const FermiBose = Tuple{FermionicFreq,BosonicFreq}
"Frequency argument `(ν, ν′, ω)` of four-point quantities."
const FermiFermiBose = Tuple{FermionicFreq,FermionicFreq,BosonicFreq}

# The functions of the package, each with the Unicode name of the paper and an exported
# ASCII alias. The methods for the two atoms are in half_filling.jl and doped.jl; those that
# hold for any atom are here.

"""
    bare_vertex(r::SpinChannel, atom::AbstractHubbardAtom)

Bare vertex `Uᵣ` of channel `r`: `U`, `-U`, `2U` and `0` for `d`, `m`, `s` and `t`. Because
of the spin combinations that define the channels, it is not always `U`.
"""
bare_vertex(::d, at::AbstractHubbardAtom) = at.U
bare_vertex(::m, at::AbstractHubbardAtom) = -at.U
bare_vertex(::s, at::AbstractHubbardAtom) = 2 * at.U
bare_vertex(::t, at::AbstractHubbardAtom) = zero(at.U)

"""
    gf(atom::AbstractHubbardAtom, ν::FermionicFreq)
    G(atom, ν)

Green's function `G(ν) = -∫₀^β dτ exp(iντ) ⟨T c↑(τ) c↑⁺(0)⟩` (Equation 2). For
`DopedHubbardAtom`, `G(ν) = (1 - ⟨n↑⟩)/(iν + μ + U/2) + ⟨n↑⟩/(iν + μ - U/2)`. Complex for
both atoms.
"""
function G end
const gf = G

"""
    chi(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    chi(r::SpinChannel, atom::AbstractHubbardAtom, ω::BosonicFreq)
    χ(…)

Generalized susceptibility `χᵣ(ν, ν′, ω)` of channel `r` (Equations 3-5 and 10; away from
half filling, Eßl et al., Phys. Rev. Research 6, 033061 (2024), Eq. (D3)), and the
susceptibility `χᵣ(ω) = -2/β² Σ_νν′ χᵣ(ν, ν′, ω)`.

For `r = d, m`, `χᵣ(ω)` is the physical susceptibility `-⟨ρ; ρ⟩(ω)` of the charge
(`ρ = n↑ + n↓`) or spin (`ρ = n↑ - n↓`) density; both are conserved, so it vanishes for
`ω ≠ 0`. For `r = s` it is `-⟨ρ; ρ⁺⟩(-ω)` with the pair density `ρ = c↓c↑`, which equals
`-(⟨n⟩ - 1)/(2μ - iω)`; the pair density is conserved only at half filling. For `r = t` it
is not an observable (the local triplet pair density vanishes) but the bare
particle-particle bubble `1/β Σ_ν G(ν) G(-ν - ω)`, as the vertex part drops out.
"""
function χ end
const chi = χ

"""
    chi0(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    chi0(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ω)::FermiBose)
    χ₀(…)

Bare generalized susceptibility (Equation 6): `χ₀ᵣ(ν, ω) = -G(ν) G(ν + ω)` for `r = d, m`
and `-G(ν) G(-ν - ω)/2` for `r = s, t`, and `χ₀ᵣ(ν, ν′, ω) = β δ_νν′ χ₀ᵣ(ν, ω)`.
"""
function χ₀ end
const chi0 = χ₀

"""
    full_vertex(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    F(…)

Full vertex `Fᵣ(ν, ν′, ω)` (Equation 27 at half filling), defined by
`χᵣ(ν, ν′, ω) = ηᵣ χ₀ᵣ(ν, ν′, ω) - χ₀ᵣ(ν, ω) Fᵣ(ν, ν′, ω) χ₀ᵣ(ν′, ω)` with `ηᵣ = -1` for
`r = s` and `ηᵣ = 1` for `r = d, m, t`.
"""
function F end
const full_vertex = F

"""
    gamma(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    Γ(…)

Channel-irreducible vertex `Γᵣ = β² (χᵣ⁻¹ - ηᵣ χ₀ᵣ⁻¹)`, with the inverses of matrices in the
fermionic frequencies and `ηᵣ` as for `full_vertex`: Equation 19 at half filling, and away
from half filling the closed forms of `derivation/doped-gamma-derivation.pdf`.

At the removable singularities of these closed forms, where terms diverge separately but `Γ`
is finite (e.g. Equation 21), the cancellation is handled analytically or by evaluating in
higher precision, so `Γ` stays accurate there. At genuine divergences it is infinite or
large.
"""
function Γ end
const gamma = Γ

"""
    irreducible_vertex(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    Λ(…)

Fully irreducible vertex `Λᵣ(ν, ν′, ω)` (Equation 26). It is evaluated as the sum of `Γ` and
`F` in Equation 26, so its absolute accuracy is that of the largest of these terms; where
`Λ` is much smaller (e.g. in the triplet channel at large frequencies), its relative
accuracy is lower.
"""
function Λ(::d, at::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    w = (ν, ν′, ω)
    w_phbar = (ν, ν + ω, ν′ - ν)
    w_pp = (ν, ν′, -ν - ν′ - ω)

    (Γ(d(), at, w) - Γ(d(), at, w_phbar) / 2 - 3Γ(m(), at, w_phbar) / 2 +
     Γ(s(), at, w_pp) / 2 + 3Γ(t(), at, w_pp) / 2 - 2F(d(), at, w))
end
function Λ(::m, at::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    w = (ν, ν′, ω)
    w_phbar = (ν, ν + ω, ν′ - ν)
    w_pp = (ν, ν′, -ν - ν′ - ω)

    (Γ(m(), at, w) - Γ(d(), at, w_phbar) / 2 + Γ(m(), at, w_phbar) / 2 -
     Γ(s(), at, w_pp) / 2 + Γ(t(), at, w_pp) / 2 - 2F(m(), at, w))
end
function Λ(::s, at::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    w = (ν, ν′, ω)
    w_phbar = (ν, -ν′ - ω, ν′ - ν)
    w_pp = (ν, ν′, -ν - ν′ - ω)

    (Γ(s(), at, w) + Γ(d(), at, w_pp) / 2 - 3Γ(m(), at, w_pp) / 2 +
     Γ(d(), at, w_phbar) / 2 - 3Γ(m(), at, w_phbar) / 2 - 2F(s(), at, w))
end
function Λ(::t, at::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    w = (ν, ν′, ω)
    w_phbar = (ν, -ν′ - ω, ν′ - ν)
    w_pp = (ν, ν′, -ν - ν′ - ω)

    (Γ(t(), at, w) + Γ(d(), at, w_pp) / 2 + Γ(m(), at, w_pp) / 2 -
     Γ(d(), at, w_phbar) / 2 - Γ(m(), at, w_phbar) / 2 - 2F(t(), at, w))
end

const irreducible_vertex = Λ

"""
    channel_reducible_vertex(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ν′, ω)::FermiFermiBose)
    Φ(…)

Channel-reducible vertex `Φᵣ = Fᵣ - Γᵣ`; its relative accuracy is lower where it is much
smaller than `F`.
"""
Φ(r::SpinChannel, at::AbstractHubbardAtom, w::FermiFermiBose) = F(r, at, w) - Γ(r, at, w)

const channel_reducible_vertex = Φ

"""
    g3(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ω)::FermiBose)
    G₃(…)

Three-point (fermion-boson) Green's function, `G₃ᵣ(ν, ω) = -1/β Σ_ν′ χᵣ(ν, ν′, ω)` for
`r = d, m, s`; for `r = d, m` it follows from the Ward identities of Krien and Valli, Phys.
Rev. B 100, 245147 (2019), Eqs. (C1)-(C3). At half filling, η-pairing symmetry gives
`G₃(s) = G₃(d)/2`. Zero for `r = t`, as `c↑c↑ = 0`.
"""
function G₃ end
const g3 = G₃

"""
    hedin(r::SpinChannel, atom::AbstractHubbardAtom, (ν, ω)::FermiBose)

Hedin vertex `λᵣ(ν, ω)`, the interaction-irreducible three-point vertex,
`λᵣ = -G₃ᵣ / (χ₀ᵣ (1 + Uᵣ χᵣ(ω) / 2))` with the bare vertex `Uᵣ` (`bare_vertex`). This is
the convention of Krien, Valli and Capone, Phys. Rev. B 100, 155149 (2019), Eqs. (8) and
(15): `λ` tends to `1` for `d, m` and to `-1` for `s`, both for `U → 0` and `|ν| → ∞`, and
at half filling `hedin(s, …) = -hedin(d, …)`. There is none in the triplet channel, where
the bare vertex vanishes; `hedin` throws an `ArgumentError` there.
"""
function hedin(r::SpinChannel, at::AbstractHubbardAtom, (n, k)::FermiBose)
    -G₃(r, at, (n, k)) / (χ₀(r, at, (n, k)) * (1 + bare_vertex(r, at) * χ(r, at, k) / 2))
end
function hedin(::t, at::AbstractHubbardAtom, w::FermiBose)
    throw(ArgumentError("there is no Hedin vertex in the triplet channel, where the bare interaction vanishes"))
end

# Matsubara frequency nπ/β for an index n or a frequency (SparseIR's `value`, which is not public)
_freq(i::Integer, β) = i * (π / β)
_freq(x::Union{FermionicFreq,BosonicFreq}, β) = _freq(Int(x), β)

"Kronecker delta"
δ(a) = δ(a, zero(a))
δ(a, b) = Int(a == b)
