"""
Analytic expressions for vertices in the half-filled Hubbard atom.

All equation numbers refer to Phys. Rev. B 98, 235107 (2018) by Thunström et al.:
https://journals.aps.org/prb/abstract/10.1103/PhysRevB.98.235107
"""
module HubbardAtoms

using SparseIR: FermionicFreq, BosonicFreq, value, valueim

export HubbardAtom, CHANNELS, FermiBose, FermiFermiBose,
    DensityChannel, MagneticChannel, SingletChannel, TripletChannel,
    bare_vertex, gf, chi, chi0, full_vertex, gamma, irreducible_vertex,
    channel_reducible_vertex, hedin

abstract type SpinChannel end

Base.broadcastable(ch::SpinChannel) = Ref(ch)

abstract type PHChannel <: SpinChannel end
abstract type PPChannel <: SpinChannel end

struct DensityChannel <: PHChannel end
struct MagneticChannel <: PHChannel end
struct SingletChannel <: PPChannel end
struct TripletChannel <: PPChannel end

# Convenient shorthands
const d = DensityChannel
const m = MagneticChannel
const s = SingletChannel
const t = TripletChannel

const CHANNELS = (d(), m(), s(), t())

const FermiBose = Tuple{FermionicFreq,BosonicFreq}
const FermiFermiBose = Tuple{FermionicFreq,FermionicFreq,BosonicFreq}


"""
Represents a half-filled Hubbard atom embedded. Its Hamiltonian is:

    H = U * (c'[↑] * c[↑] - 1/2) * (c'[↓] * c[↓] - 1/2)
 
where `c[σ]` annihilates a spin-σ electron. We assume that the atom is
connected to a large heat bath with temperature `1/β`. (Equation 1)
"""
struct HubbardAtom
    U::Float64              # Hubbard interaction
    beta::Float64           # inverse temperature

    _expuhalfbeta::Float64  # exp(βU/2)
    _uhalf2::Float64        # (U/2)^2
    _psum::Float64          # shifted partition function

    function HubbardAtom(U, beta)
        0 <= beta || throw(DomainError("beta must be non-negative"))

        expuhalfbeta = exp(beta * U / 2)
        uhalf2 = (U / 2)^2
        psum = 2 + 2 / expuhalfbeta
        isfinite(expuhalfbeta) || error(beta * U / 2, "exp(βU/2) must be finite")
        new(U, beta, expuhalfbeta, uhalf2, psum)
    end
end

Base.broadcastable(atom::HubbardAtom) = Ref(atom)

"""
    bare_vertex(::SpinChannel, atom::HubbardAtom)

Bare vertex entering diagrammatic equations in the different spin channels.
Note that due to the rotations and multiplicities, these are not always equal
to `U`.
"""
bare_vertex(::d, at::HubbardAtom) = at.U
bare_vertex(::m, at::HubbardAtom) = -at.U
bare_vertex(::s, at::HubbardAtom) = 2 * at.U
bare_vertex(::t, at::HubbardAtom) = zero(at.U)

"""
    G(atom::HubbardAtom, n::FermionicFreq)
    gf(atom::HubbardAtom, n::FermionicFreq)

Two-point propagator. (Equation 2)
"""
function G(at::HubbardAtom, n::FermionicFreq)
    U²₄ = at._uhalf2
    iν = valueim(n, at.beta)

    1 / (iν - U²₄ / iν)
end

const gf = G

"""
    χ(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    chi(::SpinChannel, atom::HubbardAtom, (n, n´, m)::FermiFermiBose)

Two-point susceptibility in each channel for the Hubbard atom. (Equation 10)
"""
χ(r::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose) =
    a₀(r, at, (n, m)) * (δ(n, n´) - δ(n, -n´ - m)) +
    b₀(r, at, (n, m)) * (δ(n, n´) + δ(n, -n´ - m)) +
    b₁(r, at, (n, m)) * b₁(r, at, (n´, m)) + b₂(r, at, (n, m)) * b₂(r, at, (n´, m))

const chi = χ

"Equation 11a"
function a₀(r::SpinChannel, at::HubbardAtom, (n, m)::FermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    ν = value(n, β)
    ω = value(m, β)
    Aᵣ = A(r, at)

    𝒜₀(r) * β / 2 * (ν * (ν + ω) - Aᵣ^2) / ((ν^2 + U²₄) * ((ν + ω)^2 + U²₄))
end

"Equation 11b"
function b₀(r::SpinChannel, at::HubbardAtom, (n, m)::FermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    ν = value(n, β)
    ω = value(m, β)
    Bᵣ = B(r, at)

    ℬ₀(r) * β / 2 * (ν * (ν + ω) - Bᵣ^2) / ((ν^2 + U²₄) * ((ν + ω)^2 + U²₄))
end

"Equation 11c"
function b₁(r::SpinChannel, at::HubbardAtom, (n, m)::FermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    U = at.U
    ν = value(n, β)
    ω = value(m, β)
    Cᵣʷ = C(r, at, m)
    Dᵣʷ = D(r, at, m)

    ℬ₁(r) * √(Complex(U * (1 - Cᵣʷ))) * (ν * (ν + ω) - Dᵣʷ) / ((ν^2 + U²₄) * ((ν + ω)^2 + U²₄))
end

"Equation 11d"
function b₂(r::SpinChannel, at::HubbardAtom, (n, m)::FermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    U = at.U
    ν = value(n, β)
    ω = value(m, β)
    Cᵣʷ = C(r, at, m)

    ℬ₂(r) * √(Complex(U * U²₄)) * √(U^2 / (1 - Cᵣʷ) + ω^2) / ((ν^2 + U²₄) * ((ν + ω)^2 + U²₄))
end

"Equation 12"
function D(r::SpinChannel, at::HubbardAtom, m::BosonicFreq)
    U²₄ = at._uhalf2
    Cᵣʷ = C(r, at, m)

    U²₄ * (1 + Cᵣʷ) / (1 - Cᵣʷ)
end

# Table I

A(::d, at::HubbardAtom) = at.U / 2 * √3
A(::m, at::HubbardAtom) = im * at.U / 2
A(::s, at::HubbardAtom) = 0.0
A(::t, at::HubbardAtom) = im * at.U / 2

B(::d, at::HubbardAtom) = at.U / 2 * √(Complex((-1 + 3at._expuhalfbeta) / (1 + at._expuhalfbeta)))
B(::m, at::HubbardAtom) = -at.U / 2 * √(Complex((-at._expuhalfbeta + 3) / (at._expuhalfbeta + 1)))
B(::s, at::HubbardAtom) = at.U / 2 * √(Complex((-1 + 3at._expuhalfbeta) / (1 + at._expuhalfbeta)))
B(::t, at::HubbardAtom) = 0.0

C(::d, at::HubbardAtom, m::BosonicFreq) = at.beta * at.U / 2 * δ(m) / (1 + at._expuhalfbeta)
C(::m, at::HubbardAtom, m::BosonicFreq) = -at.beta * at.U / 2 * δ(m) / (1 + 1 / at._expuhalfbeta)
C(::s, at::HubbardAtom, m::BosonicFreq) = at.beta * at.U / 2 * δ(m) / (1 + at._expuhalfbeta)
C(::t, at::HubbardAtom, m::BosonicFreq) = 0.0

𝒜₀(::d) = +1.0
𝒜₀(::m) = +1.0
𝒜₀(::s) = +0.5
𝒜₀(::t) = -0.5

ℬ₀(::d) = +1.0
ℬ₀(::m) = +1.0
ℬ₀(::s) = +0.5
ℬ₀(::t) = -0.5

ℬ₁(::d) = 1.0im
ℬ₁(::m) = 1.0
ℬ₁(::s) = 1.0im / √2
ℬ₁(::t) = 0.0

ℬ₂(::d) = 1.0
ℬ₂(::m) = 1.0im
ℬ₂(::s) = 1.0 / √2
ℬ₂(::t) = 0.0


"""
    χ(::SpinChannel, atom::HubbardAtom, m::BosonicFreq)
    chi(::SpinChannel, atom::HubbardAtom, m::BosonicFreq)

Sum over all fermionic frequencies of the susceptibility, i.e.
`-sum(χ(r, atom, (n, n´, m)) for n in -∞:+∞, n´ in -∞:+∞) * atom.beta^2/2 * tanh(atom.U * atom.beta / 4)^2`.
This is an original calculation.
"""
χ(::d, at::HubbardAtom, m::BosonicFreq) = -2at.beta * δ(m) / (at._expuhalfbeta * at._psum)
χ(::m, at::HubbardAtom, m::BosonicFreq) = -2at.beta * δ(m) / at._psum
χ(::s, at::HubbardAtom, m::BosonicFreq) = -at.beta * δ(m) / (at._expuhalfbeta * at._psum)
χ(::t, at::HubbardAtom, m::BosonicFreq) = at.U / ((8 / at._psum - 2) * (value(m, at.beta)^2 + 4at._uhalf2))

"""
    χ₀(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    χ₀(::SpinChannel, at::HubbardAtom, (n, m)::Tuple{FermionicFreq, BosonicFreq})
    chi0(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    chi0(::SpinChannel, at::HubbardAtom, (n, m)::Tuple{FermionicFreq, BosonicFreq})

Bare generalized susceptibility. The 2- and 3-frequency versions are related by
`β * χ₀(r, at, (n, m)) = sum(χ₀(r, at, (n, n´, m)) for n´ in -∞:+∞)`. (Equation 6)
"""
function χ₀(r::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    β = at.beta

    β * δ(n, n´) * χ₀(r, at, (n, m))
end

χ₀(::PHChannel, at::HubbardAtom, (n, m)::FermiBose) = -G(at, n) * G(at, n + m)
χ₀(::PPChannel, at::HubbardAtom, (n, m)::FermiBose) = -1 / 2 * G(at, n) * G(at, -n - m)

const chi0 = χ₀

"""
    F(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    full_vertex(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)

Full two-particle scattering amplitude `F(ν, ν´, ω)`. (Equation 27)
"""
F(::DensityChannel, at::HubbardAtom, w::FermiFermiBose) = F_up_up(at, w) + F_up_down(at, w)
F(::MagneticChannel, at::HubbardAtom, w::FermiFermiBose) = F_up_up(at, w) - F_up_down(at, w)
F(::SingletChannel, at::HubbardAtom, w::FermiFermiBose) = -F_up_up(at, _to_pp(w)) + 2F_up_down(at, _to_pp(w))
F(::TripletChannel, at::HubbardAtom, w::FermiFermiBose) = F_up_up(at, _to_pp(w))

const full_vertex = F

_to_pp((n, n´, m)::FermiFermiBose) = (n, n´, -n - n´ - m)

# These formulas can be found in Fully_irreducible_vertex.nb from the paper's supplementary material
function F_up_up(at::HubbardAtom, w::FermiFermiBose)
    (n, n´, m) = w
    β = at.beta
    U²₄ = at._uhalf2

    res = zero(β)

    deltas = δ(n, n´) - δ(m)

    if !iszero(deltas)
        ν = value(n, β)
        ν´ = value(n´, β)
        ω = value(m, β)

        res += β * U²₄ * (ν^2 + U²₄) * ((ν´ + ω)^2 + U²₄) / (ν^2 * (ν´ + ω)^2) * deltas
    end

    res
end

function F_up_down(at::HubbardAtom, w::FermiFermiBose)
    (n, n´, m) = w
    β = at.beta
    U = at.U
    U²₄ = at._uhalf2

    ν = value(n, β)
    ν´ = value(n´, β)
    ω = value(m, β)

    res = U - U^3 / 8 * (ν^2 + (ν + ω)^2 + (ν´ + ω)^2 + ν´^2) / (ν * (ν + ω) * (ν´ + ω) * ν´) - 3U^5 / (16 * ν * (ν + ω) * (ν´ + ω) * ν´)

    deltas1 = 2 * δ(n, -(n´ + m)) + δ(m)
    if !iszero(deltas1)
        res -= β * U²₄ / (1 + at._expuhalfbeta) * ((ν + ω)^2 + U²₄) * ((ν´ + ω)^2 + U²₄) / ((ν + ω)^2 * (ν´ + ω)^2) * deltas1
    end

    deltas2 = 2 * δ(n, n´) + δ(m)
    if !iszero(deltas2)
        res += β * U²₄ / (1 + 1 / at._expuhalfbeta) * (ν^2 + U²₄) * ((ν´ + ω)^2 + U²₄) / (ν^2 * (ν´ + ω)^2) * deltas2
    end

    res
end

"""
    Γ(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    gamma(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)

Gives channel-irreducible four-point vertex `Γ(ν, ν', ω)`. (Equation 19)
"""
function Γ(r::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    ν = value(n, β)
    ω = value(m, β)

    Aᵣ² = A(r, at)^2
    Γᵣ = β * Aᵣ² / 2𝒜₀(r) * (ν^2 + U²₄) * ((ν + ω)^2 + U²₄) / ((ν * (ν + ω) - Aᵣ²) * ν * (ν + ω)) *
         (δ(n, n´) - δ(n, -n´ - m))
    r isa TripletChannel && return Γᵣ

    Bᵣ² = B(r, at)^2
    Γᵣ += β * Bᵣ² / 2ℬ₀(r) * (ν^2 + U²₄) * ((ν + ω)^2 + U²₄) / ((ν * (ν + ω) - Bᵣ²) * ν * (ν + ω)) *
          (δ(n, n´) + δ(n, -n´ - m))

    U = at.U
    ν´ = value(n´, β)
    sqrtω = √(4Bᵣ² + ω^2)
    ± = ifelse(r isa MagneticChannel, -, +)

    Γᵣ -= U * abs2(ℬ₂(r)) / ℬ₀(r)^2 * U²₄ * (U²₄ * (Bᵣ² / U²₄ + 1)^2 + ω^2) /
          ((U * tan(β / 4 * (sqrtω + ω)) / sqrtω ± 1) *
           (ν * (ν + ω) - Bᵣ²) * (ν´ * (ν´ + ω) - Bᵣ²))

    Γᵣ -= (ℬ₁(r) / ℬ₀(r))^2 * U

    Γᵣ
end

const gamma = Γ

"""
    Λ(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    irreducible_vertex(::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)

Fully irreducible four-point vertex `Λ(ν, ν', ω)`. (Equation 26)
"""
function Λ(::d, at::HubbardAtom, (ν, ν´, ω)::FermiFermiBose)
    w = (ν, ν´, ω)
    w_phbar = (ν, ν + ω, ν´ - ν)
    w_pp = (ν, ν´, -ν - ν´ - ω)

    (Γ(d(), at, w) - 0.5Γ(d(), at, w_phbar) - 1.5Γ(m(), at, w_phbar)
     + 0.5Γ(s(), at, w_pp) + 1.5Γ(t(), at, w_pp) - 2F(d(), at, w))
end
function Λ(::m, at::HubbardAtom, (ν, ν´, ω)::FermiFermiBose)
    w = (ν, ν´, ω)
    w_phbar = (ν, ν + ω, ν´ - ν)
    w_pp = (ν, ν´, -ν - ν´ - ω)

    (Γ(m(), at, w) - 0.5Γ(d(), at, w_phbar) + 0.5Γ(m(), at, w_phbar)
     -
     0.5Γ(s(), at, w_pp) + 0.5Γ(t(), at, w_pp) - 2F(m(), at, w))
end
function Λ(::s, at::HubbardAtom, (ν, ν´, ω)::FermiFermiBose)
    w = (ν, ν´, ω)
    w_phbar = (ν, -ν´ - ω, ν´ - ν)
    w_pp = (ν, ν´, -ν - ν´ - ω)

    (Γ(s(), at, w) + 0.5Γ(d(), at, w_pp) - 1.5Γ(m(), at, w_pp) +
     0.5Γ(d(), at, w_phbar) - 1.5Γ(m(), at, w_phbar) - 2F(s(), at, w))
end
function Λ(::t, at::HubbardAtom, (ν, ν´, ω)::FermiFermiBose)
    w = (ν, ν´, ω)
    w_phbar = (ν, -ν´ - ω, ν´ - ν)
    w_pp = (ν, ν´, -ν - ν´ - ω)

    (Γ(t(), at, w) + 0.5Γ(d(), at, w_pp) + 0.5Γ(m(), at, w_pp) -
     0.5Γ(d(), at, w_phbar) - 0.5Γ(m(), at, w_phbar) - 2F(t(), at, w))
end

const irreducible_vertex = Λ

"""
    Φ(r::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)
    channel_reducible_vertex(r::SpinChannel, at::HubbardAtom, (n, n´, m)::FermiFermiBose)

Channel reducible four-point vertex in channel `r` `Φʳ(ν, ν', ω)`.
"""
Φ(r::SpinChannel, at::HubbardAtom, w::FermiFermiBose) = F(r, at, w) - Γ(r, at, w)

const channel_reducible_vertex = Φ

"""
    G₃(::SpinChannel, atom::HubbardAtom, (n, m)::Tuple{FermionicFreq, BosonicFreq})
    g3(::SpinChannel, atom::HubbardAtom, (n, m)::Tuple{FermionicFreq, BosonicFreq})

Three-point Green's function.
"""
G₃(::d, at::HubbardAtom, (n, m)::FermiBose) = iszero(m) ? -∂G∂μ(at, n) : -∂G∂ν(at, n, m)
G₃(::m, at::HubbardAtom, (n, m)::FermiBose) = iszero(m) ? -∂G∂H(at, n) : -∂G∂ν(at, n, m)
function G₃(::s, at::HubbardAtom, (n, m)::FermiBose)
    β = at.beta
    U = at.U
    iν = valueim(n, β)
    iω = valueim(m, β)

    r = (1 + at._expuhalfbeta) / ((iν + U / 2) * (iω - iν - U / 2))
    r += (1 + at._expuhalfbeta) / ((iν + U / 2) * (iω - iν + U / 2))
    r += (δ(m) * β * U) / ((iν + U / 2) * (iν - U / 2))
    r / (-at._expuhalfbeta * at._psum)
end
G₃(::t, at::HubbardAtom, (n, m)::FermiBose) = 0.0

const g3 = G₃

"Finite difference of Green's function with respect to frequency"
function ∂G∂ν(at::HubbardAtom, n::FermionicFreq, m::BosonicFreq)
    β = at.beta
    iω = valueim(m, β)

    (G(at, n + m) - G(at, n)) / iω
end

"Derivative of the Green's function with respect to the chemical potential"
function ∂G∂μ(at::HubbardAtom, n::FermionicFreq)
    psum = at._psum
    U = at.U
    β = at.beta
    iν = valueim(n, β)

    # dgdμ enters the charge channel, while dgdh enters the spin channel.
    # There is an additional factor exp(-beta*U/2) here while it is absent
    # in the spin channel. This factor suppresses the beta dependence and thus
    # the "Curie-like" term.
    r = -β * U / (at._expuhalfbeta * (iν^2 - at._uhalf2))
    r += (psum / 2) / (iν + U / 2)^2
    r += (psum / 2) / (iν - U / 2)^2
    -r / psum
end

"Derivative of the Green's function with respect to the magnetic field"
function ∂G∂H(at::HubbardAtom, n::FermionicFreq)
    psum = at._psum
    U = at.U
    β = at.beta
    iν = valueim(n, β)

    r = β * U / (iν^2 - at._uhalf2)
    r += (psum / 2) / (iν + U / 2)^2
    r += (psum / 2) / (iν - U / 2)^2
    -r / psum
end

"""
    hedin(::SpinChannel, atom::HubbardAtom, (n, m)::Tuple{FermionicFreq, BosonicFreq})

Hedin vertex `γ(ν, ω)`, i.e. the interaction-irreducible three-point vertex.
"""
hedin(r::SpinChannel, at::HubbardAtom, (n, m)::FermiBose) =
    G₃(r, at, (n, m)) /
    (χ₀(r, at, (n, m)) * (1 + bare_vertex(r, at) * χ(r, at, m) / 2))

"Kronecker delta"
δ(a) = δ(a, zero(a))
δ(a, b) = δ(Float64, a, b)
δ(::Type{T}, a, b) where {T} = ifelse(a == b, one(T), zero(T))

end
