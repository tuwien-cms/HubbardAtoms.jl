# Field-sector labels are separate from the SU(2) channels: the latter's generic methods
# (e.g. Λ) do not apply to the coupled longitudinal or oriented pairing problems.
"Supertype of the spin-labelled sectors of `FieldHubbardAtom`; spins are +1 (↑), -1 (↓)."
abstract type FieldSector end
Base.Broadcast.broadcastable(r::FieldSector) = Ref(r)

function _field_spin(σ::Integer)
    σ in (1, -1) || throw(ArgumentError("spin must be +1 (↑) or -1 (↓)"))
    Int(σ)
end

"""
    LongitudinalSector(sigma, tau)

Entry (σ, τ) of the connected longitudinal particle-hole matrix in the spin basis.
Spins are +1 (↑), -1 (↓). Its inverse couples both spins and all fermionic frequencies;
inverting a charge or spin diagonal block separately is not the longitudinal BSE.
"""
struct LongitudinalSector <: FieldSector
    sigma::Int
    tau::Int
    LongitudinalSector(σ::Integer, τ::Integer) = new(_field_spin(σ), _field_spin(τ))
end

"""
    TransverseSector(sigma)

Transverse spin T_σ, with σ = +1 (↑), -1 (↓): the full Fourier transform of
⟨T cσ†(τ₁)c₋σ(τ₂)c₋σ†(τ₃)cσ(0)⟩, with the package's particle-hole Fourier factor.
"""
struct TransverseSector <: FieldSector
    sigma::Int
    TransverseSector(σ::Integer) = new(_field_spin(σ))
end

"""
    OppositeSpinPairSector(sigma)

Oriented opposite-spin pair A_σ, σ = +1 (↑), -1 (↓), in pp notation:
χ_Aσ(ν,ν′,ω) = -χ_Tσ(ν,ν′,-ω-ν-ν′). The two orientations obey χ_A↓ = M χ_A↑ M,
where M reflects ν to -ν-ω; they are not singlet/triplet scalar channels at finite field.
"""
struct OppositeSpinPairSector <: FieldSector
    sigma::Int
    OppositeSpinPairSector(σ::Integer) = new(_field_spin(σ))
end

"""
    EqualSpinPairSector(sigma)

Equal-spin pair E_σ, σ = +1 (↑), -1 (↓): -1/2 times the full (not connected) same-spin
ph four-point function at transfer -ω-ν-ν′. The factor 1/2 removes identical-leg double
counting. χ, χ₀ and Γ vanish on the mirror-even subspace; Γ uses inverses restricted
to the mirror-odd subspace M₋, extended by zero (including mirror fixed points).
"""
struct EqualSpinPairSector <: FieldSector
    sigma::Int
    EqualSpinPairSector(σ::Integer) = new(_field_spin(σ))
end

struct FieldLSetup{T}
    w::Complex{T}
    q0::Complex{T}
    E8::Vector{Complex{T}}  # monic quartic in s = z², ascending coefficients
    s::Vector{Complex{T}}
    u::Vector{Complex{T}}
    keep::Vector{Bool}     # omit the zero-coefficient atomic root at k = 0
    f::Matrix{Complex{T}}  # pole-pair modes (↑+, ↑-, ↓+, ↓-)
    𝔠::Matrix{Complex{T}}  # A_tj/K_jt = W_tj/K_j,-t
end

"""
    FieldHubbardAtom(U, beta, mu, h)

Hubbard atom with doping and Zeeman field (splitting 2h),

    H = U(n↑ - 1/2)(n↓ - 1/2) - mu(n↑ + n↓ - 1) - h(n↑ - n↓).

Parameters must be finite, with beta > 0; they are promoted to one floating-point type T.
`gf(atom, sigma, ν)`, `chi`, `chi0` and `gamma` in the four [`FieldSector`](@ref)s return
Complex{T}, including for BigFloat. The longitudinal per-transfer data are cached thread-safely.
No d/m/s/t, full vertex or three-point API is implied for this spin-asymmetric atom.

The closed forms are those of the companion paper (see the README). The longitudinal
theorem covers μh(μ²-h²) ≠ 0, simple nonzero non-atomic roots off the lattice, and nonzero
thermal denominators. On μ±h = 0 the exact Cauchy-kernel limits give its continuous extension,
verified numerically, not proved there. At h = 0 and μ = 0 the established zero-field/Shiba
identities are used. Near longitudinal root confluences or lattice roots the reference's
symmetric μ±δ evaluation in extended precision is retained; it is meaningful only at
removable singularities, **not** at genuine divergences. Vanishing thermal denominators and
equal-spin susceptibility zeros are not regularized. The generic transverse formula likewise
does not continue confluent/lattice roots; use away from these exceptional sets.
"""
struct FieldHubbardAtom{T<:AbstractFloat} <: AbstractHubbardAtom
    U::T
    beta::T
    mu::T
    h::T
    _a::T
    _lambda::T
    _p0::T
    _pu::T
    _pd::T
    _p2::T
    _rho::T
    _mag::T
    _piP::T
    _piS::T
    _cP::T
    _cS::T
    _D::NTuple{2,T}
    _c::NTuple{2,T}
    _setups::Dict{Int,FieldLSetup{T}}
    _lock::ReentrantLock

    function FieldHubbardAtom(U::Real, beta::Real, mu::Real, h::Real)
        U, beta, mu, h = float.(promote(U, beta, mu, h))
        for (name, x) in (("U", U), ("mu", mu), ("h", h))
            isfinite(x) || throw(DomainError(x, "$name must be finite"))
        end
        isfinite(beta) && beta > 0 ||
            throw(DomainError(beta, "beta must be positive and finite"))
        E = (U / 4 + mu, -U / 4 - h, -U / 4 + h, U / 4 - mu)
        weights = map(e -> exp(-beta * (e - minimum(E))), E)
        p0, pu, pd, p2 = weights ./ sum(weights)
        ρ = _field_difference(p2, p0, 2beta * mu)
        mag = _field_difference(pu, pd, 2beta * h)
        D = (_field_difference(p2 * pu, p0 * pd, 2beta * (mu + h)),
             _field_difference(p2 * pd, p0 * pu, 2beta * (mu - h)))
        T = typeof(U)
        new{T}(U, beta, mu, h, U / 2, U^2, p0, pu, pd, p2, ρ, mag, p0 * p2, pu * pd,
               beta * (p0 + p2) / 2, beta * (pu + pd) / 2, D,
               (p2 * pu + p0 * pd, p2 * pd + p0 * pu), Dict{Int,FieldLSetup{T}}(),
               ReentrantLock())
    end
end

# p-q with log(p/q) = x, without cancellation or an overflowing exponential.
_field_difference(p, q, x) = x >= 0 ? -p * expm1(-x) : q * expm1(x)

function Base.:(==)(a::FieldHubbardAtom, b::FieldHubbardAtom)
    (a.U, a.beta, a.mu, a.h) == (b.U, b.beta, b.mu, b.h)
end
function Base.hash(a::FieldHubbardAtom, h::UInt)
    hash((a.U, a.beta, a.mu, a.h), hash(:FieldHubbardAtom, h))
end
function Base.show(io::IO, a::FieldHubbardAtom)
    print(io, "FieldHubbardAtom(", a.U, ", ", a.beta, ", ", a.mu, ", ", a.h, ")")
end

include("field_longitudinal.jl")
include("field_transverse.jl")

"""
    gf(atom::FieldHubbardAtom, sigma::Integer, ν::FermionicFreq)

Spin-resolved Green's function, σ = +1 (↑), -1 (↓):
Gσ(ν) = (p₀+pσ)/(iν+μ+σh+U/2) + (p₋σ+p₂)/(iν+μ+σh-U/2).
"""
function G(at::FieldHubbardAtom, σ::Integer, n::FermionicFreq)
    σ = _field_spin(σ)
    x = Complex(at.mu + σ * at.h, _freq(n, at.beta))
    pσ, pmσ = σ == 1 ? (at._pu, at._pd) : (at._pd, at._pu)
    (at._p0 + pσ) / (x + at._a) + (pmσ + at._p2) / (x - at._a)
end

"""
    chi0(r::FieldSector, atom::FieldHubbardAtom, (ν, ω)::FermiBose)
    chi0(r::FieldSector, atom::FieldHubbardAtom, (ν, ν′, ω)::FermiFermiBose)

Physical field-sector bubbles. The two-frequency value excludes β:
L_στ: -δστ Gσ(ν)Gσ(ν+ω); Tσ: -Gσ(ν)G₋σ(ν+ω);
Aσ: Gσ(ν)G₋σ(-ν-ω); Eσ: Gσ(ν)Gσ(-ν-ω)/2.
The matrix value includes βδνν′, or β(δνν′-δν′,-ν-ω) for Eσ.
"""
function χ₀(r::LongitudinalSector, at::FieldHubbardAtom, (n, k)::FermiBose)
    r.sigma == r.tau ? -G(at, r.sigma, n) * G(at, r.sigma, n + k) : zero(complex(at.U))
end
function χ₀(r::TransverseSector, at::FieldHubbardAtom, (n, k)::FermiBose)
    -G(at, r.sigma, n) * G(at, -r.sigma, n + k)
end
function χ₀(r::OppositeSpinPairSector, at::FieldHubbardAtom, (n, k)::FermiBose)
    G(at, r.sigma, n) * G(at, -r.sigma, -n - k)
end
function χ₀(r::EqualSpinPairSector, at::FieldHubbardAtom, (n, k)::FermiBose)
    G(at, r.sigma, n) * G(at, r.sigma, -n - k) / 2
end
function χ₀(r::FieldSector, at::FieldHubbardAtom, (n, m, k)::FermiFermiBose)
    at.beta * δ(n, m) * χ₀(r, at, (n, k))
end
function χ₀(r::EqualSpinPairSector, at::FieldHubbardAtom, (n, m, k)::FermiFermiBose)
    at.beta * (δ(n, m) - δ(m, -n - k)) * χ₀(r, at, (n, k))
end

_field_flip(at::FieldHubbardAtom) = FieldHubbardAtom(at.U, at.beta, at.mu, -at.h)
_field_shiba(at::FieldHubbardAtom) = FieldHubbardAtom(-at.U, at.beta, at.h, at.mu)
_field_indices((n, m, k)::FermiFermiBose) = ((Int(n) - 1) ÷ 2, (Int(m) - 1) ÷ 2, Int(k) ÷ 2)

"""
    gamma(r::FieldSector, atom::FieldHubbardAtom, (ν, ν′, ω)::FermiFermiBose)

Closed-form Γ_X = β²(χ_X⁻¹ - χ₀_X⁻¹), η = +1 in every field sector. L is inverted in
both spins and frequencies together; E only on M₋ and then extended by zero. See
[`FieldHubbardAtom`](@ref) for the exceptional-set restrictions and continuous extensions.
At h = 0: Γ_Lσσ = (Γ_d+Γ_m)/2, Γ_Lσ,-σ = (Γ_d-Γ_m)/2, Γ_Tσ = Γ_m,
Γ_Aσ = (Γ_s-Γ_t)/2, Γ_Eσ = -Γ_t/2, all at DopedHubbardAtom(U,β,μ).
"""
function Γ(r::LongitudinalSector, at::FieldHubbardAtom{T}, w::FermiFermiBose) where {T}
    iszero(at.U) && return zero(Complex{T})
    σ, τ = r.sigma, r.tau
    if iszero(at.h) || iszero(at.mu)
        source = iszero(at.h) ? DopedHubbardAtom(at.U, at.beta, at.mu) :
                 DopedHubbardAtom(-at.U, at.beta, at.h)
        n, m, k = w
        sign = iszero(at.h) ? 1 : σ * τ
        v = iszero(at.h) ? w : (σ == 1 ? n : -n - k, τ == 1 ? m : -m - k, k)
        return sign *
               (Γ(DensityChannel(), source, v) + σ * τ * Γ(MagneticChannel(), source, v)) /
               2
    end
    n, m, k = _field_indices(w)
    s, t = (3 - σ) ÷ 2, (3 - τ) ÷ 2
    _field_l_needs_limit(at, s, t, n, m, k) && return _field_l_limit(at, s, t, n, m, k)
    _field_l_gammaL(at, s, t, n, m, k)
end
function Γ(r::Union{TransverseSector,OppositeSpinPairSector,EqualSpinPairSector},
           at::FieldHubbardAtom{T}, w::FermiFermiBose) where {T}
    iszero(at.U) && return zero(Complex{T})
    a = r.sigma == 1 ? at : _field_flip(at)
    n, m, k = _field_indices(w)
    r isa EqualSpinPairSector && return _field_t_gammaE_up(a, n, m, k)
    r isa OppositeSpinPairSector && (a = _field_shiba(a))
    _field_t_gammaT_up(a, n, m, k)
end

"""
    chi(r::FieldSector, atom::FieldHubbardAtom, (ν, ν′, ω)::FermiFermiBose)

Generalized susceptibility in the field sectors, from the exact four-state Lehmann input
(compact longitudinal/transverse forms and the conserved-occupation mixture for equal-spin
pairs). The Fourier factor is exp(-iντ₁+i(ν+ω)τ₂-i(ν′+ω)τ₃). Only L subtracts the product
of same-bilinear averages; T and A have no disconnected term, and E uses the full correlator.
"""
function χ(r::LongitudinalSector, at::FieldHubbardAtom, w::FermiFermiBose)
    n, m, k = _field_indices(w)
    σ, t = (3 - r.sigma) ÷ 2, (3 - r.tau) ÷ 2
    z, v = _field_l_zlat(at, σ, n, k), _field_l_zlat(at, t, m, k)
    β, a = at.beta, at._a
    q0 = a^2 + _freq(w[3], β)^2 / 4
    Pz, Pv = _field_l_Ppol(at, z, q0), _field_l_Ppol(at, v, q0)
    if σ == t
        pσ, pmσ = σ == 1 ? (at._pu, at._pd) : (at._pd, at._pu)
        val = -β * _field_l_Hpol(at, σ, z, q0) * δ(n, m)
        iszero(k) &&
            (val += 4β * a^2 * (at._p0 + pσ) * (pmσ + at._p2) * (z^2 - a^2) * (v^2 - a^2) /
                    Pv)
    else
        ζ = iszero(k) ? 4β * a^2 * (at._piP - at._piS) : zero(a)
        N2 = -2a * _field_l_Hpol(at, t, v, q0) + ζ * (v^2 - a^2)
        N0 = (v^2 + 2q0 - 4a^2) * N2 + (2a - ζ) * Pv
        val = at._lambda * (_field_l_kerP(at, z, v, n + m + 1 + k == 0) -
                            r.sigma * _field_l_kerS(at, σ, z, v, n == m)) +
              (N0 + z^2 * N2) / Pv
    end
    val / Pz
end
function χ(r::TransverseSector, at::FieldHubbardAtom, w::FermiFermiBose)
    a = r.sigma == 1 ? at : _field_flip(at)
    n, m, k = _field_indices(w)
    d = _field_t_tdata(a, k)
    z, v = _field_t_zlat(a, n, k), _field_t_zlat(a, m, k)
    # m/(2h-iω) has an exact static limit, including h = 0.
    ζ = iszero(k) ? 4a._a^2 * _field_l_kerS(a, 1, z, v, true) : -4a._a^2 * a._mag / d.W
    N2 = 2a._a * (v^2 + 2a._a * d.ρ * v + d.q0) + ζ * (v^2 - d.q0)
    N0 = (v^2 + 2d.q0 - 4a._a^2) * N2 - (2a._a + ζ) * _field_t_Pp(d, v)
    C = _field_l_kerP(a, z, v, n + m + 1 + k == 0)
    (d.λ * C + (N0 + z^2 * N2) / _field_t_Pp(d, v) - a.beta * _field_t_Hp(d, z) * δ(n, m)) /
    _field_t_Pp(d, z)
end
function χ(r::OppositeSpinPairSector, at::FieldHubbardAtom, (n, m, k)::FermiFermiBose)
    -χ(TransverseSector(r.sigma), at, (n, m, -k - n - m))
end
function χ(r::EqualSpinPairSector, at::FieldHubbardAtom, (n, m, k)::FermiFermiBose)
    D = δ(n, m) - δ(m, -n - k)
    iszero(D) && return zero(complex(at.U))
    x = Complex(at.mu + r.sigma * at.h, _freq(n, at.beta))
    xb = Complex(at.mu + r.sigma * at.h, _freq(-n - k, at.beta))
    pσ, pmσ = r.sigma == 1 ? (at._pu, at._pd) : (at._pd, at._pu)
    at.beta / 2 * D *
    ((at._p0 + pσ) / ((x + at._a) * (xb + at._a)) +
     (pmσ + at._p2) / ((x - at._a) * (xb - at._a)))
end
