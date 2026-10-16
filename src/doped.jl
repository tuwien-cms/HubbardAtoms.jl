#=
The Hubbard atom away from half filling. The formulas for Γ are derived in the note
derivation/doped-gamma-derivation.pdf of this repository (source: derivation/doped-gamma-derivation.typ);
the theorem, lemma and proposition names below refer to it. The susceptibility is the Lehmann representation of Eßl et al., Phys. Rev. Research
6, 033061 (2024), Eq. (D3), in the compact form of that note.

Notation: a = U/2, ε = +1 (d) or -1 (m), x = μ + iν, w = iω, z = x + w/2 (the centered lattice),
p₀, p₁, p₂ the weights of |0⟩, of each singly occupied state, and of |↑↓⟩, q₊ = p₁ + p₂,
q₋ = p₀ + p₁, ρ = p₂ - p₀ = ⟨n⟩ - 1, w₁₁ = p₁² - p₀p₂, h = β√(p₀p₂), c̄ = β(p₀ + p₂)/2.
=#

# Precomputed scalars of Theorem 4 for one transfer and channel (see `_ph_setup`)
struct PHSetup{T<:AbstractFloat}
    δ1::Complex{T}    # roots s = u² of the quartic e in z², as δ = u - w/2:
    δ2::Complex{T}    # z² - s = (x - δ)(x + w + δ) with x = z - w/2
    Y1::Complex{T}    # separable coefficients Y₁ and Y_d
    Yd::Complex{T}
end

"""
    DopedHubbardAtom(U, beta, mu)

Hubbard atom at arbitrary filling (zero magnetic field). Its Hamiltonian is

    H = U * (n[↑] - 1/2) * (n[↓] - 1/2) - mu * (n[↑] + n[↓] - 1)

with `mu` the chemical potential measured from half filling; `mu = 0` is the half-filled atom of
`HubbardAtom`. `U` and `mu` must be finite and `beta` finite and positive; all three are promoted
to a common floating-point type `T`. Results are complex (`Complex{T}`).

Γ in the density and magnetic channels needs a few scalars per bosonic transfer, which are
computed once and cached in the atom (thread-safely), so evaluating many entries at the same
transfer is cheap.
"""
struct DopedHubbardAtom{T<:AbstractFloat} <: AbstractHubbardAtom
    U::T              # Hubbard interaction
    beta::T           # inverse temperature
    mu::T             # chemical potential, measured from half filling

    _a::T             # U/2
    _p0::T            # weight of |0⟩
    _p1::T            # weight of each of |↑⟩, |↓⟩
    _p2::T            # weight of |↑↓⟩
    _qp::T            # p₁ + p₂ = ⟨n↑⟩
    _qm::T            # p₀ + p₁ = 1 - ⟨n↑⟩
    _rho::T           # p₂ - p₀ = ⟨n↑ + n↓⟩ - 1
    _w11::T           # p₁² - p₀p₂

    _setups::Dict{Tuple{Int,Int},PHSetup{T}}  # (Int(k), ε) => setup of Theorem 4
    _pp_setups::Dict{Int,Complex{T}}          # Int(k) => 𝒫(-Aₛ)/Kₛ of Theorem 3
    _lock::ReentrantLock

    function DopedHubbardAtom(U::Real, beta::Real, mu::Real)
        U, beta, mu = float.(promote(U, beta, mu))
        isfinite(U) || throw(DomainError(U, "U must be finite"))
        isfinite(mu) || throw(DomainError(mu, "mu must be finite"))
        (isfinite(beta) && beta > 0) ||
            throw(DomainError(beta, "beta must be positive and finite"))

        # Boltzmann factors relative to the ground state, each from a directly computed gap
        # (E₀ - E₁ = a + μ, E₂ - E₁ = a - μ, E₀ - E₂ = 2μ), so none of them overflows.
        a = U / 2
        if a + mu >= 0 && a - mu >= 0      # a singly occupied ground state
            l0, l1, l2 = -beta * (a + mu), zero(U), -beta * (a - mu)
        elseif mu <= 0                     # |0⟩
            l0, l1, l2 = zero(U), beta * (a + mu), 2beta * mu
        else                               # |↑↓⟩
            l0, l1, l2 = -2beta * mu, beta * (a - mu), zero(U)
        end
        w0, w1, w2 = exp(l0), exp(l1), exp(l2)
        Z = w0 + 2w1 + w2
        p0, p1, p2 = w0 / Z, w1 / Z, w2 / Z
        # p₂ - p₀ = -p₂ expm1(-2βμ) = p₀ expm1(2βμ) and p₁² - p₀p₂ = -p₁² expm1(-2βa) = p₀p₂ expm1(2βa),
        # without cancellation for small βμ or βa and without overflow.
        rho = mu >= 0 ? -p2 * expm1(-2beta * mu) : p0 * expm1(2beta * mu)
        w11 = a >= 0 ? -p1^2 * expm1(-2beta * a) : p0 * p2 * expm1(2beta * a)
        T = typeof(U)
        new{T}(U, beta, mu, a, p0, p1, p2, (w1 + w2) / Z, (w0 + w1) / Z, rho, w11,
               Dict{Tuple{Int,Int},PHSetup{T}}(), Dict{Int,Complex{T}}(), ReentrantLock())
    end
end

function Base.:(==)(a::DopedHubbardAtom, b::DopedHubbardAtom)
    (a.U, a.beta, a.mu) == (b.U, b.beta, b.mu)
end
function Base.hash(a::DopedHubbardAtom, h::UInt)
    hash((a.U, a.beta, a.mu), hash(:DopedHubbardAtom, h))
end
function Base.show(io::IO, a::DopedHubbardAtom)
    print(io, "DopedHubbardAtom(", a.U, ", ", a.beta, ", ", a.mu, ")")
end

# expm1(x)/x, = 1 at x = 0
_expm1x(x) = iszero(x) ? one(x) : expm1(x) / x

# tanh(y)/y for complex y, = 1 at y = 0
_tanhc(y) = iszero(y) ? one(y) : tanh(y) / y

"""
Coefficient ρ/(2μ + iπj/β) of the pair term (the Cauchy matrix C of the note), for an even
integer `j`; at j = 0 it is (p₂ - p₀)/(2μ), which tends to βp₀ for μ → 0.
"""
function _pair(at::DopedHubbardAtom, j::Integer)
    β, μ = at.beta, at.mu
    if iszero(j)
        return complex(β * (μ >= 0 ? at._p2 : at._p0) * _expm1x(-2β * abs(μ)))
    end
    at._rho / Complex(2μ, _freq(j, β))
end

_kodd(k::BosonicFreq) = isodd(Int(k) ÷ 2)

#=
Thermal function αₖ(x) = ρSₖ(x) - c̄, from its closed form (Lemma "thermal function")
    αₖ(x) = -h cosh(y₊)/cosh(y₋) (even k), -h sinh(y₊)/sinh(y₋) (odd k), y± = β(x ± μ)/2,
which avoids the cancellation of ρS and c̄ at low temperature. With σ± = sign(Re y±),
cosh(y) = e^{σy}(1 + e^{-2σy})/2 and sinh(y) = σe^{σy}(1 - e^{-2σy})/2, and the prefactor
h e^{σ₊y₊ - σ₋y₋} is βp₂ or βp₀ (σ₊ = σ₋ = ±1) or h e^{±βx}, with |Re x| < |μ|. It is taken from
the weights, so that α cancels against them as it should, e.g. in Kⱼ = X + λα. (Where p₀p₂ is not a
normal number, Γ is evaluated in BigFloat; see `_adaptive`.)
=#
function _α(at::DopedHubbardAtom, kodd::Bool, x)
    β, μ = at.beta, at.mu
    y₊, y₋ = β * (x + μ) / 2, β * (x - μ) / 2
    σ₊, σ₋ = real(y₊) >= 0 ? 1 : -1, real(y₋) >= 0 ? 1 : -1
    pre = β * (σ₊ == σ₋ ? (σ₊ > 0 ? at._p2 : at._p0) : √(at._p0 * at._p2) * exp(σ₊ * β * x))
    if kodd
        -σ₊ * σ₋ * pre * expm1(-2σ₊ * y₊) / expm1(-2σ₋ * y₋)
    else
        -pre * (1 + exp(-2σ₊ * y₊)) / (1 + exp(-2σ₋ * y₋))
    end
end

# x = μ + iν for a fermionic index
_x(at::DopedHubbardAtom, n::FermionicFreq) = Complex(at.mu, _freq(Int(n), at.beta))

# z = x + w/2 = μ + i(ν + ω/2)
function _z(at::DopedHubbardAtom, n::FermionicFreq, k::BosonicFreq)
    Complex(at.mu, _freq(Int(n) + Int(k) ÷ 2, at.beta))
end

function G(at::DopedHubbardAtom, n::FermionicFreq)
    x = _x(at, n)
    a = at._a
    at._qm / (x + a) + at._qp / (x - a)
end

χ₀(::PHChannel, at::DopedHubbardAtom, (n, k)::FermiBose) = -G(at, n) * G(at, n + k)
χ₀(::PPChannel, at::DopedHubbardAtom, (n, k)::FermiBose) = -G(at, n) * G(at, -n - k) / 2

function χ₀(r::SpinChannel, at::DopedHubbardAtom, (n, n′, k)::FermiFermiBose)
    at.beta * δ(n, n′) * χ₀(r, at, (n, k))
end

#=
Particle-hole channels. With x and x + w (= μ + iν_{n+k}) the one-particle factors are
Q₋(z) = (x - a)(x + w - a), Q₊(z) = (x + a)(x + w + a), P = Q₊Q₋, G₂(z) = (x + aρ)(x + w + aρ),
and the bubble is B = -β G₂/P.
=#
function _ph_factors(at::DopedHubbardAtom, n::FermionicFreq, k::BosonicFreq)
    a, ρ = at._a, at._rho
    x, xw = _x(at, n), _x(at, n + k)
    Qm, Qp = (x - a) * (xw - a), (x + a) * (xw + a)
    (; z=_z(at, n, k), Qm, Qp, P=Qm * Qp, G2=(x + a * ρ) * (xw + a * ρ))
end

_ε(::DensityChannel) = 1
_ε(::MagneticChannel) = -1

# Connected part χ - B δₙₙ′ of χ_d, χ_m (Proposition "structure" of the note), B the bubble.
function _χconn(r::PHChannel, at::DopedHubbardAtom, (n, n′, k)::FermiFermiBose)
    T = typeof(at.U)
    β, a, ρ = at.beta, at._a, at._rho
    ε = _ε(r)
    fn, fm = _ph_factors(at, n, k), _ph_factors(at, n′, k)
    z, v = fn.z, fm.z
    q0 = a^2 + _freq(Int(k), β)^2 / 4
    C = _pair(at, Int(n) + Int(n′) + Int(k))
    # ε𝒩 (Lemma "compact input"), with 4a²(1 - ρ) = 8a²q₋
    # divided early, so that no intermediate exceeds degree four in the energies (Float32 range)
    res = ε * ((4a^2 * C - 2a) / fn.Qm / fm.Qm +
               (8a^2 * at._qm * (z * v + q0) / fn.P) * ((z + v + 2a) / fm.P))
    if iszero(k)
        # ζᵣ J(z) J(v), ζᵣ = βa²[1 - ρ² + 4ε(p₀p₂ - p₁²)] with 1 - ρ² = 4q₊q₋
        ζ = β * a^2 * 4 * (at._qp * at._qm - ε * at._w11)
        res += ζ / (z^2 - a^2) / (v^2 - a^2)
    end
    if n == n′
        # χ - B on the diagonal: -β(H - G₂)/P with H - G₂ = Δᵣ = a²(1 - ρ² + 4εp₁)
        res -= β * _Δ(r, at) / fn.P
    end
    Complex{T}(res)
end

# Δᵣ = H(z) - G₂(z): 4a²(2p₁ - w₁₁) (d), -4a²w₁₁ (m)
_Δ(::DensityChannel, at::DopedHubbardAtom) = 4at._a^2 * (2at._p1 - at._w11)
_Δ(::MagneticChannel, at::DopedHubbardAtom) = -4at._a^2 * at._w11

# Eßl et al., Eq. (D3), in the compact form of the note
function χ(r::SpinChannel, at::DopedHubbardAtom, w::FermiFermiBose)
    _χconn(r, at, w) + _η(r) * χ₀(r, at, w)
end

_η(::SpinChannel) = 1
_η(::SingletChannel) = -1

# from the connected part of χ directly
function F(r::SpinChannel, at::DopedHubbardAtom, (n, n′, k)::FermiFermiBose)
    β = at.beta
    B, B′ = β * χ₀(r, at, (n, k)), β * χ₀(r, at, (n′, k))
    -(β * (_χconn(r, at, (n, n′, k)) / B)) * (β / B′)
end

#=
Particle-particle channels (Proposition in "Proofs for the singlet and triplet channels" of the note). With b = 2μ - w and x̄ = b - x = μ + iν_{n̄},
n̄ = -n - k - 1, t = x x̄, 𝒫(t) = (x - a)(x + a)(x̄ - a)(x̄ + a), 𝒢(t) = (x + aρ)(x̄ + aρ); the bubble
is B = -β𝒢/(2𝒫).
=#
function _pp_factors(at::DopedHubbardAtom, n::FermionicFreq, k::BosonicFreq)
    a, ρ = at._a, at._rho
    x, x̄ = _x(at, n), _x(at, -n - k)
    𝒫 = (x - a) * (x + a) * (x̄ - a) * (x̄ + a)
    (; x, x̄, t=x * x̄, 𝒫, 𝒢=(x + a * ρ) * (x̄ + a * ρ))
end

# Dₛ = a²(ρ² - 1 + 8p₁) = 4a²(p₁ + w₁₁), so that t + Aₛ = 𝒢(t) - Dₛ
_Ds(at::DopedHubbardAtom) = 4at._a^2 * (at._p1 + at._w11)

function _χconn(::TripletChannel, at::DopedHubbardAtom, (n, n′, k)::FermiFermiBose)
    T = typeof(at.U)
    D = δ(n, n′) - δ(n, -n′ - k)
    iszero(D) && return zero(Complex{T})
    f = _pp_factors(at, n, k)
    # (L_t - B)(δ - δ̄)/2 with L_t - B = -βa²(1 - ρ²)/(2𝒫)
    Complex{T}(-at.beta * at._a^2 * at._qp * at._qm * D / f.𝒫)
end

function _χconn(::SingletChannel, at::DopedHubbardAtom, (n, n′, k)::FermiFermiBose)
    T = typeof(at.U)
    a = at._a
    fn, fm = _pp_factors(at, n, k), _pp_factors(at, n′, k)
    b = Complex(2at.mu, -_freq(Int(k), at.beta))
    c = _pair(at, -Int(k))  # ρ/b
    # χ_s,reg, with (t - t₋)/𝒫 = 1/((x - a)(x̄ - a)) and 1 - ρ = 2q₋
    res = a * (2a * c - 1) / ((fn.x - a) * (fn.x̄ - a)) / ((fm.x - a) * (fm.x̄ - a)) +
          (2a^2 * at._qm * (fn.t + fm.t + 2a^2) / fn.𝒫) * ((b + 2a) / fm.𝒫)
    D = δ(n, n′) + δ(n, -n′ - k)
    # (L_s + B)(δ + δ̄)/2 with L_s + B = -βDₛ/(2𝒫)
    iszero(D) || (res -= at.beta * _Ds(at) * D / (4fn.𝒫))
    Complex{T}(res)
end

# Γ from the closed forms of the note: triplet (Theorem 2), singlet (Theorem 3), density and
# magnetic (Theorem 4, and its limit at mu = 0, Corollary 5)
function Γ(::TripletChannel, at::DopedHubbardAtom, (n, n′, k)::FermiFermiBose)
    T = typeof(at.U)
    D = δ(n, n′) - δ(n, -n′ - k)
    iszero(D) && return zero(Complex{T})
    a = at._a
    f = _pp_factors(at, n, k)
    # βa²(1 - ρ²)𝒫/(𝒢(t + aρb + a²)) with t + aρb + a² = 𝒢 + a²(1 - ρ²), 1 - ρ² = 4q₊q₋
    g = 4a^2 * at._qp * at._qm
    Complex{T}(at.beta * g * (f.𝒫 / f.𝒢) / (f.𝒢 + g) * D)
end

function Γ(r::Union{PHChannel,SingletChannel}, at::DopedHubbardAtom, w::FermiFermiBose)
    val, amp = _Γraw(r, at, w)
    amp <= 2^16 ? val : _Γbig(r, at, w, log2(amp) / 2)
end

#=
Near the removable singularities of the closed forms (Theorem 3 at t + Aₛ = 0, Theorem 4 at lattice zeros
of e, see the note), single entries are sums of large terms that cancel; `_Γraw` returns the squared
factor by which this amplifies rounding errors, and entries that would lose more than 8 bits are
re-evaluated in high precision. Only the entries of one mirror orbit are
affected, at isolated parameters.
=#
function _Γbig(r, at::DopedHubbardAtom{T}, w, loss) where {T}
    prec = min(precision(T) + 32 + _bits(3loss), 1 << 14)
    val = _withprecision(prec) do
        _Γraw(r, DopedHubbardAtom(BigFloat(at.U), BigFloat(at.beta), BigFloat(at.mu)), w)[1]
    end
    Complex{T}(val)
end

function _Γraw(::SingletChannel, at::DopedHubbardAtom, (n, n′, k)::FermiFermiBose)
    T = typeof(at.U)
    β, a, ρ = at.beta, at._a, at._rho
    iszero(a) && return zero(Complex{T}), zero(T)
    fn, fm = _pp_factors(at, n, k), _pp_factors(at, n′, k)
    Ds = _Ds(at)
    tAn, tAm = fn.𝒢 - Ds, fm.𝒢 - Ds  # t + Aₛ
    c = _pp_setup(at, k)
    res = 4a - 4a * (c / tAn) / tAm
    D = δ(n, n′) + δ(n, -n′ - k)
    iszero(D) || (res += β * Ds * (fn.𝒫 / fn.𝒢) / tAn * D)
    # c/(tAn tAm) has a double pole where n′ ∈ {n, n̄}
    Complex{T}(res), _amp(tAn, fn.𝒢, Ds) * _amp(tAm, fm.𝒢, Ds)
end

#=
Setup of Theorem 3 for one transfer: c = 𝒫(-Aₛ)/Kₛ, with 𝒫(-Aₛ) = 4a²(4ap₁ - bq₊)(4ap₁ + bq₋) and
Kₛ = 1 + (2a/β)𝒯ₖ(-Aₛ), 𝒯ₖ(-Aₛ) = (β/2ξ) tanh(βξ/2) (coth for odd k), ξ² = b²/4 + Aₛ. Thus
ξKₛ = ξ + a tanh(y), y = βξ/2, which cancels if ξ ≈ -a and tanh y ≈ 1. There ξKₛ = (ξ + a) - a(1 - tanh y)
with 1 - tanh y = 2/(1 + e^{2y}) (or (ξ + a) + 2a/expm1(2y) for coth), and ξ + a = (ξ² - a²)/(ξ - a),
ξ² - a² = b²/4 + aρb - 8a²p₁. For U < 0 at low temperature both terms are still ≈ 2|a|e^{-β|a|};
`loss` estimates the cancellation, and `_adaptive` repeats the setup in high precision if needed.
At large |k|, ξ ≈ ±b/2 and y ≈ ±βb/4 = ±(βμ - iπk)/2 is large: its rounding would cost log₂|y| bits. Then
tanh y (coth y) = tanh Y with Y = y ± iπk/2 = β(ξ ∓ b/2 ± μ)/2 and ξ ∓ b/2 = Aₛ/(ξ ± b/2).
A small 𝒫(-Aₛ) costs no accuracy in Γ, as it multiplies c.
=#
function _pp_setup(at::DopedHubbardAtom{T}, k::BosonicFreq) where {T}
    _cached(at, at._pp_setups, Int(k)) do
        _adaptive(at, (bat, _) -> _pp_setup_raw(bat, k), x -> x.loss,
                  x -> Complex{T}(x.c))
    end::Complex{T}
end

function _pp_setup_raw(at::DopedHubbardAtom{T}, k::BosonicFreq) where {T}
    β, a, ρ, p1 = at.beta, at._a, at._rho, at._p1
    b = Complex(2at.mu, -_freq(Int(k), β))
    As = a * ρ * b + a^2 * (1 - 8p1)
    𝒫A = 4a^2 * (4a * p1 - b * at._qp) * (4a * p1 + b * at._qm)
    ξ = sqrt(b^2 / 4 + As)  # Re ξ ≥ 0
    y = β * ξ / 2
    kodd = _kodd(k)
    sg = real(ξ * conj(b)) >= 0 ? 1 : -1  # ξ ≈ sg b/2 at large |k|
    Y = β * (As / (ξ + sg * b / 2) + sg * at.mu) / 2
    red = abs(Y) < abs(y)                 # then tanh Y replaces tanh y (coth y)
    if !iszero(ξ) && abs(ξ + a) < abs(ξ - a)
        num = b^2 / 4 + a * ρ * b - 8a^2 * p1
        ξa = num / (ξ - a)  # ξ + a
        corr = red ? -2a / (1 + exp(2Y)) : kodd ? 2a / expm1(2y) : -2a / (1 + exp(2y))
        Ks = (ξa + corr) / ξ
        loss = _cancel(num, b^2 / 4, a * ρ * b, 8a^2 * p1) + _cancel(ξa + corr, ξa, corr)
    else
        τ = red ? a / ξ * tanh(Y) : a * β / 2 * (kodd ? inv(y * tanh(y)) : _tanhc(y))
        Ks = 1 + τ
        loss = _cancel(Ks, 1, τ)
    end
    (; c=𝒫A / Ks, loss)
end

#=
`f()` with the BigFloat precision set to `prec`. From Julia 1.12 on, `setprecision` with a function is
task-local; before, it sets a global value, so concurrent calls are serialized (they still change the
precision seen by other code running concurrently).
=#
const _PRECISION_LOCK = ReentrantLock()
if VERSION >= v"1.12"
    _withprecision(f, prec) = setprecision(f, BigFloat, prec)
else
    _withprecision(f, prec) = lock(() -> setprecision(f, BigFloat, prec), _PRECISION_LOCK)
end

# Value cached in `dict[key]`, computed by `f()` if absent (thread-safe).
function _cached(f, at::DopedHubbardAtom, dict, key)
    lock(at._lock)
    hit = try
        get(dict, key, nothing)
    finally
        unlock(at._lock)
    end
    isnothing(hit) || return hit
    val = f()
    lock(at._lock)
    try
        dict[key] = val
    finally
        unlock(at._lock)
    end
    val
end

#=
`raw(at, T)` computes a setup in the precision of `at`, `loss(x)` its estimated loss in bits, and
`result(x)` the converted value. If the loss exceeds 6 bits, the setup is repeated in `BigFloat`
at a precision covering it; the estimate is a heuristic that can be off by several bits, hence the
generous margins. Where `_floatok(at)` fails, it is computed in `BigFloat` right away.
=#
function _adaptive(at::DopedHubbardAtom{T}, raw, loss, result) where {T}
    l = zero(T)
    if _floatok(at)
        x = raw(at, T)
        l = loss(x)
        l <= 6 && return result(x)
    end
    prec = precision(T) + 48 + min(_bits(5l / 4), 1024)
    while true
        y = _withprecision(prec) do
            raw(DopedHubbardAtom(BigFloat(at.U), BigFloat(at.beta), BigFloat(at.mu)),
                BigFloat)
        end
        ly = loss(y)
        (precision(T) + 40 + 5ly / 4 <= prec || prec >= 1 << 16) && return result(y)
        prec = max(2prec, precision(T) + 48 + min(_bits(5ly / 4), 1 << 16))
    end
end

#=
Setup of Theorem 4 for one transfer k and channel: the roots s₁, s₂ of e (in z²) and the two
separable coefficients. With G(s) = 1/((z² - s)(v² - s)), the separable part Σⱼ γⱼ gⱼ(z) gⱼ(v) is
λρ times the divided difference of Z(s)G(s) at s₁, s₂, i.e. λ[Y₁ G[s₁, s₂] + Y_d G(s₂)] with
Yⱼ = ρZⱼ, Y_d = (Y₁ - Y₂)/(s₁ - s₂) and G[s₁, s₂] = (z² + v² - s₁ - s₂)/(e(z) e(v)). Thus nearly
coinciding roots only affect the scalar Y_d.

The naive formulas cancel in several places, mostly at low temperature, where p₁ or p₀p₂ is
small. The discriminant of e and s₁ - s₂ are taken from the weights. P(u) = (s - t₊²)(s - t₋²) and
δ = u - w/2 follow from the offsets d = s - t² of the roots from t² = t₊², t₋², a², w²/4 (t± = ±a + w/2
the zeros of P): d² + 2(t² + m)d + e(t²) = 0, with t² + m and e(t²) from the weights without
cancellation. Sums like tanh then take the small δ instead of u: Sₖ(u) = S₀(δ), αₖ(u) = α₀(δ).

Each Yⱼ avoids the cancellations of Kⱼ = A + λρS and Nⱼ = B - λρS (S = Sₖ(uⱼ)) at small ρ: with
X = βH(-u), A = X - λc̄ and B = X + λc̄ satisfy AB = -ρM, M = 4aβuX + λ²β²ρ/4, at a root (proof of
Corollary "half filling" of the note), so the smaller of A, B is computed as -ρM over the larger
one. Kⱼ and Nⱼ are also evaluated as X ± λα with the closed form of α, Kⱼ/ρ also from the offset
of uⱼ from ±εa (small ρ at low temperature), and for k = 0, r = d from the offset from the double
zero ±a of P (U < 0 at low temperature, where Kⱼ and P(uⱼ) are O(p₁²)). The form that cancels
least is used.

Some cancellations remain, e.g. in Y_d for nearly coinciding roots and in the magnetic static case
(`_Y_atomic`). `_ph_setup_raw` estimates the bits they cost (`loss`); if this exceeds 6 bits,
the setup is repeated in `BigFloat` at a precision covering the loss. Only these few scalars are
computed in higher precision (once per transfer, as they are cached); the evaluation of each
element is not.
=#
function _ph_setup(at::DopedHubbardAtom{T}, k::BosonicFreq, ε::Int) where {T}
    _cached(at, at._setups, (Int(k), ε)) do
        _adaptive(at, (bat, _) -> _ph_setup_exact(bat, k, ε), x -> x.loss,
                  x -> PHSetup{T}(x.δ1, x.δ2, x.Y1, x.Yd))
    end::PHSetup{T}
end

# _ph_setup_raw, with exactly coinciding roots (only possible at constructed parameters) avoided
# by shifting μ: Γ is continuous in μ there.
function _ph_setup_exact(at::DopedHubbardAtom{T}, k::BosonicFreq, ε::Int) where {T}
    x = _ph_setup_raw(at, k, ε)
    iszero(x.Δs) || return x
    δμ = max(abs(at.mu), abs(at._a)) * T(2)^(-precision(T) ÷ 2)
    _ph_setup_raw(DopedHubbardAtom(at.U, at.beta, at.mu + δμ), k, ε)
end

_bits(loss) = isfinite(loss) ? ceil(Int, loss) : 1 << 16

#=
Whether the setups can be computed in the type of `at`: not if a weight is so small that products
of two weights are not normal numbers (e.g. p₀p₂ for βU/2 ≳ 350 in Float64, or p₁ = 0 by
underflow). Terms of that order decide quotients there (e.g. Kₛ and 𝒫(-Aₛ) are both O(p₁²)), and
the loss estimates cannot see underflow; the exponent range of BigFloat keeps them.
=#
function _floatok(at::DopedHubbardAtom{T}) where {T}
    min(at._p0, at._p1, at._p2) >= sqrt(floatmin(T)) * T(2)^32
end

# tanh(B + Δ) - tanh(B) without cancellation for small Δ. For |Re B| ≥ 1/2 and B, B + Δ with real parts
# of the same sign (≥ 0, else use oddness) it is -2e^{-2B} expm1(-2Δ)/((1 + e^{-2(B + Δ)})(1 + e^{-2B})),
# which is also accurate when both tanh saturate; otherwise tanh Δ (1 - tanh²B)/(1 + tanh B tanh Δ).
function _tanhdiff(B, Δ)
    A = B + Δ
    σ = real(B) >= 0 ? 1 : -1
    if abs(real(B)) >= 1 / 2 && (real(A) >= 0 ? 1 : -1) == σ
        A, B, Δ = σ * A, σ * B, σ * Δ
        return -2σ * exp(-2B) * expm1(-2Δ) / ((1 + exp(-2A)) * (1 + exp(-2B)))
    end
    abs(real(Δ)) >= 1 / 2 && return tanh(A) - tanh(B)
    tB, tΔ = tanh(B), tanh(Δ)
    tΔ * (1 - tB^2) / (1 + tB * tΔ)
end

# the (value, loss) pair with the smaller loss; non-finite values never win
_better(x, y) = !isfinite(y[1]) || (isfinite(x[1]) && x[2] <= y[2]) ? x : y

# log2 of the cancellation in a sum: |terms|/|result| (≥ 0, Inf if only the result vanishes)
function _cancel(result, terms...)
    s = sum(abs, terms)
    iszero(s) ? zero(s) : log2(max(1, s / abs(result)))
end

# the roots -f ± ζ of d² + 2fd + R = 0, the smaller one from the product R
function _offsets(f, R, ζ)
    d1, d2 = -f + ζ, -f - ζ
    abs(d1) >= abs(d2) ? (iszero(d1) || (d2 = R / d1)) : (d1 = R / d2)
    d1, d2
end

function _ph_setup_raw(at::DopedHubbardAtom{T}, k::BosonicFreq, ε::Int) where {T}
    β, a, ρ, μ = at.beta, at._a, at._rho, at.mu
    p0, p1, p2, qp, qm = at._p0, at._p1, at._p2, at._qp, at._qm
    if ε == -1 && iszero(k)
        # magnetic, ω = 0: s₁ = a² is atomic (P(a) = 0), so Y₁ = 0 and only s₂ enters
        Y2, d2, u2, loss = _Y_atomic(at)
        return (; s1=complex(a^2), s2=complex(a^2 + d2), δ1=complex(a), δ2=u2, Y1=zero(Y2),
                Yd=Y2 / d2, loss, Δs=complex(-d2))
    end
    λ = 4ε * a^2
    c̄ = β * (p0 + p2) / 2
    ω = _freq(Int(k), β)
    w = Complex(zero(ω), ω)
    q1 = a^2 + ω^2 / 4 + 4ε * a^2 * p1
    Q = qp * qm
    κε = ε == 1 ? p1 : p0 * p1 + 2p0 * p2 + p1 * p2  # q₊q₋ + εw₁₁
    # e(s) = s² + 2ms + e₀, with discriminant m² - e₀ = 4a²D and D from the weights (as
    # p₀ + 2p₁ + p₂ = 1), without the cancellation of m² - e₀ at small p₁ or p₀p₂. The roots are
    # s = -m ± ζ and σ = s + q₁ = 2a²ρ² ± ζ; for real ζ the smaller ones follow from the products
    # s₁s₂ = e₀ and σ₁σ₂ = 4a²(ρ²q₁ - 4a²p₀p₂), which can cancel: l1, l2 are the bits lost.
    m = q1 - 2a^2 * ρ^2
    h4 = 4a^2 * √(p0 * p2)
    e0 = (q1 - h4) * (q1 + h4)
    D = 4a^2 * (4p0 * p2 * Q + ρ^2 * p1 * (p1 - 1 - ε)) - (ρ * ω)^2 / 4
    ζ = D >= 0 ? complex(2abs(a) * √D) : Complex(zero(D), 2abs(a) * √(-D))
    s1, s2 = -m + ζ, -m - ζ
    σ1, σ2 = 2a^2 * ρ^2 + ζ, 2a^2 * ρ^2 - ζ
    l1 = l2 = zero(D)
    if D >= 0
        le = _cancel(q1 - h4, q1, h4) + _cancel(q1 + h4, q1, h4)
        if abs(s1) < abs(s2)
            s1, l1 = e0 / s2, l1 + le
        elseif !iszero(s1)
            s2, l2 = e0 / s1, l2 + le
        end
        σp = 4a^2 * (ρ^2 * q1 - 4a^2 * p0 * p2)
        lp = _cancel(σp, 4a^2 * ρ^2 * q1, 16a^4 * p0 * p2)
        if abs(σ1) < abs(σ2)
            σ1, l1 = σp / σ2, l1 + lp
        elseif !iszero(σ1)
            σ2, l2 = σp / σ1, l2 + lp
        end
    end
    # offsets d = s - t² of both roots from t² = t₊², t₋², a², w²/4, with t² + m = h ± aw, h + ω²/4,
    # a²(1 + 4εp₁ - 2ρ²) and e(t₊²) = 8a²(w + c)(aκε + q₊q₋w/2), e(t₋²) = 8a²(w - c)(q₊q₋w/2 - aκε),
    # c = 2(1 + ε)a, e(a²) = (ω²/4 + 4(1 + ε)a²p₁)(ω²/4 + 4a²(1 - (1 - ε)p₁)),
    # e(w²/4) = a⁴(1 + 4εp₁)² - 16a⁴p₀p₂ + a²ρ²ω²
    tp, tm = a + w / 2, -a + w / 2
    h = 4a^2 * (ε == 1 ? 2Q + p1 : κε)  # 4a²(2q₊q₋ + εp₁)
    c = 2(1 + ε) * a
    dp1, dp2 = _offsets(h + a * w, 8a^2 * (w + c) * (a * κε + Q * w / 2), ζ)
    dm1, dm2 = _offsets(h - a * w, 8a^2 * (w - c) * (Q * w / 2 - a * κε), ζ)
    da1, da2 = _offsets(h + ω^2 / 4,
                        (ω^2 / 4 + 2c * a * p1) *
                        (ω^2 / 4 + 4a^2 * (ε == 1 ? one(p1) : p0 + p2)), ζ)
    Rw1, Rw2 = 1 + 4ε * p1 - 4√(p0 * p2), 1 + 4ε * p1 + 4√(p0 * p2)
    Rw = a^4 * Rw1 * Rw2 + (a * ρ * ω)^2
    dw1, dw2 = _offsets(a^2 * (1 + 4ε * p1 - 2ρ^2), Rw, ζ)
    lw = _cancel(Rw, a^4 * Rw1 * Rw2, (a * ρ * ω)^2) +
         _cancel(Rw1, 1 + 4ε * p1, 4√(p0 * p2)) +
         _cancel(Rw2, 1 + 4ε * p1, 4√(p0 * p2))
    # Y = ρuPN/K at the root s = u² (σ = s + q₁, offsets dp, dm, da, dw, l bits lost in s and σ),
    # its loss, and δ = u - w/2.
    # P = Q₊Q₋ is the product of the offsets from t₊², t₋². X = βH(-u) = β(σ - 2aρu) and
    # X₊ = βH(u) = β(σ + 2aρu) satisfy XX₊ = λ²h² = 16a⁴β²p₀p₂, so the smaller one is computed from
    # the larger.
    function Y(s, σ, dp, dm, da, dw, l)
        u = sqrt(s)
        (iszero(ω) ? real(u) < 0 : ω * imag(u) < 0) && (u = -u)  # u near t₊ or t₋
        # u - t = d/(u + t) for t = t±, and δ = ±a + (u - t±) from the nearer zero or, at large |ω|
        # where δ can be ≪ a, δ = (s - w²/4)/(u + w/2); with the bits lost in δ
        ηp = abs(u + tp) >= abs(u - tp) ? dp / (u + tp) : u - tp
        ηm = abs(u + tm) >= abs(u - tm) ? dm / (u + tm) : u - tm
        σt, η = abs(ηp) <= abs(ηm) ? (1, ηp) : (-1, ηm)  # the nearer t = σt a + w/2
        if abs(ω) > 4abs(a)
            δu = dw / (u + w / 2)
            l += lw
        else
            δu = σt * a + η
            l += _cancel(δu, a, η)
        end
        P = dp * dm
        y = β * (δu - μ) / 2
        Su = -β / 2 * tanh(y)  # Sₖ(u) = S₀(δu), the Matsubara sum PV Σₙ 1/(zₙ - u)
        λα = λ * _α(at, false, δu)
        X, X₊ = β * (σ - 2a * ρ * u), β * (σ + 2a * ρ * u)
        abs(X) < abs(X₊) && (X = 16a^4 * β^2 * p0 * p2 / X₊)
        A, B = X - λ * c̄, X + λ * c̄
        M = 4a * β * u * X + λ^2 * β^2 * ρ / 4
        lM = _cancel(M, 4a * β * u * X, λ^2 * β^2 * ρ / 4)
        # K/ρ = λS - M/B = -2aβ(u + εaτ) + 4a²β²ρ(s - a²)/(B + 2aβρu), τ = tanh y. With the nearer
        # u★ = ∓εa, u + εaτ = (u - u★) + εa(τ ∓ 1) and u - u★ = (s - a²)/(u + u★), so nothing cancels
        # as τ → ±1, at low temperature.
        u★ = abs(u + ε * a) <= abs(u - ε * a) ? -ε * a : ε * a
        th = u★ == -ε * a ? 2ε * a^2 * β * 2 / (1 + exp(2y)) :
             -2ε * a^2 * β * 2 / (1 + exp(-2y))
        Bs = B - 2a * β * ρ * u★
        Bu = B + 2a * β * ρ * u
        t1 = -2a * β * da / (u + u★) * (Bs / Bu)
        κa = (t1 + th,
              _cancel(t1 + th, t1, th) + _cancel(Bs, B, 2a * β * ρ * u★) +
              _cancel(Bu, B, 2a * β * ρ * u))
        if ε == 1 && iszero(ω)
            # P has double zeros t = σt a. With Y = (u - t)/t, from e(u²) = 0 and the weights,
            # K - λρ(S(u) - S(t)) = βa²σtρY²(Y² + 4q₋Y + 4)/(2q(Y² + 2Y + 4)), q = q₊ for t = a, q₋ for
            # t = -a: both terms are O(p₁²) as p₁ → 0 (U < 0), like P(u).
            qt, qo = σt == 1 ? (qp, qm) : (qm, qp)
            Yt = η / (σt * a)
            f1, f2 = Yt^2 + 4qo * Yt + 4, Yt^2 + 2Yt + 4
            g = β * a^2 * σt * Yt^2 * f1 / (2qt * f2)
            ΔS = -β / 2 * _tanhdiff(complex(β * (σt * a - μ) / 2), β * η / 2)
            κg = (g + λ * ΔS,
                  _cancel(g + λ * ΔS, g, λ * ΔS) + _cancel(f1, Yt^2, 4qo * Yt, 4) +
                  _cancel(f2, Yt^2, 2Yt, 4))
            κa = _better(κa, κg)
        end
        if abs(A) >= abs(B)
            K, lK = _better((X + λα, _cancel(X + λα, X, λα)), (ρ * κa[1], κa[2]))
            Nρ = -M / A - λ * Su
            N, lN = _better((ρ * Nρ, _cancel(Nρ, M / A, λ * Su) + lM),
                            (X - λα, _cancel(X - λα, X, λα)))
            return u * P * (N / K) * ρ, lK + lN + l, δu
        end
        # A = -ρM/B is small: Y = uPN/κ with κ = K/ρ = -M/B + λS
        κ, lκ = _better((-M / B + λ * Su, _cancel(-M / B + λ * Su, M / B, λ * Su) + lM), κa)
        iszero(ρ) || ((κ, lκ) = _better((κ, lκ), ((X + λα) / ρ, _cancel(X + λα, X, λα))))
        N, lN = _better((B - λ * ρ * Su, _cancel(B - λ * ρ * Su, B, λ * ρ * Su)),
                        (X - λα, _cancel(X - λα, X, λα)))
        u * P * (N / κ), lκ + lN + l, δu
    end
    (Y1, l1, δ1), (Y2, l2, δ2) = Y(s1, σ1, dp1, dm1, da1, dw1, l1),
                                 Y(s2, σ2, dp2, dm2, da2, dw2, l2)
    # nearly coinciding roots cancel in Y₁ - Y₂
    (; s1, s2, δ1, δ2, Y1, Yd=(Y1 - Y2) / 2ζ, loss=max(l1, l2) + _cancel(Y1 - Y2, Y1, Y2),
     Δs=2ζ)
end

#=
Y₂ for the magnetic channel at ω = 0, where e(z) = (z² - a²)(z² - s₂) with
s₂ - a² = (e₀ - a⁴)/a² = d₂ = -8a²(p₁(p₀ + p₂) + 2p₀p₂). Here P(u₂) = d₂², and in the cold limit u₂
approaches the atomic pole a, where the terms of K₂ = X + λα cancel to O(β²p₂³). With x = d₂/a²,
u₂ = a√(1 + x), η = β(u₂ - a)/2 and t = tanh(β(a - μ)/2) = (p₁ - p₂)/q₊, one finds exactly
    K₂/ρ = 2a²β[ρ(p₀ + p₂ - ρ²)/q₊ + x²/(2(1 + √(1 + x))²)] - 4a²(S(u₂) - S(a)),
    S(u₂) - S(a) = -(β/2)[tanh(y + η) - tanh(y)],   y = β(a - μ)/2,
in which nothing cancels (p₀ + p₂ - ρ² = Var(n), and the tanh difference is taken by `_tanhdiff`).
Returns (Y₂, d₂, u₂, loss).
=#
function _Y_atomic(at::DopedHubbardAtom)
    β, a, ρ, μ = at.beta, at._a, at._rho, at.mu
    p0, p1, p2, qp, qm = at._p0, at._p1, at._p2, at._qp, at._qm
    d2 = -8a^2 * (p1 * (p0 + p2) + 2p0 * p2)
    x = d2 / a^2
    r = sqrt(complex(1 + x))
    δu = a * x / (1 + r)                       # u₂ - a
    u = a + δu
    var = 4p0 * qp^2 + 2p1 * ρ^2 + 4p2 * qm^2  # p₀ + p₂ - ρ²
    κ1 = 2a^2 * β * (ρ * var / qp + x^2 / (2(1 + r)^2))
    y = complex(β * (a - μ) / 2)
    κ2 = 2a^2 * β * _tanhdiff(y, β * δu / 2)
    κ = κ1 + κ2
    λα = -4a^2 * _α(at, false, u)
    X = β * (d2 + 4a^2 * p0 - 2a * ρ * δu)       # βH(-u₂)
    N = X - λα
    Y2 = u * d2^2 * (N / κ)
    loss = _cancel(κ, κ1, κ2) + _cancel(N, X, λα) + _cancel(X, d2, 4a^2 * p0, 2a * ρ * δu)
    Y2, d2, u, loss
end

# (|a|² + |b|²)/|r|² for r = a ± b: the squared factor by which r amplifies the rounding errors of a, b
_amp(r, a, b) = (abs2(a) + abs2(b)) / abs2(r)

# For z = zₙ: x = z - w/2 and x + w (= μ + iνₙ₊ₖ), the factors z² - sⱼ = (x - δⱼ)(x + w + δⱼ) and their
# (squared) error amplification near a lattice zero of e, where the poles cancel in Γ
function _efactors(at::DopedHubbardAtom, st::PHSetup, n::FermionicFreq, k::BosonicFreq)
    x, xw = _x(at, n), _x(at, n + k)
    x1, x2, xw1, xw2 = x - st.δ1, x - st.δ2, xw + st.δ1, xw + st.δ2
    amp = max(_amp(x1, x, st.δ1) * _amp(xw1, xw, st.δ1),
              _amp(x2, x, st.δ2) * _amp(xw2, xw, st.δ2))
    (; x, xw, e1=x1 * xw1, e2=x2 * xw2, amp)
end

function _Γraw(r::PHChannel, at::DopedHubbardAtom, (n, n′, k)::FermiFermiBose)
    T = typeof(at.U)
    β, a, ρ = at.beta, at._a, at._rho
    iszero(a) && return zero(Complex{T}), zero(T)
    ε = _ε(r)
    λ = 4ε * a^2
    st = _ph_setup(at, k, ε)
    fn = _ph_factors(at, n, k)
    z, v = fn.z, _z(at, n′, k)
    fz, fv = _efactors(at, st, n, k), _efactors(at, st, n′, k)
    ez, ev = fz.e1 * fz.e2, fv.e1 * fv.e2

    res = complex(2ε * a)
    # separable part, with z² + v² - s₁ - s₂ = (z² - s₁) + (v² - s₂)
    res += λ * ((st.Y1 / ez) * ((fz.e1 + fv.e2) / ev) + st.Yd / fz.e2 / fv.e2)
    # dense part -λρ𝔻(z, v) = -λ C(z, v) Q(z, v), C = ρ/(z + v)
    C = _pair(at, Int(n) + Int(n′) + Int(k))
    iszero(C) || (res -= λ * C * _Q(at, ε, z, v, fz, fv))
    if n == n′
        # βP[1/G₂ - H(-z)/e] = βP(Δᵣ H(-z) - 16a⁴p₀p₂)/(G₂ e), H(-z) = x(x + w) + a²(1 + 4εp₁) - 2aρz
        H = fz.x * fz.xw + a^2 * (1 + 4ε * at._p1) - 2a * ρ * z
        res += β * (fn.P / ez) * ((_Δ(r, at) * H - 16a^4 * at._p0 * at._p2) / fn.G2)
    end
    # lattice zeros of e (removable): terms ∝ 1/e(z), 1/e(v) cancel, doubly so on the diagonal
    Complex{T}(res), fz.amp * fv.amp
end

#=
Q(z, v) = (zF(z) - vF(v))/(z - v), F = P/e, and F(z) + zF′(z) for z = v. With φ(z) = z(F(z) - 1) =
zR(z²)/e(z), R = P - e = r₁z² + r₀, Q = 1 + (φ(z) - φ(v))/(z - v), which avoids the cancellation of
zF(z) - vF(v) for large frequencies. For nearby z, v (and z = v), φ = zg(z²), g = R/e, gives
Q = 1 + g(z²) + v(z + v) g[z², v²], with the divided difference g[s, t] = (r₁e(t) - R(t)e[s, t])/(e(s)e(t)),
e[s, t] = (s - s₁) + (t - s₂), so that φ(z) - φ(v) does not cancel. `fz`, `fv` are the `_efactors` of
z, v; R uses z² + q₀ = x(x + w) + a², q₀ = a² + ω²/4.
=#
function _Q(at::DopedHubbardAtom, ε::Int, z, v, fz, fv)
    a = at._a
    r1 = -4a^2 * (4at._qp * at._qm + 2ε * at._p1)
    R(z, f) = -4a^2 *
              (4at._qp * at._qm * z^2 + 2ε * at._p1 * (f.x * f.xw + a^2) + 4a^2 * at._w11)
    # divided early: no intermediate exceeds degree four in the energies (Float32 range)
    gz, gv = R(z, fz) / fz.e1 / fz.e2, R(v, fv) / fv.e1 / fv.e2
    abs(z - v) > abs(z + v) / 2 && return 1 + (z * gz - v * gv) / (z - v)
    1 + gz + v / fz.e1 * ((z + v) / fz.e2) * (r1 - gv * (fz.e1 + fv.e2))
end

function χ(::DensityChannel, at::DopedHubbardAtom, k::BosonicFreq)
    # variance of n = n↑ + n↓: Σ pᵢ(nᵢ - ⟨n⟩)², with 1 ± ρ = 2q±
    var = 4at._p0 * at._qp^2 + 2at._p1 * at._rho^2 + 4at._p2 * at._qm^2
    complex(-at.beta * δ(k) * var)
end
function χ(::MagneticChannel, at::DopedHubbardAtom, k::BosonicFreq)
    complex(-at.beta * δ(k) * 2at._p1)
end
χ(::SingletChannel, at::DopedHubbardAtom, k::BosonicFreq) = -_pair(at, -Int(k))
function χ(::TripletChannel, at::DopedHubbardAtom, k::BosonicFreq)
    # Σ over the pole pairs of G(ν) G(-ν - ω): 2Sₖ(∓a)/(b ∓ 2a) for both poles at ±a, and 2βp₁ρ/b for mixed ones
    β, μ, a = at.beta, at.mu, at._a
    w = Complex(zero(β), _freq(Int(k), β))
    function f(y)  # β tanh(y)/(4y/β - w), = (β²/4) tanh(y)/y at w = 0
        iszero(k) ? β^2 / 4 * _tanhc(y) : β * tanh(y) / (4y / β - w)
    end
    (at._qp^2 * f(complex(β * (μ - a) / 2)) + at._qm^2 * f(complex(β * (μ + a) / 2))) / β +
    2at._p1 * _pair(at, -Int(k))
end

function G₃(r::PHChannel, at::DopedHubbardAtom, (n, k)::FermiBose)
    β, a, ρ = at.beta, at._a, at._rho
    f = _ph_factors(at, n, k)
    z = f.z
    q0 = a^2 + _freq(Int(k), β)^2 / 4
    res = (z^2 + 2a * ρ * z + q0) / f.P
    if iszero(k)
        # -ζᵣ(z² - a²)/(2εaP) = -β a (2εq₊q₋ - 2w₁₁)/(z² - a²), as P = (z² - a²)² at ω = 0
        ε = _ε(r)
        res -= β * a * 2(ε * at._qp * at._qm - at._w11) / (z^2 - a^2)
    end
    res
end
function G₃(::SingletChannel, at::DopedHubbardAtom, (n, k)::FermiBose)
    a = at._a
    f = _pp_factors(at, n, k)
    c = _pair(at, -Int(k))
    (-(f.𝒢 - _Ds(at)) / 2 + a * (f.t + a^2) * c + a^2 * (at._p0 + at._p2 - 2at._p1)) / f.𝒫
end
G₃(::TripletChannel, at::DopedHubbardAtom, (n, k)::FermiBose) = zero(Complex{typeof(at.U)})
