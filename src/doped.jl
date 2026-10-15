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
    q1::T             # q₁ = a² + ω²/4 + 4εa²p₁
    s1::Complex{T}    # roots of the quartic e in z²
    s2::Complex{T}
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
    _logh::T          # log(h), h = β√(p₀p₂)

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
        logh = log(beta) + (l0 + l2) / 2 - log(Z)
        T = typeof(U)
        new{T}(U, beta, mu, a, p0, p1, p2, (w1 + w2) / Z, (w0 + w1) / Z, rho, w11, logh,
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

# Matsubara frequency from an integer index i: πi/β (as SparseIR.value)
_freq(i::Integer, β) = i * (π / β)

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

"Matsubara sum S_k(x) = PV Σₙ 1/(zₙ - x) on the lattice zₙ of transfer `k` (Lemma \"basic sum\")."
function _S(at::DopedHubbardAtom, kodd::Bool, x)
    y = at.beta * (x - at.mu) / 2
    -at.beta / 2 * (kodd ? coth(y) : tanh(y))
end

_kodd(k::BosonicFreq) = isodd(Int(k) ÷ 2)

#=
Thermal function αₖ(x) = ρSₖ(x) - c̄, from its closed form (Lemma "thermal function")
    αₖ(x) = -h cosh(y₊)/cosh(y₋) (even k), -h sinh(y₊)/sinh(y₋) (odd k), y± = β(x ± μ)/2,
which avoids the cancellation of ρS and c̄ at low temperature. With σ± = sign(Re y±),
cosh(y) = e^{σy}(1 + e^{-2σy})/2 and sinh(y) = σe^{σy}(1 - e^{-2σy})/2, and the prefactor
h e^{σ₊y₊ - σ₋y₋} is one of βp₂, βp₀, h e^{±βx}, evaluated from log h.
=#
function _α(at::DopedHubbardAtom, kodd::Bool, x)
    β, μ = at.beta, at.mu
    y₊, y₋ = β * (x + μ) / 2, β * (x - μ) / 2
    σ₊, σ₋ = real(y₊) >= 0 ? 1 : -1, real(y₋) >= 0 ? 1 : -1
    pre = exp(at._logh + σ₊ * y₊ - σ₋ * y₋)
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
    val, loss = _Γraw(r, at, w)
    loss <= 8 ? val : _Γbig(r, at, w, loss)
end

#=
Near the removable singularities of the closed forms (Theorem 3 at t + Aₛ = 0, Theorem 4 at lattice zeros
of e, see the note), single entries are sums of large terms that cancel; `_Γraw` returns the bits this
costs, and such entries are re-evaluated in high precision. Only the entries of one mirror orbit are
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
    loss = max(_cancel(tAn, fn.𝒢, Ds), _cancel(tAm, fm.𝒢, Ds))
    Complex{T}(res), loss
end

#=
Setup of Theorem 3 for one transfer: c = 𝒫(-Aₛ)/Kₛ, with 𝒫(-Aₛ) = 4a²(4ap₁ - bq₊)(4ap₁ + bq₋) and
Kₛ = 1 + (2a/β)𝒯ₖ(-Aₛ), 𝒯ₖ(-Aₛ) = (β/2ξ) tanh(βξ/2) (coth for odd k), ξ² = b²/4 + Aₛ. Thus
ξKₛ = ξ + a tanh(y), y = βξ/2, which cancels if ξ ≈ -a and tanh y ≈ 1. There ξKₛ = (ξ + a) - a(1 - tanh y)
with 1 - tanh y = 2/(1 + e^{2y}) (or (ξ + a) + 2a/expm1(2y) for coth), and ξ + a = (ξ² - a²)/(ξ - a),
ξ² - a² = b²/4 + aρb - 8a²p₁. For U < 0 at low temperature both terms are still ≈ 2|a|e^{-β|a|} and Kₛ is
O(βp₁²), as is 𝒫(-Aₛ); `loss` estimates the cancellation, and `_adaptive` repeats the setup in high
precision if needed.
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
    loss = _cancel(4a * p1 - b * at._qp, 4a * p1, b * at._qp) +
           _cancel(4a * p1 + b * at._qm, 4a * p1, b * at._qm)
    ξ = sqrt(b^2 / 4 + As)  # Re ξ ≥ 0
    y = β * ξ / 2
    kodd = _kodd(k)
    if !iszero(ξ) && abs(ξ + a) < abs(ξ - a)
        num = b^2 / 4 + a * ρ * b - 8a^2 * p1
        ξa = num / (ξ - a)  # ξ + a
        corr = kodd ? 2a / expm1(2y) : -2a / (1 + exp(2y))
        Ks = (ξa + corr) / ξ
        loss += _cancel(num, b^2 / 4, a * ρ * b, 8a^2 * p1) + _cancel(ξa + corr, ξa, corr)
    else
        τ = a * β / 2 * (kodd ? inv(y * tanh(y)) : _tanhc(y))
        Ks = 1 + τ
        loss += _cancel(Ks, 1, τ)
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
`result(x)` the converted value. If the loss exceeds 4 bits, the setup is repeated in `BigFloat`
at a precision covering it; the estimate is a heuristic that can be off by ~20 bits either way,
hence the generous margins.
=#
function _adaptive(at::DopedHubbardAtom{T}, raw, loss, result) where {T}
    x = raw(at, T)
    l = loss(x)
    l <= 4 && return result(x)
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

Each Yⱼ avoids the cancellations of Kⱼ = A + λρS and Nⱼ = B - λρS (S = Sₖ(uⱼ)) at small ρ: with
X = βH(-u), A = X - λc̄ and B = X + λc̄ satisfy AB = -ρM, M = 4aβuX + λ²β²ρ/4, at a root (proof of
Corollary "half filling" of the note), so the smaller of A, B is computed as -ρM over the larger
one. Kⱼ and Nⱼ are also evaluated as X ± λα with the closed form of α (better at low temperature),
and the form that cancels less is used.

Some cancellations remain, e.g. in Kⱼ for the cold magnetic static case, where uⱼ approaches the
atomic pole a and the leading thermal terms of Kⱼ cancel identically, and in Y_d for nearly
coinciding roots. `_ph_setup_raw` estimates the bits they cost (`loss`); if this exceeds 4 bits,
the setup is repeated in `BigFloat` at a precision covering the loss. Only these few scalars are
computed in higher precision (once per transfer, as they are cached); the evaluation of each
element is not.
=#
function _ph_setup(at::DopedHubbardAtom{T}, k::BosonicFreq, ε::Int) where {T}
    _cached(at, at._setups, (Int(k), ε)) do
        _adaptive(at, (bat, _) -> _ph_setup_exact(bat, k, ε), x -> x.loss,
                  x -> PHSetup{T}(x.q1, x.s1, x.s2, x.Y1, x.Yd))
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

# the (value, loss) pair with the smaller loss
_better(x, y) = x[2] <= y[2] ? x : y

# log2 of the cancellation in a sum: |terms|/|result| (≥ 0, Inf if the result vanishes)
_cancel(result, terms...) = log2(max(1, sum(abs, terms) / abs(result)))

function _ph_setup_raw(at::DopedHubbardAtom{T}, k::BosonicFreq, ε::Int) where {T}
    β, a, ρ = at.beta, at._a, at._rho
    p0, p1, p2 = at._p0, at._p1, at._p2
    λ = 4ε * a^2
    c̄ = β * (p0 + p2) / 2
    ω = _freq(Int(k), β)
    q0 = a^2 + ω^2 / 4
    q1 = q0 + 4ε * a^2 * p1
    e0 = (q1 - 4a^2 * √(p0 * p2)) * (q1 + 4a^2 * √(p0 * p2))  # q₁² - 16a⁴p₀p₂
    kodd = _kodd(k)
    # Y at the root s = u², given σ = s + q₁ accurately; P(u) = Q₊(u)Q₋(u) as a product of four linear
    # factors u ± a ± iω/2. X = βH(-u) = β(σ - 2aρu) and X₊ = βH(u) = β(σ + 2aρu) satisfy
    # XX₊ = λ²h² = 16a⁴β²p₀p₂, so the smaller one is computed from the larger.
    # Returns (Y, estimated loss in bits).
    function Y(s, σ)
        u = sqrt(complex(s))
        iω2 = Complex(zero(ω), ω / 2)
        f = (u + a + iω2, u + a - iω2, u - a + iω2, u - a - iω2)
        Pu = prod(f)
        lossP = sum(_cancel(fi, u, a, ω / 2) for fi in f)
        X, X₊ = β * (σ - 2a * ρ * u), β * (σ + 2a * ρ * u)
        abs(X) < abs(X₊) && (X = 16a^4 * β^2 * p0 * p2 / X₊)
        A, B = X - λ * c̄, X + λ * c̄
        M = 4a * β * u * X + λ^2 * β^2 * ρ / 4
        Su = _S(at, kodd, u)
        λα = λ * _α(at, kodd, u)
        loss = lossP + _cancel(M, 4a * β * u * X, λ^2 * β^2 * ρ / 4)
        # K = X + λα and N = X - λα, each either directly or from the identity (whichever cancels less)
        # B = -ρM/A is small: Y = uPρN/K with N = B - λρS = ρ(-M/A - λS)
        if abs(A) >= abs(B)
            K, lK = _better((A + λ * ρ * Su, _cancel(A + λ * ρ * Su, A, λ * ρ * Su)),
                            (X + λα, _cancel(X + λα, X, λα)))
            Nρ = -M / A - λ * Su
            N, lN = _better((ρ * Nρ, _cancel(Nρ, M / A, λ * Su)),
                            (X - λα, _cancel(X - λα, X, λα)))
            return u * Pu * (N / K) * ρ, loss + lK + lN
        else                 # A = -ρM/B is small: Y = uPN/κ with κ = K/ρ = -M/B + λS
            κ, lκ = -M / B + λ * Su, _cancel(-M / B + λ * Su, M / B, λ * Su)
            if !iszero(ρ)
                K = X + λα
                κ, lκ = _better((κ, lκ), (K / ρ, _cancel(K, X, λα)))
            end
            N, lN = _better((B - λ * ρ * Su, _cancel(B - λ * ρ * Su, B, λ * ρ * Su)),
                            (X - λα, _cancel(X - λα, X, λα)))
            return u * Pu * (N / κ), loss + lκ + lN
        end
    end
    if ε == -1 && iszero(k)
        # magnetic, ω = 0: s₁ = a² is atomic (P(a) = 0), so Y₁ = 0 and only s₂ enters
        Y2, d2, loss = _Y_atomic(at)
        s1, s2 = complex(a^2), complex(a^2 + d2)
        return (; q1, s1, s2, Y1=zero(Y2), Yd=Y2 / d2, loss, Δs=s1 - s2)
    end
    # s² + 2m s + e₀ = 0, m = q₁ - 2a²ρ², with discriminant m² - e₀ = 4a²D; the roots are
    # s = -m ± r and σ = s + q₁ = 2a²ρ² ± r, r = 2|a|√D. For real r, the smaller root and the smaller σ
    # follow from the products s₁s₂ = e₀ and σ₁σ₂ = 4a²(ρ²q₁ - 4a²p₀p₂).
    m = q1 - 2a^2 * ρ^2
    D = a^2 * ρ^4 - ρ^2 * q1 + 4a^2 * p0 * p2
    if D >= 0
        r = 2abs(a) * √D
        sg = m >= 0 ? -1 : 1          # s₁ = -m + sg r is the larger root in magnitude
        s1 = complex(-m + sg * r)
        s2 = iszero(s1) ? s1 : e0 / s1
        σa, σb = 2a^2 * ρ^2 + r, 2a^2 * ρ^2 - r
        σp = 4a^2 * (ρ^2 * q1 - 4a^2 * p0 * p2)
        abs(σa) >= abs(σb) ? (σb = iszero(σa) ? σb : σp / σa) : (σa = σp / σb)
        σ1, σ2 = complex(sg > 0 ? σa : σb), complex(sg > 0 ? σb : σa)
        Δs = complex(2sg * r)
    else
        r = 2abs(a) * √(-D)
        s1, s2 = Complex(-m, r), Complex(-m, -r)
        σ1, σ2 = Complex(2a^2 * ρ^2, r), Complex(2a^2 * ρ^2, -r)
        Δs = Complex(zero(r), 2r)
    end
    (Y1, l1), (Y2, l2) = Y(s1, σ1), Y(s2, σ2)
    # D carries the error of the roots, Y₁ - Y₂ that of the divided difference
    loss = max(l1, l2) + _cancel(D, a^2 * ρ^4, ρ^2 * q1, 4a^2 * p0 * p2) / 2 +
           _cancel(Y1 - Y2, Y1, Y2)
    (; q1, s1, s2, Y1, Yd=(Y1 - Y2) / Δs, loss, Δs)
end

#=
Y₂ for the magnetic channel at ω = 0, where e(z) = (z² - a²)(z² - s₂) with
s₂ - a² = (e₀ - a⁴)/a² = d₂ = -8a²(p₁(p₀ + p₂) + 2p₀p₂). Here P(u₂) = d₂², and in the cold limit u₂
approaches the atomic pole a, where the terms of K₂ = X + λα cancel to O(β²p₂³). With x = d₂/a²,
u₂ = a√(1 + x), η = β(u₂ - a)/2 and t = tanh(β(a - μ)/2) = (p₁ - p₂)/q₊, one finds exactly
    K₂/ρ = 2a²β[ρ(p₀ + p₂ - ρ²)/q₊ + x²/(2(1 + √(1 + x))²)] - 4a²(S(u₂) - S(a)),
    S(u₂) - S(a) = -(β/2)[tanh(y + η) - tanh(y)],   y = β(a - μ)/2,
in which nothing cancels (p₀ + p₂ - ρ² = Var(n), and the tanh difference is taken by `_tanhdiff`).
Returns (Y₂, d₂, loss).
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
    Y2, d2, _cancel(κ, κ1, κ2) + _cancel(N, X, λα) + _cancel(X, d2, 4a^2 * p0, 2a * ρ * δu)
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
    z², v² = z^2, v^2
    ez, ev = (z² - st.s1) * (z² - st.s2), (v² - st.s1) * (v² - st.s2)

    res = complex(2ε * a)
    # separable part
    res += λ * ((st.Y1 / ez) * ((z² + v² - st.s1 - st.s2) / ev) +
                st.Yd / (z² - st.s2) / (v² - st.s2))
    # dense part -λρ𝔻(z, v) = -λ C(z, v) Q(z, v), C = ρ/(z + v)
    C = _pair(at, Int(n) + Int(n′) + Int(k))
    iszero(C) || (res -= λ * C * _Q(at, st, k, ε, z, v, n == n′))
    if n == n′
        # βP[1/G₂ - H(-z)/e] = βP(Δᵣ H(-z) - 16a⁴p₀p₂)/(G₂ e)
        H = z² - 2a * ρ * z + st.q1
        res += β * (fn.P / ez) * ((_Δ(r, at) * H - 16a^4 * at._p0 * at._p2) / fn.G2)
    end
    # lattice zeros of e (removable): terms ∝ 1/e(z), 1/e(v) cancel
    R = abs(st.s1) + abs(st.s2) + a^2
    loss = max(_cancel(min(abs(z² - st.s1), abs(z² - st.s2)), abs(z²), R),
               _cancel(min(abs(v² - st.s1), abs(v² - st.s2)), abs(v²), R))
    Complex{T}(res), loss
end

#=
Q(z, v) = (zF(z) - vF(v))/(z - v), F = P/e, and F(z) + zF′(z) for z = v. With φ(z) = z(F(z) - 1) =
zR(z²)/e(z), R = P - e = r₁z² + r₀, Q = 1 + (φ(z) - φ(v))/(z - v), which avoids the cancellation of
zF(z) - vF(v) for large, nearby frequencies.
=#
function _Q(at::DopedHubbardAtom, st, k::BosonicFreq, ε::Int, z, v, diagonal::Bool)
    a = at._a
    q0 = a^2 + _freq(Int(k), at.beta)^2 / 4
    r1 = -4a^2 * (4at._qp * at._qm + 2ε * at._p1)
    r0 = -8ε * a^2 * at._p1 * q0 - 16a^4 * at._w11
    s1, s2 = st.s1, st.s2
    # z == v also for distinct frequencies that round to the same value
    if diagonal || z == v
        s = z^2
        es = (s - s1) * (s - s2)
        Rs = r1 * s + r0
        return 1 + (Rs + 2s * r1) / es - 2s * (Rs / es) * ((2s - s1 - s2) / es)
    end
    z², v² = z^2, v^2
    φz = z / (z² - s1) * ((r1 * z² + r0) / (z² - s2))
    φv = v / (v² - s1) * ((r1 * v² + r0) / (v² - s2))
    1 + (φz - φv) / (z - v)
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
