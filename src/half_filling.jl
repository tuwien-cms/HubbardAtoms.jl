# The half-filled atom; equation numbers refer to Thunström et al., Phys. Rev. B 98, 235107
# (2018).

"""
    HubbardAtom(U, beta)

Half-filled Hubbard atom with the Hamiltonian (Equation 1)

    H = U * (n[↑] - 1/2) * (n[↓] - 1/2)

in contact with a heat bath at temperature `1/beta`. `U` must be finite and `beta` finite
and positive. Both are promoted to a common floating-point type `T`, which is used for all
results, e.g. `HubbardAtom(big"2.0", big"10.0")` evaluates everything in `BigFloat`. All
quantities except `gf` are real; `DopedHubbardAtom(U, beta, 0)` gives the same values as
complex numbers.
"""
struct HubbardAtom{T<:AbstractFloat} <: AbstractHubbardAtom
    U::T              # Hubbard interaction
    beta::T           # inverse temperature

    _uhalf2::T        # (U/2)^2
    _p::T             # 1/(1 + exp(βU/2)), probability of an empty or doubly occupied atom
    _q::T             # 1/(1 + exp(-βU/2)) = 1 - p, probability of a singly occupied atom

    function HubbardAtom(U::Real, beta::Real)
        U, beta = float.(promote(U, beta))
        isfinite(U) || throw(DomainError(U, "U must be finite"))
        (isfinite(beta) && beta > 0) ||
            throw(DomainError(beta, "beta must be positive and finite"))

        # Both weights are computed directly, so each is accurate (and neither
        # overflows) for arbitrarily large |βU|.
        x = beta * U / 2
        new{typeof(U)}(U, beta, (U / 2)^2, 1 / (1 + exp(x)), 1 / (1 + exp(-x)))
    end
end

Base.:(==)(a::HubbardAtom, b::HubbardAtom) = (a.U, a.beta) == (b.U, b.beta)
Base.hash(a::HubbardAtom, h::UInt) = hash((a.U, a.beta), hash(:HubbardAtom, h))
Base.show(io::IO, a::HubbardAtom) = print(io, "HubbardAtom(", a.U, ", ", a.beta, ")")

# Equation 2
function G(at::HubbardAtom, n::FermionicFreq)
    U²₄ = at._uhalf2
    iν = im * _freq(n, at.beta)

    1 / (iν - U²₄ / iν)
end

# Equation 10
function χ(r::SpinChannel, at::HubbardAtom, (n, n′, k)::FermiFermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    ν = _freq(n, β)
    νω = _freq(n + k, β)  # from the indices, as adding ν and ω cancels for ν ≈ -ω
    δ₁, δ₂ = δ(n, n′), δ(n, -n′ - k)

    # a₀ (δ₁ - δ₂) + b₀ (δ₁ + δ₂) with a₀, b₀ from Equations 11a, b and 𝒜₀ = ℬ₀; both deltas hold
    # only for 2ν + ω = 0. For ν′ = -ν - ω ≠ ν only b₀ - a₀ ∝ Bᵣ² - Aᵣ² remains, which is small
    # compared to a₀ and b₀ for |ν(ν + ω)| ≫ U² and hence evaluated directly.
    ab = δ₂ == 0 ? δ₁ * (2ν * νω - A²(r, at) - B²(r, at)) :
         δ₁ == 0 ? -B²mA²(r, at) : 2(ν * νω - B²(r, at))
    χ_ab = ℬ₀(r) * β / 2 * ab / ((ν^2 + U²₄) * (νω^2 + U²₄))
    # each bᵢ is purely real or purely imaginary, so these products are real
    χ_ab +
    real(b₁(r, at, (n, k)) * b₁(r, at, (n′, k)) + b₂(r, at, (n, k)) * b₂(r, at, (n′, k)))
end

"Equation 11c"
function b₁(r::SpinChannel, at::HubbardAtom, (n, k)::FermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    U = at.U
    ν = _freq(n, β)
    νω = _freq(n + k, β)
    Cᵣʷ = C(r, at, k)
    Dᵣʷ = D(r, at, k)

    ℬ₁(r, at) * √(Complex(U * (1 - Cᵣʷ))) * (ν * νω - Dᵣʷ) / ((ν^2 + U²₄) * (νω^2 + U²₄))
end

"Equation 11d"
function b₂(r::SpinChannel, at::HubbardAtom, (n, k)::FermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    U = at.U
    ν = _freq(n, β)
    ω = _freq(k, β)
    νω = _freq(n + k, β)
    Cᵣʷ = C(r, at, k)

    ℬ₂(r, at) * √(Complex(U * U²₄)) * √(U^2 / (1 - Cᵣʷ) + ω^2) /
    ((ν^2 + U²₄) * (νω^2 + U²₄))
end

"Equation 12"
function D(r::SpinChannel, at::HubbardAtom, k::BosonicFreq)
    U²₄ = at._uhalf2
    Cᵣʷ = C(r, at, k)

    U²₄ * (1 + Cᵣʷ) / (1 - Cᵣʷ)
end

# Table I

# Aᵣ and Bᵣ only enter squared, and their squares are real.
A²(::d, at::HubbardAtom) = 3at._uhalf2
A²(::m, at::HubbardAtom) = -at._uhalf2
A²(::s, at::HubbardAtom) = zero(at.U)
A²(::t, at::HubbardAtom) = -at._uhalf2

B²(::d, at::HubbardAtom) = at._uhalf2 * (3 - 4at._p)
B²(::m, at::HubbardAtom) = at._uhalf2 * (4at._p - 1)
B²(::s, at::HubbardAtom) = at._uhalf2 * (3 - 4at._p)
B²(::t, at::HubbardAtom) = zero(at.U)

# Bᵣ² + U²/4, without the cancellation of computing it from Bᵣ² when p → 1 (d, s) or p → 0 (m)
B²U²₄(::d, at::HubbardAtom) = 4at._uhalf2 * at._q
B²U²₄(::m, at::HubbardAtom) = 4at._uhalf2 * at._p
B²U²₄(::s, at::HubbardAtom) = 4at._uhalf2 * at._q

# Bᵣ² - Aᵣ², without cancellation for p → 0 (r = d, m)
B²mA²(::d, at::HubbardAtom) = -4at._uhalf2 * at._p
B²mA²(::m, at::HubbardAtom) = 4at._uhalf2 * at._p
B²mA²(::s, at::HubbardAtom) = at._uhalf2 * (3 - 4at._p)
B²mA²(::t, at::HubbardAtom) = at._uhalf2

C(::d, at::HubbardAtom, k::BosonicFreq) = at.beta * at.U / 2 * δ(k) * at._p
C(::m, at::HubbardAtom, k::BosonicFreq) = -at.beta * at.U / 2 * δ(k) * at._q
C(::s, at::HubbardAtom, k::BosonicFreq) = at.beta * at.U / 2 * δ(k) * at._p
C(::t, at::HubbardAtom, k::BosonicFreq) = zero(at.U)

𝒜₀(::d) = +1
𝒜₀(::m) = +1
𝒜₀(::s) = +1 // 2
𝒜₀(::t) = -1 // 2

ℬ₀(::d) = +1
ℬ₀(::m) = +1
ℬ₀(::s) = +1 // 2
ℬ₀(::t) = -1 // 2

ℬ₁(::d, at::HubbardAtom) = im
ℬ₁(::m, at::HubbardAtom) = 1
ℬ₁(::s, at::HubbardAtom) = im / √(2one(at.U))
ℬ₁(::t, at::HubbardAtom) = 0

# ℬ₁² exactly; note also |ℬ₂|² = ℬ₀ for r = d, m, s
ℬ₁²(::d) = -1
ℬ₁²(::m) = 1
ℬ₁²(::s) = -1 // 2
ℬ₁²(::t) = 0

ℬ₂(::d, at::HubbardAtom) = 1
ℬ₂(::m, at::HubbardAtom) = im
ℬ₂(::s, at::HubbardAtom) = 1 / √(2one(at.U))
ℬ₂(::t, at::HubbardAtom) = 0

χ(::d, at::HubbardAtom, k::BosonicFreq) = -at.beta * δ(k) * at._p
χ(::m, at::HubbardAtom, k::BosonicFreq) = -at.beta * δ(k) * at._q
χ(::s, at::HubbardAtom, k::BosonicFreq) = -at.beta * δ(k) * at._p / 2
# With G(ν) = (1/(iν - U/2) + 1/(iν + U/2))/2 the bubble is
# U/2 tanh(βU/4)/(U² + ω²) + δ_ω0 β/(8 cosh²(βU/4)), evaluated without 0/0 at U = 0 for ω = 0.
function χ(::t, at::HubbardAtom, k::BosonicFreq)
    β = at.beta
    U = at.U
    x = β * U / 4
    iszero(k) || return U / 2 * tanh(x) / (U^2 + _freq(k, β)^2)
    β / 8 * ((iszero(x) ? one(x) : tanh(x) / x) + sech(x)^2)
end

# Equation 6
function χ₀(r::SpinChannel, at::HubbardAtom, (n, n′, k)::FermiFermiBose)
    β = at.beta

    β * δ(n, n′) * χ₀(r, at, (n, k))
end

# G is purely imaginary, so these products are real
χ₀(::PHChannel, at::HubbardAtom, (n, k)::FermiBose) = -real(G(at, n) * G(at, n + k))
χ₀(::PPChannel, at::HubbardAtom, (n, k)::FermiBose) = -real(G(at, n) * G(at, -n - k)) / 2

# Equation 27
F(::DensityChannel, at::HubbardAtom, w::FermiFermiBose) = F_up_up(at, w) + F_up_down(at, w)
F(::MagneticChannel, at::HubbardAtom, w::FermiFermiBose) = F_up_up(at, w) - F_up_down(at, w)
function F(::SingletChannel, at::HubbardAtom, w::FermiFermiBose)
    -F_up_up(at, _to_pp(w)) + 2F_up_down(at, _to_pp(w))
end
F(::TripletChannel, at::HubbardAtom, w::FermiFermiBose) = F_up_up(at, _to_pp(w))

_to_pp((n, n′, k)::FermiFermiBose) = (n, n′, -n - n′ - k)

# These formulas can be found in Fully_irreducible_vertex.nb from the paper's supplementary material
function F_up_up(at::HubbardAtom, w::FermiFermiBose)
    (n, n′, k) = w
    β = at.beta
    U²₄ = at._uhalf2

    res = zero(β)

    deltas = δ(n, n′) - δ(k)

    if !iszero(deltas)
        ν = _freq(n, β)
        ν′ω = _freq(n′ + k, β)

        res += β * U²₄ * (ν^2 + U²₄) * (ν′ω^2 + U²₄) / (ν^2 * ν′ω^2) * deltas
    end

    res
end

function F_up_down(at::HubbardAtom, w::FermiFermiBose)
    (n, n′, k) = w
    β = at.beta
    U = at.U
    U²₄ = at._uhalf2

    ν = _freq(n, β)
    ν′ = _freq(n′, β)
    νω = _freq(n + k, β)
    ν′ω = _freq(n′ + k, β)

    res = U - U^3 / 8 * (ν^2 + νω^2 + ν′ω^2 + ν′^2) / (ν * νω * ν′ω * ν′) -
          3U^5 / (16 * ν * νω * ν′ω * ν′)

    deltas1 = 2 * δ(n, -(n′ + k)) + δ(k)
    if !iszero(deltas1)
        res -= β * U²₄ * at._p * (νω^2 + U²₄) * (ν′ω^2 + U²₄) / (νω^2 * ν′ω^2) * deltas1
    end

    deltas2 = 2 * δ(n, n′) + δ(k)
    if !iszero(deltas2)
        res += β * U²₄ * at._q * (ν^2 + U²₄) * (ν′ω^2 + U²₄) / (ν^2 * ν′ω^2) * deltas2
    end

    res
end

# Equation 19. The removable singularities at ν(ν + ω) = Bᵣ² (Equation 21), where two terms of
# Equation 19 diverge separately, are cancelled analytically (see Γ_B).
function Γ(r::SpinChannel, at::HubbardAtom, (n, n′, k)::FermiFermiBose)
    β = at.beta
    U²₄ = at._uhalf2
    ν = _freq(n, β)
    νω = _freq(n + k, β)

    Aᵣ² = A²(r, at)
    Γᵣ = β * Aᵣ² / 2𝒜₀(r) * (ν^2 + U²₄) * (νω^2 + U²₄) / ((ν * νω - Aᵣ²) * ν * νω) *
         (δ(n, n′) - δ(n, -n′ - k))
    r isa TripletChannel && return Γᵣ

    Γᵣ = Γ_B(r, at, (n, n′, k), Γᵣ)
    Γᵣ -= at.U * ℬ₁²(r) / ℬ₀(r)^2

    Γᵣ
end

#=
Second and third term of Eq. 19 (r = d, m, s). With

    X(ν) = ν(ν + ω) - B²,   s = √(4B² + ω²),   T = U tan(β(s + ω)/4) / s ± 1,
    P(ν) = (ν² + U²/4)((ν + ω)² + U²/4) = (X + B² + U²/4)² + U²ω²/4,
    K = U |ℬ₂|²/ℬ₀² P|_{X=0} = U Q / ℬ₀,   Q = (B² + U²/4)² + U²ω²/4,

they read

    β B² P(ν) / (2ℬ₀ ν(ν + ω) X(ν)) [δ(ν, ν′) + δ(ν, -ν′ - ω)]  -  K / (T X(ν) X(ν′)).

Let z = βs/4 and z_ν = β|2ν + ω|/4. Since 2ν + ω = 2π(2j + 1 + l)/β for ν = (2j + 1)π/β, ω = 2πl/β, z_ν is
always a pole of tan(β(s + ω)/4), which hence equals cot(w) with w = z_ν - z. As
X = (4/β²)(z_ν² - z²), w = β²X / (4(z_ν + z)) can be computed from X without cancellation and

    R(ν) := T X(ν) = ±X + (U/β)(1 + z_ν/z) w cot(w)

is smooth at X(ν) = 0, where T diverges (Eq. 21). For ν′ ∈ {ν, -ν - ω}, X(ν′) = X(ν) and the two
1/X(ν) poles cancel. With w cot(w) = 1 + w² ψ(w²), ψ(t) = (√t cot √t - 1)/t, the sum becomes

    [±β B² P - 2UQ + 2U B² (P ρ η + 2(B² + U²/4) + X)] / (2ℬ₀ ν(ν + ω) R)

with ρ = w/X = β²/(4(z_ν + z)) and η = (1 + w² ψ)/(2z) + w ψ, which contains no 1/X.

For 2ν + ω = 0 (z_ν = 0, odd l) both deltas hold, X = -s²/4, and the pole of T at s = 0 makes
R = ±X + (U/β) z cot(z) = ±X + (U/β)(1 + z² ψ(z²)); the sum then becomes

    [±β B² P - UQ + U B² (2(B² + U²/4) + X - β² P ψ/4)] / (ℬ₀ ν(ν + ω) R).

Γ_B returns ΓA plus these terms, where ΓA is the first term of Eq. 19. For ν′ = -ν - ω, ΓA is
-β A² P / (2ℬ₀ ν(ν + ω) X_A) with X_A = ν(ν + ω) - A² (𝒜₀ = ℬ₀), which cancels the second term
for A² ≈ B², i.e. r = d, m for βU ≫ 1; both are ∝ β/ν², large for β ≫ 1. Their sum is
β P (B² - A²) / (2ℬ₀ X_A X), with B² - A² = ∓U² p for r = d, m.
=#
function Γ_B(r::SpinChannel, at::HubbardAtom, (n, n′, k)::FermiFermiBose, ΓA)
    β = at.beta
    U = at.U
    U²₄ = at._uhalf2
    ν = _freq(n, β)
    ν′ = _freq(n′, β)
    ω = _freq(k, β)
    νω = _freq(n + k, β)
    ν′ω = _freq(n′ + k, β)
    σ = r isa MagneticChannel ? -1 : 1

    b = B²(r, at)
    bU = B²U²₄(r, at)
    Q = bU^2 + U²₄ * ω^2
    s² = 4b + ω^2
    z = β * √(max(s², zero(s²))) / 4
    T, scaled = _T(σ, U, β, b, s², bU, ω, isodd(Int(k) ÷ 2))
    # If `scaled`, T and K are divided by w₀² (see _T); only K/T enters then.
    K = scaled ? U^5 / ℬ₀(r) : U * Q / ℬ₀(r)

    a = _TX(ν, νω, _freq(n + (n + k), β), β, U, σ, b, s², z, T)
    D = δ(n, n′) + δ(n, -n′ - k)
    if iszero(D)
        # Only the third term contributes; use T X of whichever frequency is closer to its pole.
        a′ = _TX(ν′, ν′ω, _freq(n′ + (n′ + k), β), β, U, σ, b, s², z, T)
        return ΓA + (abs(a.w) <= abs(a′.w) ? -K / (a.R * a′.X) : -K / (a′.R * a.X))
    end

    # Both deltas hold (D = 2) only for 2ν + ω = 0, i.e. z_ν = 0.
    (; X, R, w, ψ, zν) = a
    P = (X + bU)^2 + U²₄ * ω^2
    if isfinite(w) && iszero(zν)
        ΓA +
        (σ * β * b * P - U * Q + U * b * (2bU + X - β^2 * P * ψ / 4)) / (ℬ₀(r) * ν * νω * R)
    elseif isfinite(w)
        ρ = β^2 / (4(zν + z))
        η = (1 + w^2 * ψ) / (2z) + w * ψ
        ΓA +
        (σ * β * b * P - 2U * Q + 2U * b * (P * ρ * η + 2bU + X)) / (2ℬ₀(r) * ν * νω * R)
    elseif D == 1 && iszero(δ(n, n′))  # ν′ = -ν - ω
        β * P * B²mA²(r, at) / (2ℬ₀(r) * (ν * νω - A²(r, at)) * X) - K / (R * X)
    else
        ΓA + (D * β * b * P / (2ℬ₀(r) * ν * νω) - K / R) / X
    end
end

# T X(ν), evaluated via w if ν is close to the pole of T (if any) that belongs to it; νω = ν + ω, y = 2ν + ω
function _TX(ν, νω, y, β, U, σ, b, s², z, T)
    X = ν * νω - b
    zν = β * abs(y) / 4
    if iszero(zν)
        t = β^2 * s² / 16  # z², the pole is at z = 0
        if abs(t) <= 1
            ψ = _ψ(t)
            return (; X, R=σ * X + U / β * (1 + t * ψ), w=(√abs(t)), ψ, zν)
        end
    elseif s² > 0
        w = β^2 * X / (4(zν + z))
        if abs(w) <= 1
            ψ = _ψ(w^2)
            return (; X, R=σ * X + U / β * (1 + zν / z) * (1 + w^2 * ψ), w, ψ, zν)
        end
    end
    (; X, R=T * X, w=oftype(z, Inf), ψ=zero(z), zν)
end

#=
T = U tan(β(s + ω)/4) / s + σ, evaluated without complex arithmetic. For ω = 2πl/β, β(ω + |ω|)/4
is a multiple of π, so tan(β(s + ω)/4) = tan(δ) with δ = β(s - |ω|)/4 = βB²/(s + |ω|); this avoids
the large argument βs/4 for |ω| ≫ 1/β. Equivalently it is tan(z) for even l and -cot(z) for odd l,
with z = βs/4, so T is even in s and real. For s² < 0, z = iζ and T = σ + U tanh(ζ)/a or
σ + U coth(ζ)/a with a = √(-s²). Then B² < 0,
which requires U < 0 for r = d, s and U > 0 for r = m, i.e. U = -σ|U|: the two terms cancel for
βU → ±∞, which is avoided by U² - a² = 4(B² + U²/4) + ω².

Returns (T, scaled). For ω = 0 and |βU| ≫ 1, T and K = U (B² + U²/4)²/ℬ₀ are both ∝ w₀², which
underflows for |βU| ≳ 700; then T/w₀² is returned with scaled = true, and K/w₀² = U⁵/ℬ₀.
=#
function _T(σ, U, β, b, s², bU, ω, kodd)
    if s² >= 0
        s = √s²
        iszero(ω) || return σ + U * tan(β * b / (s + abs(ω))) / s, false
        z = β * s / 4
        return σ + U * β / 4 * (iszero(z) ? one(z) : tan(z) / z), false
    end
    a = √(-s²)
    ζ = β * a / 4
    w₀ = bU / U^2  # q (d, s) or p (m), 1/(1 + exp(β|U|/2))
    if iszero(ω) && w₀ < 1 // 8
        # Here both terms below are ≈ 2|U| w₀ while T ∝ w₀². With ε = 4w₀, c = a/|U| = √(1 - ε)
        # and |x| = β|U|/2: 1/(1 + exp(|x|c)) = w₀ (1 + g expm1(y)), y = |x|(1 - c), g = 1/(1 + exp(-|x|c)).
        ε = 4w₀
        c = √(1 - ε)
        x = β * abs(U) / 2
        g = 1 / (1 + exp(-x * c))
        y = x * ε / (1 + c)
        expm1y_y = iszero(y) ? one(y) : expm1(y) / y
        return 2σ * (4g * expm1y_y * x / (1 + c) - 4 / (1 + c)^2) / c, true
    end
    th1 = kodd ? 2 / expm1(2ζ) : -2 / (exp(2ζ) + 1)  # coth(ζ) - 1 or tanh(ζ) - 1
    (-σ * (4bU + ω^2) / (a + abs(U)) + U * th1) / a, false
end

#=
ψ(t) = (√t cot √t - 1)/t for real |t| ≤ 1 (analytic, ψ(0) = -1/3), without cancellation: from
cot(z) = (cot(z/2) - tan(z/2))/2 follows ψ(t) = ψ(t/4)/4 - tanc(t/4)/4 with tanc(t) = tan(√t)/√t
(= tanh(√-t)/√-t for t < 0). Iterating gives a sum of terms of equal sign; the remainder is
4⁻ʲ ψ(t/4ʲ) ≈ -4⁻ʲ/3.
=#
function _ψ(t)
    tanc(t) = t > 0 ? tan(√t) / √t : t < 0 ? tanh(√-t) / √-t : one(t)
    ψ = zero(t)
    c = one(t)
    while true
        t /= 4
        c /= 4
        ψ -= c * tanc(t)
        c * abs(t) < eps(typeof(t)) && break
    end
    ψ - c / 3
end

# The Ward identities of Krien and Valli; at half filling, the singlet pair and the charge density
# are related by η-pairing symmetry, which gives G₃(s) = G₃(d)/2.
function G₃(::d, at::HubbardAtom, (n, k)::FermiBose)
    -real(iszero(k) ? ∂G∂μ(at, n) : ∂G∂ν(at, n, k))
end
function G₃(::m, at::HubbardAtom, (n, k)::FermiBose)
    -real(iszero(k) ? ∂G∂H(at, n) : ∂G∂ν(at, n, k))
end
G₃(::s, at::HubbardAtom, w::FermiBose) = G₃(d(), at, w) / 2
G₃(::t, at::HubbardAtom, (n, k)::FermiBose) = zero(at.U)

"Finite difference of Green's function with respect to frequency"
function ∂G∂ν(at::HubbardAtom, n::FermionicFreq, k::BosonicFreq)
    β = at.beta
    iω = im * _freq(k, β)

    (G(at, n + k) - G(at, n)) / iω
end

"Derivative of the Green's function with respect to the chemical potential"
function ∂G∂μ(at::HubbardAtom, n::FermionicFreq)
    U = at.U
    β = at.beta
    iν = im * _freq(n, β)

    # dgdμ enters the charge channel, while dgdh enters the spin channel.
    # The "Curie-like" term ∝ β is weighted by p here and by q = 1 - p in the
    # spin channel, which suppresses it in the charge channel for βU ≫ 1.
    r = -β * U * at._p / (iν^2 - at._uhalf2)
    r += 1 / (iν + U / 2)^2
    r += 1 / (iν - U / 2)^2
    -r / 2
end

"Derivative of the Green's function with respect to the magnetic field"
function ∂G∂H(at::HubbardAtom, n::FermionicFreq)
    U = at.U
    β = at.beta
    iν = im * _freq(n, β)

    r = β * U * at._q / (iν^2 - at._uhalf2)
    r += 1 / (iν + U / 2)^2
    r += 1 / (iν - U / 2)^2
    -r / 2
end
