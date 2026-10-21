# Transverse-spin, opposite-spin pair and equal-spin pair sectors of FieldHubbardAtom.
_field_nu(at::FieldHubbardAtom{T}, n) where {T} = (2n + 1) * T(π) / at.beta
function _field_t_zlat(at::FieldHubbardAtom{T}, n, k) where {T}
    Complex{T}(at.mu, T(π) * (2n + 1 + k) / at.beta)
end
function _field_t_sum(at, k, x)
    y = at.beta * (x - at.mu) / 2
    -at.beta / 2 * (iseven(k) ? tanh(y) : coth(y))
end

struct _field_t_TData{T}
    a::T
    ρ::T
    mg::T
    S1::T
    p0p2::T
    W::Complex{T}
    q0::Complex{T}
    q1::Complex{T}
    λ::T
end
function _field_t_tdata(at::FieldHubbardAtom{T}, k::Integer) where {T}
    a = at.U / 2
    W = complex(zero(T), 2 * T(π) * k / at.beta) - 2 * at.h
    q0 = a^2 - W^2 / 4
    mg = at._mag
    S1 = at._pu + at._pd
    q1 = q0 - 2 * a^2 * S1 - a * W * mg
    _field_t_TData{T}(a, at._rho, mg, S1, at._p0 * at._p2, W, q0, q1, -4 * a^2)
end
_field_t_Hp(d::_field_t_TData, x) = x^2 + 2 * d.a * d.ρ * x + d.q1
_field_t_Pp(d::_field_t_TData, x) = ((x + d.a)^2 - d.W^2 / 4) * ((x - d.a)^2 - d.W^2 / 4)
_field_t_ep(d::_field_t_TData, x) = _field_t_Hp(d, x) * _field_t_Hp(d, -x) - d.λ^2 * d.p0p2
_field_t_G2p(d::_field_t_TData, x) = (x + d.a * d.ρ)^2 - (d.W / 2 + d.a * d.mg)^2
function _field_t_dPp(d::_field_t_TData, x)
    2 * (x + d.a) * ((x - d.a)^2 - d.W^2 / 4) + 2 * (x - d.a) * ((x + d.a)^2 - d.W^2 / 4)
end
_field_t_dHp(d::_field_t_TData, x) = 2 * x + 2 * d.a * d.ρ
function _field_t_dep(d::_field_t_TData, x)
    _field_t_dHp(d, x) * _field_t_Hp(d, -x) - _field_t_Hp(d, x) * _field_t_dHp(d, -x)
end

# Transverse vertex (μ ≠ 0): the doped magnetic-channel closed form (derivation note, theorem "density and
# magnetic") with the data above.
function _field_t_gammaT_up(at::FieldHubbardAtom{T}, n::Integer, m::Integer,
                            k::Integer) where {T}
    w = (FermionicFreq(2n + 1), FermionicFreq(2m + 1), BosonicFreq(2k))
    iszero(at.h) && return Γ(MagneticChannel(), DopedHubbardAtom(at.U, at.beta, at.mu), w)
    if iszero(at.mu)
        source = DopedHubbardAtom(-at.U, at.beta, at.h)
        return (Γ(SingletChannel(), source, w) - Γ(TripletChannel(), source, w)) / 2
    end
    val, loss = _field_t_gammaT_raw(at, n, m, k)
    loss <= 6 && return val
    p = precision(T)
    prec = p + 48 + min(_bits(loss), p)
    while true
        val, loss = _withprecision(prec) do
            bat = FieldHubbardAtom(big(at.U), big(at.beta), big(at.mu), big(at.h))
            _field_t_gammaT_raw(bat, n, m, k)
        end
        # Atomic factors and thermal denominators can still vanish to exponential order.
        # Cover their measured loss, including weights that underflowed in the input type.
        (p + 32 + 2loss <= prec || prec >= 1 << 16) && return Complex{T}(val)
        prec = min(max(2prec, p + 48 + min(_bits(2loss), prec)), 1 << 16)
    end
end

# tanh divided difference, including Δ = 0 and saturated tanh.
function _field_t_tanhdd(y, Δ)
    abs(Δ) >= 1 / 2 && return _tanhdiff(y, Δ) / Δ
    σ = real(y) >= 0 ? 1 : -1
    e = exp(-2σ * y)
    _tanhc(Δ) * (4e / (1 + e)^2) / (1 + tanh(y) * tanh(Δ))
end

function _field_t_gammaT_raw(at::FieldHubbardAtom{T}, n, m, k) where {T}
    d = _field_t_tdata(at, k)
    β, a, ρ, λ = at.beta, d.a, d.ρ, d.λ
    z = _field_t_zlat(at, n, k)
    v = _field_t_zlat(at, m, k)
    F(x) = _field_t_Pp(d, x) / _field_t_ep(d, x)
    if n == m
        dF = (_field_t_dPp(d, z) * _field_t_ep(d, z) -
              _field_t_Pp(d, z) * _field_t_dep(d, z)) / _field_t_ep(d, z)^2
        DD = (F(z) + z * dF) / (2z)
    else
        DD = (z * F(z) - v * F(v)) / (z^2 - v^2)
    end
    out = -2a - λ * ρ * DD
    # e - G2 H(-z): cancel the degree-four terms before evaluating the diagonal.
    n == m && (out += β * _field_t_Pp(d, z) *
                      (4a^2 * ((d.p0p2 - at._piS) * _field_t_Hp(d, -z) -
                               4a^2 * d.p0p2)) /
                      (_field_t_G2p(d, z) * _field_t_ep(d, z)))
    # roots s_j of e in s = z²: s² + 2(q1 - 2a²ρ²)s + q1² - 16a⁴p0p2
    B = d.q1 - 2a^2 * ρ^2
    r = sqrt(4a^4 * ρ^4 - 4a^2 * ρ^2 * d.q1 + 16a^4 * d.p0p2)
    s1, s2 = -B + r, -B - r
    u1, u2 = sqrt(s1), sqrt(s2)
    abs(u1 + u2) < abs(u1 - u2) && (u2 = -u2)
    Δs = 2r
    # Pairs (f(s1), f[s1,s2]) in the quadratic quotient algebra; no inverse root gap.
    loss = Ref(zero(T))
    # Inlining the repeated loss estimates triggers an LLVM SLP crash on Julia 1.11.
    @noinline function total(x, y)
        res = x + y
        loss[] = max(loss[], _cancel(res, x, y))
        res
    end
    add(x, y) = (total(x[1], y[1]), total(x[2], y[2]))
    scale(c, x) = c .* x
    function mul(x, y)
        x2 = total(x[1], -Δs * x[2])
        (x[1] * y[1], total(x[2] * y[1], x2 * y[2]))
    end
    function divide(x, y)
        den2 = total(y[1], -Δs * y[2])
        mul(x, (inv(y[1]), -y[2] / (y[1] * den2)))
    end
    u = (u1, inv(u1 + u2))
    s = (s1, one(s1))
    c(c0) = (complex(c0), zero(s1))
    P = mul(add(s, c(-(a + d.W / 2)^2)), add(s, c(-(a - d.W / 2)^2)))
    X = scale(β, add((2a^2 * ρ^2 + r, one(s1)), scale(-2a * ρ, u)))
    y2 = β * (u2 - at.mu) / 2 + (iseven(k) ? zero(s1) : complex(0, T(π) / 2))
    Δy = β * Δs * u[2] / 2
    S = (_field_t_sum(at, k, u1), -β^2 / 4 * u[2] * _field_t_tanhdd(y2, Δy))
    α = add(scale(ρ, S), c(-β * (at._p0 + at._p2) / 2))
    K, N = add(X, scale(λ, α)), add(X, scale(-λ, α))
    Y = scale(ρ, divide(mul(mul(u, P), N), K))
    # (Y G)[s1,s2] = Y(s1) G[s1,s2] + Y[s1,s2] G(s2).
    term = λ * (Y[1] * (z^2 + v^2 - s1 - s2) /
                (_field_t_ep(d, z) * _field_t_ep(d, v)) +
                Y[2] / ((z^2 - s2) * (v^2 - s2)))
    result = out + term
    loss[] = max(loss[], _cancel(result, out, term))
    isfinite(result) || (loss[] = T(Inf))
    min(at._p0, at._pu, at._pd, at._p2) < sqrt(floatmin(T)) && (loss[] = T(Inf))
    result, loss[]
end

# Equal-spin (↑↑) pair vertex, explicit
function _field_t_gammaE_up(at::FieldHubbardAtom{T}, n::Integer, m::Integer,
                            k::Integer) where {T}
    nb = -n - k - 1
    (m == n || m == nb) && n != nb || return zero(Complex{T})
    a = at.U / 2
    r = at._pd + at._rho - at._pu          # 2⟨n↓⟩ - 1
    x = im * _field_nu(at, n) + at.mu + at.h
    x̄ = im * _field_nu(at, nb) + at.mu + at.h
    PE = (x^2 - a^2) * (x̄^2 - a^2)
    GE = (x + a * r) * (x̄ + a * r)
    LE = x * x̄ + a^2 + a * r * (x + x̄)
    val = -at.beta * a^2 * (1 - r^2) * PE / (2 * LE * GE)
    m == n ? val : -val
end
