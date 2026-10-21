# Port of FieldL.jl, one-sided theorem (fieldL.tex); n,m,k below use ν=(2n+1)π/β, ω=2kπ/β.
_field_l_sgn(σ) = 3 - 2σ

_field_l_μs(at, σ) = at.mu + _field_l_sgn(σ) * at.h
_field_l_rs(at, σ) = at._rho - _field_l_sgn(σ) * at._mag
_field_l_pw(at, σ) = σ == 1 ? at._pu : at._pd

"Centred lattice point z_{σn} = μ_σ + iπ(2n+1+k)/β."
function _field_l_zlat(at::FieldHubbardAtom{T}, σ, n, k) where {T}
    Complex{T}(_field_l_μs(at, σ), T(pi) * (2n + 1 + k) / at.beta)
end

"Thermal sum S_τ(x) = PV Σ_{n} 1/(z_{τn} - x) = -β/2 tanh (even k) or coth (odd k) of β(x-μ_τ)/2."
function _field_l_Sfun(at, τ, x, k)
    y = at.beta * (x - _field_l_μs(at, τ)) / 2
    -at.beta / 2 * (iseven(k) ? tanh(y) : coth(y))
end
_field_l_Hpol(at, σ, x, q0) = x^2 + 2at._a * _field_l_rs(at, σ) * x + q0
_field_l_Ppol(at, x, q0) = (x^2 + q0)^2 - 4at._a^2 * x^2
_field_l_G2pol(at, σ, x, w) = (x + at._a * _field_l_rs(at, σ))^2 - w^2 / 4

function _field_l_e8coeffs(at::FieldHubbardAtom, q0)
    a, λ = at._a, at._lambda
    ru, rd = _field_l_rs(at, 1), _field_l_rs(at, 2)
    pmul(x, y) = [sum(x[i] * y[j - i + 1]
                      for i in max(1, j - length(y) + 1):min(j, length(x)))
                  for j in 1:(length(x) + length(y) - 1)]
    padd(x, y) = (n=max(length(x), length(y));
                  [get(x, i, zero(q0)) + get(y, i, zero(q0)) for i in 1:n])
    eσ(r) = [q0^2, 2q0 - 4a^2 * r^2, one(q0)]           # H_σ(x)H_σ(-x) in s = x²
    sq2 = [q0^2, 2q0, one(q0)]
    t2 = padd(at._piS .* padd(2 .* sq2, [0, 8a^2 * ru * rd]),
              at._piP .* padd(2 .* sq2, [0, -8a^2 * ru * rd]))
    padd(padd(pmul(eσ(ru), eσ(rd)), -λ^2 .* t2), [λ^4 * (at._piP - at._piS)^2])
end
_field_l_E8val(c, s) = sum(c[i] * s^(i - 1) for i in eachindex(c))

function _field_l_quarticroots(c::Vector{Complex{T}}) where {T}
    f(s) = _field_l_E8val(c, s)
    df(s) = sum((i - 1) * c[i] * s^(i - 2) for i in 2:5)
    R = maximum(abs, c[1:4])^(1 / 4) + 1
    r = [R * cis(T(2) * T(pi) * (j - 1) / 4 + T(0.4)) for j in 1:4]
    for _ in 1:5000
        rn = let r = r
            [r[j] - f(r[j]) / prod(r[j] - r[i] for i in 1:4 if i != j) for j in 1:4]
        end
        done = maximum(abs.(rn .- r)) <= 64eps(T) * R
        r = rn
        done && break
    end
    for _ in 1:4, j in 1:4
        d = df(r[j])
        iszero(d) || (r[j] -= f(r[j]) / d)
    end
    r
end

function _field_l_Lsetup(at::FieldHubbardAtom{T}, k::Integer) where {T}
    lock(at._lock) do
        get!(at._setups, Int(k)) do
            T === BigFloat && return _field_l_buildsetup(at, k, ones(Int, 4))
            # Thermal denominators cancel strongly even at ordinary parameters: round only the completed setup.
            _withprecision(256) do
                bat = FieldHubbardAtom(big(at.U), big(at.beta), big(at.mu), big(at.h))
                st = _field_l_buildsetup(bat, k, ones(Int, 4))
                FieldLSetup{T}(st.w, st.q0, st.E8, st.s, st.u, st.keep, st.f, st.𝔠)
            end
        end
    end
end
# branch[j] = ±1 selects the square root u_j = ±√s_j (the vertex does not depend on it)
function _field_l_buildsetup(at::FieldHubbardAtom{T}, k::Integer, branch) where {T}
    β, λ, a, ρ, m = at.beta, at._lambda, at._a, at._rho, at._mag
    w = Complex{T}(0, 2T(pi) * k / β)
    q0 = a^2 - w^2 / 4
    c = _field_l_e8coeffs(at, q0)
    s = _field_l_quarticroots(c)
    u = branch .* sqrt.(s)
    keep = [!(k == 0 && abs(s[j] - a^2) <= sqrt(eps(T)) * (1 + abs(a^2))) for j in 1:4]
    E8p = [prod(s[j] - s[i] for i in 1:4 if i != j) for j in 1:4]
    𝔞 = [2λ^2 * _field_l_Ppol(at, u[j], q0) * (s[j] + q0) / (β^2 * E8p[j]) for j in 1:4]
    f = zeros(Complex{T}, 4, 4)
    K = zeros(Complex{T}, 4, 2)
    𝔠 = zeros(Complex{T}, 4, 2)
    dπ = at._piP - at._piS
    ru, rd = _field_l_rs(at, 1), _field_l_rs(at, 2)
    for j in 1:4
        keep[j] || continue
        x = u[j]
        h1, h2, h3, h4 = _field_l_Hpol(at, 1, x, q0), _field_l_Hpol(at, 1, -x, q0),
                         _field_l_Hpol(at, 2, x, q0), _field_l_Hpol(at, 2, -x, q0)
        ν1 = -β * (h2 * h3 * h4 - λ^2 * (at._piS * h3 + at._piP * h4))
        ν2 = λ^2 * (h3 + h4)
        ν3 = -λ * (h2 * h4 + λ^2 * dπ)
        ν4 = -λ * (h2 * h3 - λ^2 * dπ)
        S = _field_l_Sfun(at, 1, x, k)                                   # the thermal sum S_↑(u_j)
        f[j, :] = [ν1, ν2 * (at._D[1] * S - β * at._c[1] / 2), ν3 * (m * S - at._cS),
                   ν4 * (ρ * S - at._cP)]
        for σ in 1:2
            K[j, σ] = f[j, 2σ-1] + f[j, 2σ]
        end
        # 𝔠_{jt} = A_{tj}/K_{jt} = W_{tj}/K_{j,-t} (Lemma fieldL-lem-ratio); take the larger denominator,
        # which keeps it regular on the axes μ_t = 0 where A_{tj} and K_{jt} vanish together
        Ξ = (s[j] + q0)^2 - 4a^2 * ru * rd * s[j] - λ^2 * dπ
        Π = (s[j] + q0)^2 + 4a^2 * ru * rd * s[j] + λ^2 * dπ
        for t in 1:2
            A = at._D[t] * 𝔞[j]
            W = -λ * _field_l_Ppol(at, x, q0) * (ρ * Ξ + _field_l_sgn(t) * m * Π) /
                (β^2 * E8p[j])
            𝔠[j, t] = abs(K[j, t]) >= abs(K[j, 3-t]) ? A / K[j, t] : W / K[j, 3-t]
        end
    end
    FieldLSetup{T}(w, q0, c, s, u, keep, f, 𝔠)
end

# ---------------------------------------------------------------- moving-pole coefficients
# F = P/E8; coefficient functions of the column (t, v) (hatted: without the D_t, m, ρ prefactors)
_field_l_Fv(at, st, v) = _field_l_Ppol(at, v, st.q0) / _field_l_E8val(st.E8, v^2)
function _field_l_Qt(at, t, v, q0)
    (o=3 - t;
     _field_l_Hpol(at, t, -v, q0) * _field_l_Hpol(at, o, v, q0) *
     _field_l_Hpol(at, o, -v, q0) -
     at._lambda^2 *
     (at._piP * _field_l_Hpol(at, o, -v, q0) + at._piS * _field_l_Hpol(at, o, v, q0)))
end
function _field_l_ahat(at, st, t, v)
    2at._lambda^2 * (v^2 + st.q0) * _field_l_Fv(at, st, v) / at.beta^2
end
function _field_l_bhat(at, st, v)
    -at._lambda *
    (_field_l_Hpol(at, 1, -v, st.q0) * _field_l_Hpol(at, 2, -v, st.q0) +
     at._lambda^2 * (at._piP - at._piS)) * _field_l_Fv(at, st, v) / at.beta^2
end
function _field_l_chat(at, st, t, v)
    -at._lambda *
    (_field_l_Hpol(at, t, -v, st.q0) * _field_l_Hpol(at, 3 - t, v, st.q0) -
     at._lambda^2 * (at._piP - at._piS)) * _field_l_Fv(at, st, v) / at.beta^2
end

# Cauchy kernels with their exact limits where the denominator vanishes (μ_t = 0, μ = 0, h = 0 axes)
function _field_l_kerH(at, t, z, v, mirror)        # D_t/(z+v), z, v on Λ_t; on the mirror z+v = 2μ_t
    mirror && return at.beta *
           (_field_l_μs(at, t) >= 0 ? at._p2 * _field_l_pw(at, t) :
            at._p0 * _field_l_pw(at, 3 - t)) *
           _expm1x(-2at.beta * abs(_field_l_μs(at, t)))
    at._D[t] / (z + v)
end
function _field_l_kerP(at, z, v, mirror)           # ρ/(z+v), z on Λ_σ, v on Λ_{-σ}; on the mirror z+v = 2μ
    mirror &&
        return at.beta * (at.mu >= 0 ? at._p2 : at._p0) * _expm1x(-2at.beta * abs(at.mu))
    at._rho / (z + v)
end
function _field_l_kerS(at, σ, z, v, same)          # m/(z-v), z on Λ_σ, v on Λ_{-σ}; for n = m, z-v = 2σh
    same && return _field_l_sgn(σ) * at.beta * (at.h >= 0 ? at._pu : at._pd) *
           _expm1x(-2at.beta * abs(at.h))
    at._mag / (z - v)
end

function _field_l_gammaL(at::FieldHubbardAtom{T}, σ::Integer, t::Integer, n::Integer,
                         m::Integer, k::Integer) where {T}
    st = _field_l_Lsetup(at, k)
    β = at.beta
    z = _field_l_zlat(at, σ, n, k)
    v = _field_l_zlat(at, t, m, k)
    mirror = n + m + 1 + k == 0
    val = zero(Complex{T})
    if σ == t
        val += _field_l_ahat(at, st, t, v) * _field_l_kerH(at, t, z, v, mirror)
    else
        val += _field_l_sgn(t) * _field_l_bhat(at, st, v) *
               _field_l_kerS(at, σ, z, v, n == m) +
               _field_l_chat(at, st, t, v) * _field_l_kerP(at, z, v, mirror)
        val += at.U / β^2
    end
    for j in 1:4
        st.keep[j] || continue
        Φ = st.f[j, 2σ-1] / (z - st.u[j]) + st.f[j, 2σ] / (z + st.u[j])
        val -= st.𝔠[j, t] * Φ / (v^2 - st.s[j])
    end
    Γ = β^2 * val
    if σ == t && n == m      # diagonal: β²(d_t - 1/b_t) = βP(z)[1/G2_t(z) - Q_t(z)/E8(z)]
        Γ += β * _field_l_Ppol(at, z, st.q0) *
             (1 / _field_l_G2pol(at, t, z, st.w) -
              _field_l_Qt(at, t, z, st.q0) / _field_l_E8val(st.E8, z^2))
    end
    Γ
end

function _field_l_needs_limit(at::FieldHubbardAtom{T}, σ, t, n, m, k) where {T}
    st = _field_l_Lsetup(at, k)
    sc = 1 + maximum(abs, st.s)
    tol = sqrt(eps(T)) * sc
    any(abs(st.s[i] - st.s[j]) < tol for i in 1:4
        for j in (i + 1):4 if st.keep[i] && st.keep[j]) && return true
    z = _field_l_zlat(at, σ, n, k)
    v = _field_l_zlat(at, t, m, k)
    any(st.keep[j] && (abs(z^2 - st.s[j]) < tol || abs(v^2 - st.s[j]) < tol) for j in 1:4)
end
function _field_l_limit(at::FieldHubbardAtom{T}, σ, t, n, m, k) where {T}
    p = T === BigFloat ? precision(BigFloat) : 53
    _withprecision(3p + 64) do
        δ = BigFloat(2)^(-(p + 16))
        pair = Tuple(FieldHubbardAtom(BigFloat(at.U), BigFloat(at.beta),
                                      BigFloat(at.mu) + s * δ, BigFloat(at.h))
                     for s in (1, -1))
        Complex{T}((_field_l_gammaL(pair[1], σ, t, n, m, k) +
                    _field_l_gammaL(pair[2], σ, t, n, m, k)) / 2)
    end
end
