# Exact Matsubara correlators of the Hubbard atom from its four Fock states |0⟩, |↑⟩, |↓⟩, |↑↓⟩ = c↑⁺c↓⁺|0⟩.
module ED
using LinearAlgebra

const cup⁺ = [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
const cdn⁺ = [0 0 0 0; 0 0 0 0; 1 0 0 0; 0 -1 0 0]
const cup, cdn = Matrix(cup⁺'), Matrix(cdn⁺')
const nup, ndn = cup⁺ * cup, cdn⁺ * cdn
const δμ = Ref(0.0)
energies(U) = U .* (diag(nup) .- 1 / 2) .* (diag(ndn) .- 1 / 2) .- δμ[] .* (diag(nup) .+ diag(ndn) .- 1)

#=
⟨T O₁(τ₁) ⋯ Oₖ(τₖ) Oₖ₊₁(0)⟩ Fourier transformed with ∫dτⱼ exp(i φⱼ τⱼ) over (0, β)ᵏ, where
`fermionic[j]` tells whether Oⱼ anticommutes. As H is diagonal, the integrand on each ordered
simplex is a sum of exponentials whose integral is the divided difference of exp(βx) at the
cumulative exponents (Hermite-Genocchi), i.e. an entry of exp(βJ) for bidiagonal J (Opitz).
=#
function correlator(U, β, ops, φ, fermionic)
    E = energies(U)
    w = exp.(-β .* (E .- minimum(E)))
    Z = sum(w)
    k = length(φ)
    res = zero(ComplexF64)
    for P in permutations(k)
        sgn = permsign(P, fermionic)
        sgn == 0 && continue
        string = push!([ops[j] for j in P], ops[end])
        for idx in Iterators.product(ntuple(_ -> 1:4, k + 1)...)
            amp = w[idx[1]] * prod(string[j][idx[j], idx[mod1(j + 1, k + 1)]] for j in 1:k+1)
            iszero(amp) && continue
            a = [E[idx[j]] - E[idx[j+1]] + im * φ[P[j]] for j in 1:k]
            J = diagm(0 => [0; cumsum(a)], 1 => ones(k))
            res += sgn * amp * exp(β * J)[1, k+1]
        end
    end
    res / Z
end

permutations(k) = k == 1 ? [[1]] : [insert!(copy(p), i, k) for p in permutations(k - 1) for i in 1:k]

# sign of the time-ordering permutation P (descending times) restricted to fermionic operators
function permsign(P, fermionic)
    f = [j for j in P if fermionic[j]]
    s = 1
    for i in eachindex(f), j in i+1:lastindex(f)
        f[i] > f[j] && (s = -s)
    end
    s
end

ν(n, β) = (2n + 1) * π / β
ω(k, β) = 2k * π / β

# G(ν) = -∫dτ e^{iντ} ⟨T c(τ) c⁺(0)⟩
G(U, β, n) = -correlator(U, β, (cup, cup⁺), (ν(n, β),), (true,))

# χ(ω) = -∫dτ e^{iωτ} ⟨T ρ(τ) ρ(0)⟩ + β⟨ρ⟩² δ_ω0 for a bosonic density ρ
function χ(U, β, ρ, k)
    E = energies(U); w = exp.(-β .* (E .- minimum(E))); avg = tr(Diagonal(w) * ρ) / sum(w)
    -correlator(U, β, (ρ, ρ), (ω(k, β),), (false,)) + (k == 0 ? β * avg^2 : 0)
end
χ₊₋(U, β, k) = -correlator(U, β, (cdn * cup, cup⁺ * cdn⁺), (ω(k, β),), (false,))  # -⟨ρ⁻; ρ⁺⟩, ρ⁻ = c↓c↑

#=
Eq. 3: χ_ph,σσ′(ν, ν′, ω) = ∫ e^{-iντ₁} e^{i(ν+ω)τ₂} e^{-i(ν′+ω)τ₃}
[⟨T cσ⁺(τ₁) cσ(τ₂) cσ′⁺(τ₃) cσ′(0)⟩ - ⟨T cσ⁺(τ₁) cσ(τ₂)⟩⟨T cσ′⁺(τ₃) cσ′(0)⟩]
=#
function χph(U, β, σσ′, n, n′, k)
    c⁺1, c1 = σσ′[1] == :up ? (cup⁺, cup) : (cdn⁺, cdn)
    c⁺2, c2 = σσ′[2] == :up ? (cup⁺, cup) : (cdn⁺, cdn)
    v, v′, w = ν(n, β), ν(n′, β), ω(k, β)
    full = correlator(U, β, (c⁺1, c1, c⁺2, c2), (-v, v + w, -(v′ + w)), (true, true, true))
    # disconnected: β δ_ω0 ∫ e^{-iντ} ⟨T cσ⁺(τ) cσ(0)⟩ ∫ e^{-iν′τ} ⟨T cσ′⁺(τ) cσ′(0)⟩
    disc = k == 0 ? β * correlator(U, β, (c⁺1, c1), (-v,), (true,)) * correlator(U, β, (c⁺2, c2), (-v′,), (true,)) : 0
    full - disc
end
end
