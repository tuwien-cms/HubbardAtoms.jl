@testsetup module ExactDiagonalization
#=
Exact Matsubara correlators of the Hubbard atom from its four Fock states |0⟩, |↑⟩, |↓⟩ and
|↑↓⟩ = c↑⁺c↓⁺|0⟩, independent of the formulas implemented in the package.
=#
using LinearAlgebra

export ν, ω, cup, cup⁺, cdn, cdn⁺, nup, ndn, G_ed, χ_ed, χpair_ed, χph_ed

const cup⁺ = [0 0 0 0; 1 0 0 0; 0 0 0 0; 0 0 1 0]
const cdn⁺ = [0 0 0 0; 0 0 0 0; 1 0 0 0; 0 -1 0 0]
const cup, cdn = Matrix(cup⁺'), Matrix(cdn⁺')
const nup, ndn = cup⁺ * cup, cdn⁺ * cdn

# Eq. 1, H = U(n↑ - 1/2)(n↓ - 1/2), is diagonal in this basis.
energies(U) = U .* (diag(nup) .- 1 / 2) .* (diag(ndn) .- 1 / 2)

ν(n, β) = (2n + 1) * π / β
ω(k, β) = 2k * π / β

#=
⟨T O₁(τ₁) ⋯ Oₖ(τₖ) Oₖ₊₁(0)⟩, Fourier transformed with ∫dτⱼ exp(iφⱼτⱼ) over (0, β)ᵏ; `fermionic[j]`
tells whether Oⱼ anticommutes. As H is diagonal, the integrand on each time-ordered simplex is a
sum of exponentials, whose integral is the divided difference of exp(βx) at the cumulative
exponents (Hermite-Genocchi formula), i.e. the corner entry of exp(βJ) for a bidiagonal J (Opitz).
=#
function correlator(U, β, ops, φ, fermionic)
    E = energies(U)
    w = exp.(-β .* (E .- minimum(E)))
    k = length(φ)
    res = zero(ComplexF64)
    for P in permutations(k)
        sgn = permsign(P, fermionic)
        string = push!([ops[j] for j in P], ops[end])
        for idx in Iterators.product(ntuple(_ -> 1:4, k + 1)...)
            amp = w[idx[1]] * prod(string[j][idx[j], idx[mod1(j + 1, k + 1)]] for j in 1:k+1)
            iszero(amp) && continue
            a = [E[idx[j]] - E[idx[j+1]] + im * φ[P[j]] for j in 1:k]
            J = diagm(0 => [0; cumsum(a)], 1 => ones(k))
            res += sgn * amp * exp(β * J)[1, k+1]
        end
    end
    res / sum(w)
end

# all orderings τ_P[1] > τ_P[2] > ⋯ of the k free times
permutations(k) = k == 1 ? [[1]] : [insert!(copy(p), i, k) for p in permutations(k - 1) for i in 1:k]

# sign of the time-ordering permutation restricted to the fermionic operators
function permsign(P, fermionic)
    f = [j for j in P if fermionic[j]]
    count(f[i] > f[j] for i in eachindex(f) for j in i+1:lastindex(f); init=0) |> isodd ? -1 : 1
end

# G(ν) = -∫dτ e^{iντ} ⟨T c(τ) c⁺(0)⟩
G_ed(U, β, n) = -correlator(U, β, (cup, cup⁺), (ν(n, β),), (true,))

# χ(ω) = -∫dτ e^{iωτ} ⟨T ρ(τ) ρ(0)⟩ + β⟨ρ⟩² δ_ω0 for a density ρ
function χ_ed(U, β, ρ, k)
    E = energies(U)
    w = exp.(-β .* (E .- minimum(E)))
    avg = tr(Diagonal(w) * ρ) / sum(w)
    -correlator(U, β, (ρ, ρ), (ω(k, β),), (false,)) + (k == 0 ? β * avg^2 : 0)
end

# -⟨ρ⁻; ρ⁺⟩(ω) for the singlet pair ρ⁻ = c↓c↑, ρ⁺ = c↑⁺c↓⁺
χpair_ed(U, β, k) = -correlator(U, β, (cdn * cup, cup⁺ * cdn⁺), (ω(k, β),), (false,))

#=
Eq. 3: χ_ph,σσ´(ν, ν´, ω) = ∫dτ₁dτ₂dτ₃ e^{-iντ₁} e^{i(ν+ω)τ₂} e^{-i(ν´+ω)τ₃}
    [⟨T cσ⁺(τ₁) cσ(τ₂) cσ´⁺(τ₃) cσ´(0)⟩ - ⟨T cσ⁺(τ₁) cσ(τ₂)⟩⟨T cσ´⁺(τ₃) cσ´(0)⟩]
with ν = νₙ, ν´ = νₙ´, ω = ωₖ and σ´ = ↑ (`same = true`) or ↓.
=#
function χph_ed(U, β, same, n, n´, k)
    v, v´, w = ν(n, β), ν(n´, β), ω(k, β)
    c⁺2, c2 = same ? (cup⁺, cup) : (cdn⁺, cdn)
    full = correlator(U, β, (cup⁺, cup, c⁺2, c2), (-v, v + w, -(v´ + w)), (true, true, true))
    # the disconnected term is β δ_ω0 ∫dτ e^{-iντ} ⟨T c↑⁺(τ) c↑(0)⟩ ∫dτ e^{-iν´τ} ⟨T cσ´⁺(τ) cσ´(0)⟩
    k == 0 || return full
    full - β * correlator(U, β, (cup⁺, cup), (-v,), (true,)) * correlator(U, β, (c⁺2, c2), (-v´,), (true,))
end
end

@testitem "Exact diagonalization: G and χ(ω)" setup = [ExactDiagonalization] begin
    using SparseIR, LinearAlgebra

    (d, m, s, t) = CHANNELS
    for (U, β) in ((1.3, 2.1), (-0.7, 3.0), (5.0, 2.0), (-4.0, 6.0), (0.0, 1.0))
        at = HubbardAtom(U, β)
        for n in -3:2
            @test gf(at, FermionicFreq(2n + 1)) ≈ G_ed(U, β, n) rtol = 1e-12
        end
        for k in -2:2
            w = BosonicFreq(2k)
            @test chi(d, at, w) ≈ χ_ed(U, β, nup + ndn - I, k) atol = 1e-12
            @test chi(m, at, w) ≈ χ_ed(U, β, nup - ndn, k) atol = 1e-12
            @test chi(s, at, w) ≈ χpair_ed(U, β, k) atol = 1e-12
        end
    end
end

@testitem "Exact diagonalization: generalized susceptibility" setup = [ExactDiagonalization] begin
    using SparseIR

    (d, m, s, t) = CHANNELS
    F(n) = FermionicFreq(2n + 1)
    for (U, β) in ((1.3, 2.1), (-0.7, 3.0), (5.0, 2.0), (-4.0, 6.0))
        at = HubbardAtom(U, β)
        for n in -2:1, n´ in -2:1, k in -1:1
            χ_uu, χ_ud = χph_ed(U, β, true, n, n´, k), χph_ed(U, β, false, n, n´, k)
            # Eq. 4: pp notation is ph notation at bosonic frequency -ω - ν - ν´ (index shift n + n´ + 1)
            kpp = -k - (n + n´ + 1)
            χpp_uu, χpp_ud = χph_ed(U, β, true, n, n´, kpp), χph_ed(U, β, false, n, n´, kpp)
            χ0pp = n == n´ ? -β / 2 * G_ed(U, β, n) * G_ed(U, β, -n - k - 1) : 0  # Eq. 6b
            w = (F(n), F(n´), BosonicFreq(2k))
            scale = max(abs(χ_uu), abs(χ_ud), abs(χpp_uu), abs(χpp_ud), abs(χ0pp))
            # Eqs. 5a-d
            @test chi(d, at, w) ≈ χ_uu + χ_ud atol = 1e-12 * scale
            @test chi(m, at, w) ≈ χ_uu - χ_ud atol = 1e-12 * scale
            @test chi(s, at, w) ≈ (-χpp_uu + 2χpp_ud - 2χ0pp) / 4 atol = 1e-12 * scale
            @test chi(t, at, w) ≈ (χpp_uu + 2χ0pp) / 4 atol = 1e-12 * scale
        end
    end
end
