import HubbardAtomLean.PHSums

/-!
# The rational identities of the density and magnetic channels

Identities (A6) and (A7) of Appendix A, in the form in which the lattice sums of `PHSums` produce
them (with `S±` from the weights, `S₊ = β(p₁ - p₀)/(2q₋)`, `S₋ = β(p₂ - p₁)/(2q₊)`), and the
partial fractions in `s = v²` of Step 4.

* `Params.identity_A6`: the image of the constant, `-βH(z) + λc̄ + ∑ⱼ cNⱼ(z)ℓⱼ = βN₂(z)/(2εa)`.
* `Params.identity_A7`: Eq. (source), with `W(z; s)` of (A7).
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- The weights in the form used by the identities: `q₋ = p₀ + p₁ ≠ 0`, `q₊ = p₁ + p₂ ≠ 0`,
`p₀ + 2p₁ + p₂ = 1`, `ρ = p₂ - p₀`. -/
theorem qm_ne : (A.p₀ : ℂ) + A.p₁ ≠ 0 := by exact_mod_cast (add_pos A.p₀_pos A.p₁_pos).ne'

theorem qp_ne : (A.p₁ : ℂ) + A.p₂ ≠ 0 := by exact_mod_cast (add_pos A.p₁_pos A.p₂_pos).ne'

theorem weights_sum' : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₂ = 1 := by exact_mod_cast A.weights_sum

theorem ρ_eq : (A.ρ : ℂ) = A.p₂ - A.p₀ := by rw [ρ]; push_cast; ring

/-- Identity (A6), Step 2 of the proof of Theorem "density and magnetic": the image of the constant
function, `-βH(z) + λc̄ + PV ∑ₗ N(z, zₗ)/P(zₗ) = βN₂(z)/(2εa)`, the sum being
`λc̄ + ∑ⱼ cNⱼ(z)ℓⱼ - λc̄` by `hasSum_const`. -/
theorem identity_A6 (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (z : ℂ) :
    -A.β * A.Hph k ε z + A.lam ε * A.cbar +
        (A.cN₀ k ε z * A.ℓ₀ k + A.cN₁ k ε z * A.ℓ₁ k + A.cN₂ k ε z * A.ℓ₂ k + A.cN₃ ε * A.ℓ₃ k) =
      A.β * A.N₂ k ε z / (2 * ε * A.a) := by
  have hq := A.q₀_ne_zero k ha
  have hβ := A.β_ne_zero
  have hqm := A.qm_ne
  have hqp := A.qp_ne
  have hs := A.weights_sum'
  have hρ := A.ρ_eq
  simp only [Hph, lam, cbar, cN₀, cN₁, cN₂, cN₃, ℓ₀, ℓ₁, ℓ₂, ℓ₃, Tp, Tm, Sp, Sm, N₂, ζ, κ,
    A.S₀_neg_a, A.S₀_a]
  rw [q₀] at hq ⊢
  push_cast
  rw [hρ]
  generalize (A.p₂ : ℂ) = p₂ at *
  obtain rfl : p₂ = 1 - A.p₀ - 2 * A.p₁ := by linear_combination hs
  rcases hε with rfl | rfl <;> by_cases hk : k = 0
  · subst hk
    simp only [kron_self, A.w_zero] at hq ⊢
    field_simp
    ring
  · rw [kron_of_ne hk]
    have hq4 : (A.a : ℂ) ^ 2 * 4 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    have hq4' : 4 * (A.a : ℂ) ^ 2 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    field_simp
    ring
  · subst hk
    simp only [kron_self, A.w_zero] at hq ⊢
    field_simp
    ring
  · rw [kron_of_ne hk]
    have hq4 : (A.a : ℂ) ^ 2 * 4 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    have hq4' : 4 * (A.a : ℂ) ^ 2 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    field_simp
    ring

/-- `ϖ = 4(p₀p₂ - p₁²)` of (A7). -/
noncomputable def ϖ : ℂ := 4 * (A.p₀ * A.p₂ - A.p₁ ^ 2)

/-- `ψ(s)` of (A7). -/
noncomputable def ψ (s : ℂ) : ℂ :=
  -4 * ε * A.p₁ * (A.p₀ + A.p₂ - A.ρ ^ 2) * (s + (1 + 2 * ε) * A.a ^ 2) +
    (1 + ε) * A.ρ ^ 2 * A.ϖ * (s - A.a ^ 2)

/-- `W₀(s)` of (A7). -/
noncomputable def W₀ (s : ℂ) : ℂ :=
  ε * A.a * A.q₀ k * (A.ϖ - 2 * ε * (1 - A.ρ ^ 2)) * (s + A.q₀ k + 2 * ε * A.a ^ 2) +
    kron k 0 * A.β * A.a ^ 4 * A.ψ ε s

/-- `W₂(s)` of (A7). -/
noncomputable def W₂ (s : ℂ) : ℂ :=
  ε * A.a * A.ϖ * (s + A.q₀ k + 2 * ε * A.a ^ 2) - kron k 0 * A.β * A.a ^ 2 * A.ψ ε s

/-- `W(z; s) = W₀(s) + z²W₂(s)`, a polynomial of degree one in `s`. -/
noncomputable def Wph (z s : ℂ) : ℂ := A.W₀ k ε s + z ^ 2 * A.W₂ k ε s

/-- Identity (A7), Eq. (source): with `P(v)Ī(z; -v) = c̄N(z, -v)/ρ + R̂(z, -v)` (`P` is even),
`λρ(z - v + 2aρ)P(v) - H(-v)N(z, v) - (λρ/β)P(v)Ī(z; -v)
  = -e(v)N₂(z) + λρz[P(v) - e(v)] + λW(z; v²)`, for all `z`, `v`. -/
theorem identity_A7 (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (z v : ℂ) :
    A.lam ε * A.ρ * (z - v + 2 * A.a * A.ρ) * A.Pph k v - A.Hph k ε (-v) * A.Nph k ε z v -
        A.lam ε / A.β * (A.cbar * A.Nph k ε z (-v) + A.ρ * A.Rhat k ε z (-v)) =
      -A.eph k ε v * A.N₂ k ε z + A.lam ε * A.ρ * z * (A.Pph k v - A.eph k ε v) +
        A.lam ε * A.Wph k ε z (v ^ 2) := by
  have hq := A.q₀_ne_zero k ha
  have hβ := A.β_ne_zero
  have hqm := A.qm_ne
  have hqp := A.qp_ne
  have hs := A.weights_sum'
  have hρ := A.ρ_eq
  simp only [Rhat, A.Nph_eq_coeffs, A.Pph_eq_poly, eph, Hph, Wph, W₀, W₂, ψ, ϖ, lam, cbar, cN₀,
    cN₁, cN₂, cN₃, ℓ₀, ℓ₁, ℓ₂, ℓ₃, Tp, Tm, Sp, Sm, N₂, ζ, κ, A.S₀_neg_a, A.S₀_a]
  rw [q₀] at hq ⊢
  push_cast
  rw [hρ]
  generalize (A.p₂ : ℂ) = p₂ at *
  obtain rfl : p₂ = 1 - A.p₀ - 2 * A.p₁ := by linear_combination hs
  rcases hε with rfl | rfl <;> by_cases hk : k = 0
  · subst hk
    simp only [kron_self, A.w_zero] at hq ⊢
    field_simp
    ring
  · rw [kron_of_ne hk]
    have hq4 : (A.a : ℂ) ^ 2 * 4 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    have hq4' : 4 * (A.a : ℂ) ^ 2 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    field_simp
    ring
  · subst hk
    simp only [kron_self, A.w_zero] at hq ⊢
    field_simp
    ring
  · rw [kron_of_ne hk]
    have hq4 : (A.a : ℂ) ^ 2 * 4 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    have hq4' : 4 * (A.a : ℂ) ^ 2 - A.w k ^ 2 ≠ 0 := fun h ↦ hq (by linear_combination h / 4)
    field_simp
    ring

end Params

end HubbardAtom
