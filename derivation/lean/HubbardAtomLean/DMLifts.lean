import HubbardAtomLean.DensityMagnetic

/-!
# Theorem "density and magnetic": the assumptions of the note

The generic case `Params.DMGeneric` asks, for each root `uⱼ` of `e`, that `P(uⱼ) ≠ 0` and `Kⱼ ≠ 0`
unless the root is the atomic pair. The note asks only `Kⱼ ≠ 0`, with the atomic pair omitted. Both
agree: at a zero of `P = Q₊Q₋`, `e` is a multiple of `K` (`beta_mul_eph_of_Qp`,
`beta_mul_eph_of_Qm`), so a root of `e` with `P(u) = 0` and `K(u) ≠ 0` is the atomic pair of
`r = m`, `k = 0` (`modeOK_of_K`).

* `Params.DMHyp`: the assumptions of Theorem "density and magnetic" as in the note (for `s₁ ≠ s₂`),
  and `Params.DMHyp.toDMGeneric`.
* `Params.DMGeneric.zph_neg`, `Params.DMGeneric.gammaR_neg₁`, `Params.DMGeneric.gammaR_neg₂`: the
  closed form does not depend on the signs of the roots `uⱼ` (Step 5 of the note, Eq. branch).
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-! ### Zeros of `P` -/

/-- At a zero of `Q₊`, `S_k(u) = S₀(-a)`. -/
theorem S_of_Qp_eq_zero {u : ℂ} (hu : A.Qp k u = 0) : A.S k u = A.S 0 (-A.a) := by
  have h : (u + A.a - A.w k / 2) * (u + A.a + A.w k / 2) = 0 := by
    rw [Qp] at hu
    linear_combination hu
  rcases mul_eq_zero.mp h with h | h
  · rw [A.S_shift k u]
    congr 1
    linear_combination h
  · rw [A.S_shift' k u]
    congr 1
    linear_combination h

/-- At a zero of `Q₋`, `S_k(u) = S₀(a)`. -/
theorem S_of_Qm_eq_zero {u : ℂ} (hu : A.Qm k u = 0) : A.S k u = A.S 0 A.a := by
  have h : (u - A.a - A.w k / 2) * (u - A.a + A.w k / 2) = 0 := by
    rw [Qm] at hu
    linear_combination hu
  rcases mul_eq_zero.mp h with h | h
  · rw [A.S_shift k u]
    congr 1
    linear_combination h
  · rw [A.S_shift' k u]
    congr 1
    linear_combination h

/-- At a zero of `Q₊`: `βe(u) = -2a(1 - ρ)(u - εa)K(u)`. -/
theorem beta_mul_eph_of_Qp (hε : ε = 1 ∨ ε = -1) {u : ℂ} (hu : A.Qp k u = 0) :
    (A.β : ℂ) * A.eph k ε u = -2 * A.a * (1 - A.ρ) * (u - ε * A.a) * A.Kph k ε u := by
  have hw : A.w k ^ 2 = 4 * (u + A.a) ^ 2 := by
    rw [Qp] at hu
    linear_combination -4 * hu
  have hs := A.weights_sum'
  have hq := A.qm_ne
  have hS := A.S_of_Qp_eq_zero k hu
  rw [A.S₀_neg_a] at hS
  rcases hε with rfl | rfl <;>
  · simp only [Kph, α, hS, eph, Hph, lam, q₀, hw, A.ρ_eq, cbar]
    push_cast
    field_simp
    generalize (A.p₂ : ℂ) = p₂ at hs hq ⊢
    obtain rfl : p₂ = 1 - A.p₀ - 2 * A.p₁ := by linear_combination hs
    ring

/-- At a zero of `Q₋`: `βe(u) = 2a(1 + ρ)(u + εa)K(u)`. -/
theorem beta_mul_eph_of_Qm (hε : ε = 1 ∨ ε = -1) {u : ℂ} (hu : A.Qm k u = 0) :
    (A.β : ℂ) * A.eph k ε u = 2 * A.a * (1 + A.ρ) * (u + ε * A.a) * A.Kph k ε u := by
  have hw : A.w k ^ 2 = 4 * (u - A.a) ^ 2 := by
    rw [Qm] at hu
    linear_combination -4 * hu
  have hs := A.weights_sum'
  have hq := A.qp_ne
  have hS := A.S_of_Qm_eq_zero k hu
  rw [A.S₀_a] at hS
  rcases hε with rfl | rfl <;>
  · simp only [Kph, α, hS, eph, Hph, lam, q₀, hw, A.ρ_eq, cbar]
    push_cast
    field_simp
    generalize (A.p₂ : ℂ) = p₂ at hs hq ⊢
    obtain rfl : p₂ = 1 - A.p₀ - 2 * A.p₁ := by linear_combination hs
    ring

theorem one_sub_rho_ne : (1 : ℂ) - A.ρ ≠ 0 := by
  rw [A.ρ_eq, show (1 : ℂ) - (A.p₂ - A.p₀) = 2 * (A.p₀ + A.p₁) by
    linear_combination -A.weights_sum']
  exact mul_ne_zero two_ne_zero A.qm_ne

theorem one_add_rho_ne : (1 : ℂ) + A.ρ ≠ 0 := by
  rw [A.ρ_eq, show (1 : ℂ) + (A.p₂ - A.p₀) = 2 * (A.p₁ + A.p₂) by
    linear_combination -A.weights_sum']
  exact mul_ne_zero two_ne_zero A.qp_ne

/-- `w² = 16a²` is impossible for `a ≠ 0`: `w` is imaginary. -/
theorem w_sq_ne (ha : (A.a : ℂ) ≠ 0) : A.w k ^ 2 ≠ 16 * A.a ^ 2 := by
  intro h
  have h' : ((-(A.ω k) ^ 2 : ℝ) : ℂ) = ((16 * A.a ^ 2 : ℝ) : ℂ) := by
    push_cast
    rw [← h, w]
    ring_nf
    rw [I_sq]
    ring
  have h'' := ofReal_injective h'
  have ha' : A.a ≠ 0 := fun h0 ↦ ha (by rw [h0, ofReal_zero])
  have : 0 < A.a ^ 2 := by positivity
  nlinarith [sq_nonneg (A.ω k)]

theorem k_eq_zero_of_w (h : A.w k = 0) : k = 0 := by
  have hβ := A.β_pos.ne'
  simp only [w, ω, mul_eq_zero, I_ne_zero, or_false, ofReal_eq_zero, div_eq_zero_iff, hβ,
    or_false, Real.pi_ne_zero, Int.cast_eq_zero, two_ne_zero, false_or] at h
  exact h

/-- A root of `e` with `P(u) = 0` and `K(u) ≠ 0` is the atomic pair: `r = m` (`ε = -1`), `w = 0`
and `u² = a²`. -/
theorem atomic_of_Pph_eq_zero (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) {u : ℂ}
    (he : A.eph k ε u = 0) (hK : A.Kph k ε u ≠ 0) (hP : A.Pph k u = 0) :
    ε = -1 ∧ A.w k = 0 ∧ u ^ 2 = A.a ^ 2 := by
  rcases mul_eq_zero.mp hP with hQ | hQ
  · have h₁ := A.beta_mul_eph_of_Qp k ε hε hQ
    rw [he, mul_zero] at h₁
    have hu : u - ε * A.a = 0 := by
      have h₂ : (-2 * A.a * (1 - A.ρ) : ℂ) ≠ 0 :=
        mul_ne_zero (mul_ne_zero (by norm_num) ha) A.one_sub_rho_ne
      rcases mul_eq_zero.mp h₁.symm with h₃ | h₃
      · exact (mul_eq_zero.mp h₃).resolve_left h₂
      · exact absurd h₃ hK
    have hw : A.w k ^ 2 = 4 * (u + A.a) ^ 2 := by
      rw [Qp] at hQ
      linear_combination -4 * hQ
    obtain rfl : u = ε * A.a := by linear_combination hu
    rcases hε with rfl | rfl
    · exact absurd (by linear_combination hw) (A.w_sq_ne k ha)
    · exact ⟨rfl, pow_eq_zero_iff two_ne_zero |>.mp (by linear_combination hw), by ring⟩
  · have h₁ := A.beta_mul_eph_of_Qm k ε hε hQ
    rw [he, mul_zero] at h₁
    have hu : u + ε * A.a = 0 := by
      have h₂ : (2 * A.a * (1 + A.ρ) : ℂ) ≠ 0 :=
        mul_ne_zero (mul_ne_zero two_ne_zero ha) A.one_add_rho_ne
      rcases mul_eq_zero.mp h₁.symm with h₃ | h₃
      · exact (mul_eq_zero.mp h₃).resolve_left h₂
      · exact absurd h₃ hK
    have hw : A.w k ^ 2 = 4 * (u - A.a) ^ 2 := by
      rw [Qm] at hQ
      linear_combination -4 * hQ
    obtain rfl : u = -(ε * A.a) := by linear_combination hu
    rcases hε with rfl | rfl
    · exact absurd (by linear_combination hw) (A.w_sq_ne k ha)
    · exact ⟨rfl, pow_eq_zero_iff two_ne_zero |>.mp (by linear_combination hw), by ring⟩

/-- The note's assumption on a root `u` of `e` (`K(u) ≠ 0`, unless it is the atomic pair) implies
`ModeOK`: a root with `P(u) = 0` and `K(u) ≠ 0` is the atomic pair of `r = m`, `k = 0`. -/
theorem modeOK_of_K (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) {u : ℂ} (he : A.eph k ε u = 0)
    (h : A.Kph k ε u ≠ 0 ∨ (A.Pph k u = 0 ∧ ∀ z, A.Wph k ε z (u ^ 2) = 0)) :
    A.ModeOK k ε u := by
  rcases h with hK | hat
  · by_cases hP : A.Pph k u = 0
    · -- `u` is the atomic pair
      right
      refine ⟨hP, ?_⟩
      obtain ⟨rfl, hw, hu⟩ := A.atomic_of_Pph_eq_zero k ε hε ha he hK hP
      obtain rfl := A.k_eq_zero_of_w k hw
      intro z
      rw [hu]
      simp only [Wph, W₀, W₂, ψ, q₀, hw, kron_self]
      ring
    · exact Or.inl ⟨hP, hK⟩
  · exact Or.inr hat

/-- The assumptions of Theorem "density and magnetic" as stated in the note, for roots `u₁`, `u₂` of
`e` with `s₁ ≠ s₂`: `Kⱼ ≠ 0`, except for the atomic pair (`P(uⱼ) = 0`, `W(z; sⱼ) = 0`). -/
structure DMHyp (u₁ u₂ : ℂ) : Prop where
  ε_eq : ε = 1 ∨ ε = -1
  a_ne : (A.a : ℂ) ≠ 0
  μ_ne : A.μ ≠ 0
  e_ne : ∀ n, A.eph k ε (A.z k n) ≠ 0
  e_u₁ : A.eph k ε u₁ = 0
  e_u₂ : A.eph k ε u₂ = 0
  s_ne : u₁ ^ 2 ≠ u₂ ^ 2
  K₁ : A.Kph k ε u₁ ≠ 0 ∨ (A.Pph k u₁ = 0 ∧ ∀ z, A.Wph k ε z (u₁ ^ 2) = 0)
  K₂ : A.Kph k ε u₂ ≠ 0 ∨ (A.Pph k u₂ = 0 ∧ ∀ z, A.Wph k ε z (u₂ ^ 2) = 0)

variable {A k ε} {u₁ u₂ : ℂ}

theorem DMHyp.toDMGeneric (h : A.DMHyp k ε u₁ u₂) : A.DMGeneric k ε u₁ u₂ :=
  ⟨h.ε_eq, h.a_ne, h.μ_ne, h.e_ne, h.e_u₁, h.e_u₂, h.s_ne,
    A.modeOK_of_K k ε h.ε_eq h.a_ne h.e_u₁ h.K₁, A.modeOK_of_K k ε h.ε_eq h.a_ne h.e_u₂ h.K₂⟩

/-! ### The signs of the roots -/

/-- At a root, `K(u) = 0` iff `K(-u) = 0`. -/
theorem DMGeneric.Kph_neg_ne (h : A.DMGeneric k ε u₁ u₂) {u : ℂ} (hu : A.eph k ε u = 0)
    (hK : A.Kph k ε u ≠ 0) : A.Kph k ε (-u) ≠ 0 := by
  have hu' : A.eph k ε (-u) = 0 := by rw [A.eph_neg, hu]
  have hrel := h.hrel hu
  have hα := h.α_ne hu'
  intro h0
  apply hK
  simp only [Kph, neg_neg] at h0 ⊢
  have hβ := A.β_ne_zero
  have hlam := A.lam_ne_zero ε h.ε_eq h.a_ne
  -- `βH(u) = -λα(-u)`, so the root relation gives `βH(-u) = -λα(u)`
  have e₁ : (A.β : ℂ) * A.Hph k ε u = -(A.lam ε * A.α k (-u)) := by linear_combination h0
  apply mul_left_cancel₀ (mul_ne_zero hlam hα)
  linear_combination (A.β * A.Hph k ε (-u)) * h0 - hrel

/-- Eq. (branch): `Z(-u) = Z(u)` at a root satisfying `ModeOK`. -/
theorem DMGeneric.zph_neg (h : A.DMGeneric k ε u₁ u₂) {u : ℂ} (hu : A.eph k ε u = 0)
    (hmode : A.ModeOK k ε u) : A.Zph k ε (-u) = A.Zph k ε u := by
  rcases hmode with ⟨hP, hK⟩ | ⟨hP, _⟩
  · have hK' := h.Kph_neg_ne hu hK
    have hrel := h.hrel hu
    simp only [Zph, A.Pph_neg, neg_neg] at hK' ⊢
    simp only [Kph, neg_neg] at hK hK' ⊢
    field_simp
    linear_combination (-2 * u) * hrel
  · simp only [Zph, A.Pph_neg, hP, mul_zero, zero_mul, zero_div]

theorem DMGeneric.modeOK_neg (h : A.DMGeneric k ε u₁ u₂) {u : ℂ} (hu : A.eph k ε u = 0)
    (hmode : A.ModeOK k ε u) : A.ModeOK k ε (-u) := by
  rcases hmode with ⟨hP, hK⟩ | ⟨hP, hW⟩
  · exact Or.inl ⟨by rwa [A.Pph_neg], h.Kph_neg_ne hu hK⟩
  · exact Or.inr ⟨by rwa [A.Pph_neg], fun z ↦ by rw [neg_sq]; exact hW z⟩

/-- The other sign of `u₁` satisfies the assumptions too. -/
theorem DMGeneric.neg₁ (h : A.DMGeneric k ε u₁ u₂) : A.DMGeneric k ε (-u₁) u₂ :=
  ⟨h.ε_eq, h.a_ne, h.μ_ne, h.e_ne, by rw [A.eph_neg, h.e_u₁], h.e_u₂,
    by rw [neg_sq]; exact h.s_ne, h.modeOK_neg h.e_u₁ h.mode₁, h.mode₂⟩

/-- The other sign of `u₂` satisfies the assumptions too. -/
theorem DMGeneric.neg₂ (h : A.DMGeneric k ε u₁ u₂) : A.DMGeneric k ε u₁ (-u₂) :=
  ⟨h.ε_eq, h.a_ne, h.μ_ne, h.e_ne, h.e_u₁, by rw [A.eph_neg, h.e_u₂],
    by rw [neg_sq]; exact h.s_ne, h.mode₁, h.modeOK_neg h.e_u₂ h.mode₂⟩

/-- Eq. (Gamma-ph) does not depend on the sign of `u₁` (Step 5). -/
theorem DMGeneric.gammaR_neg₁ (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    A.gammaR k ε (-u₁) u₂ n m = A.gammaR k ε u₁ u₂ n m := by
  rw [gammaR, gammaR, h.zph_neg h.e_u₁ h.mode₁, neg_sq]

/-- Eq. (Gamma-ph) does not depend on the sign of `u₂` (Step 5). -/
theorem DMGeneric.gammaR_neg₂ (h : A.DMGeneric k ε u₁ u₂) (n m : ℤ) :
    A.gammaR k ε u₁ (-u₂) n m = A.gammaR k ε u₁ u₂ n m := by
  rw [gammaR, gammaR, h.zph_neg h.e_u₂ h.mode₂, neg_sq]

end Params

end HubbardAtom
