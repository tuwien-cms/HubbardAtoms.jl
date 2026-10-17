import HubbardAtomLean.LehmannInput

/-!
# The non-interacting atom

At `U = 0` the opposite-spin susceptibility vanishes and every channel is its bubble:
`χ_t = B`, `χ_s = -B`, `χ_d = χ_m = B` (diagonal). So the inverses are diagonal and unique, and
`Γ_r = 0` in all four channels.

* `Params.chiUpDn_of_a_eq_zero`, `Params.chiUpUp_of_a_eq_zero`: the input at `a = U/2 = 0`.
* `Params.gammaT_zero_of_a_eq_zero`, `Params.gammaS_zero_of_a_eq_zero`,
  `Params.gammaR_zero_of_a_eq_zero`: `Γ_r = β²(Y - η_r B⁻¹) = 0` for every right or left inverse
  `Y`, and `Params.exists_inverse_of_a_eq_zero`: such inverses exist.
-/

open Complex

namespace HubbardAtom

/-- A right inverse of a diagonal matrix `δₙₗ dₙ` with nonzero `dₙ` is `δₙₘ/dₙ`. -/
theorem eq_of_isRightInverse_diag {X Y : ℤ → ℤ → ℂ} {d : ℤ → ℂ} (hd : ∀ n, d n ≠ 0)
    (hX : ∀ n l, X n l = kron n l * d n) (hY : IsRightInverse X Y) (n m : ℤ) :
    Y n m = kron n m / d n := by
  have h := hY n m
  have h₁ : HasSum (fun l ↦ X n l * Y l m) (d n * Y n m) := by
    convert hasSum_single (f := fun l ↦ X n l * Y l m) n
      (fun l hl ↦ by rw [hX, kron_of_ne (Ne.symm hl), zero_mul, zero_mul]) using 1
    rw [hX, kron_self, one_mul]
  rw [eq_div_iff (hd n), mul_comm]
  exact h₁.unique h

/-- A left inverse of a diagonal matrix `δₙₗ dₙ` with nonzero `dₙ` is `δₙₘ/dₙ`. -/
theorem eq_of_isLeftInverse_diag {X Y : ℤ → ℤ → ℂ} {d : ℤ → ℂ} (hd : ∀ n, d n ≠ 0)
    (hX : ∀ n l, X n l = kron n l * d n) (hY : IsLeftInverse X Y) (n m : ℤ) :
    Y n m = kron n m / d n := by
  have h := hY n m
  have h₁ : HasSum (fun l ↦ Y n l * X l m) (Y n m * d m) := by
    convert hasSum_single (f := fun l ↦ Y n l * X l m) m
      (fun l hl ↦ by rw [hX, kron_of_ne hl, zero_mul, mul_zero]) using 1
    rw [hX, kron_self, one_mul]
  have h₂ := h₁.unique h
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self] at h₂ ⊢
    rw [eq_div_iff (hd n)]
    exact h₂
  · rw [kron_of_ne hnm] at h₂ ⊢
    rw [zero_div]
    exact (mul_eq_zero.mp h₂).resolve_right (hd m)

/-- The diagonal matrix `δₙₘ/dₙ` is a two-sided inverse of `δₙₘ dₙ`. -/
theorem isInverse_diag {d : ℤ → ℂ} (hd : ∀ n, d n ≠ 0) :
    IsRightInverse (fun n m ↦ kron n m * d n) (fun n m ↦ kron n m / d n) ∧
      IsLeftInverse (fun n m ↦ kron n m * d n) (fun n m ↦ kron n m / d n) := by
  refine ⟨fun n m ↦ ?_, fun n m ↦ ?_⟩
  · convert hasSum_single (f := fun l ↦ kron n l * d n * (kron l m / d l)) n
      (fun l hl ↦ by rw [kron_of_ne (Ne.symm hl), zero_mul, zero_mul]) using 1
    rw [kron_self, one_mul, mul_div_assoc', mul_comm, mul_div_assoc, div_self (hd n), mul_one]
  · convert hasSum_single (f := fun l ↦ kron n l / d n * (kron l m * d l)) m
      (fun l hl ↦ by rw [kron_of_ne hl, zero_mul, mul_zero]) using 1
    by_cases hnm : n = m
    · subst hnm
      rw [kron_self, one_mul, one_div, inv_mul_cancel₀ (hd n)]
    · rw [kron_of_ne hnm, kron_self, zero_div, zero_mul]

namespace Params

variable (A : Params)

theorem x_ne_zero (n : ℤ) : A.x n ≠ 0 := by
  have h := A.x_sub_ofReal_ne_zero n 0
  rwa [ofReal_zero, sub_zero] at h

/-- At `U = 0`, `G(x) = 1/x`. -/
theorem G_of_a_eq_zero (ha : A.a = 0) {z : ℂ} (hz : z ≠ 0) : A.G z = 1 / z := by
  rw [G, ha, ofReal_zero]
  field_simp
  ring

/-- At `U = 0` the opposite-spin susceptibility vanishes: the pair term cancels the twelve
fractions of Appendix C. -/
theorem chiUpDn_of_a_eq_zero (ha : A.a = 0) (k n m : ℤ) : A.chiUpDn k n m = 0 := by
  have hx := A.x_ne_zero n
  have hy := A.x_ne_zero m
  have hxw := A.x_ne_zero (n + k)
  have hyw := A.x_ne_zero (m + k)
  rw [chiUpDn, pairC, piPair', Rreg', ha, ofReal_zero, A.pair_denom]
  rw [← A.x_add_w n k, ← A.x_add_w m k] at *
  simp only [zero_pow two_ne_zero, mul_zero, zero_mul, sub_zero, add_zero, zero_add, J]
  by_cases hp : A.μ = 0 ∧ n + m + k + 1 = 0
  · rw [ite_eq_left hp]
    have hσ0 := (A.pair_eq_zero_iff k n m).mpr hp
    have hρ := p₂_eq_p₀_of_mu hp.1
    have hX : A.x n + A.w k = -A.x m := by linear_combination hσ0
    have hY : A.x m + A.w k = -A.x n := by linear_combination hσ0
    simp only [qm, qp, hρ]
    rw [hX, hY]
    field_simp
    ring
  · rw [ite_eq_right hp]
    have hσ0 : A.x n + A.x m + A.w k ≠ 0 := fun h0 ↦ hp ((A.pair_eq_zero_iff k n m).mp h0)
    simp only [qm, qp, ρ]
    push_cast
    field_simp
    ring

/-- At `U = 0` the same-spin susceptibility is the bubble: `χ↑↑ = δₙₘ B(νₙ)`. -/
theorem chiUpUp_of_a_eq_zero (ha : A.a = 0) (k n m : ℤ) :
    A.chiUpUp k n m = kron n m * A.Bph k n := by
  rw [chiUpUp, Bph, ha, ofReal_zero]
  ring

variable (k : ℤ)

theorem chiR_of_a_eq_zero (ha : A.a = 0) (ε : ℂ) (n m : ℤ) :
    A.chiR k ε n m = kron n m * A.Bph k n := by
  rw [chiR, A.chiUpUp_of_a_eq_zero ha, A.chiUpDn_of_a_eq_zero ha, mul_zero, add_zero]

theorem chiPPUpUp_of_a_eq_zero (ha : A.a = 0) (n m : ℤ) :
    A.chiPPUpUp k n m = kron n m * (2 * A.Bpp k n) := by
  rw [chiPPUpUp, A.chiUpUp_of_a_eq_zero ha]
  by_cases hnm : n = m
  · subst hnm
    rw [kron_self, Bph, Bpp, A.x_add_w]
    have h₁ : A.x (n + (-k - n - n - 1)) = A.μ + (-A.ν n - A.ω k) * I := by
      rw [show n + (-k - n - n - 1) = mirror k n by unfold mirror; ring, x, A.ν_mirror]
      push_cast
      ring
    rw [h₁, x]
    ring
  · rw [kron_of_ne hnm, zero_mul, zero_mul]

theorem chiT_of_a_eq_zero (ha : A.a = 0) (n m : ℤ) : A.chiT k n m = kron n m * A.Bpp k n := by
  rw [chiT, A.chiPPUpUp_of_a_eq_zero k ha]
  ring

theorem chiS_of_a_eq_zero (ha : A.a = 0) (n m : ℤ) :
    A.chiS k n m = kron n m * -A.Bpp k n := by
  rw [chiS, A.chiPPUpUp_of_a_eq_zero k ha, chiPPUpDn, A.chiUpDn_of_a_eq_zero ha]
  ring

theorem Bph_ne_zero_of_a_eq_zero (ha : A.a = 0) (n : ℤ) : A.Bph k n ≠ 0 := by
  rw [Bph, A.x_add_w, A.G_of_a_eq_zero ha (A.x_ne_zero n),
    A.G_of_a_eq_zero ha (A.x_ne_zero (n + k))]
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  exact mul_ne_zero (mul_ne_zero (neg_ne_zero.mpr hβ) (one_div_ne_zero (A.x_ne_zero n)))
    (one_div_ne_zero (A.x_ne_zero (n + k)))

/-- At `U = 0`, `Γ_t = β²(Y - B⁻¹) = 0` for every right inverse `Y` of `χ_t`. -/
theorem gammaT_zero_of_a_eq_zero (ha : A.a = 0) {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiT k) Y ∨ IsLeftInverse (A.chiT k) Y) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (Y n m - kron n m / A.Bpp k n) = 0 := by
  have hX : ∀ n l, A.chiT k n l = kron n l * A.Bpp k n := A.chiT_of_a_eq_zero k ha
  rcases hY with hY | hY
  · rw [eq_of_isRightInverse_diag (A.Bpp_ne_zero k) hX hY, sub_self, mul_zero]
  · rw [eq_of_isLeftInverse_diag (A.Bpp_ne_zero k) hX hY, sub_self, mul_zero]

/-- At `U = 0`, `Γ_s = β²(Y + B⁻¹) = 0` for every right or left inverse `Y` of `χ_s`. -/
theorem gammaS_zero_of_a_eq_zero (ha : A.a = 0) {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiS k) Y ∨ IsLeftInverse (A.chiS k) Y) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (Y n m + kron n m / A.Bpp k n) = 0 := by
  have hX : ∀ n l, A.chiS k n l = kron n l * -A.Bpp k n := A.chiS_of_a_eq_zero k ha
  have hd : ∀ n, -A.Bpp k n ≠ 0 := fun n ↦ neg_ne_zero.mpr (A.Bpp_ne_zero k n)
  rcases hY with hY | hY
  · rw [eq_of_isRightInverse_diag hd hX hY, div_neg, neg_add_cancel, mul_zero]
  · rw [eq_of_isLeftInverse_diag hd hX hY, div_neg, neg_add_cancel, mul_zero]

/-- At `U = 0`, `Γ_d = Γ_m = β²(Y - B⁻¹) = 0` for every right or left inverse `Y` of `χ_r`. -/
theorem gammaR_zero_of_a_eq_zero (ha : A.a = 0) (ε : ℂ) {Y : ℤ → ℤ → ℂ}
    (hY : IsRightInverse (A.chiR k ε) Y ∨ IsLeftInverse (A.chiR k ε) Y) (n m : ℤ) :
    (A.β : ℂ) ^ 2 * (Y n m - kron n m / A.Bph k n) = 0 := by
  have hX : ∀ n l, A.chiR k ε n l = kron n l * A.Bph k n := A.chiR_of_a_eq_zero k ha ε
  rcases hY with hY | hY
  · rw [eq_of_isRightInverse_diag (A.Bph_ne_zero_of_a_eq_zero k ha) hX hY, sub_self, mul_zero]
  · rw [eq_of_isLeftInverse_diag (A.Bph_ne_zero_of_a_eq_zero k ha) hX hY, sub_self, mul_zero]

/-- At `U = 0` the inverses exist: `χ_s` and `χ_r` are diagonal with nonzero entries. -/
theorem exists_inverse_of_a_eq_zero (ha : A.a = 0) (ε : ℂ) :
    (∃ Y, IsRightInverse (A.chiS k) Y ∧ IsLeftInverse (A.chiS k) Y) ∧
      ∃ Y, IsRightInverse (A.chiR k ε) Y ∧ IsLeftInverse (A.chiR k ε) Y := by
  have hs : A.chiS k = fun n m ↦ kron n m * -A.Bpp k n := by
    funext n m
    exact A.chiS_of_a_eq_zero k ha n m
  have hr : A.chiR k ε = fun n m ↦ kron n m * A.Bph k n := by
    funext n m
    exact A.chiR_of_a_eq_zero k ha ε n m
  rw [hs, hr]
  exact ⟨⟨_, isInverse_diag fun n ↦ neg_ne_zero.mpr (A.Bpp_ne_zero k n)⟩,
    ⟨_, isInverse_diag (A.Bph_ne_zero_of_a_eq_zero k ha)⟩⟩

end Params

end HubbardAtom
