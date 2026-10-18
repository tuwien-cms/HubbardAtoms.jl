import HubbardAtomLean.Lehmann
import HubbardAtomLean.PHInput

/-!
# The input from the Hamiltonian: evaluation

The propagator and the generalized susceptibilities defined from the Hamiltonian in
`Lehmann.lean` equal the expressions that the note takes as input from Ref. [Eßl et al. 2024].

* `Params.propagator_eq`: `G(ν_n) = G(x_n) = (x_n + aρ)/(x_n² - a²)`, Eq. (G).
* `Params.chiLehmann_upUp`: `χ↑↑(ν_n, ν_m, ω_k)` is Eq. (chi-uu) (`Params.chiUpUp`).
* `Params.chiLehmann_upDn`: `χ↑↓(ν_n, ν_m, ω_k)` is Eq. (chi-ud) with Eq. (pair) and the twelve
  fractions of Appendix C (`Params.chiUpDn`), including the pair coefficient `c_ℓ = βp₀δ_ℓ0` at
  `μ = 0`.

Of the six orderings, two contribute to `χ↑↑` (two paths each) and all six to `χ↑↓` (one path
each). A partial sum of the rates vanishes exactly when `k = 0`, `n = m`, or `μ = 0` and
`n + m + k + 1 = 0` (the transition `|0⟩ ↔ |↑↓⟩`); these are the cases of the closing algebra.
-/

open Complex Matrix

namespace HubbardAtom

theorem cUp_eq : cUp = !![0, 1, 0, 0; 0, 0, 0, 0; 0, 0, 0, 1; 0, 0, 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cUp, cUpDag]

theorem cDn_eq : cDn = !![0, 0, 1, 0; 0, 0, 0, -1; 0, 0, 0, 0; 0, 0, 0, 0] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [cDn, cDnDag]

namespace Params

variable (A : Params)

theorem weight_zero : A.weight 0 = A.p₀ := rfl
theorem weight_one : A.weight 1 = A.p₁ := rfl
theorem weight_two : A.weight 2 = A.p₁ := rfl
theorem weight_three : A.weight 3 = A.p₂ := rfl

/-- `e^{(x_n - μ)β} = e^{iν_nβ} = -1`. -/
theorem cexp_x_sub_mu (n : ℤ) : cexp ((A.x n - A.μ) * A.β) = -1 := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  have h : (A.x n - A.μ) * A.β = n * (2 * Real.pi * I) + Real.pi * I := by
    simp only [x, ν]
    push_cast
    field_simp
    ring
  rw [h, Complex.exp_add, Complex.exp_int_mul_two_pi_mul_I, Complex.exp_pi_mul_I, one_mul]

/-- Eq. (G): the propagator of the Hamiltonian is `G(x_n) = (x_n + aρ)/(x_n² - a²)`. -/
theorem propagator_eq (n : ℤ) : A.propagator (A.ν n) = A.G (A.x n) := by
  have hψ := A.cexp_x_sub_mu n
  have h : A.propagator (A.ν n) = -∫ τ in (0 : ℝ)..A.β,
      cexp ((A.x n - A.μ) * τ) * A.thermal (A.heis τ cUp * cUpᴴ) := by
    rw [propagator]
    congr 1
    refine integral_congr_Ioo A.β_pos.le fun τ hτ ↦ ?_
    rw [A.timeOrdered_12 _ _ hτ.1, heis_zero,
      show (A.ν n : ℂ) * τ * I = (A.x n - A.μ) * τ by simp only [x]; ring]
  rw [h, lineTerm_eq]
  simp [Fin.sum_univ_four, cUp_eq]
  rw [A.pathTerm₁_eq 2 3 hψ (r := A.x n - A.a) (by simp [rate, energy]; ring),
    A.pathTerm₁_eq 0 1 hψ (r := A.x n + A.a) (by simp [rate, energy]; ring),
    A.G_eq_lehmann (A.x_add_ofReal_ne_zero n A.a) (A.x_sub_ofReal_ne_zero n A.a)]
  have h₁ := A.x_add_ofReal_ne_zero n A.a
  have h₂ := A.x_sub_ofReal_ne_zero n A.a
  simp only [weight_zero, weight_one, weight_two, weight_three, qm, qp]
  push_cast
  field_simp
  ring

/-! ### The frequencies of the susceptibility -/

theorem cexp_mu_sub_x (n : ℤ) : cexp ((A.μ - A.x n) * A.β) = -1 := by
  rw [show ((A.μ : ℂ) - A.x n) * A.β = -((A.x n - A.μ) * A.β) by ring, Complex.exp_neg,
    A.cexp_x_sub_mu]
  norm_num

theorem w_eq_zero_iff (k : ℤ) : A.w k = 0 ↔ k = 0 := by
  have hβ := A.β_pos.ne'
  simp only [w, ω, mul_eq_zero, I_ne_zero, or_false, ofReal_eq_zero, div_eq_zero_iff, hβ,
    Real.pi_ne_zero, Int.cast_eq_zero, two_ne_zero, false_or]

theorem x_sub_x_eq_zero_iff (n m : ℤ) : A.x n - A.x m = 0 ↔ n = m := by
  constructor
  · intro h
    have him := congrArg Complex.im h
    simp only [sub_im, x_im, zero_im] at him
    have hβ := A.β_pos.ne'
    simp only [ν] at him
    field_simp at him
    have : ((n : ℝ) - m) * Real.pi = 0 := by linear_combination him / 2
    have hnm : (n : ℝ) - m = 0 := by
      rcases mul_eq_zero.mp this with h | h
      · exact h
      · exact absurd h Real.pi_ne_zero
    exact_mod_cast sub_eq_zero.mp hnm
  · rintro rfl
    exact sub_self _

theorem pair_eq_zero_iff (k n m : ℤ) :
    A.x n + A.x m + A.w k = 0 ↔ A.μ = 0 ∧ n + m + k + 1 = 0 := by
  have hβ := A.β_pos.ne'
  constructor
  · intro h
    have hre := congrArg Complex.re h
    have him := congrArg Complex.im h
    simp [x, w] at hre him
    refine ⟨by linarith, ?_⟩
    simp only [ν, ω] at him
    field_simp at him
    have : (2 * ((n : ℝ) + m + k + 1)) * Real.pi = 0 := by linear_combination him
    have h' : (n : ℝ) + m + k + 1 = 0 := by
      rcases mul_eq_zero.mp this with h | h
      · linarith
      · exact absurd h Real.pi_ne_zero
    exact_mod_cast h'
  · rintro ⟨hμ, hk⟩
    have hk' : (k : ℝ) = -n - m - 1 := by
      have : ((n + m + k + 1 : ℤ) : ℝ) = 0 := by exact_mod_cast hk
      push_cast at this
      linarith
    simp only [x, w, ν, ω, hμ, hk']
    push_cast
    field_simp
    ring

/-- The exponent of the susceptibility in the variables `x_n, x_m, w`. -/
theorem freq_eq (k n m : ℤ) (τ₁ τ₂ τ₃ : ℝ) :
    (((-A.ν n * τ₁ + (A.ν n + A.ω k) * τ₂ - (A.ν m + A.ω k) * τ₃ : ℝ) : ℂ) * I) =
      (A.μ - A.x n) * τ₁ + (A.x n + A.w k - A.μ) * τ₂ + (A.μ - A.x m - A.w k) * τ₃ := by
  simp only [x, w]
  push_cast
  ring

/-! ### The disconnected part -/

/-- The two-time integral of the disconnected part: `β δ_k0 G(x_n)`. -/
theorem disc_triangle (k n : ℤ) :
    triangleW A.β (A.weight 1) (A.weight 0) (A.weight 1) (-(A.x n + A.a))
          (A.x n + A.w k + A.a) (A.w k) (k = 0) +
        triangleW A.β (A.weight 3) (A.weight 2) (A.weight 3) (-(A.x n - A.a))
          (A.x n + A.w k - A.a) (A.w k) (k = 0) -
      (triangleW A.β (A.weight 0) (A.weight 1) (A.weight 0) (A.x n + A.w k + A.a)
          (-(A.x n + A.a)) (A.w k) (k = 0) +
        triangleW A.β (A.weight 2) (A.weight 3) (A.weight 2) (A.x n + A.w k - A.a)
          (-(A.x n - A.a)) (A.w k) (k = 0)) =
      if k = 0 then A.β * A.G (A.x n) else 0 := by
  have hX₁ := A.x_add_ofReal_ne_zero n A.a
  have hX₂ := A.x_sub_ofReal_ne_zero n A.a
  have hXW₁ := A.x_add_ofReal_ne_zero (n + k) A.a
  have hXW₂ := A.x_sub_ofReal_ne_zero (n + k) A.a
  rw [← x_add_w] at hXW₁ hXW₂
  simp only [triangleW, weight_zero, weight_one, weight_two, weight_three]
  by_cases hk : k = 0
  · simp only [ite_eq_left hk]
    subst hk
    rw [A.G_eq_lehmann hX₁ hX₂]
    simp only [w_zero, add_zero] at *
    simp only [qm, qp]
    push_cast
    field_simp
    ring
  · simp only [ite_eq_right hk]
    have hW := mt (A.w_eq_zero_iff k).mp hk
    field_simp
    ring

/-- The single-time integral of the disconnected part, for either spin: `G(x_m + w)`. -/
theorem disc_line (k m : ℤ) {q₀ q₁ q₂ q₃ : ℝ} (h₀ : q₀ = A.p₀) (h₁ : q₁ = A.p₁) (h₂ : q₂ = A.p₁)
    (h₃ : q₃ = A.p₂) :
    (-(q₀ : ℂ) - q₁) / -(A.x m + A.w k + A.a) + (-(q₂ : ℂ) - q₃) / -(A.x m + A.w k - A.a) =
      A.G (A.x m + A.w k) := by
  have hYW₁ := A.x_add_ofReal_ne_zero (m + k) A.a
  have hYW₂ := A.x_sub_ofReal_ne_zero (m + k) A.a
  rw [← x_add_w] at hYW₁ hYW₂
  subst h₀ h₁ h₂ h₃
  rw [A.G_eq_lehmann hYW₁ hYW₂, qm, qp]
  push_cast
  field_simp
  ring

/-! ### The same-spin susceptibility -/

theorem chiLehmann_upUp (k n m : ℤ) :
    A.chiLehmann cUp cUp (A.ν n) (A.ν m) (A.ω k) = A.chiUpUp k n m := by
  have h₁ := A.cexp_mu_sub_x n
  have h₂ : cexp ((A.x n + A.w k - A.μ) * A.β) = -1 := by
    rw [x_add_w, A.cexp_x_sub_mu]
  have h₃ : cexp ((A.μ - A.x m - A.w k) * A.β) = -1 := by
    rw [sub_sub, x_add_w, A.cexp_mu_sub_x]
  rw [A.chiLehmann_eq _ _ _ _ _ (A.freq_eq k n m)]
  simp only [orderedTerm_eq, triangleTerm_eq, lineTerm_eq]
  simp [Fin.sum_univ_four, cUp_eq]
  have hk := A.w_eq_zero_iff k
  have hnm := A.x_sub_x_eq_zero_iff n m
  rw [A.pathTerm_eq 1 0 1 0 h₁ h₂ h₃ (r₁ := -(A.x n + A.a)) (r₂ := A.x n + A.w k + A.a)
      (r₃ := -(A.x m + A.w k + A.a)) (r₁₂ := A.w k) (r₂₃ := A.x n - A.x m)
      (r₁₂₃ := -(A.x m + A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hnm,
    A.pathTerm_eq 3 2 3 2 h₁ h₂ h₃ (r₁ := -(A.x n - A.a)) (r₂ := A.x n + A.w k - A.a)
      (r₃ := -(A.x m + A.w k - A.a)) (r₁₂ := A.w k) (r₂₃ := A.x n - A.x m)
      (r₁₂₃ := -(A.x m - A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hnm,
    A.pathTerm_eq 1 0 1 0 h₃ h₂ h₁ (r₁ := -(A.x m + A.w k + A.a)) (r₂ := A.x n + A.w k + A.a)
      (r₃ := -(A.x n + A.a)) (r₁₂ := A.x n - A.x m) (r₂₃ := A.w k)
      (r₁₂₃ := -(A.x m + A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hk,
    A.pathTerm_eq 3 2 3 2 h₃ h₂ h₁ (r₁ := -(A.x m + A.w k - A.a)) (r₂ := A.x n + A.w k - A.a)
      (r₃ := -(A.x n - A.a)) (r₁₂ := A.x n - A.x m) (r₂₃ := A.w k)
      (r₁₂₃ := -(A.x m - A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hk,
    A.pathTerm₂_eq 1 0 h₁ h₂ (r₁ := -(A.x n + A.a)) (r₂ := A.x n + A.w k + A.a) (r₁₂ := A.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    A.pathTerm₂_eq 3 2 h₁ h₂ (r₁ := -(A.x n - A.a)) (r₂ := A.x n + A.w k - A.a) (r₁₂ := A.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    A.pathTerm₂_eq 0 1 h₂ h₁ (r₁ := A.x n + A.w k + A.a) (r₂ := -(A.x n + A.a)) (r₁₂ := A.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    A.pathTerm₂_eq 2 3 h₂ h₁ (r₁ := A.x n + A.w k - A.a) (r₂ := -(A.x n - A.a)) (r₁₂ := A.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    A.pathTerm₁_eq 1 0 h₃ (r := -(A.x m + A.w k + A.a)) (by simp [rate, energy]; ring),
    A.pathTerm₁_eq 3 2 h₃ (r := -(A.x m + A.w k - A.a)) (by simp [rate, energy]; ring),
    A.disc_triangle, A.disc_line k m A.weight_zero A.weight_one A.weight_two A.weight_three]
  have hX₁ := A.x_add_ofReal_ne_zero n A.a
  have hX₂ := A.x_sub_ofReal_ne_zero n A.a
  have hY₁ := A.x_add_ofReal_ne_zero m A.a
  have hY₂ := A.x_sub_ofReal_ne_zero m A.a
  have hXW₁ := A.x_add_ofReal_ne_zero (n + k) A.a
  have hXW₂ := A.x_sub_ofReal_ne_zero (n + k) A.a
  have hYW₁ := A.x_add_ofReal_ne_zero (m + k) A.a
  have hYW₂ := A.x_sub_ofReal_ne_zero (m + k) A.a
  rw [← x_add_w] at hXW₁ hXW₂ hYW₁ hYW₂
  have hJX : A.x n ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero hX₁ hX₂
  have hJY : A.x m ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero hY₁ hY₂
  have hJXW : (A.x n + A.w k) ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero hXW₁ hXW₂
  have hJYW : (A.x m + A.w k) ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero hYW₁ hYW₂
  have hw : ((A.p₀ + 2 * A.p₁ + A.p₂ : ℝ) : ℂ) = 1 := by rw [A.weights_sum, ofReal_one]
  push_cast at hw
  have hk' : k ≠ 0 → A.w k ≠ 0 := mt hk.mp
  have hnm' : n ≠ m → A.x n - A.x m ≠ 0 := mt hnm.mp
  clear h₁ h₂ h₃ hk hnm
  simp only [simplexW, chiUpUp, G, J, kron, weight_zero, weight_one, weight_two,
    weight_three, ρ]
  push_cast
  by_cases hk0 : k = 0
  · by_cases hnm0 : n = m
    · simp only [ite_eq_left hk0, ite_eq_left hnm0]
      subst hk0 hnm0
      simp only [w_zero, add_zero] at *
      generalize A.x n = X at *
      generalize (A.a : ℂ) = a at *
      generalize (A.β : ℂ) = β at *
      generalize (A.p₀ : ℂ) = p₀ at *
      generalize (A.p₁ : ℂ) = p₁ at *
      generalize (A.p₂ : ℂ) = p₂ at *
      obtain rfl : p₁ = (1 - p₀ - p₂) / 2 := by linear_combination hw / 2
      field_simp
      ring
    · simp only [ite_eq_left hk0, ite_eq_right hnm0]
      have hXY := hnm' hnm0
      subst hk0
      simp only [w_zero, add_zero] at *
      generalize A.x n = X at *
      generalize A.x m = Y at *
      generalize (A.a : ℂ) = a at *
      generalize (A.β : ℂ) = β at *
      generalize (A.p₀ : ℂ) = p₀ at *
      generalize (A.p₁ : ℂ) = p₁ at *
      generalize (A.p₂ : ℂ) = p₂ at *
      obtain rfl : p₁ = (1 - p₀ - p₂) / 2 := by linear_combination hw / 2
      field_simp
      ring
  · have hW := hk' hk0
    by_cases hnm0 : n = m
    · simp only [ite_eq_right hk0, ite_eq_left hnm0, zero_mul, sub_zero]
      subst hnm0
      generalize A.x n = X at *
      generalize A.w k = W at *
      generalize (A.a : ℂ) = a at *
      generalize (A.β : ℂ) = β at *
      generalize (A.p₀ : ℂ) = p₀ at *
      generalize (A.p₁ : ℂ) = p₁ at *
      generalize (A.p₂ : ℂ) = p₂ at *
      obtain rfl : p₁ = (1 - p₀ - p₂) / 2 := by linear_combination hw / 2
      field_simp
      ring
    · simp only [ite_eq_right hk0, ite_eq_right hnm0, zero_mul, sub_zero]
      have hXY := hnm' hnm0
      generalize A.x n = X at *
      generalize A.x m = Y at *
      generalize A.w k = W at *
      generalize (A.a : ℂ) = a at *
      generalize (A.β : ℂ) = β at *
      generalize (A.p₀ : ℂ) = p₀ at *
      generalize (A.p₁ : ℂ) = p₁ at *
      generalize (A.p₂ : ℂ) = p₂ at *
      obtain rfl : p₁ = (1 - p₀ - p₂) / 2 := by linear_combination hw / 2
      field_simp
      ring

/-! ### The opposite-spin susceptibility -/

/-- `2μ + iΩ_{n+m+k+1} = x_n + x_m + w`. -/
theorem pair_denom (k n m : ℤ) :
    2 * (A.μ : ℂ) + A.ω (n + m + k + 1) * I = A.x n + A.x m + A.w k := by
  simp only [x, w, ω, ν]
  push_cast
  ring

theorem chiLehmann_upDn (k n m : ℤ) :
    A.chiLehmann cUp cDn (A.ν n) (A.ν m) (A.ω k) = A.chiUpDn k n m := by
  have h₁ := A.cexp_mu_sub_x n
  have h₂ : cexp ((A.x n + A.w k - A.μ) * A.β) = -1 := by
    rw [x_add_w, A.cexp_x_sub_mu]
  have h₃ : cexp ((A.μ - A.x m - A.w k) * A.β) = -1 := by
    rw [sub_sub, x_add_w, A.cexp_mu_sub_x]
  rw [A.chiLehmann_eq _ _ _ _ _ (A.freq_eq k n m)]
  simp only [orderedTerm_eq, triangleTerm_eq, lineTerm_eq]
  simp [Fin.sum_univ_four, cUp_eq, cDn_eq]
  have hk := A.w_eq_zero_iff k
  have hnm := A.x_sub_x_eq_zero_iff n m
  have hp : -(A.x n + A.x m + A.w k) = 0 ↔ A.μ = 0 ∧ n + m + k + 1 = 0 :=
    neg_eq_zero.trans (A.pair_eq_zero_iff k n m)
  rw [A.pathTerm_eq 3 2 3 1 h₁ h₂ h₃ (r₁ := -(A.x n - A.a)) (r₂ := A.x n + A.w k - A.a)
      (r₃ := -(A.x m + A.w k - A.a)) (r₁₂ := A.w k) (r₂₃ := A.x n - A.x m)
      (r₁₂₃ := -(A.x m - A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hnm,
    A.pathTerm_eq 3 2 0 1 h₁ h₃ h₂ (r₁ := -(A.x n - A.a)) (r₂ := -(A.x m + A.w k + A.a))
      (r₃ := A.x n + A.w k + A.a) (r₁₂ := -(A.x n + A.x m + A.w k)) (r₂₃ := A.x n - A.x m)
      (r₁₂₃ := -(A.x m - A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hp hnm,
    A.pathTerm_eq 3 1 0 1 h₃ h₁ h₂ (r₁ := -(A.x m + A.w k - A.a)) (r₂ := -(A.x n + A.a))
      (r₃ := A.x n + A.w k + A.a) (r₁₂ := -(A.x n + A.x m + A.w k)) (r₂₃ := A.w k)
      (r₁₂₃ := -(A.x m - A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hp hk,
    A.pathTerm_eq 2 3 2 0 h₂ h₁ h₃ (r₁ := A.x n + A.w k - A.a) (r₂ := -(A.x n - A.a))
      (r₃ := -(A.x m + A.w k + A.a)) (r₁₂ := A.w k) (r₂₃ := -(A.x n + A.x m + A.w k))
      (r₁₂₃ := -(A.x m + A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hk hp,
    A.pathTerm_eq 2 3 1 0 h₂ h₃ h₁ (r₁ := A.x n + A.w k - A.a) (r₂ := -(A.x m + A.w k - A.a))
      (r₃ := -(A.x n + A.a)) (r₁₂ := A.x n - A.x m) (r₂₃ := -(A.x n + A.x m + A.w k))
      (r₁₂₃ := -(A.x m + A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hp,
    A.pathTerm_eq 2 0 1 0 h₃ h₂ h₁ (r₁ := -(A.x m + A.w k + A.a)) (r₂ := A.x n + A.w k + A.a)
      (r₃ := -(A.x n + A.a)) (r₁₂ := A.x n - A.x m) (r₂₃ := A.w k)
      (r₁₂₃ := -(A.x m + A.a)) (by simp [rate, energy]; ring) (by simp [rate, energy]; ring)
      (by simp [rate, energy]; ring) (by ring) (by ring) (by ring) hnm hk,
    A.pathTerm₂_eq 1 0 h₁ h₂ (r₁ := -(A.x n + A.a)) (r₂ := A.x n + A.w k + A.a) (r₁₂ := A.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    A.pathTerm₂_eq 3 2 h₁ h₂ (r₁ := -(A.x n - A.a)) (r₂ := A.x n + A.w k - A.a) (r₁₂ := A.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    A.pathTerm₂_eq 0 1 h₂ h₁ (r₁ := A.x n + A.w k + A.a) (r₂ := -(A.x n + A.a)) (r₁₂ := A.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    A.pathTerm₂_eq 2 3 h₂ h₁ (r₁ := A.x n + A.w k - A.a) (r₂ := -(A.x n - A.a)) (r₁₂ := A.w k)
      (by simp [rate, energy]; ring) (by simp [rate, energy]; ring) (by ring) hk,
    A.pathTerm₁_eq 2 0 h₃ (r := -(A.x m + A.w k + A.a)) (by simp [rate, energy]; ring),
    A.pathTerm₁_eq 3 1 h₃ (r := -(A.x m + A.w k - A.a)) (by simp [rate, energy]; ring)]
  rw [A.disc_triangle, A.disc_line k m A.weight_zero A.weight_two A.weight_one A.weight_three]
  have hX₁ := A.x_add_ofReal_ne_zero n A.a
  have hX₂ := A.x_sub_ofReal_ne_zero n A.a
  have hY₁ := A.x_add_ofReal_ne_zero m A.a
  have hY₂ := A.x_sub_ofReal_ne_zero m A.a
  have hXW₁ := A.x_add_ofReal_ne_zero (n + k) A.a
  have hXW₂ := A.x_sub_ofReal_ne_zero (n + k) A.a
  have hYW₁ := A.x_add_ofReal_ne_zero (m + k) A.a
  have hYW₂ := A.x_sub_ofReal_ne_zero (m + k) A.a
  rw [← x_add_w] at hXW₁ hXW₂ hYW₁ hYW₂
  have hJX : A.x n ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero hX₁ hX₂
  have hJY : A.x m ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero hY₁ hY₂
  have hJXW : (A.x n + A.w k) ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero hXW₁ hXW₂
  have hJYW : (A.x m + A.w k) ^ 2 - (A.a : ℂ) ^ 2 ≠ 0 := by
    rw [sq_sub_sq]
    exact mul_ne_zero hYW₁ hYW₂
  have hw : ((A.p₀ + 2 * A.p₁ + A.p₂ : ℝ) : ℂ) = 1 := by rw [A.weights_sum, ofReal_one]
  push_cast at hw
  have hk' : k ≠ 0 → A.w k ≠ 0 := mt hk.mp
  have hnm' : n ≠ m → A.x n - A.x m ≠ 0 := mt hnm.mp
  have hp' : A.μ = 0 ∧ n + m + k + 1 = 0 → A.x n + A.x m + A.w k = 0 :=
    (A.pair_eq_zero_iff k n m).mpr
  have hp'' : ¬(A.μ = 0 ∧ n + m + k + 1 = 0) → A.x n + A.x m + A.w k ≠ 0 :=
    mt (A.pair_eq_zero_iff k n m).mp
  clear h₁ h₂ h₃ hk hnm hp
  simp only [simplexW, chiUpDn, pairC, piPair', Rreg', G, J, kron, weight_zero,
    weight_one, weight_two, weight_three, qm, qp, ρ, ← x_add_w, pair_denom]
  push_cast
  by_cases hk0 : k = 0
  · by_cases hnm0 : n = m
    · have hp0 : ¬(A.μ = 0 ∧ n + m + k + 1 = 0) := fun h ↦ by
        have := h.2
        omega
      simp only [ite_eq_left hk0, ite_eq_left hnm0, ite_eq_right hp0]
      have hXX := hp'' hp0
      subst hk0 hnm0
      simp only [w_zero, add_zero] at *
      generalize A.x n = X at *
      generalize (A.a : ℂ) = a at *
      generalize (A.β : ℂ) = β at *
      generalize (A.p₀ : ℂ) = p₀ at *
      generalize (A.p₁ : ℂ) = p₁ at *
      generalize (A.p₂ : ℂ) = p₂ at *
      obtain rfl : p₁ = (1 - p₀ - p₂) / 2 := by linear_combination hw / 2
      have hX : X ≠ 0 := fun h ↦ hXX (by rw [h, add_zero])
      field_simp
      ring
    · by_cases hp0 : A.μ = 0 ∧ n + m + k + 1 = 0
      · simp only [ite_eq_left hk0, ite_eq_right hnm0, ite_eq_left hp0]
        have hXY := hnm' hnm0
        have hXYW := hp' hp0
        rw [p₂_eq_p₀_of_mu hp0.1] at *
        subst hk0
        simp only [w_zero, add_zero] at *
        generalize A.x n = X at *
        generalize A.x m = Y at *
        obtain rfl : Y = -X := by linear_combination hXYW
        generalize (A.a : ℂ) = a at *
        generalize (A.β : ℂ) = β at *
        generalize (A.p₀ : ℂ) = p₀ at *
        generalize (A.p₁ : ℂ) = p₁ at *
        obtain rfl : p₁ = (1 - p₀ - p₀) / 2 := by linear_combination hw / 2
        have hX : X ≠ 0 := fun h ↦ hXY (by rw [h]; ring)
        field_simp
        ring
      · simp only [ite_eq_left hk0, ite_eq_right hnm0, ite_eq_right hp0]
        have hXY := hnm' hnm0
        have hXYW := hp'' hp0
        subst hk0
        simp only [w_zero, add_zero] at *
        generalize A.x n = X at *
        generalize A.x m = Y at *
        generalize (A.a : ℂ) = a at *
        generalize (A.β : ℂ) = β at *
        generalize (A.p₀ : ℂ) = p₀ at *
        generalize (A.p₁ : ℂ) = p₁ at *
        generalize (A.p₂ : ℂ) = p₂ at *
        obtain rfl : p₁ = (1 - p₀ - p₂) / 2 := by linear_combination hw / 2
        field_simp
        ring
  · have hW := hk' hk0
    by_cases hnm0 : n = m
    · by_cases hp0 : A.μ = 0 ∧ n + m + k + 1 = 0
      · simp only [ite_eq_right hk0, ite_eq_left hnm0, ite_eq_left hp0, zero_mul, sub_zero]
        have hXYW := hp' hp0
        rw [p₂_eq_p₀_of_mu hp0.1] at *
        subst hnm0
        generalize A.x n = X at *
        generalize A.w k = W at *
        generalize (A.a : ℂ) = a at *
        generalize (A.β : ℂ) = β at *
        generalize (A.p₀ : ℂ) = p₀ at *
        generalize (A.p₁ : ℂ) = p₁ at *
        obtain rfl : p₁ = (1 - p₀ - p₀) / 2 := by linear_combination hw / 2
        field_simp
        obtain rfl : W = -X - X := by linear_combination hXYW
        ring
      · simp only [ite_eq_right hk0, ite_eq_left hnm0, ite_eq_right hp0, zero_mul, sub_zero]
        have hXYW := hp'' hp0
        -- `field_simp` rewrites `x + x` as `x * (1 + 1)`, so keep the pair sum as a variable
        generalize hS : A.x n + A.x m + A.w k = S at *
        subst hnm0
        generalize A.x n = X at *
        generalize A.w k = W at *
        generalize (A.a : ℂ) = a at *
        generalize (A.β : ℂ) = β at *
        generalize (A.p₀ : ℂ) = p₀ at *
        generalize (A.p₁ : ℂ) = p₁ at *
        generalize (A.p₂ : ℂ) = p₂ at *
        obtain rfl : p₁ = (1 - p₀ - p₂) / 2 := by linear_combination hw / 2
        field_simp
        obtain rfl : W = S - X - X := by linear_combination hS
        ring
    · by_cases hp0 : A.μ = 0 ∧ n + m + k + 1 = 0
      · simp only [ite_eq_right hk0, ite_eq_right hnm0, ite_eq_left hp0, zero_mul, sub_zero]
        have hXY := hnm' hnm0
        have hXYW := hp' hp0
        rw [p₂_eq_p₀_of_mu hp0.1] at *
        generalize A.x n = X at *
        generalize A.x m = Y at *
        generalize A.w k = W at *
        obtain rfl : Y = -X - W := by linear_combination hXYW
        generalize (A.a : ℂ) = a at *
        generalize (A.β : ℂ) = β at *
        generalize (A.p₀ : ℂ) = p₀ at *
        generalize (A.p₁ : ℂ) = p₁ at *
        obtain rfl : p₁ = (1 - p₀ - p₀) / 2 := by linear_combination hw / 2
        field_simp
        ring
      · simp only [ite_eq_right hk0, ite_eq_right hnm0, ite_eq_right hp0, zero_mul, sub_zero]
        have hXY := hnm' hnm0
        have hXYW := hp'' hp0
        generalize A.x n = X at *
        generalize A.x m = Y at *
        generalize A.w k = W at *
        generalize (A.a : ℂ) = a at *
        generalize (A.β : ℂ) = β at *
        generalize (A.p₀ : ℂ) = p₀ at *
        generalize (A.p₁ : ℂ) = p₁ at *
        generalize (A.p₂ : ℂ) = p₂ at *
        obtain rfl : p₁ = (1 - p₀ - p₂) / 2 := by linear_combination hw / 2
        field_simp
        ring

/-! ### The forms used by the channels -/

/-- The particle–particle form of `χ↑↑`: `χ_pp↑↑(ν_n, ν_m, ω_k) = χ↑↑(ν_n, ν_m, -ω_k - ν_n - ν_m)`
with `χ↑↑` from the Hamiltonian. -/
theorem chiPPUpUp_eq_lehmann (k n m : ℤ) :
    A.chiPPUpUp k n m = A.chiLehmann cUp cUp (A.ν n) (A.ν m) (-A.ω k - A.ν n - A.ν m) := by
  rw [← A.ω_pp, A.chiLehmann_upUp, chiPPUpUp]

/-- The particle–particle form of `χ↑↓`, from the Hamiltonian. -/
theorem chiPPUpDn_eq_lehmann (k n m : ℤ) :
    A.chiPPUpDn k n m = A.chiLehmann cUp cDn (A.ν n) (A.ν m) (-A.ω k - A.ν n - A.ν m) := by
  rw [← A.ω_pp, A.chiLehmann_upDn, chiPPUpDn]

/-- The particle–particle bubble of Eq. (B), `-(β/2) G(ν_n) G(-ν_n - ω_k)`, with the propagator
of the Hamiltonian. -/
theorem Bpp_eq_propagator (k n : ℤ) :
    A.Bpp k n = -((A.β : ℂ) / 2) * A.propagator (A.ν n) * A.propagator (-A.ν n - A.ω k) := by
  rw [← A.ν_mirror, A.propagator_eq, A.propagator_eq, Bpp, x, x, ν_mirror]
  push_cast
  ring_nf

/-- The particle–hole bubble of Eq. (B), `-β G(ν_n) G(ν_n + ω_k)`, with the propagator of the
Hamiltonian. -/
theorem Bph_eq_propagator (k n : ℤ) :
    A.Bph k n = -(A.β : ℂ) * A.propagator (A.ν n) * A.propagator (A.ν n + A.ω k) := by
  rw [A.ν_add_ω, A.propagator_eq, A.propagator_eq, Bph, x_add_w]

end Params

end HubbardAtom
