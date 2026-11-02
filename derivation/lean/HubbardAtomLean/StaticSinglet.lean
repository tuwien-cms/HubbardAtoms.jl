import HubbardAtomLean.Singlet

/-!
# The static singlet zeros

Item (c) of the static estimates: the static (`k = 0`) singlet denominator `Kₛ` for `U > 0`.
At `k = 0`, `ξ² = b²/4 + Aₛ` is the real `σₛ = μ² + 2aρμ + a²(1 - 8p₁)` (`xi_sq_static`), and
`Kₛ = 1 + a tanh(βξ/2)/ξ` (`Ks_static`). With `P = β(a + μ)`, `Q = β(a - μ)`:

* `Params.sigS_mul`: `β²σₛ(2 + e^{-P} + e^{-Q}) = -[P²(1 - e^{-Q}) + Q²(1 - e^{-P}) + 4PQ]`, the
  formula for `η² = -σₛ`;
* `Params.sigS_nonneg`: `σₛ ≥ 0` for `|μ| ≥ a`; `Params.Ks_ne_zero_static_of_nonneg`: for
  `σₛ ≥ 0`, `ξ` is real and `Kₛ ≥ 1`; so **`Kₛ = 0` requires `|μ| < U/2`**
  (`abs_mu_lt_of_Ks_eq_zero`);
* `Params.Ks_static_eq_zero_iff`: for `σₛ < 0`, `ξ = ±iη`, and **`Kₛ = 0` iff
  `βη/2 + arctan(η/a) ∈ πℤ_{>0}`**, as
  `η cos(βη/2) + a sin(βη/2) = √(a² + η²) sin(βη/2 + arctan(η/a))`.
-/

open Complex

namespace HubbardAtom

namespace Params

variable (A : Params)

/-- The static `ξ²`: `σₛ = μ² + 2aρμ + a²(1 - 8p₁)`. -/
noncomputable def sigS : ℝ := A.μ ^ 2 + 2 * A.a * A.ρ * A.μ + A.a ^ 2 * (1 - 8 * A.p₁)

/-- At `k = 0`, `b²/4 + Aₛ = σₛ`. -/
theorem xi_sq_static : A.b 0 ^ 2 / 4 + A.As 0 = A.sigS := by
  simp only [b, As, A.w_zero, sigS]
  push_cast
  ring

/-- `Kₛ = 1 + a tanh(βξ/2)/ξ` at `k = 0`, `ξ ≠ 0`. -/
theorem Ks_static {ξ : ℂ} (hξ : ξ ≠ 0) : A.Ks 0 ξ = 1 + A.a * tanh (A.β * ξ / 2) / ξ := by
  have hβ := A.β_ne_zero
  rw [Ks, Tpp, ite_eq_right hξ, S, ite_eq_left even_zero',
    show (A.β : ℂ) * (A.μ + ξ - A.μ) / 2 = A.β * ξ / 2 by ring]
  field_simp

/-- `Kₛ(0) = 1 + aβ/2` at `k = 0`. -/
theorem Ks_static_zero : A.Ks 0 0 = 1 + A.a * A.β / 2 := by
  have hβ := A.β_ne_zero
  rw [Ks, Tpp, ite_eq_left rfl]
  field_simp
  ring

/-- `Kₛ` is even in `ξ` at `k = 0`. -/
theorem Ks_static_neg (ξ : ℂ) : A.Ks 0 (-ξ) = A.Ks 0 ξ := by
  by_cases hξ : ξ = 0
  · rw [hξ, neg_zero]
  rw [A.Ks_static (neg_ne_zero.mpr hξ), A.Ks_static hξ, mul_neg, neg_div, tanh_neg]
  ring

/-- **The formula for `η² = -σₛ`**: `β²σₛ(2 + e^{-P} + e^{-Q}) =
-[P²(1 - e^{-Q}) + Q²(1 - e^{-P}) + 4PQ]` with `P = β(a + μ)`, `Q = β(a - μ)`. -/
theorem sigS_mul :
    A.β ^ 2 * A.sigS * (2 + Real.exp (-(A.β * (A.a + A.μ))) + Real.exp (-(A.β * (A.a - A.μ)))) =
      -((A.β * (A.a + A.μ)) ^ 2 * (1 - Real.exp (-(A.β * (A.a - A.μ)))) +
        (A.β * (A.a - A.μ)) ^ 2 * (1 - Real.exp (-(A.β * (A.a + A.μ)))) +
        4 * (A.β * (A.a + A.μ)) * (A.β * (A.a - A.μ))) := by
  -- `p₀ = e^{-P}p₁`, `p₂ = e^{-Q}p₁`, `p₁(2 + e^{-P} + e^{-Q}) = 1`
  have h₀ : A.p₀ = Real.exp (-(A.β * (A.a + A.μ))) * A.p₁ := by
    rw [A.p₁_eq_p₀_mul, mul_comm (A.p₀), ← mul_assoc, ← Real.exp_add]
    ring_nf; rw [Real.exp_zero, one_mul]
  have h₂ : A.p₂ = Real.exp (-(A.β * (A.a - A.μ))) * A.p₁ := by
    rw [A.p₁_eq_p₂_mul, mul_comm (A.p₂), ← mul_assoc, ← Real.exp_add]
    ring_nf; rw [Real.exp_zero, one_mul]
  have hs := A.weights_sum
  rw [h₀, h₂] at hs
  simp only [sigS, ρ, h₀, h₂]
  linear_combination (A.β ^ 2 * (2 * A.a * A.μ * (Real.exp (-(A.β * (A.a - A.μ))) -
    Real.exp (-(A.β * (A.a + A.μ)))) - 8 * A.a ^ 2)) * hs

/-- **`σₛ ≥ 0` for `|μ| ≥ a`** (`a > 0`): then `PQ ≤ 0`, and with `1 - e^{s} ≤ -s`,
`1 - e^{-P} ≤ P`, the bracket is `≤ Ps(s - P - 4) ≤ 0` for `s = -Q`. -/
theorem sigS_nonneg (ha : 0 < A.a) (hμ : A.a ≤ |A.μ|) : 0 ≤ A.sigS := by
  have hβ := A.β_pos
  have h := A.sigS_mul
  set P := A.β * (A.a + A.μ)
  set Q := A.β * (A.a - A.μ)
  have hN : 0 < 2 + Real.exp (-P) + Real.exp (-Q) := by
    have := Real.exp_pos (-P); have := Real.exp_pos (-Q); linarith
  have hβ2 : 0 < A.β ^ 2 := pow_pos hβ 2
  -- the bracket is `≤ 0`
  have key : P ^ 2 * (1 - Real.exp (-Q)) + Q ^ 2 * (1 - Real.exp (-P)) + 4 * P * Q ≤ 0 := by
    have hP := Real.add_one_le_exp (-P)
    have hQ := Real.add_one_le_exp (-Q)
    rcases le_abs'.mp hμ with h | h
    · -- `μ ≤ -a`: `P ≤ 0 ≤ Q`
      have hP0 : P ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hβ.le (by linarith)
      have hQ0 : 0 ≤ Q := mul_nonneg hβ.le (by linarith)
      have h1 : P ^ 2 * (1 - Real.exp (-Q)) ≤ P ^ 2 * Q :=
        mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg _)
      have h2 : Q ^ 2 * (1 - Real.exp (-P)) ≤ Q ^ 2 * P :=
        mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg _)
      have h3 : P ^ 2 * Q + Q ^ 2 * P + 4 * P * Q = P * Q * (P + Q + 4) := by ring
      have h4 : 0 ≤ P + Q + 4 := by
        have e : P + Q = 2 * (A.β * A.a) := by simp only [P, Q]; ring
        have := mul_pos hβ ha
        linarith
      have h5 : P * Q ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hP0 hQ0
      nlinarith
    · -- `μ ≥ a`: `Q ≤ 0 ≤ P`
      have hQ0 : Q ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hβ.le (by linarith)
      have hP0 : 0 ≤ P := mul_nonneg hβ.le (by linarith)
      have h1 : P ^ 2 * (1 - Real.exp (-Q)) ≤ P ^ 2 * Q :=
        mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg _)
      have h2 : Q ^ 2 * (1 - Real.exp (-P)) ≤ Q ^ 2 * P :=
        mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg _)
      have h3 : P ^ 2 * Q + Q ^ 2 * P + 4 * P * Q = P * Q * (P + Q + 4) := by ring
      have h4 : 0 ≤ P + Q + 4 := by
        have e : P + Q = 2 * (A.β * A.a) := by simp only [P, Q]; ring
        have := mul_pos hβ ha
        linarith
      have h5 : P * Q ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hP0 hQ0
      nlinarith
  have h0 : 0 ≤ A.β ^ 2 * A.sigS * (2 + Real.exp (-P) + Real.exp (-Q)) := by rw [h]; linarith
  by_contra hs
  push Not at hs
  have := mul_neg_of_neg_of_pos (mul_neg_of_pos_of_neg hβ2 hs) hN
  linarith

/-- **`Kₛ ≠ 0` for `σₛ ≥ 0`** (`a > 0`): `ξ = ±√σₛ` is real and `Kₛ = 1 + a tanh(βξ/2)/ξ ≥ 1`. -/
theorem Ks_ne_zero_static_of_nonneg (ha : 0 < A.a) (hσ : 0 ≤ A.sigS) {ξ : ℂ}
    (hξ : ξ ^ 2 = A.sigS) : A.Ks 0 ξ ≠ 0 := by
  have hβ := A.β_pos
  set r := Real.sqrt A.sigS
  have hr : 0 ≤ r := Real.sqrt_nonneg _
  have hξ2 : ξ ^ 2 = (r : ℂ) ^ 2 := by rw [hξ, ← ofReal_pow, Real.sq_sqrt hσ]
  have hK : A.Ks 0 ξ = A.Ks 0 (r : ℂ) := by
    rcases sq_eq_sq_iff_eq_or_eq_neg.mp hξ2 with h | h
    · rw [h]
    · rw [h, A.Ks_static_neg]
  rw [hK]
  rcases eq_or_lt_of_le hr with h0 | h0
  · rw [← h0, ofReal_zero, A.Ks_static_zero]
    have : (1 : ℂ) + A.a * A.β / 2 = ((1 + A.a * A.β / 2 : ℝ) : ℂ) := by push_cast; ring
    rw [this, ofReal_ne_zero]
    nlinarith
  · rw [A.Ks_static (ofReal_ne_zero.mpr h0.ne')]
    have e : (1 : ℂ) + A.a * tanh (A.β * r / 2) / r =
        ((1 + A.a * Real.tanh (A.β * r / 2) / r : ℝ) : ℂ) := by
      push_cast; ring
    rw [e, ofReal_ne_zero]
    have ht : 0 < Real.tanh (A.β * r / 2) := by
      rw [Real.tanh_eq_sinh_div_cosh]
      exact div_pos (Real.sinh_pos_iff.mpr (by positivity)) (Real.cosh_pos _)
    have : 0 < A.a * Real.tanh (A.β * r / 2) / r := div_pos (mul_pos ha ht) h0
    linarith

/-- **`Kₛ = 0` requires `|μ| < U/2`**, for `U > 0` at `k = 0`. -/
theorem abs_mu_lt_of_Ks_eq_zero (ha : 0 < A.a) {ξ : ℂ} (hξ : ξ ^ 2 = A.b 0 ^ 2 / 4 + A.As 0)
    (hK : A.Ks 0 ξ = 0) : |A.μ| < A.a := by
  by_contra h
  push Not at h
  rw [A.xi_sq_static] at hξ
  exact A.Ks_ne_zero_static_of_nonneg ha (A.sigS_nonneg ha h) hξ hK

/-- **The static singlet zeros**: for `U > 0`, `σₛ < 0` and `ξ² = σₛ`, with `η = √(-σₛ)`,
`Kₛ = 0` iff `βη/2 + arctan(η/a) = nπ` for some integer `n ≥ 1`. -/
theorem Ks_static_eq_zero_iff (ha : 0 < A.a) (hσ : A.sigS < 0) {ξ : ℂ}
    (hξ : ξ ^ 2 = A.sigS) :
    A.Ks 0 ξ = 0 ↔ ∃ n : ℕ, 0 < n ∧
      A.β * Real.sqrt (-A.sigS) / 2 + Real.arctan (Real.sqrt (-A.sigS) / A.a) = n * Real.pi := by
  have hβ := A.β_pos
  set η := Real.sqrt (-A.sigS)
  have hη : 0 < η := Real.sqrt_pos.mpr (by linarith)
  have hξ2 : ξ ^ 2 = (η * I : ℂ) ^ 2 := by
    rw [hξ, mul_pow, I_sq, ← ofReal_pow, Real.sq_sqrt (by linarith)]; push_cast; ring
  have hK : A.Ks 0 ξ = A.Ks 0 (η * I) := by
    rcases sq_eq_sq_iff_eq_or_eq_neg.mp hξ2 with h | h
    · rw [h]
    · rw [h, A.Ks_static_neg]
  rw [hK]
  set y := A.β * η / 2
  have hy : 0 < y := by positivity
  -- `Kₛ = 1 + a sin y/(η cos y)`, and `Kₛ = 0` iff `η cos y + a sin y = 0`
  have hKs : A.Ks 0 (η * I) = ((1 + A.a * (Real.sin y / Real.cos y) / η : ℝ) : ℂ) := by
    rw [A.Ks_static (mul_ne_zero (ofReal_ne_zero.mpr hη.ne') I_ne_zero),
      show (A.β : ℂ) * (η * I) / 2 = (y : ℂ) * I by simp only [y]; push_cast; ring,
      tanh_eq_sinh_div_cosh, sinh_mul_I, cosh_mul_I]
    have hI : (I : ℂ) ≠ 0 := I_ne_zero
    have hη' : (η : ℂ) ≠ 0 := ofReal_ne_zero.mpr hη.ne'
    push_cast
    by_cases hc : Complex.cos (y : ℂ) = 0
    · simp [hc]
    · field_simp
  have hiff : A.Ks 0 (η * I) = 0 ↔ η * Real.cos y + A.a * Real.sin y = 0 := by
    rw [hKs, ofReal_eq_zero]
    constructor
    · intro h0
      by_cases hc : Real.cos y = 0
      · rw [hc, div_zero, mul_zero, zero_div, add_zero] at h0; norm_num at h0
      · field_simp at h0
        linarith
    · intro h0
      have hc : Real.cos y ≠ 0 := fun hc ↦ by
        rw [hc, mul_zero, zero_add] at h0
        have := Real.sin_sq_add_cos_sq y
        rw [hc] at this
        rcases mul_eq_zero.mp h0 with h | h
        · exact ha.ne' h
        · rw [h] at this; norm_num at this
      field_simp
      linarith
  rw [hiff]
  -- `η cos y + a sin y = √(a² + η²) sin(y + arctan(η/a))`
  set θ := Real.arctan (η / A.a)
  have hsθ : Real.sin θ = η / A.a / Real.sqrt (1 + (η / A.a) ^ 2) := Real.sin_arctan _
  have hcθ : Real.cos θ = 1 / Real.sqrt (1 + (η / A.a) ^ 2) := Real.cos_arctan _
  have hR : 0 < Real.sqrt (1 + (η / A.a) ^ 2) := Real.sqrt_pos.mpr (by positivity)
  have hsin : Real.sin (y + θ) * (A.a * Real.sqrt (1 + (η / A.a) ^ 2)) =
      η * Real.cos y + A.a * Real.sin y := by
    rw [Real.sin_add, hsθ, hcθ]
    field_simp
    ring
  have hθ0 : 0 < θ := Real.arctan_pos.mpr (div_pos hη ha)
  constructor
  · intro h0
    have : Real.sin (y + θ) = 0 := by
      have hne : A.a * Real.sqrt (1 + (η / A.a) ^ 2) ≠ 0 := mul_ne_zero ha.ne' hR.ne'
      rw [h0] at hsin
      exact (mul_eq_zero.mp hsin).resolve_right hne
    obtain ⟨n, hn⟩ := Real.sin_eq_zero_iff.mp this
    have hn0 : 0 < n := by
      have : (0 : ℝ) < n * Real.pi := by rw [hn]; linarith
      have := pos_of_mul_pos_left this Real.pi_pos.le
      exact_mod_cast this
    refine ⟨n.toNat, by omega, ?_⟩
    rw [← hn]
    have : ((n.toNat : ℕ) : ℝ) = (n : ℝ) := by exact_mod_cast Int.toNat_of_nonneg hn0.le
    rw [this]
  · rintro ⟨n, -, hn⟩
    rw [← hsin, show y + θ = (n : ℤ) * Real.pi by rw [hn]; push_cast; ring, Real.sin_int_mul_pi,
      zero_mul]

end Params

end HubbardAtom
