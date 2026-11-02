import HubbardAtomLean.Singlet

/-!
# Pairing at nonzero transfer: no singlet or triplet divergence

Item (d) of the static estimates: for `μ ≠ 0` and `k ≠ 0`, neither the singlet nor the triplet
vertex diverges. With `δ = ω_k(μ + aρ)`:

* `Params.hasSum_im_Ks`: `Im Kₛ = (2aδ/β)∑ₙ|tₙ + Aₛ|⁻²`, as `Im(tₙ + Aₛ) = -δ` for all `n`;
  so `Kₛ ≠ 0` for `a ≠ 0`, `k ≠ 0` and `μ + aρ ≠ 0` (`Ks_ne_zero_of_im`), and the triplet factor
  `tₙ + aρb + a²`, with the same imaginary part, does not vanish either;
* on the curve `μ + aρ = 0` (`μ ≠ 0`), where `a < 0` and `β²a²(1 - ρ²) < 4` (`curve_bound`):
  `tₙ + Aₛ = Yₙ + c` and `tₙ + aρb + a² = Yₙ + a²(1 - ρ²)` are real, with
  `Yₙ = νₙν_{n+k}` and `c = a²(1 - ρ² - 8p₁)`. As `Yₙ ≤ -(π/β)²` or `Yₙ ≥ 3(π/β)²`, the triplet
  factor does not vanish, and since `∑ₙ 1/Yₙ = 0`, `Kₛ = 1 - (2ac/β)∑ₙ 1/(Yₙ(Yₙ + c))` with
  positive terms; with `p₁ ≤ e^{βa}/2` and `∑ₙ 1/νₙ² = β²/4` this gives
  `Kₛ ≥ 1 - 4(β|a|)³e^{-β|a|}/π² ≥ 1 - 108/(π²e³) > 0` (`Ks_ge_of_curve`);
* `Params.Ks_ne_zero_of_k_ne`, `Params.triplet_ne_zero_of_k_ne`: together, for `k ≠ 0`, `μ ≠ 0`,
  `Kₛ ≠ 0` and `tₙ + aρb + a² ≠ 0` for all `n`; `singletGeneric_of_k_ne`: the assumptions
  `SingletGeneric` of Theorem "singlet" hold for every `U ≠ 0`.
-/

open Complex

namespace HubbardAtom

/-- `x²e^{-x} ≤ 4e^{-2}` for `x ≥ 0`. -/
theorem sq_mul_exp_neg_le_two {x : ℝ} (hx : 0 ≤ x) : x ^ 2 * Real.exp (-x) ≤ 4 * Real.exp (-2) := by
  have h := Real.add_one_le_exp (x / 2 - 1)
  have h2 : (x / 2) ^ 2 ≤ Real.exp (x / 2 - 1) ^ 2 := pow_le_pow_left₀ (by linarith) (by linarith) 2
  rw [← Real.exp_nat_mul] at h2
  have e : Real.exp (-2) = Real.exp ((2 : ℕ) * (x / 2 - 1)) * Real.exp (-x) := by
    rw [← Real.exp_add]; push_cast; ring_nf
  rw [e]
  have := Real.exp_pos (-x)
  nlinarith

/-- `x³e^{-x} ≤ 27e^{-3}` for `x ≥ 0`. -/
theorem cube_mul_exp_neg_le {x : ℝ} (hx : 0 ≤ x) :
    x ^ 3 * Real.exp (-x) ≤ 27 * Real.exp (-3) := by
  have h := Real.add_one_le_exp (x / 3 - 1)
  have h3 : (x / 3) ^ 3 ≤ Real.exp (x / 3 - 1) ^ 3 := pow_le_pow_left₀ (by linarith) (by linarith) 3
  rw [← Real.exp_nat_mul] at h3
  have e : Real.exp (-3) = Real.exp ((3 : ℕ) * (x / 3 - 1)) * Real.exp (-x) := by
    rw [← Real.exp_add]; push_cast; ring_nf
  rw [e]
  have := Real.exp_pos (-x)
  nlinarith

/-- The product of two distinct odd integers is `≤ -1` or `≥ 3`. -/
theorem odd_mul_odd_le_or_ge (n k : ℤ) (hk : k ≠ 0) :
    (2 * n + 1) * (2 * (n + k) + 1) ≤ -1 ∨ 3 ≤ (2 * n + 1) * (2 * (n + k) + 1) := by
  have hodd : Odd ((2 * n + 1) * (2 * (n + k) + 1)) :=
    Int.odd_mul.mpr ⟨⟨n, rfl⟩, ⟨n + k, rfl⟩⟩
  have h1 : (2 * n + 1) * (2 * (n + k) + 1) ≠ 1 := fun h ↦ by
    rcases Int.eq_one_or_neg_one_of_mul_eq_one' h with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> omega
  obtain ⟨r, hr⟩ := hodd
  omega

namespace Params

variable (A : Params) (k : ℤ)

/-! ### Off the curve `μ + aρ = 0` -/

/-- `Im(tₙ + Aₛ) = -ω_k(μ + aρ)`. -/
theorem t_add_As_im (n : ℤ) : (A.t k n + A.As k).im = -(A.ω k * (A.μ + A.a * A.ρ)) := by
  simp only [t, x, b, w, As, add_im, mul_im, mul_re, sub_im, sub_re, add_re, ofReal_re,
    ofReal_im, I_re, I_im, sq, re_ofNat, im_ofNat, one_re, one_im]
  ring

/-- `Im(tₙ + aρb + a²) = -ω_k(μ + aρ)`. -/
theorem t_add_triplet_im (n : ℤ) :
    (A.t k n + A.a * A.ρ * A.b k + A.a ^ 2).im = -(A.ω k * (A.μ + A.a * A.ρ)) := by
  simp only [t, x, b, w, add_im, mul_im, mul_re, sub_im, sub_re, add_re, ofReal_re, ofReal_im,
    I_re, I_im, sq, re_ofNat, im_ofNat]
  ring

theorem ω_ne_zero (hk : k ≠ 0) : A.ω k ≠ 0 := by
  unfold ω
  have : (k : ℝ) ≠ 0 := by exact_mod_cast hk
  exact div_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero this) Real.pi_ne_zero) A.β_pos.ne'

/-- **`Im Kₛ = (2aδ/β)∑ₙ|tₙ + Aₛ|⁻²`**, `δ = ω_k(μ + aρ)`, for `ξ² = b²/4 + Aₛ` and
`tₙ + Aₛ ≠ 0`. -/
theorem hasSum_im_Ks {ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k)
    (ht : ∀ n, A.t k n + A.As k ≠ 0) :
    HasSum (fun n ↦ 2 * A.a * (A.ω k * (A.μ + A.a * A.ρ)) / A.β /
      normSq (A.t k n + A.As k)) (A.Ks k ξ).im := by
  have h := A.hasSum_Tpp (t := -A.As k) (by rw [hξ]; ring)
    (fun n h0 ↦ ht n (by rw [h0, neg_add_cancel]))
  have hi := ((Complex.hasSum_iff _ _).mp h).2
  have hKs : (A.Ks k ξ).im = 2 * A.a / A.β * (A.Tpp k ξ).im := by
    rw [Ks, show (2 : ℂ) * A.a * A.Tpp k ξ / A.β = ((2 * A.a / A.β : ℝ) : ℂ) * A.Tpp k ξ by
      push_cast; ring, add_im, one_im, zero_add, im_ofReal_mul]
  rw [hKs]
  convert hi.mul_left (2 * A.a / A.β) using 1
  funext n
  rw [sub_neg_eq_add, one_div, inv_im, A.t_add_As_im k n]
  ring

/-- **`Kₛ ≠ 0` off the curve**: for `a ≠ 0`, `k ≠ 0` and `μ + aρ ≠ 0`, `Im Kₛ ≠ 0`. -/
theorem Ks_ne_zero_of_im (ha : A.a ≠ 0) (hk : k ≠ 0) (hδ : A.μ + A.a * A.ρ ≠ 0) {ξ : ℂ}
    (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k) : A.Ks k ξ ≠ 0 := by
  have hC : 2 * A.a * (A.ω k * (A.μ + A.a * A.ρ)) / A.β ≠ 0 :=
    div_ne_zero (mul_ne_zero (mul_ne_zero two_ne_zero ha)
      (mul_ne_zero (A.ω_ne_zero k hk) hδ)) A.β_pos.ne'
  have ht : ∀ n, A.t k n + A.As k ≠ 0 := fun n h0 ↦ by
    have := A.t_add_As_im k n
    rw [h0, zero_im, eq_comm, neg_eq_zero] at this
    exact mul_ne_zero (A.ω_ne_zero k hk) hδ this
  have hpos : ∀ n, 0 < normSq (A.t k n + A.As k) := fun n ↦ normSq_pos.mpr (ht n)
  intro h0
  have h := A.hasSum_im_Ks k hξ ht
  rw [h0, zero_im] at h
  have hg := h.div_const (2 * A.a * (A.ω k * (A.μ + A.a * A.ρ)) / A.β)
  rw [zero_div] at hg
  have hle := le_hasSum hg 0 fun j _ ↦ by
    rw [div_div_cancel_left' hC]
    exact (inv_pos.mpr (hpos j)).le
  rw [div_div_cancel_left' hC] at hle
  exact absurd hle (not_le.mpr (inv_pos.mpr (hpos 0)))

/-- The triplet factor off the curve: `tₙ + aρb + a² ≠ 0` for `k ≠ 0`, `μ + aρ ≠ 0`. -/
theorem triplet_ne_zero_of_im (hk : k ≠ 0) (hδ : A.μ + A.a * A.ρ ≠ 0) (n : ℤ) :
    A.t k n + A.a * A.ρ * A.b k + A.a ^ 2 ≠ 0 := fun h0 ↦ by
  have := A.t_add_triplet_im k n
  rw [h0, zero_im, eq_comm, neg_eq_zero] at this
  exact mul_ne_zero (A.ω_ne_zero k hk) hδ this

/-! ### On the curve `μ + aρ = 0` -/

/-- On the curve, `a < 0`: `ρ` has the sign of `μ`. -/
theorem a_neg_of_curve (hμ : A.μ ≠ 0) (hc : A.μ + A.a * A.ρ = 0) : A.a < 0 := by
  have hρμ : 0 < A.ρ * A.μ := by
    have hρ : A.ρ = A.p₀ * (Real.exp (2 * A.β * A.μ) - 1) := by
      rw [ρ, A.p₂_eq_p₀_mul]; ring
    have hp := A.p₀_pos
    have hβ := A.β_pos
    rw [hρ, mul_assoc]
    refine mul_pos hp ?_
    rcases lt_or_gt_of_ne hμ with h | h
    · have : Real.exp (2 * A.β * A.μ) < 1 := Real.exp_lt_one_iff.mpr (by nlinarith)
      exact mul_pos_of_neg_of_neg (by linarith) h
    · have : 1 < Real.exp (2 * A.β * A.μ) := Real.one_lt_exp_iff.mpr (by nlinarith)
      exact mul_pos (by linarith) h
  have hμ2 : 0 < A.μ ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hμ))
  have e : A.a * (A.ρ * A.μ) = -A.μ ^ 2 := by linear_combination A.μ * hc
  by_contra h
  have := mul_nonneg (not_lt.mp h) hρμ.le
  linarith

/-- The weights through `x = e^{βμ}`, `y = e^{βa}`: `p₁ = p₀xy`, `p₂ = p₀x²`,
`p₀(1 + 2xy + x²) = 1`. -/
private theorem weights_xy :
    A.p₁ = A.p₀ * (Real.exp (A.β * A.μ) * Real.exp (A.β * A.a)) ∧
      A.p₂ = A.p₀ * Real.exp (A.β * A.μ) ^ 2 ∧
      A.p₀ * (1 + 2 * (Real.exp (A.β * A.μ) * Real.exp (A.β * A.a)) +
        Real.exp (A.β * A.μ) ^ 2) = 1 := by
  have h₁ : A.p₁ = A.p₀ * (Real.exp (A.β * A.μ) * Real.exp (A.β * A.a)) := by
    rw [A.p₁_eq_p₀_mul, ← Real.exp_add]; ring_nf
  have h₂ : A.p₂ = A.p₀ * Real.exp (A.β * A.μ) ^ 2 := by
    rw [A.p₂_eq_p₀_mul, ← Real.exp_nat_mul]; push_cast; ring_nf
  refine ⟨h₁, h₂, ?_⟩
  have := A.weights_sum
  rw [h₁, h₂] at this
  linear_combination this

/-- `p₁ ≤ e^{βa}/2`. -/
theorem p₁_le_exp : A.p₁ ≤ Real.exp (A.β * A.a) / 2 := by
  obtain ⟨h₁, -, hs⟩ := A.weights_xy
  set x := Real.exp (A.β * A.μ)
  set y := Real.exp (A.β * A.a)
  have hx := Real.exp_pos (A.β * A.μ)
  have hy := Real.exp_pos (A.β * A.a)
  have hp := A.p₀_pos
  rw [h₁]
  have : 0 ≤ A.p₀ * (x - 1) ^ 2 * y := mul_nonneg (mul_nonneg hp.le (sq_nonneg _)) hy.le
  nlinarith

/-- **On the curve, `β²a²(1 - ρ²) < 4`**: with `x = e^{βμ}`, `y = e^{βa} < 1`,
`(1 - ρ²)/ρ² = 4x(x + y)(1 + xy)/(x² - 1)² < 4x/(x - 1)²` and `β²μ² · 4x ≤ 4(x - 1)²`
(`|βμ| ≤ 2|sinh(βμ/2)|`), while `βa = -βμ/ρ`. -/
theorem curve_bound (hμ : A.μ ≠ 0) (hc : A.μ + A.a * A.ρ = 0) :
    A.β ^ 2 * A.a ^ 2 * (1 - A.ρ ^ 2) < 4 := by
  have ha := A.a_neg_of_curve hμ hc
  obtain ⟨h₁, h₂, hs⟩ := A.weights_xy
  have hβ := A.β_pos
  have hp := A.p₀_pos
  set m := A.β * A.μ with hm
  set x := Real.exp m with hxdef
  set y := Real.exp (A.β * A.a)
  have hx := Real.exp_pos m
  have hy := Real.exp_pos (A.β * A.a)
  have hy1 : y < 1 := Real.exp_lt_one_iff.mpr (by nlinarith)
  -- `m²x ≤ (x - 1)²`, from `|m/2| ≤ |sinh(m/2)|`
  have hsh : m ^ 2 * x ≤ (x - 1) ^ 2 := by
    have hsinh : (m / 2) ^ 2 ≤ Real.sinh (m / 2) ^ 2 := by
      rcases le_total 0 m with h | h
      · have := Real.self_le_sinh_iff.mpr (by linarith : 0 ≤ m / 2)
        nlinarith
      · have := Real.self_le_sinh_iff.mpr (by linarith : 0 ≤ -(m / 2))
        rw [Real.sinh_neg] at this
        nlinarith
    have he : x = Real.exp (m / 2) ^ 2 := by
      rw [hxdef, ← Real.exp_nat_mul]; push_cast; ring_nf
    have hsd : Real.sinh (m / 2) = (Real.exp (m / 2) - Real.exp (-(m / 2))) / 2 :=
      Real.sinh_eq _
    have hinv : Real.exp (m / 2) * Real.exp (-(m / 2)) = 1 := by
      rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    rw [hsd] at hsinh
    have hE := Real.exp_pos (m / 2)
    rw [he]
    nlinarith [hinv]
  -- `ρ = p₀(x² - 1)`, `1 - ρ² = 4p₀²x(x + y)(1 + xy)`
  have hρ : A.ρ = A.p₀ * (x ^ 2 - 1) := by rw [ρ, h₂]; ring
  have h1ρ : 1 - A.ρ ^ 2 = 4 * A.p₀ ^ 2 * x * (x + y) * (1 + x * y) := by
    rw [hρ]
    linear_combination (-(A.p₀ * (1 + 2 * (x * y) + x ^ 2)) - 1) * hs
  -- `βa ρ = -m`
  have haρ : A.β * A.a * A.ρ = -m := by rw [hm]; linear_combination A.β * hc
  have hm0 : m ≠ 0 := mul_ne_zero hβ.ne' hμ
  have hm2 : 0 < m ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hm0))
  have hstrict : x * (x + y) * (1 + x * y) < x * (x + 1) ^ 2 := by
    have : 0 < (1 - y) * (x ^ 2 + x * (1 + y) + 1) := by
      apply mul_pos <;> nlinarith
    nlinarith
  -- `β²a²(1 - ρ²)ρ² = m²(1 - ρ²) < 4ρ²`
  have key : m ^ 2 * (1 - A.ρ ^ 2) < 4 * A.ρ ^ 2 := by
    rw [h1ρ, hρ]
    have hp2 : 0 < A.p₀ ^ 2 := pow_pos hp 2
    have h3 : m ^ 2 * (x * (x + y) * (1 + x * y)) < m ^ 2 * (x * (x + 1) ^ 2) :=
      mul_lt_mul_of_pos_left hstrict hm2
    have h4 : m ^ 2 * (x * (x + 1) ^ 2) ≤ (x - 1) ^ 2 * (x + 1) ^ 2 := by
      have := mul_le_mul_of_nonneg_right hsh (sq_nonneg (x + 1))
      nlinarith
    nlinarith
  have hρ0 : A.ρ ≠ 0 := fun h0 ↦ hm0 (by rw [h0, mul_zero] at haρ; linarith)
  have hρ2 : 0 < A.ρ ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hρ0))
  have e : A.β ^ 2 * A.a ^ 2 * (1 - A.ρ ^ 2) * A.ρ ^ 2 = m ^ 2 * (1 - A.ρ ^ 2) := by
    rw [← neg_sq m, ← haρ]; ring
  nlinarith

/-- `Yₙ = νₙν_{n+k}`. -/
noncomputable def Yn (n : ℤ) : ℝ := A.ν n * A.ν (n + k)

theorem Yn_eq (n : ℤ) :
    A.Yn k n = (Real.pi / A.β) ^ 2 * ((2 * n + 1) * (2 * (n + k) + 1) : ℤ) := by
  unfold Yn ν
  push_cast
  ring

/-- `Yₙ ≤ -(π/β)²` or `Yₙ ≥ 3(π/β)²`, for `k ≠ 0`. -/
theorem Yn_le_or_ge (hk : k ≠ 0) (n : ℤ) :
    A.Yn k n ≤ -(Real.pi / A.β) ^ 2 ∨ 3 * (Real.pi / A.β) ^ 2 ≤ A.Yn k n := by
  rw [A.Yn_eq k n]
  have hπ : 0 < (Real.pi / A.β) ^ 2 := pow_pos (div_pos Real.pi_pos A.β_pos) 2
  rcases odd_mul_odd_le_or_ge n k hk with h | h
  · left
    have : (((2 * n + 1) * (2 * (n + k) + 1) : ℤ) : ℝ) ≤ -1 := by exact_mod_cast h
    nlinarith
  · right
    have : (3 : ℝ) ≤ (((2 * n + 1) * (2 * (n + k) + 1) : ℤ) : ℝ) := by exact_mod_cast h
    nlinarith

/-- On the curve, `tₙ + Aₛ = Yₙ + a²(1 - ρ² - 8p₁)`, real. -/
theorem t_add_As_of_curve (hc : A.μ + A.a * A.ρ = 0) (n : ℤ) :
    A.t k n + A.As k = ((A.Yn k n + A.a ^ 2 * (1 - A.ρ ^ 2 - 8 * A.p₁) : ℝ) : ℂ) := by
  have hμ : (A.μ : ℂ) = -(A.a * A.ρ) := by
    have : A.μ = -(A.a * A.ρ) := by linarith
    exact_mod_cast this
  simp only [t, x, b, w, As, Yn, ← A.ν_add_ω]
  push_cast
  rw [hμ]
  linear_combination (-(A.ν n * (A.ν n + A.ω k)) : ℂ) * I_sq

/-- On the curve, `tₙ + aρb + a² = Yₙ + a²(1 - ρ²)`, real. -/
theorem t_add_triplet_of_curve (hc : A.μ + A.a * A.ρ = 0) (n : ℤ) :
    A.t k n + A.a * A.ρ * A.b k + A.a ^ 2 = ((A.Yn k n + A.a ^ 2 * (1 - A.ρ ^ 2) : ℝ) : ℂ) := by
  have hμ : (A.μ : ℂ) = -(A.a * A.ρ) := by
    have : A.μ = -(A.a * A.ρ) := by linarith
    exact_mod_cast this
  simp only [t, x, b, w, Yn, ← A.ν_add_ω]
  push_cast
  rw [hμ]
  linear_combination (-(A.ν n * (A.ν n + A.ω k)) : ℂ) * I_sq

/-- `1 - ρ² > 0`. -/
theorem one_sub_rho_sq_pos : 0 < 1 - A.ρ ^ 2 := by
  rw [A.one_sub_rho_sq]
  have h₁ : 0 < A.qp := add_pos A.p₁_pos A.p₂_pos
  have h₂ : 0 < A.qm := add_pos A.p₀_pos A.p₁_pos
  exact mul_pos (mul_pos (by norm_num) h₁) h₂

/-- The triplet factor on the curve: `Yₙ + a²(1 - ρ²) ≠ 0`, as `0 < a²(1 - ρ²) < 4/β² < (π/β)²`. -/
theorem triplet_ne_zero_of_curve (hμ : A.μ ≠ 0) (hk : k ≠ 0) (hc : A.μ + A.a * A.ρ = 0)
    (n : ℤ) : A.t k n + A.a * A.ρ * A.b k + A.a ^ 2 ≠ 0 := by
  rw [A.t_add_triplet_of_curve k hc n, ofReal_ne_zero]
  have hb := A.curve_bound hμ hc
  have hβ := A.β_pos
  have ha := A.a_neg_of_curve hμ hc
  have h1 := A.one_sub_rho_sq_pos
  have hπ := Real.pi_gt_three
  have hpos : 0 < A.a ^ 2 * (1 - A.ρ ^ 2) := mul_pos (pow_pos (neg_pos.mpr ha) 2 |>.trans_eq
    (by ring)) h1
  have hlt : A.a ^ 2 * (1 - A.ρ ^ 2) < (Real.pi / A.β) ^ 2 := by
    rw [div_pow, lt_div_iff₀ (pow_pos hβ 2)]
    nlinarith
  have : 0 < (Real.pi / A.β) ^ 2 := pow_pos (div_pos Real.pi_pos hβ) 2
  rcases A.Yn_le_or_ge k hk n with h | h <;> intro h0 <;> linarith

/-- `∑ₙ 1/Yₙ = 0` for `k ≠ 0`: `Yₙ = tₙ - t₀` with `ξ₀ = w/2`, and `S_k(μ + w/2) = S₀(μ) = 0`. -/
theorem hasSum_inv_Yn (hk : k ≠ 0) : HasSum (fun n ↦ 1 / A.Yn k n) 0 := by
  have hw : A.w k ≠ 0 := mul_ne_zero (ofReal_ne_zero.mpr (A.ω_ne_zero k hk)) I_ne_zero
  have hY : ∀ n, A.t k n - (A.b k ^ 2 / 4 - (A.w k / 2) ^ 2) = (A.Yn k n : ℂ) := fun n ↦ by
    simp only [t, x, b, w, Yn, ← A.ν_add_ω]
    push_cast
    linear_combination (-(A.ν n * (A.ν n + A.ω k)) : ℂ) * I_sq
  have hYne : ∀ n, A.Yn k n ≠ 0 := fun n ↦ by
    have hπ : 0 < (Real.pi / A.β) ^ 2 := pow_pos (div_pos Real.pi_pos A.β_pos) 2
    rcases A.Yn_le_or_ge k hk n with h | h <;> intro h0 <;> rw [h0] at h <;> linarith
  have h := A.hasSum_one_div_t_sub (t := A.b k ^ 2 / 4 - (A.w k / 2) ^ 2) (ξ := A.w k / 2)
    (by ring) (div_ne_zero hw two_ne_zero) fun n h0 ↦ hYne n (by
      have := hY n
      rw [h0, sub_self] at this
      exact_mod_cast this.symm)
  rw [A.S_shift, add_sub_cancel_right, A.S_mu_of_even even_zero', neg_zero, zero_div] at h
  rw [← Complex.hasSum_ofReal]
  convert h using 1
  funext n
  rw [hY n]
  push_cast
  rfl

/-- `∑ₙ 1/νₙ² = β²/4`. -/
theorem hasSum_inv_ν_sq : HasSum (fun n ↦ 1 / A.ν n ^ 2) (A.β ^ 2 / 4) := by
  have h := (A.hasSum_sq (k := 0) (x := A.μ) (A.z_ne_mu_of_even even_zero')).neg
  rw [A.S_mu_of_even even_zero'] at h
  rw [← Complex.hasSum_ofReal]
  convert h using 1
  · funext n
    have : A.z 0 n - A.μ = (A.ν n : ℂ) * I := by
      rw [A.z_eq]
      simp only [ν]
      push_cast
      ring
    rw [this, mul_pow, I_sq]
    push_cast
    ring
  · push_cast
    ring

/-- On the curve, `-8/β² < c < 4/β²` for `c = a²(1 - ρ² - 8p₁)`: `a²(1 - ρ²) < 4/β²` and
`8a²p₁ ≤ 4a²e^{βa} ≤ 16e^{-2}/β²`. -/
theorem c_bounds_of_curve (hμ : A.μ ≠ 0) (hc : A.μ + A.a * A.ρ = 0) :
    -(8 / A.β ^ 2) < A.a ^ 2 * (1 - A.ρ ^ 2 - 8 * A.p₁) ∧
      A.a ^ 2 * (1 - A.ρ ^ 2 - 8 * A.p₁) < 4 / A.β ^ 2 ∧
      -(A.a ^ 2 * (1 - A.ρ ^ 2 - 8 * A.p₁)) ≤ 4 * A.a ^ 2 * Real.exp (A.β * A.a) := by
  have hβ := A.β_pos
  have hβ2 : 0 < A.β ^ 2 := pow_pos hβ 2
  have ha := A.a_neg_of_curve hμ hc
  have hb := A.curve_bound hμ hc
  have h1 := A.one_sub_rho_sq_pos
  have hp₁ := A.p₁_le_exp
  have hp₁0 := A.p₁_pos
  have hA : 0 ≤ -(A.β * A.a) := by nlinarith
  have hsq := sq_mul_exp_neg_le_two hA
  rw [neg_neg, neg_sq] at hsq
  have he2 : Real.exp (-2) ≤ 1 / 3 := by
    rw [Real.exp_neg, inv_eq_one_div]
    apply one_div_le_one_div_of_le (by norm_num)
    have := Real.add_one_le_exp 2
    linarith
  have h0 : 0 ≤ A.a ^ 2 * (1 - A.ρ ^ 2) := mul_nonneg (sq_nonneg _) h1.le
  have h8 : 8 * A.a ^ 2 * A.p₁ ≤ 4 * A.a ^ 2 * Real.exp (A.β * A.a) := by
    have := mul_le_mul_of_nonneg_left hp₁ (by positivity : 0 ≤ 8 * A.a ^ 2)
    linarith
  refine ⟨?_, ?_, by nlinarith⟩
  · rw [neg_lt, lt_div_iff₀ hβ2]
    nlinarith
  · rw [lt_div_iff₀ hβ2]
    nlinarith

/-- **`Kₛ` on the curve**: for `μ ≠ 0`, `k ≠ 0`, `μ + aρ = 0` and `ξ² = b²/4 + Aₛ`, `Kₛ` is real
and at least `1 - 4(β|a|)³e^{-β|a|}/π² ≥ 1 - 108/(π²e³)`. -/
theorem Ks_ge_of_curve (hμ : A.μ ≠ 0) (hk : k ≠ 0) (hc : A.μ + A.a * A.ρ = 0) {ξ : ℂ}
    (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k) :
    ∃ r : ℝ, A.Ks k ξ = r ∧ 1 - 108 / (Real.pi ^ 2 * Real.exp 3) ≤ r := by
  have hβ := A.β_pos
  have hβ2 : 0 < A.β ^ 2 := pow_pos hβ 2
  have ha := A.a_neg_of_curve hμ hc
  have hπ3 := Real.pi_gt_three
  obtain ⟨c, hcdef⟩ : ∃ c, c = A.a ^ 2 * (1 - A.ρ ^ 2 - 8 * A.p₁) := ⟨_, rfl⟩
  obtain ⟨hcl, hcu, hcc⟩ := A.c_bounds_of_curve hμ hc
  rw [← hcdef] at hcl hcu hcc
  have hL0 : 0 < (Real.pi / A.β) ^ 2 := pow_pos (div_pos Real.pi_pos hβ) 2
  have h48 : 4 / A.β ^ 2 < 8 / A.β ^ 2 := div_lt_div_of_pos_right (by norm_num) hβ2
  have hLb : 8 / A.β ^ 2 < (Real.pi / A.β) ^ 2 := by
    rw [div_pow, div_lt_div_iff_of_pos_right hβ2]
    nlinarith
  -- every `Yₙ(Yₙ + c) > 0`
  have hY := A.Yn_le_or_ge k hk
  have hYc : ∀ n, A.Yn k n + c ≠ 0 := fun n ↦ by
    rcases hY n with h | h <;> intro h0 <;> nlinarith
  have hprod : ∀ n, 0 < A.Yn k n * (A.Yn k n + c) := fun n ↦ by
    rcases hY n with h | h <;> nlinarith
  have hYne : ∀ n, A.Yn k n ≠ 0 := fun n h0 ↦ by
    have := hprod n; rw [h0, zero_mul] at this; exact lt_irrefl _ this
  -- `Kₛ = 1 + (2a/β)T`, `T = ∑ₙ 1/(Yₙ + c)`
  have ht : ∀ n, A.t k n ≠ -A.As k := fun n h0 ↦ hYc n (by
    have := A.t_add_As_of_curve k hc n
    rw [h0, neg_add_cancel, ← hcdef] at this
    exact_mod_cast this.symm)
  have hT := A.hasSum_Tpp (t := -A.As k) (by rw [hξ]; ring) ht
  have hTr : HasSum (fun n ↦ ((1 / (A.Yn k n + c) : ℝ) : ℂ)) (A.Tpp k ξ) := by
    convert hT using 1
    funext n
    rw [sub_neg_eq_add, A.t_add_As_of_curve k hc n, ← hcdef]
    push_cast
    rfl
  obtain ⟨T, hTs⟩ := Complex.summable_ofReal.mp hTr.summable
  have hTeq : A.Tpp k ξ = (T : ℂ) := hTr.unique (Complex.hasSum_ofReal.mpr hTs)
  refine ⟨1 + 2 * A.a / A.β * T, ?_, ?_⟩
  · rw [Ks, hTeq]
    push_cast
    ring
  have hB : 0 ≤ 108 / (Real.pi ^ 2 * Real.exp 3) := by
    have := Real.exp_pos 3
    positivity
  have haβ : 2 * A.a / A.β < 0 := div_neg_of_neg_of_pos (by linarith) hβ
  rcases lt_trichotomy c 0 with hc0 | hc0 | hc0
  swap
  · -- `c = 0`: `T = ∑ₙ 1/Yₙ = 0`
    have hT0 : T = 0 := by
      rw [hc0] at hTs
      simp only [add_zero] at hTs
      exact hTs.unique (A.hasSum_inv_Yn k hk)
    rw [hT0, mul_zero, add_zero]
    linarith
  -- `T = -cR`, `R = ∑ₙ 1/(Yₙ(Yₙ + c)) ≥ 0`
  all_goals
    have hR : HasSum (fun n ↦ 1 / (A.Yn k n * (A.Yn k n + c))) (T / -c) := by
      have := (hTs.sub (A.hasSum_inv_Yn k hk)).div_const (-c)
      rw [sub_zero] at this
      convert this using 1
      funext n
      have h₁ := hYc n
      have h₂ := hYne n
      have h₃ : c ≠ 0 := fun h ↦ by rw [h] at hc0; exact lt_irrefl _ hc0
      field_simp
      ring
    have hR0 : 0 ≤ T / -c := hR.nonneg fun n ↦ (one_div_pos.mpr (hprod n)).le
  · -- `c < 0`: `R ≤ 2∑ₙ 1/Yₙ² ≤ 2(β/π)²∑ₙ 1/νₙ² = β⁴/(2π²)`
    have hle : ∀ n, 1 / (A.Yn k n * (A.Yn k n + c)) ≤
        2 * (A.β / Real.pi) ^ 2 * (1 / A.ν n ^ 2) := by
      intro n
      have hνn : A.ν n ≠ 0 := A.ν_ne_zero n
      have hν2 : 0 < A.ν n ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hνn))
      have hν : (Real.pi / A.β) ^ 2 ≤ A.ν (n + k) ^ 2 := by
        rw [ν, div_pow, div_pow, mul_pow]
        apply div_le_div_of_nonneg_right _ hβ2.le
        have : (1 : ℝ) ≤ (2 * ((n + k : ℤ) : ℝ) + 1) ^ 2 := by
          have h' : (1 : ℤ) ≤ (2 * (n + k) + 1) ^ 2 := by
            have h0 : (2 * (n + k) + 1 : ℤ) ≠ 0 := by omega
            have := Int.one_le_abs h0
            nlinarith [abs_mul_abs_self (2 * (n + k) + 1 : ℤ)]
          exact_mod_cast h'
        nlinarith [sq_nonneg Real.pi]
      have hY2 : A.Yn k n ^ 2 / 2 ≤ A.Yn k n * (A.Yn k n + c) := by
        rcases hY n with h | h <;> nlinarith
      have hYsq : A.Yn k n ^ 2 = A.ν n ^ 2 * A.ν (n + k) ^ 2 := by rw [Yn]; ring
      have hπb : (A.β / Real.pi) ^ 2 * (Real.pi / A.β) ^ 2 = 1 := by
        rw [← mul_pow]; field_simp
      rw [mul_one_div, div_le_div_iff₀ (hprod n) hν2]
      nlinarith
    have hRle : T / -c ≤ 2 * (A.β / Real.pi) ^ 2 * (A.β ^ 2 / 4) :=
      hasSum_le hle hR ((A.hasSum_inv_ν_sq).mul_left (2 * (A.β / Real.pi) ^ 2))
    have hRle' : T / -c ≤ A.β ^ 4 / (2 * Real.pi ^ 2) := by
      have : 2 * (A.β / Real.pi) ^ 2 * (A.β ^ 2 / 4) = A.β ^ 4 / (2 * Real.pi ^ 2) := by
        field_simp; ring
      linarith
    -- `2aT/β = -2|a||c|R/β ≥ -4|a|³e^{βa}β³/π²`, and `(β|a|)³e^{βa} ≤ 27e^{-3}`
    have hA : 0 ≤ -(A.β * A.a) := by nlinarith
    have hcube := cube_mul_exp_neg_le hA
    rw [neg_neg] at hcube
    have hE3 := Real.exp_pos 3
    have hπ2 : 0 < Real.pi ^ 2 := by positivity
    have hc' : c ≠ 0 := hc0.ne
    have hTc : T = -c * (T / -c) := by field_simp
    have hposa : 0 ≤ -A.a := by linarith
    set R := T / -c
    have hstep : 2 * (-A.a) * (-c) * R * Real.pi ^ 2 ≤ 4 * (-(A.β * A.a)) ^ 3 *
        Real.exp (A.β * A.a) * A.β := by
      have h₁ : 2 * (-A.a) * (-c) * R ≤ 2 * (-A.a) * (4 * A.a ^ 2 * Real.exp (A.β * A.a)) * R :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hcc (by linarith)) hR0
      have hR' : R * Real.pi ^ 2 ≤ A.β ^ 4 / 2 := by
        rw [le_div_iff₀ (by positivity)] at hRle'
        nlinarith
      have hnn : 0 ≤ 2 * (-A.a) * (4 * A.a ^ 2 * Real.exp (A.β * A.a)) :=
        mul_nonneg (mul_nonneg zero_le_two hposa) (mul_nonneg (by positivity) (Real.exp_pos _).le)
      have h₂ := mul_le_mul_of_nonneg_left hR' hnn
      have h₃ := mul_le_mul_of_nonneg_right h₁ hπ2.le
      have e : 4 * (-(A.β * A.a)) ^ 3 * Real.exp (A.β * A.a) * A.β =
          2 * (-A.a) * (4 * A.a ^ 2 * Real.exp (A.β * A.a)) * (A.β ^ 4 / 2) := by ring
      rw [e]
      calc 2 * (-A.a) * (-c) * R * Real.pi ^ 2
          ≤ 2 * (-A.a) * (4 * A.a ^ 2 * Real.exp (A.β * A.a)) * R * Real.pi ^ 2 := h₃
        _ = 2 * (-A.a) * (4 * A.a ^ 2 * Real.exp (A.β * A.a)) * (R * Real.pi ^ 2) := by ring
        _ ≤ _ := h₂
    have hfin : 4 * (-(A.β * A.a)) ^ 3 * Real.exp (A.β * A.a) ≤
        108 / Real.exp 3 := by
      rw [Real.exp_neg, ← div_eq_mul_inv] at hcube
      have : (-(A.β * A.a)) ^ 3 * Real.exp (A.β * A.a) ≤ 27 / Real.exp 3 := by
        convert hcube using 1
      have e : (108 : ℝ) / Real.exp 3 = 4 * (27 / Real.exp 3) := by ring
      linarith
    have hK : 2 * A.a / A.β * T = -(2 * (-A.a) * (-c) * R / A.β) := by
      rw [hTc]; field_simp
    rw [hK]
    have : 2 * (-A.a) * (-c) * R / A.β ≤ 108 / (Real.pi ^ 2 * Real.exp 3) := by
      rw [div_le_div_iff₀ hβ (by positivity)]
      have h2 : 4 * (-(A.β * A.a)) ^ 3 * Real.exp (A.β * A.a) * Real.exp 3 ≤ 108 := by
        rw [le_div_iff₀ hE3] at hfin; linarith
      have h3 := mul_le_mul_of_nonneg_right hstep hE3.le
      have h4 := mul_le_mul_of_nonneg_right h2 hβ.le
      calc 2 * (-A.a) * (-c) * R * (Real.pi ^ 2 * Real.exp 3)
          = 2 * (-A.a) * (-c) * R * Real.pi ^ 2 * Real.exp 3 := by ring
        _ ≤ 4 * (-(A.β * A.a)) ^ 3 * Real.exp (A.β * A.a) * A.β * Real.exp 3 := h3
        _ = 4 * (-(A.β * A.a)) ^ 3 * Real.exp (A.β * A.a) * Real.exp 3 * A.β := by ring
        _ ≤ 108 * A.β := h4
    linarith
  · -- `c > 0`: `2aT/β = -2acR/β ≥ 0`
    have hTn : T ≤ 0 := by
      have hc' : c ≠ 0 := hc0.ne'
      have : T = -c * (T / -c) := by field_simp
      rw [this]
      exact mul_nonpos_of_nonpos_of_nonneg (by linarith) hR0
    have : 0 ≤ 2 * A.a / A.β * T := mul_nonneg_of_nonpos_of_nonpos haβ.le hTn
    linarith

/-- **`Kₛ ≠ 0` on the curve**, as `108/(π²e³) < 1`. -/
theorem Ks_ne_zero_of_curve (hμ : A.μ ≠ 0) (hk : k ≠ 0) (hc : A.μ + A.a * A.ρ = 0) {ξ : ℂ}
    (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k) : A.Ks k ξ ≠ 0 := by
  obtain ⟨r, hr, hle⟩ := A.Ks_ge_of_curve k hμ hk hc hξ
  rw [hr, ofReal_ne_zero]
  have h3 := Real.pi_gt_three
  have he : (12 : ℝ) < Real.exp 3 := by
    have := Real.add_one_le_exp 1
    have e : Real.exp 3 = Real.exp 1 ^ 3 := by rw [← Real.exp_nat_mul]; norm_num
    have h1 : (2.7 : ℝ) < Real.exp 1 := lt_trans (by norm_num) Real.exp_one_gt_d9
    have h2 : (2.7 : ℝ) ^ 3 < Real.exp 1 ^ 3 := pow_lt_pow_left₀ h1 (by norm_num) (by norm_num)
    rw [e]; norm_num at h2 ⊢; linarith
  have : 108 / (Real.pi ^ 2 * Real.exp 3) < 1 := by
    rw [div_lt_one (by positivity)]
    nlinarith
  intro h0
  linarith

/-! ### Every nonzero transfer -/

/-- **No singlet divergence at `k ≠ 0`**: for `μ ≠ 0`, `k ≠ 0` and `ξ² = b²/4 + Aₛ`, `Kₛ ≠ 0`. -/
theorem Ks_ne_zero_of_k_ne (hμ : A.μ ≠ 0) (hk : k ≠ 0) {ξ : ℂ}
    (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k) : A.Ks k ξ ≠ 0 := by
  by_cases hc : A.μ + A.a * A.ρ = 0
  · exact A.Ks_ne_zero_of_curve k hμ hk hc hξ
  by_cases ha : A.a = 0
  · simp [Ks, ha]
  · exact A.Ks_ne_zero_of_im k ha hk hc hξ

/-- **No triplet divergence at `k ≠ 0`**: for `μ ≠ 0` and `k ≠ 0`, `tₙ + aρb + a² ≠ 0` for all
`n`. -/
theorem triplet_ne_zero_of_k_ne (hμ : A.μ ≠ 0) (hk : k ≠ 0) (n : ℤ) :
    A.t k n + A.a * A.ρ * A.b k + A.a ^ 2 ≠ 0 := by
  by_cases hc : A.μ + A.a * A.ρ = 0
  · exact A.triplet_ne_zero_of_curve k hμ hk hc n
  · exact A.triplet_ne_zero_of_im k hk hc n

/-- `tₙ + Aₛ ≠ 0` for `μ ≠ 0`, `k ≠ 0`: off the curve by the imaginary part, on it as in
`Ks_ge_of_curve`. -/
theorem t_add_As_ne_zero_of_k_ne (hμ : A.μ ≠ 0) (hk : k ≠ 0) (n : ℤ) :
    A.t k n + A.As k ≠ 0 := by
  by_cases hc : A.μ + A.a * A.ρ = 0
  · rw [A.t_add_As_of_curve k hc n, ofReal_ne_zero]
    have hβ := A.β_pos
    have hβ2 : 0 < A.β ^ 2 := pow_pos hβ 2
    have hπ3 := Real.pi_gt_three
    obtain ⟨hcl, hcu, -⟩ := A.c_bounds_of_curve hμ hc
    have hL : 8 / A.β ^ 2 < (Real.pi / A.β) ^ 2 := by
      rw [div_pow, div_lt_div_iff_of_pos_right hβ2]
      nlinarith
    have h48 : 4 / A.β ^ 2 < 8 / A.β ^ 2 := div_lt_div_of_pos_right (by norm_num) hβ2
    have hL0 : 0 < (Real.pi / A.β) ^ 2 := pow_pos (div_pos Real.pi_pos hβ) 2
    rcases A.Yn_le_or_ge k hk n with h | h <;> intro h0 <;> nlinarith
  · intro h0
    have := A.t_add_As_im k n
    rw [h0, zero_im, eq_comm, neg_eq_zero] at this
    exact mul_ne_zero (A.ω_ne_zero k hk) hc this

/-- **Theorem "singlet" applies at every `k ≠ 0`**: for `U ≠ 0`, `μ ≠ 0`, `k ≠ 0` and
`ξ² = b²/4 + Aₛ`, the assumptions `SingletGeneric` hold. -/
theorem singletGeneric_of_k_ne (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) (hk : k ≠ 0) {ξ : ℂ}
    (hξ : ξ ^ 2 = A.b k ^ 2 / 4 + A.As k) : A.SingletGeneric k ξ where
  a_ne := ha
  b_ne := fun h0 ↦ by
    have := congrArg im h0
    simp only [b, w, sub_im, mul_im, ofReal_re, ofReal_im, I_re, I_im, re_ofNat, im_ofNat,
      zero_im] at this
    exact A.ω_ne_zero k hk (by linarith)
  t_add_As_ne := A.t_add_As_ne_zero_of_k_ne k hμ hk
  ξ_sq := hξ
  Ks_ne := A.Ks_ne_zero_of_k_ne k hμ hk hξ

end Params

end HubbardAtom
