import HubbardAtomLean.ComplexZeros
import HubbardAtomLean.StaticMagnetic

/-!
# The static density zero modes

Item (a) of the static estimates: for `U > 0`, `μ ≠ 0`, `k = 0`, where the density channel can
have a zero mode, `Φ₀(u) = K_e(u) = 0`. With `T = e^{βa}`, `C = cosh βμ`, `S = T + C`,
`e(z)` is a quadratic in `s = z²`,
`S²e = S²(s + 3a²)² - 4a²(2C² + CT - 1)(s + 3a²) + 16a⁴(C² - 1)` (`Params.eph_dens`), with
discriminant `16a⁴𝓕`, `𝓕 = (4 - 3C²)T² + 2C(3 - 2C²)T + 1` (`densF`). At a zero mode
`K_eA_e = β²e(u) = 0`, and:

* `s` is real (`ComplexZeros.lean`), and `s < 0`, as `Φ₀ > 0` for `s ≥ 0`; so `s = -y²`, and
  `Φ₀ = cos βy + C + a sin(βy)/y = 0` with `C > 1` forces `sin βy < 0`, hence `βy > π`;
* `y² < 3a²`, as the quadratic above is positive for `s ≤ -3a²`;
* `𝓕 ≥ 0`, since the root is real; then `3ρ²T² < 1`, that is `|ρ| < e^{-βU/2}/√3`, from
  `S² - 3(C² - 1)T² = 𝓕 + (C² - 1)(1 + 4CT)` and `ρS = sinh βμ`;
* `βa > π/√3`, and `𝓕(C, T)/T²` decreases in `T` where `𝓕 ≥ 0` can hold, so
  `𝓕(cosh βμ, e^{π/√3}) > 0`: this is `|βμ| < arcosh 𝖼_*(e^{π/√3}) = 0.57888` with `𝖼_*` the zero
  of `𝓕(·, e^{π/√3})` on `[1, ∞)`. As a rational bound, `cosh βμ < 47/40` (`|βμ| < 0.5833`).

* `Params.zero_mode_dens`: `u² = -y²` with `sin βy < 0`, `π < βy`, `y² < 3a²`, and `𝓕 ≥ 0`;
* `Params.rho_sq_lt_of_zero_mode`: `3(ρe^{βa})² < 1`;
* `Params.pi_div_sqrt_three_lt_of_zero_mode`: `π/√3 < βa`;
* `Params.densF_pos_of_zero_mode`, `Params.cosh_lt_of_zero_mode`: `𝓕(cosh βμ, e^{π/√3}) > 0`,
  `cosh βμ < 47/40`.
-/

open Complex

namespace HubbardAtom

/-- The discriminant factor of the static density channel,
`𝓕(C, T) = (4 - 3C²)T² + 2C(3 - 2C²)T + 1`. -/
def densF (C T : ℝ) : ℝ := (4 - 3 * C ^ 2) * T ^ 2 + 2 * C * (3 - 2 * C ^ 2) * T + 1

/-! ### The real estimates -/

/-- `𝓕(C, T)/T²` decreases in `T` where it is nonnegative: for `C ≥ 1`, `2 < T₀²`, `T₀ < T`,
`𝓕(C, T) ≥ 0` implies `𝓕(C, T₀) > 0`. -/
theorem densF_pos_of_lt {C T₀ T : ℝ} (hC : 1 ≤ C) (hT₀ : 2 < T₀ ^ 2) (h0 : 0 < T₀) (hT : T₀ < T)
    (hF : 0 ≤ densF C T) : 0 < densF C T₀ := by
  unfold densF at *
  set g := 2 * C * (3 - 2 * C ^ 2)
  have hT0 : 0 < T := by linarith
  rcases le_or_gt 0 g with hg | hg
  · -- `𝓕(C, T₀)T² - 𝓕(C, T)T₀² = gT₀T(T - T₀) + T² - T₀²`
    have e : ((4 - 3 * C ^ 2) * T₀ ^ 2 + g * T₀ + 1) * T ^ 2 -
        ((4 - 3 * C ^ 2) * T ^ 2 + g * T + 1) * T₀ ^ 2 = g * T₀ * T * (T - T₀) + (T ^ 2 - T₀ ^ 2) :=
      by ring
    have h1 : 0 ≤ g * T₀ * T * (T - T₀) := by
      have := mul_pos h0 hT0
      have := sub_pos.mpr hT
      positivity
    have h2 : 0 < T ^ 2 - T₀ ^ 2 := by nlinarith
    have h3 : 0 ≤ ((4 - 3 * C ^ 2) * T ^ 2 + g * T + 1) * T₀ ^ 2 := mul_nonneg hF (sq_nonneg _)
    have h4 : 0 < ((4 - 3 * C ^ 2) * T₀ ^ 2 + g * T₀ + 1) * T ^ 2 := by linarith
    by_contra hX
    push Not at hX
    have := mul_le_mul_of_nonneg_right hX (sq_nonneg T)
    linarith
  · -- `C² > 3/2`: `𝓕(C, T) < -T²/2 + 1 < 0`
    exfalso
    have hC2 : 3 / 2 < C ^ 2 := by
      by_contra h
      push Not at h
      have : 0 ≤ 3 - 2 * C ^ 2 := by linarith
      have : 0 ≤ g := by positivity
      linarith
    have h1 : g * T < 0 := mul_neg_of_neg_of_pos hg hT0
    have h2 : 2 < T ^ 2 := by nlinarith
    have h3 : (4 - 3 * C ^ 2) * T ^ 2 < -1 / 2 * T ^ 2 :=
      mul_lt_mul_of_pos_right (by linarith) (by linarith)
    linarith

/-- The rational bound: for `T ≥ 6` and `C ≥ 47/40`, `𝓕(C, T) < 0`. -/
theorem densF_neg {C T : ℝ} (hC : 47 / 40 ≤ C) (hT : 6 ≤ T) : densF C T < 0 := by
  unfold densF
  have hT0 : 0 < T := by linarith
  rcases le_or_gt (3 / 2) (C ^ 2) with hC2 | hC2
  · -- `2C(3 - 2C²) ≤ 0`
    have h1 : 2 * C * (3 - 2 * C ^ 2) * T ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg (mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith))
        hT0.le
    have h3 : (4 - 3 * C ^ 2) * T ^ 2 ≤ -1 / 2 * T ^ 2 :=
      mul_le_mul_of_nonneg_right (by linarith) (sq_nonneg _)
    nlinarith
  · -- `𝓕 ≤ T²h(C)` with `h(C) = 4 - 3C² + C(3 - 2C²)/3 + 1/36 < 0`
    have hg : 0 < 2 * C * (3 - 2 * C ^ 2) := by
      have : 0 < 3 - 2 * C ^ 2 := by linarith
      positivity
    have h1 : 2 * C * (3 - 2 * C ^ 2) * T ≤ 2 * C * (3 - 2 * C ^ 2) * (T ^ 2 / 6) :=
      mul_le_mul_of_nonneg_left (by nlinarith) hg.le
    have h2 : (1 : ℝ) ≤ T ^ 2 / 36 := by nlinarith
    have hh : 4 - 3 * C ^ 2 + C * (3 - 2 * C ^ 2) / 3 + 1 / 36 < 0 := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hC) (sub_nonneg.mpr hC)]
    have h3 : (4 - 3 * C ^ 2 + C * (3 - 2 * C ^ 2) / 3 + 1 / 36) * T ^ 2 < 0 :=
      mul_neg_of_neg_of_pos hh (by positivity)
    nlinarith

/-- `6 ≤ e^x` for `x ≥ 9/5`. -/
theorem six_le_exp {x : ℝ} (hx : 9 / 5 ≤ x) : 6 ≤ Real.exp x := by
  have h := Real.sum_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 9 / 5) 7
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial, Nat.cast_one, pow_zero,
    pow_one, zero_add, Nat.succ_eq_add_one] at h
  norm_num at h
  have := Real.exp_le_exp.mpr hx
  linarith

namespace Params

variable (A : Params)

/-! ### The density static channel -/

/-- `ρ(C + T) = sinh βμ`, with `T = e^{βa}`, `C = cosh βμ`. -/
theorem rho_TC :
    A.ρ * (Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a)) = Real.sinh (A.β * A.μ) := by
  have h₁ : A.p₁ = A.p₀ * (Real.exp (A.β * A.μ) * Real.exp (A.β * A.a)) := by
    rw [A.p₁_eq_p₀_mul, ← Real.exp_add]; ring_nf
  have h₂ : A.p₂ = A.p₀ * Real.exp (A.β * A.μ) ^ 2 := by
    rw [A.p₂_eq_p₀_mul, ← Real.exp_nat_mul]; push_cast; ring_nf
  have hs := A.weights_sum
  rw [h₁, h₂] at hs
  have hx := Real.exp_pos (A.β * A.μ)
  have hxi : Real.exp (-(A.β * A.μ)) = 1 / Real.exp (A.β * A.μ) := by
    rw [Real.exp_neg, one_div]
  rw [ρ, h₂, Real.cosh_eq, Real.sinh_eq, hxi]
  field_simp
  linear_combination (Real.exp (A.β * A.μ) ^ 2 - 1) * hs

/-- **`e` in the static density channel**: with `T = e^{βa}`, `C = cosh βμ`, `S = T + C`,
`S²e(z) = S²(z² + 3a²)² - 4a²(2C² + CT - 1)(z² + 3a²) + 16a⁴(C² - 1)`. -/
theorem eph_dens (z : ℂ) :
    ((Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a)) ^ 2 : ℝ) * A.eph 0 1 z =
      ((Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a)) ^ 2 : ℝ) * (z ^ 2 + 3 * A.a ^ 2) ^ 2 -
        4 * A.a ^ 2 * ((2 * Real.cosh (A.β * A.μ) ^ 2 +
          Real.cosh (A.β * A.μ) * Real.exp (A.β * A.a) - 1 : ℝ) : ℂ) * (z ^ 2 + 3 * A.a ^ 2) +
        16 * A.a ^ 4 * ((Real.cosh (A.β * A.μ) ^ 2 - 1 : ℝ) : ℂ) := by
  obtain ⟨h₁, h₂⟩ := A.p₁_p₀p₂_TC
  have hρ := A.rho_TC
  set C := Real.cosh (A.β * A.μ)
  set T := Real.exp (A.β * A.a)
  have hS : 0 < C + T := by
    have := Real.cosh_pos (A.β * A.μ); have := Real.exp_pos (A.β * A.a); linarith
  have hsh : Real.sinh (A.β * A.μ) ^ 2 = C ^ 2 - 1 := by
    have := Real.cosh_sq_sub_sinh_sq (A.β * A.μ); linarith
  -- the real coefficients
  have hB : (C + T) ^ 2 * (A.a ^ 2 + 4 * A.a ^ 2 * A.p₁ - 2 * A.a ^ 2 * A.ρ ^ 2) =
      A.a ^ 2 * (3 * T ^ 2 + 4 * C * T - C ^ 2 + 2) := by
    have hρ2 : A.ρ ^ 2 * (C + T) ^ 2 = C ^ 2 - 1 := by rw [← hsh, ← hρ]; ring
    rw [h₁]
    field_simp
    linear_combination (-4 * A.a ^ 2) * hρ2
  have hC : (C + T) ^ 2 * ((A.a ^ 2 + 4 * A.a ^ 2 * A.p₁) ^ 2 - 16 * A.a ^ 4 * (A.p₀ * A.p₂)) =
      A.a ^ 4 * ((C + 3 * T) ^ 2 - 4) := by
    rw [h₁, h₂]
    field_simp
    ring
  have hB' : ((C + T) ^ 2 : ℂ) * (A.a ^ 2 + 4 * A.a ^ 2 * A.p₁ - 2 * A.a ^ 2 * A.ρ ^ 2) =
      A.a ^ 2 * (3 * T ^ 2 + 4 * C * T - C ^ 2 + 2) := by exact_mod_cast hB
  have hC' : ((C + T) ^ 2 : ℂ) * ((A.a ^ 2 + 4 * A.a ^ 2 * A.p₁) ^ 2 -
      16 * A.a ^ 4 * (A.p₀ * A.p₂)) = A.a ^ 4 * ((C + 3 * T) ^ 2 - 4) := by exact_mod_cast hC
  have he : A.eph 0 1 z = (z ^ 2) ^ 2 +
      2 * (A.a ^ 2 + 4 * A.a ^ 2 * A.p₁ - 2 * A.a ^ 2 * A.ρ ^ 2) * z ^ 2 +
      ((A.a ^ 2 + 4 * A.a ^ 2 * A.p₁) ^ 2 - 16 * A.a ^ 4 * (A.p₀ * A.p₂)) := by
    simp only [eph, Hph, lam, q₀, A.w_zero]
    ring
  rw [he]
  push_cast
  linear_combination (2 * z ^ 2) * hB' + hC'

/-- At a zero mode, `D₀(u) ≠ 0` and `e(u) = 0`. -/
theorem Dk_ne_zero_eph_of_zero_mode (hU : A.U ≠ 0) (hμ : A.μ ≠ 0) {ε u : ℂ}
    (hε : ε = 1 ∨ ε = -1) (hΦ : A.Phik 0 ε u = 0) (hK : A.Ke 0 ε u = 0) :
    A.Dk 0 u ≠ 0 ∧ A.eph 0 ε u = 0 := by
  have ha' : (A.a : ℂ) ≠ 0 := ofReal_ne_zero.mpr (by rw [a]; exact div_ne_zero hU two_ne_zero)
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  have hu : ∀ m, A.z 0 m ≠ u := fun m h ↦
    A.Phik_z_ne_zero 0 ε hε0 ha' hμ m (h ▸ hΦ)
  have hu' : ∀ m, A.z 0 m ≠ -u := fun m h ↦
    A.Phik_z_ne_zero 0 ε hε0 ha' hμ m (by rw [h, A.Phik_neg, hΦ])
  have hD := A.Dk_ne_zero_of_off 0 hu hu'
  refine ⟨hD, ?_⟩
  have he := A.Ke_mul_Ae_of_Phi 0 ε hD hΦ
  rw [hK, zero_mul, eq_comm, mul_eq_zero] at he
  have hβ2 : (A.β : ℂ) ^ 2 ≠ 0 := pow_ne_zero 2 (ofReal_ne_zero.mpr A.β_pos.ne')
  exact he.resolve_left hβ2

/-- `Φ₀ = cosh βu + cosh βμ + a shc(β, u)` in the density channel. -/
theorem Phik_dens (v : ℂ) :
    A.Phik 0 1 v = cosh (A.β * v) + Real.cosh (A.β * A.μ) + A.a * shc A.β v := by
  simp only [Phik, Dk, τk, even_zero', ↓reduceIte, one_mul]
  push_cast
  ring

/-- **The static density zero modes**: for `U > 0`, `μ ≠ 0`, at `Φ₀(u) = K_e(u) = 0` in the
density channel, `u² = -y²` with `y > 0`, `sin βy < 0`, `π < βy` and `y² < 3a²`, and
`𝓕(cosh βμ, e^{βa}) ≥ 0`. -/
theorem zero_mode_dens (hU : 0 < A.U) (hμ : A.μ ≠ 0) {u : ℂ} (hΦ : A.Phik 0 1 u = 0)
    (hK : A.Ke 0 1 u = 0) :
    ∃ y : ℝ, 0 < y ∧ u ^ 2 = -(y : ℂ) ^ 2 ∧ Real.sin (A.β * y) < 0 ∧ Real.pi < A.β * y ∧
      y ^ 2 < 3 * A.a ^ 2 ∧ 0 ≤ densF (Real.cosh (A.β * A.μ)) (Real.exp (A.β * A.a)) := by
  have ha : 0 < A.a := by rw [a]; linarith
  have hβ := A.β_pos
  obtain ⟨hD, he⟩ := A.Dk_ne_zero_eph_of_zero_mode hU.ne' hμ (Or.inl rfl) hΦ hK
  have hS := A.eph_dens u
  rw [he, mul_zero] at hS
  set C := Real.cosh (A.β * A.μ) with hCdef
  set T := Real.exp (A.β * A.a)
  have hC1 : 1 < C := Real.one_lt_cosh.mpr (mul_ne_zero hβ.ne' hμ)
  have hT0 : 0 < T := Real.exp_pos _
  -- `s = u²` is real
  have hre : (u ^ 2).im = 0 := by
    by_contra hs
    exact A.Ke_ne_zero_of_Phi 0 1 (Or.inl rfl) ha.le hs hD hΦ hK
  set r := (u ^ 2).re
  have hur : u ^ 2 = (r : ℂ) := by
    apply Complex.ext <;> simp [hre, r]
  -- `s < 0`: for `s ≥ 0`, `Φ₀ > 0`
  have hr : r < 0 := by
    by_contra hr
    push Not at hr
    set x := Real.sqrt r
    have hx : 0 ≤ x := Real.sqrt_nonneg _
    have hu2 : u ^ 2 = (x : ℂ) ^ 2 := by rw [hur, ← ofReal_pow, Real.sq_sqrt hr]
    have hΦx : A.Phik 0 1 (x : ℂ) = 0 := by
      rcases sq_eq_sq_iff_eq_or_eq_neg.mp hu2 with h | h
      · rw [← h]; exact hΦ
      · rw [show (x : ℂ) = -u by rw [h, neg_neg], A.Phik_neg]; exact hΦ
    rcases eq_or_lt_of_le hx with hx0 | hx0
    · rw [← hx0, ofReal_zero, A.Phik_zero, Dk, mul_zero, cosh_zero, τk,
        ite_eq_left even_zero'] at hΦx
      have := mul_pos ha hβ
      rw [show (1 : ℂ) + 1 * cosh (↑A.β * ↑A.μ) + 1 * ↑A.a * ↑A.β = ((1 + C + A.a * A.β : ℝ) : ℂ)
        by simp only [C]; push_cast; ring, ofReal_eq_zero] at hΦx
      linarith
    · rw [Phik_dens, shc_of_ne (ofReal_ne_zero.mpr hx0.ne')] at hΦx
      have e : cosh (↑A.β * ↑x) + ↑C + ↑A.a * (sinh (↑A.β * ↑x) / ↑x) =
          ((Real.cosh (A.β * x) + C + A.a * (Real.sinh (A.β * x) / x) : ℝ) : ℂ) := by
        push_cast; ring
      rw [e, ofReal_eq_zero] at hΦx
      have h1 := Real.cosh_pos (A.β * x)
      have h2 : 0 < Real.sinh (A.β * x) := Real.sinh_pos_iff.mpr (mul_pos hβ hx0)
      have h3 : 0 < A.a * (Real.sinh (A.β * x) / x) := mul_pos ha (div_pos h2 hx0)
      linarith
  -- `s = -y²`
  set y := Real.sqrt (-r)
  have hy : 0 < y := Real.sqrt_pos.mpr (by linarith)
  have hry : r = -y ^ 2 := by rw [Real.sq_sqrt (by linarith)]; ring
  have hu2 : u ^ 2 = -(y : ℂ) ^ 2 := by rw [hur, hry]; push_cast; ring
  have hΦy : A.Phik 0 1 (y * I) = 0 := by
    have h2 : u ^ 2 = (y * I : ℂ) ^ 2 := by rw [hu2, mul_pow, I_sq]; ring
    rcases sq_eq_sq_iff_eq_or_eq_neg.mp h2 with h | h
    · rw [← h]; exact hΦ
    · rw [show (y * I : ℂ) = -u by rw [h, neg_neg], A.Phik_neg]; exact hΦ
  rw [Phik_dens, shc_of_ne (mul_ne_zero (ofReal_ne_zero.mpr hy.ne') I_ne_zero)] at hΦy
  have e : cosh (↑A.β * (↑y * I)) + ↑C + ↑A.a * (sinh (↑A.β * (↑y * I)) / (↑y * I)) =
      ((Real.cos (A.β * y) + C + A.a * (Real.sin (A.β * y) / y) : ℝ) : ℂ) := by
    have h1 : (A.β : ℂ) * (y * I) = ((A.β * y : ℝ) : ℂ) * I := by push_cast; ring
    rw [h1, cosh_mul_I, sinh_mul_I]
    push_cast
    field_simp
  rw [e, ofReal_eq_zero] at hΦy
  -- `sin βy < 0` and `βy > π`
  have hsin : Real.sin (A.β * y) < 0 := by
    have h1 := Real.neg_one_le_cos (A.β * y)
    have h2 : A.a * (Real.sin (A.β * y) / y) < 0 := by linarith
    have h3 : Real.sin (A.β * y) / y < 0 := by
      by_contra h; push Not at h; have := mul_nonneg ha.le h; linarith
    exact (div_neg_iff.mp h3).elim (fun h ↦ absurd h.2 (not_lt.mpr hy.le)) (fun h ↦ h.1)
  have hpi : Real.pi < A.β * y := by
    by_contra h
    push Not at h
    have := Real.sin_nonneg_of_nonneg_of_le_pi (mul_pos hβ hy).le h
    linarith
  -- the quadratic in `τ = 3a² - y²`
  have hq : (C + T) ^ 2 * (3 * A.a ^ 2 - y ^ 2) ^ 2 -
      4 * A.a ^ 2 * (2 * C ^ 2 + C * T - 1) * (3 * A.a ^ 2 - y ^ 2) +
      16 * A.a ^ 4 * (C ^ 2 - 1) = 0 := by
    rw [hu2] at hS
    have h : (((C + T) ^ 2 * (3 * A.a ^ 2 - y ^ 2) ^ 2 -
        4 * A.a ^ 2 * (2 * C ^ 2 + C * T - 1) * (3 * A.a ^ 2 - y ^ 2) +
        16 * A.a ^ 4 * (C ^ 2 - 1) : ℝ) : ℂ) = 0 := by
      rw [hS]; push_cast; ring
    exact_mod_cast h
  have hB : 0 < 2 * C ^ 2 + C * T - 1 := by nlinarith
  have ha4 : 0 < A.a ^ 4 := by positivity
  refine ⟨y, hy, hu2, hsin, hpi, ?_, ?_⟩
  · by_contra h
    push Not at h
    have h1 : 0 ≤ (C + T) ^ 2 * (3 * A.a ^ 2 - y ^ 2) ^ 2 := by positivity
    have h2 : 0 ≤ 4 * A.a ^ 2 * (2 * C ^ 2 + C * T - 1) * (y ^ 2 - 3 * A.a ^ 2) := by
      have : 0 ≤ y ^ 2 - 3 * A.a ^ 2 := by linarith
      positivity
    have h3 : 0 < 16 * A.a ^ 4 * (C ^ 2 - 1) := by
      have : 0 < C ^ 2 - 1 := by nlinarith
      positivity
    nlinarith
  · -- the discriminant: `4a⁴𝓕 = ((C + T)²τ - 2a²(2C² + CT - 1))²`
    have e : 4 * A.a ^ 4 * densF C T = ((C + T) ^ 2 * (3 * A.a ^ 2 - y ^ 2) -
        2 * A.a ^ 2 * (2 * C ^ 2 + C * T - 1)) ^ 2 := by
      unfold densF
      linear_combination (-(C + T) ^ 2) * hq
    have : 0 ≤ 4 * A.a ^ 4 * densF C T := e ▸ sq_nonneg _
    by_contra h
    push Not at h
    have := mul_neg_of_pos_of_neg (by positivity : (0 : ℝ) < 4 * A.a ^ 4) h
    linarith

/-- **`|ρ| < e^{-βU/2}/√3` at a static density zero mode**: `3(ρe^{βa})² < 1`, from `𝓕 ≥ 0` and
`(C + T)² - 3(C² - 1)T² = 𝓕 + (C² - 1)(1 + 4CT)`. -/
theorem rho_sq_lt_of_zero_mode (hU : 0 < A.U) (hμ : A.μ ≠ 0) {u : ℂ} (hΦ : A.Phik 0 1 u = 0)
    (hK : A.Ke 0 1 u = 0) : 3 * (A.ρ * Real.exp (A.β * A.a)) ^ 2 < 1 := by
  obtain ⟨-, -, -, -, -, -, hF⟩ := A.zero_mode_dens hU hμ hΦ hK
  have hρ := A.rho_TC
  set C := Real.cosh (A.β * A.μ)
  set T := Real.exp (A.β * A.a)
  have hC1 : 1 < C := Real.one_lt_cosh.mpr (mul_ne_zero A.β_pos.ne' hμ)
  have hT0 : 0 < T := Real.exp_pos _
  have hsh : Real.sinh (A.β * A.μ) ^ 2 = C ^ 2 - 1 := by
    have := Real.cosh_sq_sub_sinh_sq (A.β * A.μ); linarith
  have e : (C + T) ^ 2 - 3 * (C ^ 2 - 1) * T ^ 2 = densF C T + (C ^ 2 - 1) * (1 + 4 * C * T) := by
    unfold densF; ring
  have h1 : 0 < (C ^ 2 - 1) * (1 + 4 * C * T) := by
    have : 0 < C ^ 2 - 1 := by nlinarith
    have : 0 < 1 + 4 * C * T := by positivity
    positivity
  have h2 : 3 * (A.ρ * T) ^ 2 * (C + T) ^ 2 = 3 * (C ^ 2 - 1) * T ^ 2 := by
    rw [← hsh, ← hρ]; ring
  have h3 : 3 * (A.ρ * T) ^ 2 * (C + T) ^ 2 < 1 * (C + T) ^ 2 := by linarith
  exact lt_of_mul_lt_mul_right h3 (sq_nonneg _)

/-- `π/√3 < βa` at a static density zero mode. -/
theorem pi_div_sqrt_three_lt_of_zero_mode (hU : 0 < A.U) (hμ : A.μ ≠ 0) {u : ℂ}
    (hΦ : A.Phik 0 1 u = 0) (hK : A.Ke 0 1 u = 0) :
    Real.pi / Real.sqrt 3 < A.β * A.a := by
  obtain ⟨y, hy, -, -, hpi, hy3, -⟩ := A.zero_mode_dens hU hμ hΦ hK
  have hβ := A.β_pos
  have ha : 0 < A.a := by rw [a]; linarith
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have hs3 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  rw [div_lt_iff₀ hs3]
  by_contra h
  push Not at h
  have hab : 0 < A.β * A.a * Real.sqrt 3 := by positivity
  have h1 : (A.β * A.a * Real.sqrt 3) ^ 2 ≤ Real.pi ^ 2 := pow_le_pow_left₀ hab.le h 2
  have h2 : Real.pi ^ 2 < (A.β * y) ^ 2 := pow_lt_pow_left₀ hpi Real.pi_pos.le two_ne_zero
  have h4 : (A.β * y) ^ 2 < 3 * (A.β * A.a) ^ 2 := by
    have := mul_lt_mul_of_pos_left hy3 (pow_pos hβ 2)
    nlinarith
  have h5 : (A.β * A.a * Real.sqrt 3) ^ 2 = 3 * (A.β * A.a) ^ 2 := by rw [mul_pow, h3]; ring
  linarith

/-- **The bound on `μ` at a static density zero mode**: `𝓕(cosh βμ, e^{π/√3}) > 0`, that is
`|βμ| < arcosh 𝖼_*(e^{π/√3}) = 0.57888`. -/
theorem densF_pos_of_zero_mode (hU : 0 < A.U) (hμ : A.μ ≠ 0) {u : ℂ} (hΦ : A.Phik 0 1 u = 0)
    (hK : A.Ke 0 1 u = 0) :
    0 < densF (Real.cosh (A.β * A.μ)) (Real.exp (Real.pi / Real.sqrt 3)) := by
  obtain ⟨-, -, -, -, -, -, hF⟩ := A.zero_mode_dens hU hμ hΦ hK
  have hlt := A.pi_div_sqrt_three_lt_of_zero_mode hU hμ hΦ hK
  have hs3 : Real.sqrt 3 < 2 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hs0 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  have hx1 : 1 < Real.pi / Real.sqrt 3 := by
    rw [lt_div_iff₀ hs0]; have := Real.pi_gt_three; linarith
  have hT₀ : 2 < Real.exp (Real.pi / Real.sqrt 3) := by
    have := Real.add_one_le_exp (Real.pi / Real.sqrt 3); linarith
  exact densF_pos_of_lt (Real.one_le_cosh _) (by nlinarith) (by linarith)
    (Real.exp_lt_exp.mpr hlt) hF

/-- The rational form of the bound: `cosh βμ < 47/40` (`|βμ| < 0.5833`) at a static density zero
mode. -/
theorem cosh_lt_of_zero_mode (hU : 0 < A.U) (hμ : A.μ ≠ 0) {u : ℂ} (hΦ : A.Phik 0 1 u = 0)
    (hK : A.Ke 0 1 u = 0) : Real.cosh (A.β * A.μ) < 47 / 40 := by
  obtain ⟨-, -, -, -, -, -, hF⟩ := A.zero_mode_dens hU hμ hΦ hK
  have hlt := A.pi_div_sqrt_three_lt_of_zero_mode hU hμ hΦ hK
  have hs3 : Real.sqrt 3 < 26 / 15 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hs0 : 0 < Real.sqrt 3 := Real.sqrt_pos.mpr (by norm_num)
  have h95 : 9 / 5 ≤ A.β * A.a := by
    have : 9 / 5 < Real.pi / Real.sqrt 3 := by
      rw [lt_div_iff₀ hs0]; have := Real.pi_gt_d2; nlinarith
    linarith
  by_contra h
  push Not at h
  exact absurd hF (not_le.mpr (densF_neg h (six_le_exp h95)))

end Params

end HubbardAtom
