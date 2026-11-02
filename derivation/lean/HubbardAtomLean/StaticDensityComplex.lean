import HubbardAtomLean.StaticDensity

/-!
# No zero of `Φ₀` at the complex density roots for `U < 0`

The static density channel (`k = 0`) for `U < 0`, `μ ≠ 0`: where `𝓕(C, T) < 0`
(`T = e^{βa}`, `C = cosh βμ`, `densF`), the roots of `e` are not real, and `Φ₀` does not vanish
there (`Params.Phik_ne_zero_of_densF_neg`). With `A = -βa > 0`, `S = T + C`, `g = 3T + C`
and `βu = X + iY`:

* the root equations (`Params.eph_dens`) give `S²(X² - Y²) = A²(C² - 3T² - 4CT - 2)` and
  `S²(X² + Y²)² = A⁴(g² - 4)`, so `k = (S(X² + Y²)/A² + g)/2 > 1` solves `k + 1/k = g`, and
  `S²Y²/A² = Sk - (C² - 1) ≤ 4TS`, whence `Y² < 4A²e^{-A} ≤ 16/e² < (π/2)²` (`dens_root_k`,
  `dens_root_Y_sq`);
* for `A ≤ 2`, `Im Φ₀ = sinh X sin Y - A Im(sinh w/w) ≠ 0`, since
  `2(X cosh X sin Y - Y sinh X cos Y) < (X² + Y²) sinh X sin Y` for `X > 0`, `0 < Y ≤ π/2`
  (`two_mul_j_lt`, as `sinh(vX) sin(vY) < v sinh X sin Y` for `0 < v < 1`) (`dens_root_small`);
* for `A > 2`, `K_eA_e = β²e = 0` gives `R₀ = ±S(s + q₁)/(2a²)`, so
  `(C² - 1 + ir)(cosh w + C) = ±S(C cosh w + 1)` with `r = S²XY/A²`. On `A_e = 0` the imaginary
  part of `cosh w` has the sign opposite to that of `XY` (`dens_root_Ae`); on `K_e = 0`,
  `cosh w = (1 - 3CT + ir)/(4T)`, hence `sin²Y = S²Y²/(4TkA²)`, and `sin²Y < Y²` contradicts
  `4TkA² < S²` (`four_t_k_lt`, from `m⁴ + 1 ≥ (7/4)m³` and `A < (7/8)e^{A/2} - e^{-A/2}` for
  `A ≥ 2`) (`dens_root_Ke`).
-/

open Complex

namespace HubbardAtom

/-! ### The real estimates -/

/-- `sinh(vX) < v sinh X` for `0 < v < 1`, `X > 0`. -/
theorem sinh_mul_lt {v X : ℝ} (hv0 : 0 < v) (hv1 : v < 1) (hX : 0 < X) :
    Real.sinh (v * X) < v * Real.sinh X := by
  let φ : ℝ → ℝ := fun x ↦ v * Real.sinh x - Real.sinh (x * v)
  have hd : ∀ x, HasDerivAt φ (v * Real.cosh x - Real.cosh (x * v) * v) x := fun x ↦
    ((Real.hasDerivAt_sinh x).const_mul v).sub (hasDerivAt_mul_const v).sinh
  have hmono : StrictMonoOn φ (Set.Ici 0) := by
    apply strictMonoOn_of_deriv_pos (convex_Ici 0)
    · exact (continuous_iff_continuousAt.mpr fun x ↦ (hd x).continuousAt).continuousOn
    · intro x hx
      rw [interior_Ici] at hx
      rw [(hd x).deriv]
      have hx0 : 0 < x := hx
      have : Real.cosh (x * v) < Real.cosh x := by
        rw [Real.cosh_lt_cosh, abs_of_pos (mul_pos hx0 hv0), abs_of_pos hx0]
        nlinarith
      nlinarith
  have := hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hX.le) hX
  simp only [φ, mul_zero, zero_mul, Real.sinh_zero, sub_zero] at this
  rw [mul_comm v X]
  linarith

/-- `2(X cosh X sin Y - Y sinh X cos Y) < (X² + Y²) sinh X sin Y` for `X > 0`,
`0 < Y ≤ π/2`: the imaginary part of `sinh w/w`, `w = X + iY`, is
`∫₀¹ sinh(vX) sin(vY) dv < sinh X sin Y/2`. -/
theorem two_mul_j_lt {X Y : ℝ} (hX : 0 < X) (hY : 0 < Y) (hYp : Y ≤ Real.pi / 2) :
    2 * (X * Real.cosh X * Real.sin Y - Y * Real.sinh X * Real.cos Y) <
      (X ^ 2 + Y ^ 2) * (Real.sinh X * Real.sin Y) := by
  set K := Real.sinh X * Real.sin Y
  let G : ℝ → ℝ := fun v ↦ v ^ 2 * (X ^ 2 + Y ^ 2) * K -
    2 * (X * Real.cosh (v * X) * Real.sin (v * Y) - Y * Real.sinh (v * X) * Real.cos (v * Y))
  have hd : ∀ v, HasDerivAt G
      (2 * (X ^ 2 + Y ^ 2) * (v * K - Real.sinh (v * X) * Real.sin (v * Y))) v := fun v ↦ by
    have hx := hasDerivAt_mul_const (x := v) X
    have hy := hasDerivAt_mul_const (x := v) Y
    have h1 := (hasDerivAt_pow 2 v).mul_const ((X ^ 2 + Y ^ 2) * K)
    have h2 := (((hx.cosh).const_mul X).mul hy.sin).sub (((hx.sinh).const_mul Y).mul hy.cos)
    have h := h1.sub (h2.const_mul 2)
    convert h using 1
    · funext w; simp only [G, Pi.sub_apply, Pi.mul_apply]; ring
    · simp only [Nat.cast_ofNat]; ring
  have hmono : StrictMonoOn G (Set.Icc 0 1) := by
    apply strictMonoOn_of_deriv_pos (convex_Icc 0 1)
    · exact (continuous_iff_continuousAt.mpr fun x ↦ (hd x).continuousAt).continuousOn
    · intro v hv
      rw [interior_Icc] at hv
      obtain ⟨hv0, hv1⟩ := hv
      rw [(hd v).deriv]
      have h1 := sinh_mul_lt hv0 hv1 hX
      have h2 : 0 < Real.sin (v * Y) :=
        Real.sin_pos_of_pos_of_lt_pi (mul_pos hv0 hY) (by nlinarith [Real.pi_pos])
      have h3 : Real.sin (v * Y) ≤ Real.sin Y :=
        Real.sin_le_sin_of_le_of_le_pi_div_two (by nlinarith [Real.pi_pos]) hYp (by nlinarith)
      have h4 : 0 < Real.sinh X := Real.sinh_pos_iff.mpr hX
      have h5 : Real.sinh (v * X) * Real.sin (v * Y) < v * K := by
        calc Real.sinh (v * X) * Real.sin (v * Y) < v * Real.sinh X * Real.sin (v * Y) :=
              mul_lt_mul_of_pos_right h1 h2
          _ ≤ v * Real.sinh X * Real.sin Y :=
              mul_le_mul_of_nonneg_left h3 (by positivity)
          _ = v * K := by ring
      have : 0 < X ^ 2 + Y ^ 2 := by positivity
      nlinarith
  have := hmono (Set.left_mem_Icc.mpr zero_le_one) (Set.right_mem_Icc.mpr zero_le_one) one_pos
  simp only [G, zero_mul, Real.cosh_zero, Real.sin_zero, Real.sinh_zero, Real.cos_zero, one_mul,
    mul_zero, zero_mul, sub_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
    one_pow] at this
  linarith

/-- `m⁴ + 1 ≥ (7/4)m³`; the minimum of `(√k + k^{-3/2})/2` over `k > 0` is `2/3^{3/4} > 7/8`. -/
theorem quartic_ge (m : ℝ) : 7 / 4 * m ^ 3 ≤ m ^ 4 + 1 := by
  nlinarith [sq_nonneg (m ^ 2 - 7 / 8 * m - 29 / 50), sq_nonneg (m - 812 / 631)]

/-- For `A ≥ 2` and `τ = e^{-A/2}`: `2Aτ < 7/4 - 2τ²`, i.e. `A < (7/8)e^{A/2} - e^{-A/2}`. -/
theorem exp_half_bound {A : ℝ} (hA : 2 ≤ A) :
    2 * A * Real.exp (-(A / 2)) < 7 / 4 - 2 * Real.exp (-(A / 2)) ^ 2 := by
  set Y := Real.exp (A / 2)
  have hY0 : 0 < Y := Real.exp_pos _
  have hτ : Real.exp (-(A / 2)) = 1 / Y := by rw [Real.exp_neg, one_div]
  have hE := Real.exp_one_gt_d9
  -- `Y ≥ eA/2`
  have hYA : Real.exp 1 * (A / 2) ≤ Y := by
    have h1 := Real.add_one_le_exp ((A - 2) / 2)
    have h2 : Y = Real.exp 1 * Real.exp ((A - 2) / 2) := by
      simp only [Y]; rw [← Real.exp_add]; ring_nf
    rw [h2]
    have : A / 2 ≤ Real.exp ((A - 2) / 2) := by linarith
    exact mul_le_mul_of_nonneg_left this (Real.exp_pos 1).le
  rw [hτ]
  rw [show 2 * A * (1 / Y) = 2 * A * Y / Y ^ 2 by field_simp,
    show 7 / 4 - 2 * (1 / Y) ^ 2 = (7 / 4 * Y ^ 2 - 2) / Y ^ 2 by field_simp]
  apply div_lt_div_of_pos_right _ (by positivity)
  -- `(7/4)Y² - 2AY - 2 > 0` for `Y ≥ eA/2 > 4A/7`
  have hA0 : 0 < A := by linarith
  set E := Real.exp 1
  have hq0 : 0 < 7 / 4 * (E * (A / 2)) - 2 * A := by nlinarith
  have hq : 7 / 4 * (E * (A / 2)) - 2 * A ≤ 7 / 4 * Y - 2 * A := by linarith
  have h2 : E * (A / 2) * (7 / 4 * (E * (A / 2)) - 2 * A) ≤ Y * (7 / 4 * Y - 2 * A) :=
    mul_le_mul hYA hq hq0.le hY0.le
  have h3 : E * (A / 2) * (7 / 4 * (E * (A / 2)) - 2 * A) = A ^ 2 * (7 / 16 * E ^ 2 - E) := by ring
  have h4 : 2 < 4 * (7 / 16 * E ^ 2 - E) := by nlinarith
  have h5 : 4 * (7 / 16 * E ^ 2 - E) ≤ A ^ 2 * (7 / 16 * E ^ 2 - E) :=
    mul_le_mul_of_nonneg_right (by nlinarith) (by nlinarith)
  nlinarith

/-- `eA/2 ≤ e^{A/2}`, so `A²e^{-A} ≤ 4/e²` for `A ≥ 0`. -/
theorem sq_mul_exp_neg_le {A : ℝ} (hA : 0 ≤ A) : A ^ 2 * Real.exp (-A) ≤ 4 / Real.exp 1 ^ 2 := by
  have h1 := Real.add_one_le_exp ((A - 2) / 2)
  have h2 : Real.exp (A / 2) = Real.exp 1 * Real.exp ((A - 2) / 2) := by
    rw [← Real.exp_add]; ring_nf
  have hE := Real.exp_pos 1
  have h3 : A * Real.exp 1 / 2 ≤ Real.exp (A / 2) := by
    rw [h2]; nlinarith [Real.exp_pos ((A - 2) / 2)]
  have h8 : Real.exp (-A) * Real.exp (A / 2) ^ 2 = 1 := by
    rw [← Real.exp_nat_mul, ← Real.exp_add]; push_cast; ring_nf; exact Real.exp_zero
  have h9 : (A * Real.exp 1 / 2) ^ 2 ≤ Real.exp (A / 2) ^ 2 :=
    pow_le_pow_left₀ (by positivity) h3 2
  rw [le_div_iff₀ (by positivity)]
  have h10 := Real.exp_pos (-A)
  nlinarith

/-- The inequality of the case `A ≥ 2`: for `k > 1`, `t = e^{-A}` and `dk = k² + 1 - 2tk`,
`4tkA² < d²`. With `k = m²`, `τ = e^{-A/2}`: `dk = m⁴ + 1 - 2τ²m² > (7/4 - 2τ²)m³ > 2Aτm³`. -/
theorem four_t_k_lt {A k d : ℝ} (hA : 2 ≤ A) (hk : 1 < k)
    (hd : d * k = k ^ 2 + 1 - 2 * Real.exp (-A) * k) : 4 * Real.exp (-A) * k * A ^ 2 < d ^ 2 := by
  set τ := Real.exp (-(A / 2))
  have ht : Real.exp (-A) = τ ^ 2 := by
    rw [← Real.exp_nat_mul]; push_cast; ring_nf
  set m := Real.sqrt k
  have hm2 : m ^ 2 = k := Real.sq_sqrt (by linarith)
  have hm1 : 1 < m := by
    rw [show (1 : ℝ) = Real.sqrt 1 by simp]; exact Real.sqrt_lt_sqrt zero_le_one hk
  have hτ : 0 < τ := Real.exp_pos _
  have hb : 2 * A * τ < 7 / 4 - 2 * τ ^ 2 := exp_half_bound hA
  have hq := quartic_ge m
  rw [ht] at hd ⊢
  rw [← hm2] at hd ⊢
  have h1 : τ ^ 2 * m ^ 2 < τ ^ 2 * m ^ 3 := by
    have : m ^ 2 < m ^ 3 := by nlinarith
    exact mul_lt_mul_of_pos_left this (by positivity)
  have h2 : 2 * A * τ * m ^ 3 < (7 / 4 - 2 * τ ^ 2) * m ^ 3 :=
    mul_lt_mul_of_pos_right hb (by positivity)
  have h3 : 2 * A * τ * m ^ 3 < d * m ^ 2 := by nlinarith
  have hA0 : 0 < A := by linarith
  have h4 : 0 < 2 * A * τ * m ^ 3 := by positivity
  have h5 : (2 * A * τ * m ^ 3) ^ 2 < (d * m ^ 2) ^ 2 := pow_lt_pow_left₀ h3 h4.le two_ne_zero
  have h6 : 0 < m ^ 4 := by positivity
  have e1 : (2 * A * τ * m ^ 3) ^ 2 = 4 * τ ^ 2 * m ^ 2 * A ^ 2 * m ^ 4 := by ring
  have e2 : (d * m ^ 2) ^ 2 = d ^ 2 * m ^ 4 := by ring
  rw [e1, e2] at h5
  exact lt_of_mul_lt_mul_right h5 h6.le

/-- The imaginary part of `Φ₀ = 0` is invariant under `X ↦ |X|`, `Y ↦ |Y|`. -/
theorem im_eq_abs {A X Y : ℝ}
    (h : (X ^ 2 + Y ^ 2) * (Real.sinh X * Real.sin Y) =
      A * (X * Real.cosh X * Real.sin Y - Y * Real.sinh X * Real.cos Y)) :
    (|X| ^ 2 + |Y| ^ 2) * (Real.sinh |X| * Real.sin |Y|) =
      A * (|X| * Real.cosh |X| * Real.sin |Y| - |Y| * Real.sinh |X| * Real.cos |Y|) := by
  rcases le_or_gt 0 X with hX | hX <;> rcases le_or_gt 0 Y with hY | hY <;>
    simp only [abs_of_nonneg, abs_of_neg, hX, hY, Real.sinh_neg, Real.sin_neg, Real.cosh_neg,
      Real.cos_neg, neg_sq] <;> linarith

/-- The root equations in `k`: for `S = T + C`, `g = 3T + C` and `X + iY` with `XY ≠ 0` on the
equations `E1`, `E2` of a nonreal root, `k = (S(X² + Y²)/A² + g)/2 > 1` solves
`k² - gk + 1 = 0`, and `S²Y²/A² = Sk - (C² - 1)`, `kS²X²/A² = k(C² - 1) - S`. -/
theorem dens_root_k {A C T X Y : ℝ} (hA : 0 < A) (hC : 1 < C) (hT0 : 0 < T) (hXY : X * Y ≠ 0)
    (E1 : (T + C) ^ 2 * (X ^ 2 - Y ^ 2) = A ^ 2 * (C ^ 2 - 3 * T ^ 2 - 4 * C * T - 2))
    (E2 : (T + C) ^ 2 * (X ^ 2 + Y ^ 2) ^ 2 = A ^ 4 * ((3 * T + C) ^ 2 - 4)) :
    ∃ k : ℝ, 1 < k ∧ k ^ 2 - (3 * T + C) * k + 1 = 0 ∧
      (T + C) ^ 2 * Y ^ 2 / A ^ 2 = (T + C) * k - (C ^ 2 - 1) ∧
      k * ((T + C) ^ 2 * X ^ 2 / A ^ 2) = k * (C ^ 2 - 1) - (T + C) := by
  have hX : X ≠ 0 := left_ne_zero_of_mul hXY
  have hA2 : 0 < A ^ 2 := by positivity
  have hN : 0 < (X ^ 2 + Y ^ 2) / A ^ 2 := by
    have : 0 < X ^ 2 := by positivity
    positivity
  set N := (X ^ 2 + Y ^ 2) / A ^ 2
  have E2' : (T + C) ^ 2 * N ^ 2 = (3 * T + C) ^ 2 - 4 := by
    simp only [N]; field_simp; linear_combination E2
  have hg : 2 < 3 * T + C := by
    have h1 : 0 < (T + C) ^ 2 * N ^ 2 := by
      have : 0 < T + C := by linarith
      positivity
    nlinarith
  have E1' : (T + C) ^ 2 * (X ^ 2 - Y ^ 2) / A ^ 2 = C ^ 2 - 3 * T ^ 2 - 4 * C * T - 2 := by
    field_simp; linear_combination E1
  have hSN : 0 < (T + C) * N := mul_pos (by linarith) hN
  refine ⟨((T + C) * N + (3 * T + C)) / 2, by linarith, by linear_combination E2' / 4, ?_, ?_⟩
  · have : (T + C) ^ 2 * Y ^ 2 / A ^ 2 =
        ((T + C) ^ 2 * N - (T + C) ^ 2 * (X ^ 2 - Y ^ 2) / A ^ 2) / 2 := by
      simp only [N]; field_simp; ring
    rw [this, E1']; ring
  · have : (T + C) ^ 2 * X ^ 2 / A ^ 2 =
        ((T + C) ^ 2 * N + (T + C) ^ 2 * (X ^ 2 - Y ^ 2) / A ^ 2) / 2 := by
      simp only [N]; field_simp; ring
    rw [this, E1']
    linear_combination (T + C) / 4 * E2'

/-- On a nonreal root, `Y² < (π/2)²`: `S²Y²/A² ≤ 4TS` from `k²(4TS - S²Y²/A²) = (1 - Tk)²`,
and `4TA² = 4A²e^{-A} ≤ 16/e² < π²/4`. -/
theorem dens_root_Y_sq {A C T Y k : ℝ} (hA : 0 < A) (hC : 1 < C) (hT : T = Real.exp (-A))
    (hk1 : 1 < k) (hk : k ^ 2 - (3 * T + C) * k + 1 = 0)
    (hQe : (T + C) ^ 2 * Y ^ 2 / A ^ 2 = (T + C) * k - (C ^ 2 - 1)) :
    Y ^ 2 < (Real.pi / 2) ^ 2 := by
  have hT0 : 0 < T := hT ▸ Real.exp_pos _
  have hA2 : 0 < A ^ 2 := by positivity
  have hYb : k ^ 2 * (4 * T * (T + C) - (T + C) ^ 2 * Y ^ 2 / A ^ 2) = (1 - T * k) ^ 2 := by
    rw [hQe]; linear_combination (-(T + C) * k - 1) * hk
  have hQe4 : (T + C) ^ 2 * Y ^ 2 / A ^ 2 ≤ 4 * T * (T + C) := by
    have h0 : 0 ≤ k ^ 2 * (4 * T * (T + C) - (T + C) ^ 2 * Y ^ 2 / A ^ 2) := hYb ▸ sq_nonneg _
    have hk2 : 0 < k ^ 2 := by positivity
    have := (mul_nonneg_iff_of_pos_left hk2).mp h0
    linarith
  have h1 : (T + C) * ((T + C) * Y ^ 2) ≤ (T + C) * (4 * T * A ^ 2) := by
    have := (div_le_iff₀ hA2).mp hQe4; linarith
  have h2 : (T + C) * Y ^ 2 ≤ 4 * T * A ^ 2 := le_of_mul_le_mul_left h1 (by linarith)
  have h3 : Y ^ 2 ≤ (T + C) * Y ^ 2 := le_mul_of_one_le_left (sq_nonneg _) (by linarith)
  have h4 := sq_mul_exp_neg_le hA.le
  rw [← hT] at h4
  have hE := Real.exp_one_gt_d9
  have hpi := Real.pi_gt_d2
  have h5 : 16 / Real.exp 1 ^ 2 < (Real.pi / 2) ^ 2 := by
    rw [div_lt_iff₀ (by positivity)]
    have h := mul_lt_mul'' hpi hE (by norm_num) (by norm_num)
    have h' : (8 : ℝ) < Real.pi * Real.exp 1 := by linarith
    have h'' : (8 : ℝ) * 8 < (Real.pi * Real.exp 1) * (Real.pi * Real.exp 1) :=
      mul_self_lt_mul_self (by norm_num) h'
    nlinarith
  have h6 : 4 * T * A ^ 2 ≤ 16 / Real.exp 1 ^ 2 := by
    have : 4 * T * A ^ 2 = 4 * (A ^ 2 * T) := by ring
    rw [this, show 16 / Real.exp 1 ^ 2 = 4 * (4 / Real.exp 1 ^ 2) by ring]
    linarith
  linarith

/-- `XY sinh X sin Y > 0` for `XY ≠ 0`, `|Y| < π`. -/
theorem mul_sinh_sin_pos {X Y : ℝ} (hX : X ≠ 0) (hY : Y ≠ 0) (hYp : |Y| < Real.pi) :
    0 < X * Y * (Real.sinh X * Real.sin Y) := by
  have hsY : 0 < |Y| * Real.sin |Y| :=
    mul_pos (abs_pos.mpr hY) (Real.sin_pos_of_pos_of_lt_pi (abs_pos.mpr hY) hYp)
  have hsX : 0 < |X| * Real.sinh |X| :=
    mul_pos (abs_pos.mpr hX) (Real.sinh_pos_iff.mpr (abs_pos.mpr hX))
  have e : X * Y * (Real.sinh X * Real.sin Y) = (|X| * Real.sinh |X|) * (|Y| * Real.sin |Y|) := by
    rcases le_or_gt 0 X with h | h <;> rcases le_or_gt 0 Y with h' | h' <;>
      simp only [abs_of_nonneg, abs_of_neg, h, h', Real.sinh_neg, Real.sin_neg] <;> ring
  rw [e]; positivity

/-- The case `A ≤ 2`: the imaginary part of `Φ₀` does not vanish. -/
theorem dens_root_small {A X Y : ℝ} (hA : 0 < A) (hA2 : A ≤ 2) (hX : X ≠ 0) (hY : Y ≠ 0)
    (hYp : |Y| ≤ Real.pi / 2)
    (hIm : (X ^ 2 + Y ^ 2) * (Real.sinh X * Real.sin Y) =
      A * (X * Real.cosh X * Real.sin Y - Y * Real.sinh X * Real.cos Y)) : False := by
  have h := im_eq_abs hIm
  have hj := two_mul_j_lt (abs_pos.mpr hX) (abs_pos.mpr hY) hYp
  have hK : 0 < Real.sinh |X| * Real.sin |Y| := mul_pos (Real.sinh_pos_iff.mpr (abs_pos.mpr hX))
    (Real.sin_pos_of_pos_of_lt_pi (abs_pos.mpr hY) (by linarith [Real.pi_pos]))
  have hXY2 : 0 < |X| ^ 2 + |Y| ^ 2 := by
    have := abs_pos.mpr hX
    positivity
  set J := |X| * Real.cosh |X| * Real.sin |Y| - |Y| * Real.sinh |X| * Real.cos |Y|
  set K := Real.sinh |X| * Real.sin |Y|
  have h0 : 0 < (|X| ^ 2 + |Y| ^ 2) * K := mul_pos hXY2 hK
  rcases le_or_gt J 0 with hJ | hJ
  · have : A * J ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hA.le hJ
    linarith
  · have : A * J ≤ 2 * J := mul_le_mul_of_nonneg_right hA2 hJ.le
    linarith

/-- The branch `A_e = 0` (`σ = -1`): there `(e² + r²) Im cosh w = -rS(C² - 1)` has the sign
opposite to `XY`. -/
theorem dens_root_Ae {C S P1 Q1 r : ℝ} (hC : 1 < C) (hS : 0 < S) (hrQ : 0 < r * Q1)
    (hRe' : (C ^ 2 - 1) * (P1 + C) - r * Q1 = -1 * S * (C * P1 + 1))
    (hIm' : r * (P1 + C) + (C ^ 2 - 1) * Q1 = -1 * S * (C * Q1)) : False := by
  set e := C ^ 2 - 1 + S * C
  have hQ : (e ^ 2 + r ^ 2) * (r * Q1) = -(r ^ 2 * S * (C ^ 2 - 1)) := by
    simp only [e]; linear_combination r * ((C ^ 2 - 1 + S * C) * hIm' - r * hRe')
  have hr : 0 < r ^ 2 := by
    have : r ≠ 0 := fun h ↦ by rw [h, zero_mul] at hrQ; exact lt_irrefl 0 hrQ
    positivity
  have h1 : 0 < (e ^ 2 + r ^ 2) * (r * Q1) := by positivity
  have h2 : 0 ≤ r ^ 2 * S * (C ^ 2 - 1) := by
    have : 0 ≤ C ^ 2 - 1 := by nlinarith
    positivity
  linarith

/-- The branch `K_e = 0` (`σ = 1`) for `A > 2`: there `cosh w = (1 - 3CT + ir)/(4T)`, hence
`sin²Y = S²Y²/(4TkA²)`, and `sin²Y < Y²` contradicts `4TkA² < S²` (`four_t_k_lt`). -/
theorem dens_root_Ke {A C T X Y k : ℝ} (hA : 2 < A) (hC : 1 < C) (hT : T = Real.exp (-A))
    (hX : X ≠ 0) (hY : Y ≠ 0) (hk1 : 1 < k) (hk : k ^ 2 - (3 * T + C) * k + 1 = 0)
    (hQe : (T + C) ^ 2 * Y ^ 2 / A ^ 2 = (T + C) * k - (C ^ 2 - 1))
    (hPe : k * ((T + C) ^ 2 * X ^ 2 / A ^ 2) = k * (C ^ 2 - 1) - (T + C))
    (hRe' : (C ^ 2 - 1) * (Real.cosh X * Real.cos Y + C) -
      (T + C) ^ 2 * X * Y / A ^ 2 * (Real.sinh X * Real.sin Y) =
        1 * (T + C) * (C * (Real.cosh X * Real.cos Y) + 1))
    (hIm' : (T + C) ^ 2 * X * Y / A ^ 2 * (Real.cosh X * Real.cos Y + C) +
      (C ^ 2 - 1) * (Real.sinh X * Real.sin Y) = 1 * (T + C) * (C * (Real.sinh X * Real.sin Y))) :
    False := by
  have hT0 : 0 < T := hT ▸ Real.exp_pos _
  have hA2 : 0 < A ^ 2 := by positivity
  set S := T + C with hSdef
  set Pe := S ^ 2 * X ^ 2 / A ^ 2
  set Qe := S ^ 2 * Y ^ 2 / A ^ 2
  set r := S ^ 2 * X * Y / A ^ 2
  set P1 := Real.cosh X * Real.cos Y
  set Q1 := Real.sinh X * Real.sin Y
  have hr2 : r ^ 2 = Pe * Qe := by simp only [r, Pe, Qe]; field_simp
  have hPQ : Pe * Qe = 4 * T * S * (C ^ 2 - 1) - (1 + C * T) ^ 2 := by
    have : k * (Pe * Qe) = k * (4 * T * S * (C ^ 2 - 1) - (1 + C * T) ^ 2) := by
      rw [← mul_assoc, hPe, hQe]; simp only [hSdef]
      linear_combination (T + C) * (C ^ 2 - 1) * hk
    exact mul_left_cancel₀ (by linarith) this
  have hC2 : 0 < S * (C ^ 2 - 1) := mul_pos (by linarith) (by nlinarith)
  -- `Im cosh w = r/(4T)`, `Re cosh w = (1 - 3CT)/(4T)`
  have hQ : ((1 + C * T) ^ 2 + r ^ 2) * Q1 = r * S * (C ^ 2 - 1) := by
    simp only [hSdef]
    linear_combination (-1 - C * T) * hIm' - r * hRe'
  have hQ1 : Q1 = r / (4 * T) := by
    rw [eq_div_iff (by positivity)]
    rw [hr2, hPQ] at hQ
    have : (4 * T * Q1 - r) * (S * (C ^ 2 - 1)) = 0 := by linear_combination hQ
    rcases mul_eq_zero.mp this with h | h
    · linarith
    · linarith
  have hP1 : P1 = (1 - 3 * C * T) / (4 * T) := by
    rw [eq_div_iff (by positivity)]
    have h1 : (-1 - C * T) * P1 = r * Q1 + S - C * (C ^ 2 - 1) := by
      simp only [hSdef]; linear_combination hRe'
    rw [hQ1] at h1
    have h3 : r * (r / (4 * T)) * (4 * T) = Pe * Qe := by rw [← hr2]; field_simp
    rw [hPQ] at h3
    have h2 : (-1 - C * T) * (P1 * (4 * T)) = (-1 - C * T) * (1 - 3 * C * T) := by
      simp only [hSdef] at h3 ⊢
      linear_combination (4 * T) * h1 + h3
    exact mul_left_cancel₀ (by nlinarith) h2
  -- `sinh²X - sin²Y = P1² + Q1² - 1` and `sinh²X sin²Y = Q1²`
  have hdiff : Real.sinh X ^ 2 - Real.sin Y ^ 2 = P1 ^ 2 + Q1 ^ 2 - 1 := by
    simp only [P1, Q1]
    have h1 := Real.cosh_sq X
    have h2 := Real.cos_sq_add_sin_sq Y
    linear_combination (-(Real.cos Y ^ 2)) * h1 - (Real.sinh X ^ 2 + 1) * h2
  have hprod : Real.sinh X ^ 2 * Real.sin Y ^ 2 = Q1 ^ 2 := by simp only [Q1]; ring
  have hk0 : 0 < k := by linarith
  set Zs := k * Pe / (4 * T)
  set Qs := Qe / (4 * T * k)
  have hZQ : Zs * Qs = Q1 ^ 2 := by
    rw [hQ1, div_pow, hr2]; simp only [Zs, Qs]; field_simp
  have hZQd : Zs - Qs = P1 ^ 2 + Q1 ^ 2 - 1 := by
    rw [hP1, hQ1]
    simp only [div_pow]
    rw [hr2, hPQ]
    simp only [Zs, Qs]
    rw [hPe, hQe]
    field_simp
    simp only [hSdef]
    linear_combination (4 * T * (C ^ 2 - 1)) * hk
  have hZ0 : 0 ≤ Zs := by simp only [Zs, Pe]; positivity
  have hQs0 : 0 ≤ Qs := by simp only [Qs, Qe]; positivity
  have hsum : Real.sinh X ^ 2 + Real.sin Y ^ 2 = Zs + Qs := by
    have h1 : (Real.sinh X ^ 2 + Real.sin Y ^ 2) ^ 2 = (Zs + Qs) ^ 2 := by
      have e1 : (Real.sinh X ^ 2 + Real.sin Y ^ 2) ^ 2 =
          (Real.sinh X ^ 2 - Real.sin Y ^ 2) ^ 2 + 4 * (Real.sinh X ^ 2 * Real.sin Y ^ 2) := by
        ring
      have e2 : (Zs + Qs) ^ 2 = (Zs - Qs) ^ 2 + 4 * (Zs * Qs) := by ring
      rw [e1, e2, hdiff, hprod, hZQ, hZQd]
    exact (pow_left_inj₀ (by positivity) (by positivity) two_ne_zero).mp h1
  have hsY : Real.sin Y ^ 2 = Qs := by
    have := hdiff.trans hZQd.symm; linarith
  have hlt := Real.sin_sq_lt_sq hY
  rw [hsY] at hlt
  have hSk : S * k = k ^ 2 + 1 - 2 * Real.exp (-A) * k := by
    rw [← hT]; simp only [hSdef]; linear_combination (-1) * hk
  have h4 := four_t_k_lt hA.le hk1 hSk
  rw [← hT] at h4
  have hY0 : 0 < Y ^ 2 := by positivity
  simp only [Qs, Qe] at hlt
  rw [div_div, div_lt_iff₀ (by positivity)] at hlt
  have : S ^ 2 * Y ^ 2 < 4 * T * k * A ^ 2 * Y ^ 2 := by linarith
  have := lt_of_mul_lt_mul_right this hY0.le
  linarith

/-! ### The density static channel at a complex root -/

theorem cosh_re_im (X Y : ℝ) : cosh ((X : ℂ) + Y * I) =
    ((Real.cosh X * Real.cos Y : ℝ) : ℂ) + ((Real.sinh X * Real.sin Y : ℝ) : ℂ) * I := by
  rw [Complex.cosh_add, cosh_mul_I, sinh_mul_I]; push_cast; ring

theorem sinh_re_im (X Y : ℝ) : sinh ((X : ℂ) + Y * I) =
    ((Real.sinh X * Real.cos Y : ℝ) : ℂ) + ((Real.cosh X * Real.sin Y : ℝ) : ℂ) * I := by
  rw [Complex.sinh_add, cosh_mul_I, sinh_mul_I]; push_cast; ring

theorem re_im_eq_zero {x y : ℝ} (h : (x : ℂ) + y * I = 0) : x = 0 ∧ y = 0 := by
  have h1 := congrArg Complex.re h
  have h2 := congrArg Complex.im h
  simp at h1 h2
  exact ⟨h1, h2⟩

namespace Params

variable (A : Params)

/-- **No zero of `Φ₀` at the complex density roots for `U < 0`**: for `U < 0`, `μ ≠ 0` and
`𝓕(cosh βμ, e^{βa}) < 0`, where the roots of `e` in the static density channel are not real,
`Φ₀(u) ≠ 0` at every root `u` of `e`. -/
theorem Phik_ne_zero_of_densF_neg (hU : A.U < 0) (hμ : A.μ ≠ 0)
    (hF : densF (Real.cosh (A.β * A.μ)) (Real.exp (A.β * A.a)) < 0) {u : ℂ}
    (he : A.eph 0 1 u = 0) : A.Phik 0 1 u ≠ 0 := by
  intro hΦ
  have hβ := A.β_pos
  have ha : A.a < 0 := by rw [a]; linarith
  have ha' : (A.a : ℂ) ≠ 0 := ofReal_ne_zero.mpr ha.ne
  -- `D₀(u) ≠ 0`, and `K_eA_e = β²e = 0`
  have hu : ∀ m, A.z 0 m ≠ u := fun m h ↦ A.Phik_z_ne_zero 0 1 one_ne_zero ha' hμ m (h ▸ hΦ)
  have hu' : ∀ m, A.z 0 m ≠ -u := fun m h ↦
    A.Phik_z_ne_zero 0 1 one_ne_zero ha' hμ m (by rw [h, A.Phik_neg, hΦ])
  have hD := A.Dk_ne_zero_of_off 0 hu hu'
  have hKA := A.Ke_mul_Ae_of_Phi 0 1 hD hΦ
  rw [he, mul_zero, mul_eq_zero] at hKA
  have hE := A.eph_dens u
  rw [he, mul_zero] at hE
  obtain ⟨h₁, h₂⟩ := A.p₁_p₀p₂_TC
  set C := Real.cosh (A.β * A.μ) with hCdef
  set T := Real.exp (A.β * A.a) with hTdef
  have hC : 1 < C := Real.one_lt_cosh.mpr (mul_ne_zero hβ.ne' hμ)
  have hT0 : 0 < T := Real.exp_pos _
  have hS : 0 < T + C := by linarith
  set Aa := -(A.β * A.a) with hAadef
  have hA : 0 < Aa := neg_pos.mpr (mul_neg_of_pos_of_neg hβ ha)
  have hT : T = Real.exp (-Aa) := by rw [hAadef, neg_neg]
  set X := ((A.β : ℂ) * u).re
  set Y := ((A.β : ℂ) * u).im
  have hw : (A.β : ℂ) * u = X + Y * I := (Complex.re_add_im _).symm
  -- the root equation in `w = βu = X + iY`
  have hAa2 : ((Aa : ℂ)) ^ 2 = (A.β : ℂ) ^ 2 * (A.a : ℂ) ^ 2 := by
    simp only [hAadef]; push_cast; ring
  have hw2 : (X + Y * I : ℂ) ^ 2 = (A.β : ℂ) ^ 2 * u ^ 2 := by rw [← hw]; ring
  clear_value Aa
  have hqw : ((T + C) ^ 2 : ℂ) * ((X + Y * I) ^ 2 + 3 * Aa ^ 2) ^ 2 -
      4 * Aa ^ 2 * (2 * C ^ 2 + C * T - 1) * ((X + Y * I) ^ 2 + 3 * Aa ^ 2) +
      16 * (Aa ^ 2) ^ 2 * (C ^ 2 - 1) = 0 := by
    rw [hw2, hAa2]
    push_cast at hE
    linear_combination (-(A.β : ℂ) ^ 4) * hE
  have hz : (X + Y * I : ℂ) ^ 2 = ((X ^ 2 - Y ^ 2 : ℝ) : ℂ) + ((2 * X * Y : ℝ) : ℂ) * I := by
    push_cast; linear_combination (Y : ℂ) ^ 2 * I_sq
  rw [hz] at hqw
  have hq : (((T + C) ^ 2 * ((X ^ 2 - Y ^ 2 + 3 * Aa ^ 2) ^ 2 - (2 * X * Y) ^ 2) -
      4 * Aa ^ 2 * (2 * C ^ 2 + C * T - 1) * (X ^ 2 - Y ^ 2 + 3 * Aa ^ 2) +
      16 * Aa ^ 4 * (C ^ 2 - 1) : ℝ) : ℂ) +
      ((2 * X * Y * (2 * (T + C) ^ 2 * (X ^ 2 - Y ^ 2 + 3 * Aa ^ 2) -
        4 * Aa ^ 2 * (2 * C ^ 2 + C * T - 1)) : ℝ) : ℂ) * I = 0 := by
    rw [← hqw]; push_cast
    linear_combination (-((T : ℂ) + C) ^ 2 * (2 * X * Y) ^ 2) * I_sq
  obtain ⟨hre, him⟩ := re_im_eq_zero hq
  -- the root is not real: `XY ≠ 0`, else `4Aa⁴𝓕` is a square
  have hXY : X * Y ≠ 0 := by
    intro h0
    have hq0 : 2 * X * Y = 0 := by linear_combination 2 * h0
    rw [hq0] at hre
    have : 4 * Aa ^ 4 * densF C T = ((T + C) ^ 2 * (X ^ 2 - Y ^ 2 + 3 * Aa ^ 2) -
        2 * Aa ^ 2 * (2 * C ^ 2 + C * T - 1)) ^ 2 := by
      unfold densF; linear_combination (-(T + C) ^ 2) * hre
    have h1 : 0 ≤ 4 * Aa ^ 4 * densF C T := this ▸ sq_nonneg _
    have h2 : 4 * Aa ^ 4 * densF C T < 0 := mul_neg_of_pos_of_neg (by positivity) hF
    linarith
  have hG1 : (T + C) ^ 2 * (X ^ 2 - Y ^ 2 + 3 * Aa ^ 2) -
      2 * Aa ^ 2 * (2 * C ^ 2 + C * T - 1) = 0 := by
    have : 2 * X * Y ≠ 0 := by
      rw [mul_assoc]; exact mul_ne_zero two_ne_zero hXY
    have := (mul_eq_zero.mp him).resolve_left this
    linear_combination this / 2
  have E1 : (T + C) ^ 2 * (X ^ 2 - Y ^ 2) = Aa ^ 2 * (C ^ 2 - 3 * T ^ 2 - 4 * C * T - 2) := by
    linear_combination hG1
  have E2 : (T + C) ^ 2 * (X ^ 2 + Y ^ 2) ^ 2 = Aa ^ 4 * ((3 * T + C) ^ 2 - 4) := by
    linear_combination 2 * (X ^ 2 - Y ^ 2) * hG1 - hre
  have hX : X ≠ 0 := left_ne_zero_of_mul hXY
  have hY : Y ≠ 0 := right_ne_zero_of_mul hXY
  -- the imaginary part of `wΦ₀ = w(cosh w + C) - Aa sinh w = 0`
  have hΦw : ((X * (Real.cosh X * Real.cos Y + C) - Y * (Real.sinh X * Real.sin Y) -
      Aa * (Real.sinh X * Real.cos Y) : ℝ) : ℂ) +
      ((X * (Real.sinh X * Real.sin Y) + Y * (Real.cosh X * Real.cos Y + C) -
        Aa * (Real.cosh X * Real.sin Y) : ℝ) : ℂ) * I = 0 := by
    have h := A.mul_Phik 0 1 u
    rw [hΦ, mul_zero] at h
    simp only [Dk, τk, even_zero', ↓reduceIte, one_mul] at h
    have hCc : (C : ℂ) = cosh (A.β * A.μ) := by rw [hCdef]; push_cast; rfl
    have h' : (A.β * u) * (cosh (A.β * u) + C) - Aa * sinh (A.β * u) = 0 := by
      rw [hAadef]; push_cast; linear_combination (-(A.β : ℂ)) * h + (A.β * u) * hCc
    rw [hw, cosh_re_im, sinh_re_im] at h'
    push_cast at h' ⊢
    linear_combination h' - ((Y : ℂ) * sinh X * sin Y) * I_sq
  obtain ⟨hRe0, hIm0⟩ := re_im_eq_zero hΦw
  have hIm : (X ^ 2 + Y ^ 2) * (Real.sinh X * Real.sin Y) =
      Aa * (X * Real.cosh X * Real.sin Y - Y * Real.sinh X * Real.cos Y) := by
    linear_combination X * hIm0 - Y * hRe0
  -- the branch: `K_e = 0` or `A_e = 0`
  have hsq : Real.sqrt (A.p₀ * A.p₂) = 1 / (2 * (C + T)) := by
    rw [h₂, show 1 / (4 * (C + T) ^ 2) = (1 / (2 * (C + T))) ^ 2 by field_simp; ring]
    exact Real.sqrt_sq (by positivity)
  obtain ⟨σ, hσ, hbr⟩ : ∃ σ : ℝ, (σ = 1 ∨ σ = -1) ∧
      A.β * (u ^ 2 + (A.a ^ 2 + 4 * A.a ^ 2 * A.p₁)) =
        σ * (4 * A.a ^ 2 * (A.β * (1 / (2 * (C + T))))) * A.Rk 0 u := by
    rcases hKA with hK | hK
    · refine ⟨1, Or.inl rfl, ?_⟩
      rw [A.Ke_eq 0 1 hD] at hK
      simp only [q₀, A.w_zero, lam, hth, hsq] at hK
      push_cast at hK ⊢
      linear_combination hK
    · refine ⟨-1, Or.inr rfl, ?_⟩
      rw [A.Ae_eq 0 1 hD] at hK
      simp only [q₀, A.w_zero, lam, hth, hsq] at hK
      push_cast at hK ⊢
      linear_combination hK
  -- in `w`: `(Sw² + Aa²g)(cosh w + C) = 2σAa²(C cosh w + 1)`
  have hCc : (C : ℂ) = cosh (A.β * A.μ) := by rw [hCdef]; push_cast; rfl
  have hDk : A.Dk 0 u = cosh (A.β * u) + C := by
    simp only [Dk, τk, even_zero', ↓reduceIte, one_mul, hCc]
  have hRD : A.Rk 0 u * (cosh (A.β * u) + C) = C * cosh (A.β * u) + 1 := by
    rw [← hDk, Rk, div_mul_cancel₀ _ hD, hCc]
    simp only [τk, even_zero', ↓reduceIte]
  have hI : (2 * ((C : ℂ) + T)) * (1 / (2 * (C + T))) = 1 := by
    have : ((C : ℂ) + T) ≠ 0 := by exact_mod_cast (by positivity : (C + T) ≠ 0)
    field_simp
  have hp₁ : (A.p₁ : ℂ) = T * (1 / (2 * (C + T))) := by rw [h₁]; push_cast; ring
  have hbw : ((T + C) * (X + Y * I) ^ 2 + Aa ^ 2 * (3 * T + C)) * (cosh (X + Y * I) + C) =
      σ * 2 * Aa ^ 2 * (C * cosh (X + Y * I) + 1) := by
    rw [← hw, hAa2]
    linear_combination (A.β * (T + C) * (cosh (A.β * u) + C)) * hbr +
      (-4 * A.a ^ 2 * A.β ^ 2 * (T + C) * (cosh (A.β * u) + C)) * hp₁ +
      (2 * A.a ^ 2 * A.β ^ 2 * (cosh (A.β * u) + C) * (σ * A.Rk 0 u - T)) * hI +
      (2 * σ * A.a ^ 2 * A.β ^ 2) * hRD
  rw [hz, cosh_re_im] at hbw
  set P1 := Real.cosh X * Real.cos Y
  set Q1 := Real.sinh X * Real.sin Y
  have hbw' : ((((T + C) * (X ^ 2 - Y ^ 2) + Aa ^ 2 * (3 * T + C)) * (P1 + C) -
      (T + C) * (2 * X * Y) * Q1 - σ * 2 * Aa ^ 2 * (C * P1 + 1) : ℝ) : ℂ) +
      (((T + C) * (2 * X * Y) * (P1 + C) + ((T + C) * (X ^ 2 - Y ^ 2) + Aa ^ 2 * (3 * T + C)) * Q1 -
        σ * 2 * Aa ^ 2 * (C * Q1) : ℝ) : ℂ) * I = 0 := by
    push_cast at hbw ⊢
    linear_combination hbw - ((T + C) * (2 * X * Y) * Q1 : ℂ) * I_sq
  obtain ⟨hBRe, hBIm⟩ := re_im_eq_zero hbw'
  have hA2 : 0 < Aa ^ 2 := by positivity
  set r := (T + C) ^ 2 * X * Y / Aa ^ 2 with hrdef
  have hr : r * Aa ^ 2 = (T + C) ^ 2 * X * Y := by rw [hrdef]; field_simp
  have hRe' : (C ^ 2 - 1) * (P1 + C) - r * Q1 = σ * (T + C) * (C * P1 + 1) := by
    have : 2 * Aa ^ 2 * ((C ^ 2 - 1) * (P1 + C) - r * Q1 - σ * (T + C) * (C * P1 + 1)) = 0 := by
      linear_combination (T + C) * hBRe - (P1 + C) * E1 - 2 * Q1 * hr
    have h2 : (2 * Aa ^ 2) ≠ 0 := by positivity
    linear_combination (mul_eq_zero.mp this).resolve_left h2
  have hIm' : r * (P1 + C) + (C ^ 2 - 1) * Q1 = σ * (T + C) * (C * Q1) := by
    have : 2 * Aa ^ 2 * (r * (P1 + C) + (C ^ 2 - 1) * Q1 - σ * (T + C) * (C * Q1)) = 0 := by
      linear_combination (T + C) * hBIm - Q1 * E1 + 2 * (P1 + C) * hr
    have h2 : (2 * Aa ^ 2) ≠ 0 := by positivity
    linear_combination (mul_eq_zero.mp this).resolve_left h2
  -- the real estimates
  obtain ⟨k, hk1, hk, hQe, hPe⟩ := dens_root_k hA hC hT0 hXY E1 E2
  have hY2 := dens_root_Y_sq hA hC hT hk1 hk hQe
  have hYp : |Y| ≤ Real.pi / 2 := by
    have := abs_lt_of_sq_lt_sq' hY2 (by positivity); rw [abs_le]; constructor <;> linarith
  rcases le_or_gt Aa 2 with hA2' | hA2'
  · exact dens_root_small hA hA2' hX hY hYp hIm
  rcases hσ with rfl | rfl
  · exact dens_root_Ke hA2' hC hT hX hY hk1 hk hQe hPe (by linear_combination hRe')
      (by linear_combination hIm')
  · have hrQ : 0 < r * Q1 := by
      have h := mul_sinh_sin_pos hX hY (by linarith [Real.pi_pos])
      have : r * Q1 = (T + C) ^ 2 / Aa ^ 2 * (X * Y * Q1) := by rw [hrdef]; ring
      rw [this]; positivity
    exact dens_root_Ae (P1 := P1) hC hS hrQ (by linear_combination hRe')
      (by linear_combination hIm')

end Params

end HubbardAtom
