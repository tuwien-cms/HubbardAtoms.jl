import HubbardAtomLean.DivergenceCriterion

/-!
# No static magnetic zero mode

Item (b) of the static estimates: for `U > 0`, `μ ≠ 0`, `k = 0`, the magnetic channel has no zero
mode, `Φ₀(s) = K_e(s) = 0`. With `α = βa`, `T = e^α`, `C = cosh βμ`, `S = T + C`, `D = T - C`, the
roots of `e` are the atomic `s = a²`, where `P = 0` and so `Φ₀ ≠ 0` (`Phik_ne_zero_of_Pph`), and
`s_b`, with `β²s_bS² = α²(D² - 4)`; and `Φ₀(s_b) > 0`. In `X = βu`,
`Φ₀ = cosh X + C - α sinh X/X`, and the proof has three cases:

* `D > 2` (`s_b > 0`): with `κ = X/α`, `κ²S² = D² - 4`, so `(D - κS)(D + κS) = 4`, whence
  `C(1 + κ) > T(1 - κ) - (1 + κ)/(Tκ)` and `C(1 + κ) ≥ T(1 - κ) - 2`; for `κ ≥ 1/(2α)` the first,
  with `e^{X}(1 + κ) ≤ 2κT` and `2e^X ≤ (1 + κ)T`, and for `κ < 1/(2α)` the second, with
  `e^X ≤ 1 + X + 3X²/4`, give `e^X(1 - κ) - (1 + κ)e^{-X} < 2κC` (`mag_pos_D_gt`);
* `D < -2` (`s_b > 0`): `C(1 - κ) > T(1 + κ)`, and `e^X ≤ T` for `κ ≥ 1/4`, `sinh X ≤ Xe^X` and
  `αe^{α/4} ≤ T` for `κ < 1/4` (`mag_pos_D_lt`);
* `|D| ≤ 2` (`s_b = -y²/β² ≤ 0`): `y ≤ 2α/(T + 1) ≤ 1`, `sin y ≤ y`, `cos y ≥ 1 - y²/2` and
  `C ≥ max(1, T - 2)` (`mag_pos_D_le`).

The margin is small, of order `(U - 1)cosh²μ e^{-U/2}` for large `U`, but the first case is
reduced to the exact relation between `κ`, `T` and `C` with comfortable margins.

* `Params.Phik_mag_ne_zero`: `Φ₀(u) ≠ 0` (real and positive) for `u² = s_b` (`sMag`), `U > 0`;
* `Params.Ke_ne_zero_mag`: **no static magnetic zero mode**, `Φ₀(u) = 0 → K_e(u) ≠ 0`, for
  `U > 0`, `μ ≠ 0`;
* `Params.isUnit_onePlusK_mag`: hence `Γ_m` does not diverge at `ω = 0` (`1 + 𝒦` is invertible),
  if the roots of `Φ₀` are simple (the assumption of `exists_Ke_of_not_isUnit`).
-/

open Complex

namespace HubbardAtom

/-! ### The real estimates -/

/-- `e^x ≤ 1 + x + 3x²/4` for `0 ≤ x ≤ 1`. -/
private theorem exp_le_quad {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    Real.exp x ≤ 1 + x + 3 / 4 * x ^ 2 := by
  have h := Real.exp_bound (x := x) (by rw [abs_of_nonneg h0]; exact h1) (n := 2) (by norm_num)
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, pow_zero, Nat.factorial_zero,
    Nat.cast_one, div_one, zero_add, pow_one, Nat.factorial_one, abs_of_nonneg h0] at h
  norm_num at h
  have := (abs_le.mp h).2
  linarith

/-- The estimates in exponentials: `e^X(1 - κ) - (1 + κ)e^{-X} < 2κC` gives
`α sinh X < X(cosh X + C)` for `X = ακ`. -/
private theorem sinh_lt_of_exp {α κ C X : ℝ} (hα : 0 < α) (hX : X = α * κ)
    (h : Real.exp X * (1 - κ) - (1 + κ) / Real.exp X < 2 * κ * C) :
    α * Real.sinh X < X * (Real.cosh X + C) := by
  obtain ⟨E, hE⟩ : ∃ E, E = Real.exp X := ⟨_, rfl⟩
  rw [← hE] at h
  have hE0 : 0 < E := hE ▸ Real.exp_pos X
  have hinv : Real.exp (-X) = 1 / E := by rw [Real.exp_neg, hE, one_div]
  rw [Real.sinh_eq, Real.cosh_eq, ← hE, hinv, hX]
  have e : α * κ * ((E + 1 / E) / 2 + C) - α * ((E - 1 / E) / 2) =
      α / 2 * (2 * κ * C - (E * (1 - κ) - (1 + κ) / E)) := by ring
  have := mul_pos (half_pos hα) (sub_pos.mpr h)
  linarith

/-- The relation in `κ = X/α`: `κ²(T + C)² = (T - C)² - 4`. -/
private theorem rel_kappa {α C X T : ℝ} (hα : 0 < α)
    (hrel : X ^ 2 * (T + C) ^ 2 = α ^ 2 * ((T - C) ^ 2 - 4)) :
    (X / α) ^ 2 * (T + C) ^ 2 = (T - C) ^ 2 - 4 := by
  have hα2 : α ^ 2 ≠ 0 := pow_ne_zero 2 hα.ne'
  rw [div_pow, div_mul_eq_mul_div, hrel, mul_div_cancel_left₀ _ hα2]

/-- `1 + x + x²/2 + x³/6 ≤ e^x` for `x ≥ 0`. -/
private theorem exp_ge_cubic {x : ℝ} (hx : 0 ≤ x) : 1 + x + x ^ 2 / 2 + x ^ 3 / 6 ≤ Real.exp x := by
  have := Real.sum_le_exp_of_nonneg hx 4
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial, Nat.cast_one, pow_zero,
    pow_one, zero_add, Nat.succ_eq_add_one] at this
  norm_num at this
  linarith

/-- The polynomial step of the case `D > 2`, `κ < 1/(2α)`: for `α ≥ 42/25`,
`(19α/16 - 1)(2α + 1) + 4α ≤ (1 + α + α²/2 + α³/6)(2α - 1)`. -/
private theorem poly_mag {α : ℝ} (h : 42 / 25 ≤ α) :
    (19 / 16 * α - 1) * (2 * α + 1) + 4 * α ≤ (1 + α + α ^ 2 / 2 + α ^ 3 / 6) * (2 * α - 1) := by
  have ht : 0 ≤ α - 42 / 25 := by linarith
  have hq : 0 ≤ α ^ 3 / 3 + 5 / 6 * α ^ 2 - 7 / 8 * α - 35 / 16 := by
    nlinarith [mul_nonneg ht ht, mul_nonneg (mul_nonneg ht ht) ht]
  have e : (1 + α + α ^ 2 / 2 + α ^ 3 / 6) * (2 * α - 1) -
      ((19 / 16 * α - 1) * (2 * α + 1) + 4 * α) =
      α * (α ^ 3 / 3 + 5 / 6 * α ^ 2 - 7 / 8 * α - 35 / 16) := by ring
  have := mul_nonneg (by linarith : (0 : ℝ) ≤ α) hq
  linarith

/-- The case `D > 2`, `κ ≥ 1/(2α)`. -/
private theorem key_D_gt_large {α κ T C E : ℝ}
    (hα1 : 1 < α) (hκ0 : 0 < κ) (hκ1 : κ < 1) (hT : T = Real.exp α)
    (hE : E = Real.exp (α * κ)) (hS : 0 < T + C)
    (h4 : (T - C - κ * (T + C)) * (T - C + κ * (T + C)) = 4)
    (hpos : 0 < T - C - κ * (T + C)) (h2ακ : 1 ≤ 2 * α * κ) :
    E * (1 - κ) - (1 + κ) / E < 2 * κ * C := by
  have hα : 0 < α := by linarith
  have hT0 : 0 < T := hT ▸ Real.exp_pos α
  have hE0 : 0 < E := hE ▸ Real.exp_pos _
  -- `C(1 + κ) > T(1 - κ) - (1 + κ)/(Tκ)`
  have hid : (1 + κ) * (T - C + κ * (T + C)) - 4 * T * κ =
      (1 - κ) * (T - C - κ * (T + C)) := by ring
  have hstr : 4 * T * κ < (1 + κ) * (T - C + κ * (T + C)) := by
    have := mul_pos (sub_pos.mpr hκ1) hpos
    linarith
  have hTκ : 0 < T * κ := mul_pos hT0 hκ0
  have hδ : T - C - κ * (T + C) < (1 + κ) / (T * κ) := by
    rw [lt_div_iff₀ hTκ]
    have hq : 0 < T - C + κ * (T + C) := by have := mul_pos hκ0 hS; linarith
    have e : (T - C - κ * (T + C)) * (T * κ) * (T - C + κ * (T + C)) = 4 * T * κ := by
      linear_combination (T * κ) * h4
    have : (T - C - κ * (T + C)) * (T * κ) * (T - C + κ * (T + C)) <
        (1 + κ) * (T - C + κ * (T + C)) := by rw [e]; exact hstr
    exact lt_of_mul_lt_mul_right this hq.le
  -- `e^X(1 + κ) ≤ 2κT` and `2e^X ≤ (1 + κ)T`
  have hTE : T = E * Real.exp (α * (1 - κ)) := by
    rw [hE, ← Real.exp_add, hT]; ring_nf
  have he1 := Real.add_one_le_exp (α * (1 - κ))
  have hq1 : 0 ≤ (1 - κ) * (2 * α * κ - 1) := mul_nonneg (by linarith) (by linarith)
  have hq2 : 0 ≤ (1 - κ) * (α * (1 + κ) - 1) := mul_nonneg (by linarith) (by nlinarith)
  have hi : E * (1 + κ) ≤ 2 * κ * T := by
    have : 1 + κ ≤ 2 * κ * Real.exp (α * (1 - κ)) :=
      calc 1 + κ ≤ 2 * κ * (α * (1 - κ) + 1) := by linarith
        _ ≤ 2 * κ * Real.exp (α * (1 - κ)) :=
          mul_le_mul_of_nonneg_left he1 (by linarith)
    have h := mul_le_mul_of_nonneg_left this hE0.le
    rw [hTE]; linarith
  have hii : 2 * E ≤ (1 + κ) * T := by
    have : 2 ≤ (1 + κ) * Real.exp (α * (1 - κ)) :=
      calc (2 : ℝ) ≤ (1 + κ) * (α * (1 - κ) + 1) := by linarith
        _ ≤ (1 + κ) * Real.exp (α * (1 - κ)) :=
          mul_le_mul_of_nonneg_left he1 (by linarith)
    have h := mul_le_mul_of_nonneg_left this hE0.le
    rw [hTE]; linarith
  have f1 : E * (1 - κ) ≤ 2 * κ * T * (1 - κ) / (1 + κ) := by
    rw [le_div_iff₀ (by linarith)]
    have := mul_le_mul_of_nonneg_right hi (by linarith : (0 : ℝ) ≤ 1 - κ)
    linarith
  have f2 : 2 / T ≤ (1 + κ) / E := by
    rw [div_le_div_iff₀ hT0 hE0]; linarith
  have f3 : 2 * κ * T * (1 - κ) / (1 + κ) - 2 / T < 2 * κ * C := by
    have e : 2 * κ * T * (1 - κ) / (1 + κ) - 2 / T =
        2 * κ * (T * (1 - κ) - (1 + κ) / (T * κ)) / (1 + κ) := by
      field_simp
    rw [e, div_lt_iff₀ (by linarith)]
    nlinarith
  linarith

/-- The case `D > 2`, `κ < 1/(2α)`. -/
private theorem key_D_gt_small {α κ T C E : ℝ}
    (hα1 : 1 < α) (hκ0 : 0 < κ) (hκ1 : κ < 1) (hT : T = Real.exp α)
    (hE : E = Real.exp (α * κ)) (hS : 0 < T + C)
    (h4 : (T - C - κ * (T + C)) * (T - C + κ * (T + C)) = 4)
    (hpos : 0 < T - C - κ * (T + C)) (hC : 1 ≤ C) (hκα' : 2 * α * κ < 1)
    (hD : 2 < T - C) :
    E * (1 - κ) - (1 + κ) / E < 2 * κ * C := by
  have hα : 0 < α := by linarith
  have hT0 : 0 < T := hT ▸ Real.exp_pos α
  have hE0 : 0 < E := hE ▸ Real.exp_pos _
  obtain ⟨X, hX'⟩ : ∃ X, X = α * κ := ⟨_, rfl⟩
  rw [← hX'] at hE
  have hX : 0 < X := hX' ▸ mul_pos hα hκ0
  have hX1 : X ≤ 1 := by rw [hX']; linarith
  have hEq := exp_le_quad hX.le hX1
  rw [← hE] at hEq
  have hEi : 1 - X ≤ 1 / E := by
    have h := Real.add_one_le_exp (-X)
    rw [Real.exp_neg, ← hE, ← one_div] at h; linarith
  have hup : E * (1 - κ) - (1 + κ) / E ≤ 2 * κ * (α - 1) + 3 / 4 * α ^ 2 * κ ^ 2 := by
    have h1 : E * (1 - κ) ≤ (1 + X + 3 / 4 * X ^ 2) * (1 - κ) :=
      mul_le_mul_of_nonneg_right hEq (by linarith)
    have h2 : (1 + κ) * (1 - X) ≤ (1 + κ) / E := by
      rw [div_eq_mul_one_div]; exact mul_le_mul_of_nonneg_left hEi (by linarith)
    have e : (1 + X + 3 / 4 * X ^ 2) * (1 - κ) - (1 + κ) * (1 - X) =
        2 * κ * (α - 1) + 3 / 4 * α ^ 2 * κ ^ 2 - 3 / 4 * α ^ 2 * κ ^ 3 := by rw [hX']; ring
    have : 0 ≤ α ^ 2 * κ ^ 3 := by positivity
    linarith
  -- `α - 1 + 3α²κ/8 < 19α/16 - 1 < C`
  have hb : 2 * κ * (α - 1) + 3 / 4 * α ^ 2 * κ ^ 2 < 2 * κ * (19 / 16 * α - 1) := by
    have : α ^ 2 * κ ^ 2 < α * κ / 2 := by nlinarith
    nlinarith
  suffices hfin : 19 / 16 * α - 1 ≤ C by nlinarith
  by_cases hα2 : α ≤ (42 / 25 : ℝ)
  · linarith
  · push Not at hα2
    -- `C(1 + κ) ≥ T(1 - κ) - 2`
    have hC2 : T * (1 - κ) - 2 ≤ C * (1 + κ) := by
      have hq : 2 < T - C + κ * (T + C) := by have := mul_pos hκ0 hS; linarith
      have : (T - C - κ * (T + C)) * 2 < (T - C - κ * (T + C)) * (T - C + κ * (T + C)) :=
        mul_lt_mul_of_pos_left hq hpos
      rw [h4] at this
      linarith
    have hT4 : 1 + α + α ^ 2 / 2 + α ^ 3 / 6 ≤ T := hT ▸ exp_ge_cubic hα.le
    have hpoly := poly_mag hα2.le
    -- monotonicity in `κ < 1/(2α)`, multiplied by `2α`
    have hc0 : 0 < 19 / 16 * α - 1 := by linarith
    have hk1 : (1 + κ) * (2 * α) < 2 * α + 1 := by
      have e : (1 + κ) * (2 * α) = 2 * α + 2 * α * κ := by ring
      rw [e]; linarith
    have hk2 : 2 * α - 1 ≤ (1 - κ) * (2 * α) := by
      have e : (1 - κ) * (2 * α) = 2 * α - 2 * α * κ := by ring
      rw [e]; linarith
    have h1 : (19 / 16 * α - 1) * ((1 + κ) * (2 * α)) <
        (19 / 16 * α - 1) * (2 * α + 1) := mul_lt_mul_of_pos_left hk1 hc0
    have h2 : T * (2 * α - 1) ≤ T * ((1 - κ) * (2 * α)) :=
      mul_le_mul_of_nonneg_left hk2 hT0.le
    have h5 : (1 + α + α ^ 2 / 2 + α ^ 3 / 6) * (2 * α - 1) ≤ T * (2 * α - 1) :=
      mul_le_mul_of_nonneg_right hT4 (by linarith)
    have h3 : ((19 / 16 * α - 1) * (1 + κ)) * (2 * α) < (T * (1 - κ) - 2) * (2 * α) := by
      linarith
    have h6 : (19 / 16 * α - 1) * (1 + κ) < C * (1 + κ) :=
      lt_of_lt_of_le (lt_of_mul_lt_mul_right h3 (by linarith)) hC2
    exact (lt_of_mul_lt_mul_right h6 (by linarith)).le


/-- **Case `D > 2`**: for `α > 0`, `C ≥ 1`, `X > 0` with `X²(T + C)² = α²((T - C)² - 4)` and
`T - C > 2` (`T = e^α`), `α sinh X < X(cosh X + C)`. -/
theorem mag_pos_D_gt {α C X : ℝ} (hα : 0 < α) (hC : 1 ≤ C) (hX : 0 < X)
    (hrel : X ^ 2 * (Real.exp α + C) ^ 2 = α ^ 2 * ((Real.exp α - C) ^ 2 - 4))
    (hD : 2 < Real.exp α - C) : α * Real.sinh X < X * (Real.cosh X + C) := by
  obtain ⟨T, hT⟩ : ∃ T, T = Real.exp α := ⟨_, rfl⟩
  rw [← hT] at hrel hD
  have hT0 : 0 < T := hT ▸ Real.exp_pos α
  obtain ⟨κ, hκ⟩ : ∃ κ, κ = X / α := ⟨_, rfl⟩
  have hX' : X = α * κ := by rw [hκ]; field_simp
  have hκ0 : 0 < κ := hκ ▸ div_pos hX hα
  have hS : 0 < T + C := by linarith
  have hrel' : κ ^ 2 * (T + C) ^ 2 = (T - C) ^ 2 - 4 := hκ ▸ rel_kappa hα hrel
  have h4 : (T - C - κ * (T + C)) * (T - C + κ * (T + C)) = 4 := by linear_combination -hrel'
  have hκS : κ * (T + C) < T - C := by
    have h1 : (κ * (T + C)) ^ 2 < (T - C) ^ 2 := by nlinarith
    exact (pow_lt_pow_iff_left₀ (by positivity) (by linarith) two_ne_zero).mp h1
  have hκ1 : κ < 1 := (mul_lt_iff_lt_one_left hS).mp (by linarith)
  have hα1 : 1 < α := by
    by_contra h
    have : T ≤ Real.exp 1 := hT ▸ Real.exp_le_exp.mpr (not_lt.mp h)
    have := Real.exp_one_lt_d9
    linarith
  apply sinh_lt_of_exp hα hX'
  obtain ⟨E, hE⟩ : ∃ E, E = Real.exp X := ⟨_, rfl⟩
  rw [← hE]
  have hE0 : 0 < E := hE ▸ Real.exp_pos X
  have hpos : 0 < T - C - κ * (T + C) := by linarith
  by_cases hκα : 1 / (2 * α) ≤ κ
  · have h2ακ : 1 ≤ 2 * α * κ := by
      rw [div_le_iff₀ (by linarith)] at hκα; linarith
    exact key_D_gt_large hα1 hκ0 hκ1 hT (by rw [hE, hX']) hS h4 hpos h2ακ
  · have hκα' : 2 * α * κ < 1 := by
      push Not at hκα
      rw [lt_div_iff₀ (by linarith)] at hκα; linarith
    exact key_D_gt_small hα1 hκ0 hκ1 hT (by rw [hE, hX']) hS h4 hpos hC hκα' hD

/-- **Case `D < -2`**: for `α > 0`, `X > 0` with `X²(T + C)² = α²((T - C)² - 4)` and
`C - T > 2`, `α sinh X < X(cosh X + C)`. -/
theorem mag_pos_D_lt {α C X : ℝ} (hα : 0 < α) (hX : 0 < X)
    (hrel : X ^ 2 * (Real.exp α + C) ^ 2 = α ^ 2 * ((Real.exp α - C) ^ 2 - 4))
    (hD : 2 < C - Real.exp α) : α * Real.sinh X < X * (Real.cosh X + C) := by
  obtain ⟨T, hT⟩ : ∃ T, T = Real.exp α := ⟨_, rfl⟩
  rw [← hT] at hrel hD
  have hT0 : 0 < T := hT ▸ Real.exp_pos α
  obtain ⟨κ, hκ⟩ : ∃ κ, κ = X / α := ⟨_, rfl⟩
  have hX' : X = α * κ := by rw [hκ]; field_simp
  have hκ0 : 0 < κ := hκ ▸ div_pos hX hα
  have hS : 0 < T + C := by linarith
  have hrel' : κ ^ 2 * (T + C) ^ 2 = (T - C) ^ 2 - 4 := hκ ▸ rel_kappa hα hrel
  have hκS : κ * (T + C) < C - T := by
    have h1 : (κ * (T + C)) ^ 2 < (C - T) ^ 2 := by nlinarith
    exact (pow_lt_pow_iff_left₀ (by positivity) (by linarith) two_ne_zero).mp h1
  have hκ1 : κ < 1 := (mul_lt_iff_lt_one_left hS).mp (by linarith)
  obtain ⟨E, hE⟩ : ∃ E, E = Real.exp X := ⟨_, rfl⟩
  have hE0 : 0 < E := hE ▸ Real.exp_pos X
  by_cases hκ4 : 1 / 4 ≤ κ
  · apply sinh_lt_of_exp hα hX'
    rw [← hE]
    have hET : E ≤ T := by
      rw [hE, hT]; exact Real.exp_le_exp.mpr (by rw [hX']; nlinarith)
    have hC : T * (1 + κ) < C * (1 - κ) := by linarith
    have hq : (1 - κ) ^ 2 ≤ 2 * κ * (1 + κ) := by nlinarith
    have h1 : E * (1 - κ) ≤ T * (1 - κ) := mul_le_mul_of_nonneg_right hET (by linarith)
    have h2 : T * (1 - κ) < 2 * κ * C := by
      have : T * (1 - κ) ^ 2 < 2 * κ * C * (1 - κ) := by nlinarith
      have h3 : 0 < 1 - κ := by linarith
      nlinarith
    have : 0 < (1 + κ) / E := div_pos (by linarith) hE0
    linarith
  · push Not at hκ4
    have hsh : Real.sinh X ≤ X * E := by
      rw [Real.sinh_eq, ← hE]
      have h1 := Real.add_one_le_exp (-(2 * X))
      have h2 : Real.exp (-X) = Real.exp (-(2 * X)) * E := by
        rw [hE, ← Real.exp_add]; ring_nf
      rw [h2]
      nlinarith
    have hEα : α * E ≤ T := by
      have h1 : E ≤ Real.exp (α / 4) := by
        rw [hE]; exact Real.exp_le_exp.mpr (by rw [hX']; nlinarith)
      have h2 : T = Real.exp (α / 4) * Real.exp (3 * α / 4) := by
        rw [hT, ← Real.exp_add]; ring_nf
      have h3 := Real.quadratic_le_exp_of_nonneg (by linarith : 0 ≤ 3 * α / 4)
      have h4 : α ≤ Real.exp (3 * α / 4) := by nlinarith
      rw [h2]
      have := Real.exp_pos (α / 4)
      nlinarith
    have hcosh : 0 < Real.cosh X := Real.cosh_pos X
    have h1 : α * Real.sinh X ≤ X * (α * E) := by nlinarith
    have h2 : X * (α * E) < X * C := mul_lt_mul_of_pos_left (by linarith) hX
    nlinarith

/-- **Case `|D| ≤ 2`**: for `α > 0`, `C ≥ 1`, `y ≥ 0` with `y²(T + C)² = α²(4 - (T - C)²)` and
`|T - C| ≤ 2`, `α sin y < y(cos y + C)` for `y > 0`, and `α < 1 + C`. -/
theorem mag_pos_D_le {α C y : ℝ} (hα : 0 < α) (hC : 1 ≤ C) (hy : 0 ≤ y)
    (hrel : y ^ 2 * (Real.exp α + C) ^ 2 = α ^ 2 * (4 - (Real.exp α - C) ^ 2))
    (hD : |Real.exp α - C| ≤ 2) :
    α < 1 / 2 + C ∧ y ≤ 1 ∧ (0 < y → α * Real.sin y < y * (Real.cos y + C)) := by
  obtain ⟨T, hT⟩ : ∃ T, T = Real.exp α := ⟨_, rfl⟩
  rw [← hT] at hrel hD
  have hT1 := Real.quadratic_le_exp_of_nonneg hα.le
  rw [← hT] at hT1
  have hαC : α < 1 / 2 + C := by
    rcases le_or_gt α 1 with h | h
    · linarith
    · have : T - 2 ≤ C := by have := (abs_le.mp hD).2; linarith
      nlinarith
  -- `y ≤ 2α/(T + C) ≤ 2α/(T + 1) ≤ 1`
  have hy1 : y ≤ 1 := by
    have hS : 0 < T + C := by have := Real.exp_pos α; rw [← hT] at this; linarith
    have h1 : (y * (T + C)) ^ 2 ≤ (2 * α) ^ 2 := by
      have : 0 ≤ (T - C) ^ 2 := sq_nonneg _
      nlinarith
    have h2 : y * (T + C) ≤ 2 * α :=
      (pow_le_pow_iff_left₀ (mul_nonneg hy hS.le) (by linarith) two_ne_zero).mp h1
    by_contra h
    push Not at h
    have : T + C < y * (T + C) := by nlinarith
    nlinarith
  refine ⟨hαC, hy1, fun hy0 ↦ ?_⟩
  have hs := Real.sin_le hy
  have hc := Real.one_sub_sq_div_two_le_cos (x := y)
  have : α * Real.sin y ≤ α * y := mul_le_mul_of_nonneg_left hs hα.le
  have : y * (1 / 2 + C) ≤ y * (Real.cos y + C) := by
    apply mul_le_mul_of_nonneg_left _ hy
    nlinarith
  nlinarith

namespace Params

variable (A : Params)

/-! ### The magnetic static channel -/

/-- The weights through `T = e^{βa}`, `C = cosh βμ`: `p₁ = T/(2(C + T))`,
`p₀p₂ = 1/(4(C + T)²)`. -/
theorem p₁_p₀p₂_TC :
    A.p₁ = Real.exp (A.β * A.a) / (2 * (Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a))) ∧
      A.p₀ * A.p₂ =
        1 / (4 * (Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a)) ^ 2) := by
  have h₁ : A.p₁ = A.p₀ * (Real.exp (A.β * A.μ) * Real.exp (A.β * A.a)) := by
    rw [A.p₁_eq_p₀_mul, ← Real.exp_add]; ring_nf
  have h₂ : A.p₂ = A.p₀ * Real.exp (A.β * A.μ) ^ 2 := by
    rw [A.p₂_eq_p₀_mul, ← Real.exp_nat_mul]; push_cast; ring_nf
  have hs := A.weights_sum
  rw [h₁, h₂] at hs
  have hx := Real.exp_pos (A.β * A.μ)
  have hy := Real.exp_pos (A.β * A.a)
  have hxi : Real.exp (A.β * A.μ) * Real.exp (-(A.β * A.μ)) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hC : Real.cosh (A.β * A.μ) = (Real.exp (A.β * A.μ) + Real.exp (-(A.β * A.μ))) / 2 :=
    Real.cosh_eq _
  -- `p₀ = 1/(2x(C + T))`
  have hp₀ : A.p₀ = 1 / (2 * Real.exp (A.β * A.μ) *
      (Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a))) := by
    have hpos : 0 < Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a) := by
      have := Real.cosh_pos (A.β * A.μ); linarith
    rw [eq_div_iff (by positivity), hC]
    linear_combination hs + A.p₀ * hxi
  have hpos : 0 < Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a) := by
    have := Real.cosh_pos (A.β * A.μ); linarith
  refine ⟨?_, ?_⟩
  · rw [h₁, hp₀]; field_simp
  · rw [h₂, hp₀]; field_simp; ring

/-- The non-atomic root of `e` in the static magnetic channel:
`s_b = a²[(1 - 4p₁)² - 16p₀p₂]`. -/
noncomputable def sMag : ℝ := A.a ^ 2 * ((1 - 4 * A.p₁) ^ 2 - 16 * (A.p₀ * A.p₂))

/-- `e(z) = (z² - a²)(z² - s_b)` for `r = m`, `k = 0`. -/
theorem eph_mag (z : ℂ) : A.eph 0 (-1) z = (z ^ 2 - A.a ^ 2) * (z ^ 2 - A.sMag) := by
  have hs : (A.p₀ : ℂ) + 2 * A.p₁ + A.p₂ = 1 := by exact_mod_cast A.weights_sum
  simp only [eph, Hph, lam, q₀, A.w_zero, sMag, ρ]
  push_cast
  rw [show (A.p₂ : ℂ) = 1 - A.p₀ - 2 * A.p₁ by linear_combination hs]
  ring

/-- `β²s_b(T + C)² = α²((T - C)² - 4)` with `α = βa`, `T = e^α`, `C = cosh βμ`. -/
theorem sMag_rel : A.β ^ 2 * A.sMag * (Real.exp (A.β * A.a) + Real.cosh (A.β * A.μ)) ^ 2 =
    (A.β * A.a) ^ 2 * ((Real.exp (A.β * A.a) - Real.cosh (A.β * A.μ)) ^ 2 - 4) := by
  obtain ⟨h₁, h₂⟩ := A.p₁_p₀p₂_TC
  have hpos : 0 < Real.cosh (A.β * A.μ) + Real.exp (A.β * A.a) := by
    have := Real.cosh_pos (A.β * A.μ); have := Real.exp_pos (A.β * A.a); linarith
  rw [sMag, h₁, h₂]
  field_simp
  ring

/-- **`Φ₀(s_b) > 0`**: for `U > 0` and `u² = s_b`, `Φ₀(u)` is real and positive in the magnetic
channel. -/
theorem Phik_mag_ne_zero (hU : 0 < A.a) {u : ℂ} (hu : u ^ 2 = A.sMag) :
    A.Phik 0 (-1) u ≠ 0 := by
  set α := A.β * A.a with hα
  set T := Real.exp α
  set C := Real.cosh (A.β * A.μ) with hCdef
  have hβ := A.β_pos
  have hα0 : 0 < α := mul_pos hβ hU
  have hC1 : 1 ≤ C := Real.one_le_cosh _
  have hrel := A.sMag_rel
  rw [← hα] at hrel
  have hβc : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr hβ.ne'
  -- `Φ₀(u) = cosh βu + C - a shc(β, u)`
  have hΦ : ∀ v, A.Phik 0 (-1) v = cosh (A.β * v) + C - A.a * shc A.β v := fun v ↦ by
    simp only [Phik, Dk, τk, even_zero', ↓reduceIte, one_mul, hCdef]
    push_cast; ring
  rcases lt_or_ge 0 A.sMag with hs | hs
  · -- `u = ±√s_b`
    set r := Real.sqrt A.sMag
    have hr : 0 < r := Real.sqrt_pos.mpr hs
    have hu2 : u ^ 2 = (r : ℂ) ^ 2 := by
      rw [hu, ← ofReal_pow, Real.sq_sqrt hs.le]
    have hΦr : A.Phik 0 (-1) u = A.Phik 0 (-1) (r : ℂ) := by
      rcases sq_eq_sq_iff_eq_or_eq_neg.mp hu2 with h | h
      · rw [h]
      · rw [h, A.Phik_neg]
    rw [hΦr, hΦ, shc_of_ne (ofReal_ne_zero.mpr hr.ne')]
    set X := A.β * r
    have hX : 0 < X := mul_pos hβ hr
    have hXrel : X ^ 2 * (T + C) ^ 2 = α ^ 2 * ((T - C) ^ 2 - 4) := by
      rw [← hrel]; simp only [X, mul_pow, r, Real.sq_sqrt hs.le]; ring
    have hD : 4 < (T - C) ^ 2 := by
      have : 0 < X ^ 2 * (T + C) ^ 2 := by
        have : 0 < T + C := by have := Real.exp_pos α; linarith
        positivity
      nlinarith
    have key : α * Real.sinh X < X * (Real.cosh X + C) := by
      rcases lt_or_ge 2 (T - C) with h | h
      · exact mag_pos_D_gt hα0 hC1 hX hXrel h
      · have : 2 < C - T := by nlinarith
        exact mag_pos_D_lt hα0 hX hXrel this
    have e : cosh (↑A.β * ↑r) + ↑C - ↑A.a * (sinh (↑A.β * ↑r) / ↑r) =
        ((Real.cosh X + C - α * Real.sinh X / X : ℝ) : ℂ) := by
      simp only [X, α]
      push_cast
      field_simp
    rw [e, ofReal_ne_zero]
    have : α * Real.sinh X / X < Real.cosh X + C := by rw [div_lt_iff₀ hX]; linarith
    linarith
  · -- `u = ±i√(-s_b)`
    set r := Real.sqrt (-A.sMag)
    have hr : 0 ≤ r := Real.sqrt_nonneg _
    have hu2 : u ^ 2 = (r * I : ℂ) ^ 2 := by
      rw [hu, mul_pow, I_sq, ← ofReal_pow, Real.sq_sqrt (by linarith)]; push_cast; ring
    have hΦr : A.Phik 0 (-1) u = A.Phik 0 (-1) (r * I) := by
      rcases sq_eq_sq_iff_eq_or_eq_neg.mp hu2 with h | h
      · rw [h]
      · rw [h, A.Phik_neg]
    rw [hΦr]
    set y := A.β * r
    have hy : 0 ≤ y := mul_nonneg hβ.le hr
    have hyrel : y ^ 2 * (T + C) ^ 2 = α ^ 2 * (4 - (T - C) ^ 2) := by
      have : y ^ 2 = -(A.β ^ 2 * A.sMag) := by
        simp only [y, mul_pow, r, Real.sq_sqrt (by linarith : 0 ≤ -A.sMag)]; ring
      rw [this]; linarith
    have hDle : |T - C| ≤ 2 := by
      have : 0 ≤ y ^ 2 * (T + C) ^ 2 := by positivity
      have hα2 : 0 < α ^ 2 := pow_pos hα0 2
      have h4 : (T - C) ^ 2 ≤ 4 := by
        by_contra h
        have := mul_neg_of_pos_of_neg hα2 (by linarith : 4 - (T - C) ^ 2 < 0)
        linarith
      exact (pow_le_pow_iff_left₀ (abs_nonneg _) (by norm_num) two_ne_zero).mp
        (by rw [sq_abs]; linarith)
    obtain ⟨hαC, -, hsin⟩ := mag_pos_D_le hα0 hC1 hy hyrel hDle
    rcases eq_or_lt_of_le hy with hy0 | hy0
    · -- `u = 0`: `Φ₀(0) = 1 + C - α`
      have hr0 : r = 0 := by
        rcases mul_eq_zero.mp hy0.symm with h | h
        · exact absurd h hβ.ne'
        · exact h
      rw [hr0, ofReal_zero, zero_mul, A.Phik_zero, Dk, mul_zero, cosh_zero, τk,
        ite_eq_left even_zero']
      have e : (1 : ℂ) + 1 * cosh (↑A.β * ↑A.μ) + -1 * ↑A.a * ↑A.β = ((1 + C - α : ℝ) : ℂ) := by
        simp only [C, α]; push_cast; ring
      rw [e, ofReal_ne_zero]
      linarith
    · have hr0 : r ≠ 0 := fun h ↦ by simp [y, h] at hy0
      rw [hΦ, shc_of_ne (mul_ne_zero (ofReal_ne_zero.mpr hr0) I_ne_zero)]
      have e : cosh (↑A.β * (↑r * I)) + ↑C - ↑A.a * (sinh (↑A.β * (↑r * I)) / (↑r * I)) =
          ((Real.cos y + C - α * Real.sin y / y : ℝ) : ℂ) := by
        have h1 : (A.β : ℂ) * (r * I) = (y : ℂ) * I := by simp only [y]; push_cast; ring
        rw [h1, cosh_mul_I, sinh_mul_I]
        simp only [y, α]
        push_cast
        field_simp
      rw [e, ofReal_ne_zero]
      have h := hsin hy0
      have : α * Real.sin y / y < Real.cos y + C := by rw [div_lt_iff₀ hy0]; linarith
      linarith

/-- **No static magnetic zero mode**: for `U > 0` and `μ ≠ 0`, no `u` has `Φ₀(u) = K_e(u) = 0` in
the magnetic channel. At such a `u`, `K_eA_e = β²e(u) = 0`, so `u² = a²`, where `P(u) = 0`
and `Φ₀ ≠ 0`, or `u² = s_b`, where `Φ₀ > 0`. -/
theorem Ke_ne_zero_mag (hU : 0 < A.U) (hμ : A.μ ≠ 0) {u : ℂ} (hΦ : A.Phik 0 (-1) u = 0) :
    A.Ke 0 (-1) u ≠ 0 := by
  have ha : 0 < A.a := by rw [a]; linarith
  have ha' : (A.a : ℂ) ≠ 0 := ofReal_ne_zero.mpr ha.ne'
  have hε : (-1 : ℂ) = 1 ∨ (-1 : ℂ) = -1 := Or.inr rfl
  have hu : ∀ m, A.z 0 m ≠ u := fun m h ↦
    A.Phik_z_ne_zero 0 (-1) (by norm_num) ha' hμ m (h ▸ hΦ)
  have hu' : ∀ m, A.z 0 m ≠ -u := fun m h ↦
    A.Phik_z_ne_zero 0 (-1) (by norm_num) ha' hμ m (by rw [h, A.Phik_neg, hΦ])
  have hD := A.Dk_ne_zero_of_off 0 hu hu'
  intro hK
  have he := A.Ke_mul_Ae_of_Phi 0 (-1) hD hΦ
  rw [hK, zero_mul, eq_comm, mul_eq_zero, A.eph_mag] at he
  have hβ2 : (A.β : ℂ) ^ 2 ≠ 0 := pow_ne_zero 2 (ofReal_ne_zero.mpr A.β_pos.ne')
  rcases he with h | h
  · exact hβ2 h
  rcases mul_eq_zero.mp h with h | h
  · -- `u² = a²`: `P(u) = (u² - a²)² = 0`
    have hP : A.Pph 0 u = 0 := by
      simp only [Pph, Qp, Qm, A.w_zero]
      linear_combination (u ^ 2 - A.a ^ 2) * h
    exact A.Phik_ne_zero_of_Pph 0 (-1) hε ha' hP hΦ
  · exact A.Phik_mag_ne_zero ha (by linear_combination h) hΦ

/-- **`Γ_m` does not diverge at `ω = 0`** for `U > 0`, `μ ≠ 0`, if the roots of `Φ₀` are
simple: `1 + 𝒦` is invertible, by the divergence criterion. -/
theorem isUnit_onePlusK_mag (hU : 0 < A.U) (hμ : A.μ ≠ 0)
    (hsimple : ∀ ρ, A.Phik 0 (-1) ρ = 0 →
      analyticOrderAt (A.Phik 0 (-1)) ρ ≤ if ρ = 0 then 2 else 1) :
    IsUnit (A.onePlusK 0 (-1) hμ) := by
  have ha : (A.a : ℂ) ≠ 0 := ofReal_ne_zero.mpr (by rw [a]; linarith)
  by_contra h
  obtain ⟨u, hΦ, hK⟩ := A.exists_Ke_of_not_isUnit 0 (-1) (Or.inr rfl) ha hμ hsimple h
  exact A.Ke_ne_zero_mag hU hμ hΦ hK

end Params

end HubbardAtom
