import HubbardAtomLean.DivergenceAlgebra

/-!
# No divergence at complex zeros of `Φ_k`

For the density and magnetic channels, at a nonreal `s = u²` with `Φ_k(s) = 0` (and `D_k ≠ 0`):
`Im s · Im K_e(s) > 0` for `U ≥ 0` (`Params.im_mul_im_Ke_pos`). So `K_e(s) ≠ 0`
(`Params.Ke_ne_zero_of_Phi`), and such a zero of `Φ_k` is not a divergence, which needs
`K_e = 0` (`Ke_eq_zero_of_Phi`). With `u = x + iy` the sign comes from
`x²(y sin 2βy) ≤ y²(x sinh 2βx)` (`mul_sin_le_sq`, `sq_le_mul_sinh`).
-/

open Complex

namespace HubbardAtom

/-- `t sin t ≤ t²`. -/
theorem mul_sin_le_sq (t : ℝ) : t * Real.sin t ≤ t ^ 2 := by
  rcases le_or_gt 0 t with ht | ht
  · nlinarith [Real.sin_le ht]
  · nlinarith [Real.le_sin ht.le]

/-- `t² ≤ t sinh t`. -/
theorem sq_le_mul_sinh (t : ℝ) : t ^ 2 ≤ t * Real.sinh t := by
  rcases le_or_gt 0 t with ht | ht
  · nlinarith [Real.self_le_sinh_iff.mpr ht]
  · nlinarith [Real.sinh_le_self_iff.mpr ht.le]

/-- `xy(x sin 2βy - y sinh 2βx) ≤ 0` for `β > 0`. -/
theorem xy_mul_sin_sub_sinh_nonpos {β : ℝ} (hβ : 0 < β) (x y : ℝ) :
    x * y * (x * Real.sin (2 * β * y) - y * Real.sinh (2 * β * x)) ≤ 0 := by
  have h₁ := mul_le_mul_of_nonneg_left (mul_sin_le_sq (2 * β * y)) (sq_nonneg x)
  have h₂ := mul_le_mul_of_nonneg_left (sq_le_mul_sinh (2 * β * x)) (sq_nonneg y)
  have key : 2 * β * (x * y * (x * Real.sin (2 * β * y) - y * Real.sinh (2 * β * x))) ≤ 0 := by
    nlinarith
  nlinarith

/-- `sinh(βu) conj(cosh(βu)) = (sinh(2βx) + i sin(2βy))/2` for `u = x + iy`, real `β`. -/
theorem sinh_mul_conj_cosh (β : ℝ) (u : ℂ) :
    sinh (β * u) * (starRingEnd ℂ) (cosh (β * u)) =
      ((Real.sinh (2 * β * u.re) / 2 : ℝ) : ℂ) + ((Real.sin (2 * β * u.im) / 2 : ℝ) : ℂ) * I := by
  rw [← Complex.cosh_conj, map_mul, conj_ofReal]
  have e₁ : (β : ℂ) * u + β * (starRingEnd ℂ) u = ((2 * β * u.re : ℝ) : ℂ) := by
    rw [← mul_add, add_conj]
    push_cast
    ring
  have e₂ : (β : ℂ) * u - β * (starRingEnd ℂ) u = ((2 * β * u.im : ℝ) : ℂ) * I := by
    rw [← mul_sub, sub_conj]
    push_cast
    ring
  have h : sinh (β * u) * cosh (β * (starRingEnd ℂ) u) =
      (sinh (β * u + β * (starRingEnd ℂ) u) + sinh (β * u - β * (starRingEnd ℂ) u)) / 2 := by
    rw [Complex.sinh_add, Complex.sinh_sub]
    ring
  rw [h, e₁, e₂, Complex.sinh_mul_I, ← ofReal_sinh, ← ofReal_sin]
  push_cast
  ring

namespace Params

variable (A : Params) (k : ℤ) (ε : ℂ)

/-- **No divergence at complex zeros of `Φ_k`**: for `U ≥ 0`, at a nonreal `s = u²` with
`Φ_k(s) = 0` and `D_k(s) ≠ 0`, `Im s · Im K_e(s) > 0`. -/
theorem im_mul_im_Ke_pos (hε : ε = 1 ∨ ε = -1) (ha : 0 ≤ A.a) {u : ℂ} (hs : (u ^ 2).im ≠ 0)
    (hD : A.Dk k u ≠ 0) (hΦ : A.Phik k ε u = 0) : 0 < (u ^ 2).im * (A.Ke k ε u).im := by
  obtain ⟨e, rfl, he⟩ : ∃ e : ℝ, ε = e ∧ e ^ 2 = 1 := by
    rcases hε with rfl | rfl
    · exact ⟨1, by simp, by norm_num⟩
    · exact ⟨-1, by simp, by norm_num⟩
  obtain ⟨t, ht⟩ : ∃ t : ℝ, τk k = t := by
    rw [τk]
    split_ifs
    · exact ⟨1, by simp⟩
    · exact ⟨-1, by simp⟩
  have hβ := A.β_pos
  set x := u.re with hx
  set y := u.im with hy
  have hsq : (u ^ 2).im = 2 * x * y := by rw [sq, mul_im]; ring
  have hxy : x * y ≠ 0 := fun h0 ↦ hs (by rw [hsq]; linear_combination 2 * h0)
  have hu : u ≠ 0 := fun h0 ↦ hs (by rw [h0]; simp)
  set c := Real.cosh (A.β * A.μ) with hcdef
  have hc : cosh ((A.β : ℂ) * A.μ) = c := by rw [hcdef, ofReal_cosh]; push_cast; ring_nf
  have hc1 : 1 ≤ c := Real.one_le_cosh _
  set C := cosh ((A.β : ℂ) * u) with hC
  set Sh := sinh ((A.β : ℂ) * u) with hSh
  -- `Φ_k = 0`: `(C + τc)u = -εa sinh(βu)`
  have E : (C + t * c) * u = -(e * A.a * Sh) := by
    rw [A.Phik_of_ne k _ hu, Dk, ht, hc] at hΦ
    field_simp at hΦ
    linear_combination hΦ
  have hPQ := sinh_mul_conj_cosh A.β u
  rw [← hSh, ← hC, ← hx, ← hy] at hPQ
  set P := Real.sinh (2 * A.β * x) / 2
  set Q := Real.sin (2 * A.β * y) / 2
  have hre := congrArg re E
  have him := congrArg im E
  have hP := congrArg re hPQ
  have hQ := congrArg im hPQ
  simp only [add_re, add_im, mul_re, mul_im, ofReal_re, ofReal_im, neg_re, neg_im, conj_re,
    conj_im, I_re, I_im, mul_zero, sub_zero, zero_mul, add_zero, mul_one, ← hx,
    ← hy] at hre him hP hQ
  -- the imaginary part of `Φ_k = 0` times `conj(u) conj(C)`
  have key : (x ^ 2 + y ^ 2) * t * c * C.im = e * A.a * (x * Q - y * P) := by
    linear_combination (e * A.a * x) * hQ - (e * A.a * y) * hP + (x * C.im + y * C.re) * hre +
      (y * C.im - x * C.re) * him
  -- `xy e τ Im C ≤ 0`
  have hM : x * y * (x * Q - y * P) ≤ 0 := by
    have := xy_mul_sin_sub_sinh_nonpos hβ x y
    simp only [P, Q]
    nlinarith
  have hnu : 0 < x ^ 2 + y ^ 2 := by
    have : x ≠ 0 := left_ne_zero_of_mul hxy
    positivity
  have hZ : x * y * e * t * C.im ≤ 0 := by
    have h1 : (x ^ 2 + y ^ 2) * c * (x * y * e * t * C.im) = A.a * (x * y * (x * Q - y * P)) := by
      linear_combination (x * y * e) * key + A.a * x * y * (x * Q - y * P) * he
    have h2 : (x ^ 2 + y ^ 2) * c * (x * y * e * t * C.im) ≤ 0 := by
      rw [h1]
      exact mul_nonpos_of_nonneg_of_nonpos ha hM
    have h3 : 0 < (x ^ 2 + y ^ 2) * c := by positivity
    by_contra h0
    linarith [mul_pos h3 (lt_of_not_ge h0)]
  -- `Im K_e = β Im s - λh Im R_k`, `Im R_k = τ(c² - 1) Im C/|D_k|²`
  have hq : A.q₀ k = ((A.a ^ 2 + A.ω k ^ 2 / 4 : ℝ) : ℂ) := by
    rw [q₀, w]
    push_cast
    ring_nf
    rw [I_sq]
    ring
  have hD' : C + t * c ≠ 0 := by rwa [Dk, ht, hc] at hD
  have hnD := normSq_pos.mpr hD'
  have hKe := A.Ke_eq k (e : ℂ) hD
  rw [Rk, Dk, ht, hc, ← hC, hq, lam] at hKe
  have hIm : (A.Ke k (e : ℂ) u).im = A.β * (2 * x * y) -
      4 * e * A.a ^ 2 * A.hth * (t * C.im * (c ^ 2 - 1) / normSq (C + t * c)) := by
    rw [hKe, ← hsq]
    have hR : ((c * C + t) / (C + t * c)).im = t * C.im * (c ^ 2 - 1) / normSq (C + t * c) := by
      rw [div_im]
      simp only [add_re, add_im, mul_re, mul_im, ofReal_re, ofReal_im, mul_zero, sub_zero,
        zero_mul, add_zero]
      field_simp
      ring
    rw [← hR]
    simp only [sub_im, mul_im, add_im, ofReal_re, ofReal_im, mul_re, add_re, re_ofNat, im_ofNat,
      sq, mul_zero, zero_mul, sub_zero, add_zero]
  rw [hIm, hsq]
  have hh : 0 ≤ A.hth := mul_nonneg A.β_pos.le (Real.sqrt_nonneg _)
  have hc2 : 0 ≤ c ^ 2 - 1 := by nlinarith
  have hpos : 0 < 4 * A.β * (x * y) ^ 2 := by positivity
  have hnn : 0 ≤
      -(8 * A.a ^ 2 * A.hth * (c ^ 2 - 1) / normSq (C + t * c) * (x * y * e * t * C.im)) := by
    rw [neg_nonneg]
    exact mul_nonpos_of_nonneg_of_nonpos (by positivity) hZ
  have : 2 * x * y * (A.β * (2 * x * y) -
      4 * e * A.a ^ 2 * A.hth * (t * C.im * (c ^ 2 - 1) / normSq (C + t * c))) =
      4 * A.β * (x * y) ^ 2 -
        8 * A.a ^ 2 * A.hth * (c ^ 2 - 1) / normSq (C + t * c) * (x * y * e * t * C.im) := by
    field_simp
    ring
  rw [this]
  linarith

/-- At a nonreal zero of `Φ_k`, `K_e ≠ 0` for `U ≥ 0`: the zero is not a divergence, which needs
`K_e = 0` (`Ke_eq_zero_of_Phi`). -/
theorem Ke_ne_zero_of_Phi (hε : ε = 1 ∨ ε = -1) (ha : 0 ≤ A.a) {u : ℂ} (hs : (u ^ 2).im ≠ 0)
    (hD : A.Dk k u ≠ 0) (hΦ : A.Phik k ε u = 0) : A.Ke k ε u ≠ 0 := fun h0 ↦ by
  have := A.im_mul_im_Ke_pos k ε hε ha hs hD hΦ
  rw [h0, zero_im, mul_zero] at this
  exact lt_irrefl 0 this

end Params

end HubbardAtom
