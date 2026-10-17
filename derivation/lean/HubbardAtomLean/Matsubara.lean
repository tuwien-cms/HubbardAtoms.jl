import HubbardAtomLean.Thermal

/-!
# Matsubara sums: Lemmas "basic sum" and "particle–particle sum"

Section "Lattice sums" of the note. The analytic input is Mathlib's Mittag-Leffler expansion of the
cotangent (`cot_series_rep'`); the principal-value sums of the note are sums over the symmetric
boxes of Mathlib's summation filters `SummationFilter.symmetricIcc` and `symmetricIco`.

* `hasSum_one_div_sub_int`: `π cot(πx) = ∑ₙ 1/(x - n)` over the boxes `[-N, N]`. Mathlib states the
  expansion with the terms `n` and `-n` paired; this is the same sum over `ℤ` with a summation
  filter.
* `hasSum_coth`, `hasSum_tanh`: `coth y = ∑ₙ 1/(y - nπi)` over the boxes `[-N, N]`, and
  `tanh y = ∑ₙ 1/(y - (n + 1/2)πi)` over the boxes `[-N, N - 1]`, symmetric about `-1/2`.
* `summable_inv_mul_linear`: `∑ₙ 1/((nc + w₁)(nc + w₂))` converges absolutely, for complex `w₁, w₂`
  (Mathlib's `EisensteinSeries.summable_linear_left_mul_linear_left` has integer shifts).
* `Params.hasSum_S_even`, `Params.hasSum_S_odd`: Lemma "basic sum", `PV ∑ₙ (zₙ - x)⁻¹ = S_k(x)`,
  over the boxes `|2n + 1 + k| ≤ 2M + 1` of the note: in the index `n + k/2` (even `k`) resp.
  `n + (k + 1)/2` (odd `k`) these are `[-M - 1, M]` resp. `[-M, M]`.
* `Params.hasSum_pair`: `∑ₙ 1/((zₙ - x)(zₙ - y)) = (S_k(x) - S_k(y))/(x - y)`, absolutely.
* `Params.hasSum_one_div_t_sub`: Lemma "particle–particle sum", `∑ₙ (tₙ - t)⁻¹ = -S_k(μ + ξ)/ξ` for
  `ξ² = b²/4 - t`, absolutely convergent.
* `Params.hasDerivAt_S`, `Params.hasSum_sq`: `S_k' = S_k² - β²/4` and Lemma "basic sum" for double
  poles, `∑ₙ (zₙ - x)⁻² = S_k'(x)`.
-/

open Complex Filter Topology SummationFilter Finset Asymptotics

namespace HubbardAtom

/-! ### Symmetric sums of the cotangent series -/

/-- `∑_{n ∈ [-N, N]} f n = f 0 + ∑_{n < N} (f (n + 1) + f (-(n + 1)))`. -/
theorem sum_Icc_neg_eq {M : Type*} [AddCommGroup M] (f : ℤ → M) (N : ℕ) :
    ∑ n ∈ Icc (-(N : ℤ)) N, f n = f 0 + ∑ n ∈ range N, (f (n + 1) + f (-(n + 1))) := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Nat.cast_add, Nat.cast_one, sum_Icc_succ_eq_add_endpoints, ih, sum_range_succ]
    abel

/-- The cotangent series as a sum over `ℤ` along the symmetric boxes `[-N, N]`:
`π cot(πx) = ∑ₙ 1/(x - n)`. -/
theorem hasSum_one_div_sub_int {x : ℂ} (hx : x ∈ integerComplement) :
    HasSum (fun n : ℤ ↦ 1 / (x - n)) (Real.pi * cot (Real.pi * x)) (symmetricIcc ℤ) := by
  rw [hasSum_symmetricIcc_iff]
  have h := (summable_cotTerm hx).hasSum.tendsto_sum_nat
  simp only [cotTerm] at h
  rw [← cot_series_rep' hx] at h
  have h' := h.const_add (1 / x)
  rw [add_sub_cancel] at h'
  refine h'.congr fun N ↦ ?_
  rw [sum_Icc_neg_eq]
  simp [sub_eq_add_neg]

/-- `coth(-z) = -coth z`. -/
theorem coth_neg (z : ℂ) : coth (-z) = -coth z := by
  rw [coth, coth, cosh_neg, sinh_neg, div_neg]

/-- `coth y = ∑ₙ 1/(y - nπi)` along the symmetric boxes `[-N, N]`, for `sinh y ≠ 0`. -/
theorem hasSum_coth {y : ℂ} (hy : sinh y ≠ 0) :
    HasSum (fun n : ℤ ↦ 1 / (y - n * Real.pi * I)) (coth y) (symmetricIcc ℤ) := by
  have hπ : (Real.pi : ℂ) ≠ 0 := ofReal_ne_zero.mpr Real.pi_ne_zero
  have hπI : (Real.pi : ℂ) * I ≠ 0 := mul_ne_zero hπ I_ne_zero
  have hx : y / (Real.pi * I) ∈ integerComplement := by
    rintro ⟨m, hm⟩
    apply hy
    have hy' : y = (m * Real.pi : ℂ) * I := by
      rw [eq_div_iff hπI] at hm
      rw [← hm]
      ring
    rw [hy', sinh_mul_I, sin_int_mul_pi, zero_mul]
  have h := (hasSum_one_div_sub_int hx).mul_left (Real.pi * I)⁻¹
  convert h using 1
  · funext n
    have hd : (Real.pi * I) * (y / (Real.pi * I) - n) = y - n * Real.pi * I := by
      field_simp
    rw [one_div, one_div, ← mul_inv, hd]
  · have hπx : (Real.pi : ℂ) * (y / (Real.pi * I)) = -(y * I) := by
      field_simp
      ring_nf
      rw [I_sq]
      ring
    rw [hπx, cot_eq_cos_div_sin, cos_neg, sin_neg, cos_mul_I, sin_mul_I, coth]
    field_simp
    ring_nf
    rw [I_sq]
    ring

/-- `tanh y = ∑ₙ 1/(y - (n + 1/2)πi)` along the boxes `[-N, N - 1]`, for `cosh y ≠ 0`. -/
theorem hasSum_tanh {y : ℂ} (hy : cosh y ≠ 0) :
    HasSum (fun n : ℤ ↦ 1 / (y - (n + 1 / 2) * Real.pi * I)) (tanh y) (symmetricIco ℤ) := by
  have hπ : (Real.pi : ℂ) ≠ 0 := ofReal_ne_zero.mpr Real.pi_ne_zero
  have hπI : (Real.pi : ℂ) * I ≠ 0 := mul_ne_zero hπ I_ne_zero
  -- `coth(y - iπ/2) = tanh y`, and `sinh(y - iπ/2) = -i cosh y ≠ 0`
  have hshift : y - Real.pi / 2 * I = y + ((-1 : ℤ) : ℂ) * (Real.pi / 2) * I := by push_cast; ring
  have hcoth : coth (y - Real.pi / 2 * I) = tanh y := by
    rw [hshift, coth_add_int_mul, ite_eq_right (by decide)]
  have hsinh : sinh (y - Real.pi / 2 * I) ≠ 0 := by
    rw [show y - Real.pi / 2 * I = y + (-(Real.pi / 2) : ℂ) * I by ring, sinh_add, cosh_mul_I,
      sinh_mul_I, cos_neg, sin_neg, cos_pi_div_two, sin_pi_div_two]
    simpa using hy
  have h := hasSum_coth hsinh
  rw [hcoth] at h
  have h' : HasSum (fun n : ℤ ↦ 1 / (y - (n + 1 / 2) * Real.pi * I)) (tanh y) (symmetricIcc ℤ) := by
    convert h using 3 with n
    ring
  refine h'.hasSum_symmetricIco_of_hasSum_symmetricIcc ?_
  -- the terms vanish at infinity: `1/(y - (N + 1/2)πi) = (-πi)⁻¹/(1·z₀ + N)`
  have ht := (EisensteinSeries.tendsto_zero_inv_linear (1 / 2 - y / (Real.pi * I)) 1).const_mul
    (-(-(Real.pi * I))⁻¹)
  rw [mul_zero] at ht
  refine ht.congr fun N ↦ ?_
  have hd : (y - (N + 1 / 2) * Real.pi * I) = -(Real.pi * I) * (((1 : ℤ) : ℂ) *
      (1 / 2 - y / (Real.pi * I)) + N) := by
    push_cast
    field_simp
    ring
  simp only [Int.cast_natCast]
  rw [hd]
  simp only [one_div, inv_neg, mul_inv, neg_mul, neg_neg]

/-! ### Summability -/

/-- `n ↦ nc + w` grows like `n`, for `c ≠ 0` and any complex `w`. -/
theorem isTheta_linear (w : ℂ) {c : ℂ} (hc : c ≠ 0) :
    (fun n : ℤ ↦ n * c + w) =Θ[cofinite] fun n ↦ (n : ℝ) := by
  apply IsTheta.add_isLittleO
  · simp_rw [mul_comm]
    exact IsTheta.const_mul_left hc Int.cast_complex_isTheta_cast_real
  · simp only [isLittleO_const_left,
      tendsto_norm_comp_cofinite_atTop_of_isClosedEmbedding Int.isClosedEmbedding_coe_real, or_true]

/-- `∑ₙ 1/((nc + w₁)(nc + w₂))` converges absolutely for `c ≠ 0`. -/
theorem summable_inv_mul_linear {c : ℂ} (hc : c ≠ 0) (w₁ w₂ : ℂ) :
    Summable fun n : ℤ ↦ ((n * c + w₁) * (n * c + w₂))⁻¹ := by
  apply EisensteinSeries.summable_inv_of_isBigO_rpow_inv (a := 2) (by norm_cast)
  simp only [Real.rpow_two, abs_mul_abs_self, pow_two]
  simpa using (isTheta_linear w₂ hc).inv.isBigO.mul (isTheta_linear w₁ hc).inv.isBigO

/-- An absolutely convergent sum equals its sum along any summation filter
(`tsum_eq_of_summable_unconditional`, as a `HasSum`). -/
theorem hasSum_of_hasSum_filter {f : ℤ → ℂ} {a : ℂ} {L : SummationFilter ℤ} [L.LeAtTop] [L.NeBot]
    (hf : Summable f) (hL : HasSum f a L) : HasSum f a := by
  rw [← hL.tsum_eq, tsum_eq_of_summable_unconditional hf]
  exact hf.hasSum

/-! ### The lattice sums -/

namespace Params

variable (A : Params)

theorem β_ne_zero : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'

/-- `1/(zₙ - x) = -(β/2)/(y - (zₙ - μ)β/2)` with `y = β(x - μ)/2`, also where both vanish. -/
theorem one_div_z_sub (k n : ℤ) (x : ℂ) :
    1 / (A.z k n - x) =
      -(A.β / 2 : ℂ) * (1 / (A.β * (x - A.μ) / 2 - A.β * (A.z k n - A.μ) / 2)) := by
  have hβ := A.β_ne_zero
  rw [show (A.β : ℂ) * (x - A.μ) / 2 - A.β * (A.z k n - A.μ) / 2 = -(A.β / 2) * (A.z k n - x) by
      ring, one_div, one_div, mul_inv, ← mul_assoc, mul_inv_cancel₀ (neg_ne_zero.mpr
      (div_ne_zero hβ two_ne_zero)), one_mul]

/-- Lemma "basic sum", even `k = 2j`: along the boxes `|2n + 1 + k| ≤ 2M + 1`,
`PV ∑ₙ (zₙ - x)⁻¹ = S_k(x)` for `cosh(β(x - μ)/2) ≠ 0`, i.e. `x` off the lattice. -/
theorem hasSum_S_even {k j : ℤ} (hk : k = 2 * j) {x : ℂ} (hx : cosh (A.β * (x - A.μ) / 2) ≠ 0) :
    HasSum (fun m : ℤ ↦ 1 / (A.z k (m - j) - x)) (A.S k x) (symmetricIco ℤ) := by
  have hβ := A.β_ne_zero
  have h := (hasSum_tanh hx).mul_left (-(A.β / 2 : ℂ))
  rw [S, ite_eq_left ⟨j, by omega⟩]
  convert h using 1
  funext m
  rw [A.one_div_z_sub, A.z_eq, hk]
  congr 2
  push_cast
  field_simp
  ring

/-- Lemma "basic sum", odd `k = 2j - 1`: along the boxes `|2n + 1 + k| ≤ 2M + 1`,
`PV ∑ₙ (zₙ - x)⁻¹ = S_k(x)` for `sinh(β(x - μ)/2) ≠ 0`. -/
theorem hasSum_S_odd {k j : ℤ} (hk : k = 2 * j - 1) {x : ℂ} (hx : sinh (A.β * (x - A.μ) / 2) ≠ 0) :
    HasSum (fun m : ℤ ↦ 1 / (A.z k (m - j) - x)) (A.S k x) (symmetricIcc ℤ) := by
  have hβ := A.β_ne_zero
  have h := (hasSum_coth hx).mul_left (-(A.β / 2 : ℂ))
  rw [S, ite_eq_right (by rintro ⟨r, hr⟩; omega)]
  convert h using 1
  funext m
  rw [A.one_div_z_sub, A.z_eq, hk]
  congr 2
  push_cast
  field_simp
  ring

/-- `S_k(μ - ξ) = -S_k(μ + ξ)`. -/
theorem S_mu_sub (k : ℤ) (ξ : ℂ) : A.S k (A.μ - ξ) = -A.S k (A.μ + ξ) := by
  simp only [S]
  rw [show (A.β : ℂ) * (A.μ - ξ - A.μ) / 2 = -(A.β * (A.μ + ξ - A.μ) / 2) by ring]
  split_ifs <;> simp [tanh_neg, coth_neg]

/-- Off the lattice the factor of `S_k` does not vanish: `cosh(β(x - μ)/2) ≠ 0` for even `k`. -/
theorem cosh_ne_zero_of_off {k j : ℤ} (hk : k = 2 * j) {x : ℂ} (hx : ∀ n, A.z k n ≠ x) :
    cosh (A.β * (x - A.μ) / 2) ≠ 0 := by
  have hβ := A.β_ne_zero
  intro h0
  rw [← cos_mul_I, cos_eq_zero_iff] at h0
  obtain ⟨m, hm⟩ := h0
  -- `x = z_n` with `2n + 1 + k = -(2m + 1)`
  apply hx (-m - 1 - j)
  have hI : I * I = -1 := I_mul_I
  rw [A.z_eq, hk]
  push_cast
  field_simp
  linear_combination (2 * I) * hm - (A.β * (x - A.μ)) * hI

/-- Off the lattice the factor of `S_k` does not vanish: `sinh(β(x - μ)/2) ≠ 0` for odd `k`. -/
theorem sinh_ne_zero_of_off {k j : ℤ} (hk : k = 2 * j - 1) {x : ℂ} (hx : ∀ n, A.z k n ≠ x) :
    sinh (A.β * (x - A.μ) / 2) ≠ 0 := by
  have hβ := A.β_ne_zero
  intro h0
  have h0' : sin (A.β * (x - A.μ) / 2 * I) = 0 := by rw [sin_mul_I, h0, zero_mul]
  rw [sin_eq_zero_iff] at h0'
  obtain ⟨m, hm⟩ := h0'
  -- `x = z_n` with `2n + 1 + k = -2m`
  apply hx (-m - j)
  have hI : I * I = -1 := I_mul_I
  rw [A.z_eq, hk]
  push_cast
  field_simp
  linear_combination (2 * I) * hm - (A.β * (x - A.μ)) * hI

/-- The products of two simple fractions: for `x ≠ y` off the lattice,
`∑ₙ 1/((zₙ - x)(zₙ - y)) = (S_k(x) - S_k(y))/(x - y)`, absolutely convergent. -/
theorem hasSum_pair {k : ℤ} {x y : ℂ} (hxy : x ≠ y) (hx : ∀ n, A.z k n ≠ x)
    (hy : ∀ n, A.z k n ≠ y) :
    HasSum (fun n ↦ 1 / ((A.z k n - x) * (A.z k n - y))) ((A.S k x - A.S k y) / (x - y)) := by
  have hβ := A.β_ne_zero
  have hπ : (Real.pi : ℂ) ≠ 0 := ofReal_ne_zero.mpr Real.pi_ne_zero
  have hxy' := sub_ne_zero.mpr hxy
  have hpf : ∀ n, 1 / ((A.z k n - x) * (A.z k n - y)) =
      1 / (x - y) * (1 / (A.z k n - x) - 1 / (A.z k n - y)) := fun n ↦ by
    have h₁ := sub_ne_zero.mpr (hx n)
    have h₂ := sub_ne_zero.mpr (hy n)
    field_simp
    ring
  -- absolute convergence: `zₙ - x = n c + w₁`, `zₙ - y = n c + w₂`
  have hsum : Summable fun n ↦ 1 / ((A.z k n - x) * (A.z k n - y)) := by
    have hc : (2 * Real.pi / A.β * I : ℂ) ≠ 0 :=
      mul_ne_zero (div_ne_zero (mul_ne_zero two_ne_zero hπ) hβ) I_ne_zero
    refine (summable_inv_mul_linear hc (A.μ + (1 + k) * Real.pi / A.β * I - x)
      (A.μ + (1 + k) * Real.pi / A.β * I - y)).congr fun n ↦ ?_
    rw [A.z_eq, one_div]
    congr 1
    push_cast
    ring
  -- the principal values, in an index in which the boxes are symmetric
  have hpv : ∀ (j : ℤ) (L : SummationFilter ℤ) [L.LeAtTop] [L.NeBot],
      (∀ u, u = x ∨ u = y → HasSum (fun m : ℤ ↦ 1 / (A.z k (m - j) - u)) (A.S k u) L) →
      HasSum (fun n ↦ 1 / ((A.z k n - x) * (A.z k n - y))) ((A.S k x - A.S k y) / (x - y)) := by
    intro j L _ _ hS
    have hL := ((hS _ (Or.inl rfl)).sub (hS _ (Or.inr rfl))).mul_left (1 / (x - y))
    have hL' : HasSum (fun m ↦ 1 / ((A.z k (m - j) - x) * (A.z k (m - j) - y)))
        ((A.S k x - A.S k y) / (x - y)) L := by
      convert hL using 1
      · funext m
        rw [hpf]
      · ring
    have hU := hasSum_of_hasSum_filter ((Equiv.subRight j).summable_iff.mpr hsum) hL'
    simpa [Function.comp_def] using (Equiv.addRight j).hasSum_iff.mpr hU
  obtain ⟨j, hj | hj⟩ := Int.even_or_odd' k
  · exact hpv j (symmetricIco ℤ) fun u hu ↦ A.hasSum_S_even hj
      (A.cosh_ne_zero_of_off hj (by rcases hu with rfl | rfl <;> assumption))
  · have hj' : k = 2 * (j + 1) - 1 := by omega
    exact hpv (j + 1) (symmetricIcc ℤ) fun u hu ↦ A.hasSum_S_odd hj'
      (A.sinh_ne_zero_of_off hj' (by rcases hu with rfl | rfl <;> assumption))

/-- Lemma "particle–particle sum": for `ξ² = b²/4 - t`, `ξ ≠ 0` and `t` off the lattice
`{tₙ}`, `∑ₙ (tₙ - t)⁻¹ = -S_k(μ + ξ)/ξ`, absolutely convergent. -/
theorem hasSum_one_div_t_sub {k : ℤ} {t ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 - t) (hξ0 : ξ ≠ 0)
    (ht : ∀ n, A.t k n ≠ t) :
    HasSum (fun n ↦ 1 / (A.t k n - t)) (-A.S k (A.μ + ξ) / ξ) := by
  -- `tₙ - t = -(zₙ - (μ + ξ))(zₙ - (μ - ξ))`
  have hfac : ∀ n, A.t k n - t = -((A.z k n - (A.μ + ξ)) * (A.z k n - (A.μ - ξ))) := fun n ↦ by
    rw [A.t_eq_z]
    linear_combination (-1 : ℂ) * hξ
  have hoff : ∀ n, (A.z k n - (A.μ + ξ)) * (A.z k n - (A.μ - ξ)) ≠ 0 := fun n h0 ↦
    ht n (by rw [← sub_eq_zero, hfac, h0, neg_zero])
  have h := (A.hasSum_pair (k := k) (x := A.μ + ξ) (y := A.μ - ξ)
    (fun h0 ↦ hξ0 (by linear_combination h0 / 2))
    (fun n h0 ↦ hoff n (by rw [h0, sub_self, zero_mul]))
    (fun n h0 ↦ hoff n (by rw [h0, sub_self, mul_zero]))).neg
  rw [A.S_mu_sub] at h
  convert h using 1
  · funext n
    rw [hfac, one_div_neg_eq_neg_one_div]
  · rw [show (A.μ : ℂ) + ξ - (A.μ - ξ) = 2 * ξ by ring]
    field_simp
    ring

/-- `tanh' = 1/cosh²`. -/
theorem _root_.HubbardAtom.hasDerivAt_tanh {y : ℂ} (hy : cosh y ≠ 0) :
    HasDerivAt tanh (1 / cosh y ^ 2) y := by
  have h := (hasDerivAt_sinh y).div (hasDerivAt_cosh y) hy
  rw [show (sinh / cosh : ℂ → ℂ) = tanh from
    funext fun y ↦ by rw [Pi.div_apply, tanh_eq_sinh_div_cosh]] at h
  convert h using 1
  rw [show cosh y * cosh y - sinh y * sinh y = 1 by linear_combination cosh_sq_sub_sinh_sq y]

/-- `coth' = -1/sinh²`. -/
theorem _root_.HubbardAtom.hasDerivAt_coth {y : ℂ} (hy : sinh y ≠ 0) :
    HasDerivAt coth (-1 / sinh y ^ 2) y := by
  have h := (hasDerivAt_cosh y).div (hasDerivAt_sinh y) hy
  rw [show (cosh / sinh : ℂ → ℂ) = coth from funext fun y ↦ by rw [Pi.div_apply, coth]] at h
  convert h using 1
  rw [show sinh y * sinh y - cosh y * cosh y = -1 by linear_combination -cosh_sq_sub_sinh_sq y]

/-- `S_k' = S_k² - β²/4` off the lattice. -/
theorem hasDerivAt_S {k : ℤ} {x : ℂ} (hx : ∀ n, A.z k n ≠ x) :
    HasDerivAt (A.S k) (A.S k x ^ 2 - A.β ^ 2 / 4) x := by
  have hβ := A.β_ne_zero
  have hlin : HasDerivAt (fun x : ℂ ↦ A.β * (x - A.μ) / 2) (A.β / 2) x := by
    simpa using (((hasDerivAt_id x).sub_const (A.μ : ℂ)).const_mul (A.β : ℂ)).div_const 2
  obtain ⟨j, hj | hj⟩ := Int.even_or_odd' k
  · have hk : Even k := ⟨j, by omega⟩
    have hc := A.cosh_ne_zero_of_off hj hx
    have hS : A.S k = fun x ↦ -(A.β / 2 : ℂ) * tanh (A.β * (x - A.μ) / 2) :=
      funext fun x ↦ by rw [S, ite_eq_left hk]
    rw [show A.S k x = -(A.β / 2 : ℂ) * tanh (A.β * (x - A.μ) / 2) by rw [hS], hS]
    refine (((hasDerivAt_tanh hc).comp x hlin).const_mul (-(A.β / 2 : ℂ))).congr_deriv ?_
    rw [tanh_eq_sinh_div_cosh]
    field_simp
    linear_combination 4 * cosh_sq_sub_sinh_sq (A.β * (x - A.μ) / 2)
  · have hk : ¬Even k := by rintro ⟨r, hr⟩; omega
    have hj' : k = 2 * (j + 1) - 1 := by omega
    have hs := A.sinh_ne_zero_of_off hj' hx
    have hS : A.S k = fun x ↦ -(A.β / 2 : ℂ) * coth (A.β * (x - A.μ) / 2) :=
      funext fun x ↦ by rw [S, ite_eq_right hk]
    rw [show A.S k x = -(A.β / 2 : ℂ) * coth (A.β * (x - A.μ) / 2) by rw [hS], hS]
    refine (((hasDerivAt_coth hs).comp x hlin).const_mul (-(A.β / 2 : ℂ))).congr_deriv ?_
    rw [coth]
    field_simp
    linear_combination -4 * cosh_sq_sub_sinh_sq (A.β * (x - A.μ) / 2)

/-- The lattice keeps a positive distance from every point off it. -/
theorem exists_lattice_dist {k : ℤ} {x : ℂ} (hx : ∀ n, A.z k n ≠ x) :
    ∃ r > 0, ∀ n, r ≤ ‖A.z k n - x‖ := by
  have hβ := A.β_ne_zero
  have hπ : (Real.pi : ℂ) ≠ 0 := ofReal_ne_zero.mpr Real.pi_ne_zero
  have hc : (2 * Real.pi / A.β * I : ℂ) ≠ 0 :=
    mul_ne_zero (div_ne_zero (mul_ne_zero two_ne_zero hπ) hβ) I_ne_zero
  have ht : Tendsto (fun n : ℤ ↦ ‖A.z k n - x‖) cofinite atTop := by
    have h := (isTheta_linear (A.μ + (1 + k) * Real.pi / A.β * I - x) hc).tendsto_norm_atTop_iff.mpr
      (tendsto_norm_comp_cofinite_atTop_of_isClosedEmbedding Int.isClosedEmbedding_coe_real)
    refine h.congr fun n ↦ ?_
    simp only [Function.comp_apply]
    congr 1
    rw [A.z_eq]
    push_cast
    ring
  obtain ⟨n₀, hn₀⟩ := ht.exists_forall_le
  exact ⟨_, norm_pos_iff.mpr (sub_ne_zero.mpr (hx n₀)), hn₀⟩

/-- Lemma "basic sum" for double poles: `∑ₙ 1/(zₙ - x)² = S_k'(x) = S_k(x)² - β²/4` off the
lattice, absolutely convergent. The slopes of `S_k` at `x` are the sums of `hasSum_pair`, which
differ from this sum by at most a constant times `|y - x|`. -/
theorem hasSum_sq {k : ℤ} {x : ℂ} (hx : ∀ n, A.z k n ≠ x) :
    HasSum (fun n ↦ 1 / (A.z k n - x) ^ 2) (A.S k x ^ 2 - A.β ^ 2 / 4) := by
  have hβ := A.β_ne_zero
  have hπ : (Real.pi : ℂ) ≠ 0 := ofReal_ne_zero.mpr Real.pi_ne_zero
  obtain ⟨r, hr, hdist⟩ := A.exists_lattice_dist hx
  have hsum : Summable fun n ↦ 1 / (A.z k n - x) ^ 2 := by
    have hc : (2 * Real.pi / A.β * I : ℂ) ≠ 0 :=
      mul_ne_zero (div_ne_zero (mul_ne_zero two_ne_zero hπ) hβ) I_ne_zero
    refine (summable_inv_mul_linear hc (A.μ + (1 + k) * Real.pi / A.β * I - x)
      (A.μ + (1 + k) * Real.pi / A.β * I - x)).congr fun n ↦ ?_
    rw [A.z_eq, one_div, sq]
    congr 1
    push_cast
    ring
  have hnorm := hsum.norm
  set T := ∑' n, 1 / (A.z k n - x) ^ 2
  set C := 2 / r * ∑' n, ‖1 / (A.z k n - x) ^ 2‖
  suffices hT : T = A.S k x ^ 2 - A.β ^ 2 / 4 by rw [← hT]; exact hsum.hasSum
  refine tendsto_nhds_unique ?_ (hasDerivAt_iff_tendsto_slope.mp (A.hasDerivAt_S hx))
  -- `‖slope - T‖ ≤ C ‖y - x‖` for `0 < ‖y - x‖ < r/2`
  have hbound : ∀ y, y ≠ x → ‖y - x‖ < r / 2 → ‖slope (A.S k) x y - T‖ ≤ C * ‖y - x‖ := by
    intro y hyx hyr
    have hy : ∀ n, A.z k n ≠ y := fun n h0 ↦ by
      have := hdist n
      rw [h0] at this
      linarith
    have hy' : ∀ n, r / 2 ≤ ‖A.z k n - y‖ := fun n ↦ by
      have h₁ := hdist n
      have h₂ := norm_sub_le (A.z k n - y) (x - y)
      rw [sub_sub_sub_cancel_right, ← norm_neg (x - y), neg_sub] at h₂
      linarith
    have hpair := A.hasSum_pair (Ne.symm hyx) hx hy
    have hdiff := (hsum.hasSum.sub hpair)
    have hslope : slope (A.S k) x y - T =
        -(T - (A.S k x - A.S k y) / (x - y)) := by
      rw [slope_def_field]
      have := sub_ne_zero.mpr hyx
      have := sub_ne_zero.mpr (Ne.symm hyx)
      field_simp
      ring
    rw [hslope, norm_neg]
    refine hdiff.norm_le_of_bounded ((hnorm.hasSum.mul_left (2 / r)).mul_right ‖y - x‖) ?_
    intro n
    have hn := sub_ne_zero.mpr (hx n)
    have hn' := sub_ne_zero.mpr (hy n)
    have hr2 : 0 < r / 2 := half_pos hr
    have heq : 1 / (A.z k n - x) ^ 2 - 1 / ((A.z k n - x) * (A.z k n - y)) =
        1 / (A.z k n - x) ^ 2 * ((x - y) / (A.z k n - y)) := by
      field_simp
      ring
    have hq : ‖(x - y) / (A.z k n - y)‖ ≤ ‖y - x‖ / (r / 2) := by
      rw [norm_div, ← norm_neg (x - y), neg_sub]
      exact div_le_div_of_nonneg_left (norm_nonneg _) hr2 (hy' n)
    rw [heq, norm_mul]
    calc ‖1 / (A.z k n - x) ^ 2‖ * ‖(x - y) / (A.z k n - y)‖
        ≤ ‖1 / (A.z k n - x) ^ 2‖ * (‖y - x‖ / (r / 2)) := by gcongr
      _ = 2 / r * ‖1 / (A.z k n - x) ^ 2‖ * ‖y - x‖ := by field_simp
  rw [tendsto_iff_norm_sub_tendsto_zero]
  have hC : Tendsto (fun y ↦ C * ‖y - x‖) (𝓝[≠] x) (𝓝 0) := by
    have : Tendsto (fun y ↦ C * ‖y - x‖) (𝓝 x) (𝓝 (C * ‖x - x‖)) :=
      tendsto_const_nhds.mul (tendsto_id.sub tendsto_const_nhds).norm
    rw [sub_self, norm_zero, mul_zero] at this
    exact this.mono_left nhdsWithin_le_nhds
  refine squeeze_zero' (Eventually.of_forall fun y ↦ norm_nonneg _) ?_ hC
  filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds (Metric.ball_mem_nhds x (half_pos hr))]
    with y hyx hyb
  exact hbound y hyx (by simpa [Metric.mem_ball, dist_eq_norm] using hyb)

/-- For even `k` the lattice avoids `μ`: `zₙ - μ = iπ(2n + 1 + k)/β ≠ 0`. -/
theorem z_ne_mu_of_even {k : ℤ} (hk : Even k) (n : ℤ) : A.z k n ≠ A.μ := by
  intro h
  have him := congrArg Complex.im h
  rw [A.z_eq] at him
  simp only [add_im, ofReal_im, mul_im, ofReal_re, I_im, mul_one, I_re, mul_zero, add_zero,
    zero_add] at him
  have hβ := A.β_pos.ne'
  have h2 : ((2 * n + 1 + k : ℤ) : ℝ) = 0 := by
    have : (2 * (n : ℝ) + 1 + k) * Real.pi = 0 := by
      field_simp at him
      linarith
    push_cast
    rcases mul_eq_zero.mp this with h | h
    · exact h
    · exact absurd h Real.pi_ne_zero
  have h3 : 2 * n + 1 + k = 0 := by exact_mod_cast h2
  obtain ⟨j, rfl⟩ := hk
  omega

/-- For odd `k` the mirror fixed point `n = -(k + 1)/2` has `zₙ = μ`. -/
theorem exists_z_eq_mu_of_odd {k : ℤ} (hk : Odd k) : ∃ n, A.z k n = A.μ := by
  obtain ⟨j, rfl⟩ := hk
  refine ⟨-j - 1, ?_⟩
  rw [A.z_eq]
  push_cast
  ring_nf

theorem S_mu_of_even {k : ℤ} (hk : Even k) : A.S k A.μ = 0 := by
  simp [S, hk]

/-- Lemma "particle–particle sum" at `ξ = 0`, i.e. `t = b²/4`, for even `k`:
`∑ₙ (tₙ - t)⁻¹ = -∑ₙ (zₙ - μ)⁻² = -S_k'(μ) = β²/4`. -/
theorem hasSum_one_div_t_sub_confluent {k : ℤ} (hk : Even k) :
    HasSum (fun n ↦ 1 / (A.t k n - A.b k ^ 2 / 4)) ((A.β : ℂ) ^ 2 / 4) := by
  have h := (A.hasSum_sq (k := k) (A.z_ne_mu_of_even hk)).neg
  rw [A.S_mu_of_even hk] at h
  convert h using 1
  · funext n
    rw [A.t_eq_z, show A.b k ^ 2 / 4 - (A.z k n - A.μ) ^ 2 - A.b k ^ 2 / 4 =
      -((A.z k n - A.μ) ^ 2) by ring, one_div_neg_eq_neg_one_div]
  · ring

/-- `𝒯_k(t) = -S_k(μ + ξ)/ξ` of Eq. (T), `ξ² = b²/4 - t`, with the value `β²/4` at `ξ = 0`
(Lemma "particle–particle sum"). -/
noncomputable def Tpp (k : ℤ) (ξ : ℂ) : ℂ :=
  if ξ = 0 then (A.β : ℂ) ^ 2 / 4 else -A.S k (A.μ + ξ) / ξ

/-- Lemma "particle–particle sum", including `ξ = 0`: `∑ₙ (tₙ - t)⁻¹ = 𝒯_k(t)` for `t` off the
lattice `{tₙ}`. -/
theorem hasSum_Tpp {k : ℤ} {t ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 - t) (ht : ∀ n, A.t k n ≠ t) :
    HasSum (fun n ↦ 1 / (A.t k n - t)) (A.Tpp k ξ) := by
  by_cases hξ0 : ξ = 0
  · rw [Tpp, ite_eq_left hξ0]
    have htb : t = A.b k ^ 2 / 4 := by
      rw [hξ0] at hξ
      linear_combination hξ
    rcases Int.even_or_odd k with hk | hk
    · rw [htb]
      exact A.hasSum_one_div_t_sub_confluent hk
    · obtain ⟨n, hn⟩ := A.exists_z_eq_mu_of_odd hk
      exact absurd (by rw [A.t_eq_z, hn, sub_self, htb]; ring) (ht n)
  · rw [Tpp, ite_eq_right hξ0]
    exact A.hasSum_one_div_t_sub hξ hξ0 ht

/-- The double-pole particle–particle sum: `∑ₙ (tₙ - t)⁻² = (S² - β²/4)/(2ξ²) - S/(2ξ³)` with
`S = S_k(μ + ξ)`, for `ξ² = b²/4 - t`, `ξ ≠ 0` and `t` off the lattice. -/
theorem hasSum_one_div_t_sub_sq {k : ℤ} {t ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 - t) (hξ0 : ξ ≠ 0)
    (ht : ∀ n, A.t k n ≠ t) :
    HasSum (fun n ↦ 1 / (A.t k n - t) ^ 2)
      ((A.S k (A.μ + ξ) ^ 2 - (A.β : ℂ) ^ 2 / 4) / (2 * ξ ^ 2) -
        A.S k (A.μ + ξ) / (2 * ξ ^ 3)) := by
  have hfac : ∀ n, A.t k n - t = -((A.z k n - (A.μ + ξ)) * (A.z k n - (A.μ - ξ))) := fun n ↦ by
    rw [A.t_eq_z]
    linear_combination (-1 : ℂ) * hξ
  have hoff : ∀ n, (A.z k n - (A.μ + ξ)) * (A.z k n - (A.μ - ξ)) ≠ 0 := fun n h0 ↦
    ht n (by rw [← sub_eq_zero, hfac, h0, neg_zero])
  have hx : ∀ n, A.z k n ≠ A.μ + ξ := fun n h0 ↦ hoff n (by rw [h0, sub_self, zero_mul])
  have hy : ∀ n, A.z k n ≠ A.μ - ξ := fun n h0 ↦ hoff n (by rw [h0, sub_self, mul_zero])
  have hxy : (A.μ : ℂ) + ξ ≠ A.μ - ξ := fun h0 ↦ hξ0 (by linear_combination h0 / 2)
  have h₁ := A.hasSum_sq hx
  have h₂ := A.hasSum_sq hy
  have h₃ := A.hasSum_pair hxy hx hy
  rw [A.S_mu_sub] at h₂ h₃
  convert ((h₁.add h₂).div_const (4 * ξ ^ 2)).sub (h₃.div_const (2 * ξ ^ 2)) using 1
  · funext n
    have h₄ := sub_ne_zero.mpr (hx n)
    have h₅ := sub_ne_zero.mpr (hy n)
    rw [hfac]
    field_simp
    ring
  · rw [show (A.μ : ℂ) + ξ - (A.μ - ξ) = 2 * ξ by ring]
    field_simp
    ring

end Params

end HubbardAtom
