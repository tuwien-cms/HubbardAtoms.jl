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
* `Params.hasSum_one_div_t_sub`: Lemma "particle–particle sum", `∑ₙ (tₙ - t)⁻¹ = -S_k(μ + ξ)/ξ` for
  `ξ² = b²/4 - t`, absolutely convergent.
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

/-- Lemma "particle–particle sum": for `ξ² = b²/4 - t`, `ξ ≠ 0` and `t` off the lattice
`{tₙ}`, `∑ₙ (tₙ - t)⁻¹ = -S_k(μ + ξ)/ξ`, absolutely convergent. -/
theorem hasSum_one_div_t_sub {k : ℤ} {t ξ : ℂ} (hξ : ξ ^ 2 = A.b k ^ 2 / 4 - t) (hξ0 : ξ ≠ 0)
    (ht : ∀ n, A.t k n ≠ t) :
    HasSum (fun n ↦ 1 / (A.t k n - t)) (-A.S k (A.μ + ξ) / ξ) := by
  have hβ := A.β_ne_zero
  have hπ : (Real.pi : ℂ) ≠ 0 := ofReal_ne_zero.mpr Real.pi_ne_zero
  -- `tₙ - t = (ξ - (zₙ - μ))(ξ + (zₙ - μ))`
  have hfac : ∀ n, A.t k n - t = (ξ - (A.z k n - A.μ)) * (ξ + (A.z k n - A.μ)) := fun n ↦ by
    rw [A.t_eq_z]
    linear_combination (-1 : ℂ) * hξ
  have hne₁ : ∀ n, A.z k n - (A.μ - ξ) ≠ 0 := fun n h0 ↦ ht n (by
    rw [← sub_eq_zero, hfac]
    apply mul_eq_zero_of_right
    linear_combination h0)
  have hne₂ : ∀ n, A.z k n - (A.μ + ξ) ≠ 0 := fun n h0 ↦ ht n (by
    rw [← sub_eq_zero, hfac]
    apply mul_eq_zero_of_left
    linear_combination -h0)
  -- the partial fractions
  have hpf : ∀ n, 1 / (A.t k n - t) =
      1 / (2 * ξ) * (1 / (A.z k n - (A.μ - ξ)) - 1 / (A.z k n - (A.μ + ξ))) := fun n ↦ by
    have h₁ : ξ + (A.z k n - A.μ) ≠ 0 := fun h0 ↦ hne₁ n (by linear_combination h0)
    have h₂ : ξ - (A.z k n - A.μ) ≠ 0 := fun h0 ↦ hne₂ n (by linear_combination -h0)
    have h₁' := hne₁ n
    have h₂' := hne₂ n
    rw [hfac]
    field_simp
    ring
  -- absolute convergence: `tₙ - t = -(n c + w₁)(n c + w₂)`
  have hsum : Summable fun n ↦ 1 / (A.t k n - t) := by
    have hc : (2 * Real.pi / A.β * I : ℂ) ≠ 0 :=
      mul_ne_zero (div_ne_zero (mul_ne_zero two_ne_zero hπ) hβ) I_ne_zero
    have := (summable_inv_mul_linear hc ((1 + k) * Real.pi / A.β * I - ξ)
      ((1 + k) * Real.pi / A.β * I + ξ)).neg
    refine this.congr fun n ↦ ?_
    rw [hfac, A.z_eq, one_div, ← inv_neg]
    congr 1
    push_cast
    ring
  -- the principal value, in an index in which the boxes are symmetric
  -- `zₙ - μ = ±ξ` would put `t` on the lattice
  have hoff : ∀ n, A.z k n - A.μ ≠ ξ := fun n h0 ↦ ht n (by
    rw [← sub_eq_zero, hfac]
    apply mul_eq_zero_of_left
    rw [h0, sub_self])
  -- `ξ = c I` with `β c` real: the lattice point it hits
  have hlat : ∀ m : ℤ, A.β * ξ = -(m * Real.pi) * I → (∃ n : ℤ, 2 * n + 1 + k = -m) → False := by
    rintro m hm ⟨n, hn⟩
    apply hoff n
    rw [A.z_eq]
    have hn' : ((2 * n + 1 + k : ℤ) : ℂ) = -m := by rw [hn]; push_cast; ring
    push_cast at hn' ⊢
    rw [show A.μ + (2 * (n : ℂ) + 1 + k) * Real.pi / A.β * I - A.μ =
      (2 * (n : ℂ) + 1 + k) * Real.pi * I / A.β by ring, hn', div_eq_iff hβ]
    linear_combination -hm
  have hpv : ∀ (j : ℤ) (L : SummationFilter ℤ) [L.LeAtTop] [L.NeBot],
      (∀ x, x = A.μ - ξ ∨ x = A.μ + ξ →
        HasSum (fun m : ℤ ↦ 1 / (A.z k (m - j) - x)) (A.S k x) L) →
      HasSum (fun n ↦ 1 / (A.t k n - t)) (-A.S k (A.μ + ξ) / ξ) := by
    intro j L _ _ hS
    have hL := ((hS _ (Or.inl rfl)).sub (hS _ (Or.inr rfl))).mul_left (1 / (2 * ξ))
    rw [A.S_mu_sub, show 1 / (2 * ξ) * (-A.S k (A.μ + ξ) - A.S k (A.μ + ξ)) =
      -A.S k (A.μ + ξ) / ξ by field_simp; ring] at hL
    have hL' : HasSum (fun m ↦ 1 / (A.t k (m - j) - t)) (-A.S k (A.μ + ξ) / ξ) L := by
      convert hL using 2 with m
      rw [hpf]
    have hsum' : Summable fun m ↦ 1 / (A.t k (m - j) - t) :=
      (Equiv.subRight j).summable_iff.mpr hsum
    have hU := hasSum_of_hasSum_filter hsum' hL'
    have := (Equiv.addRight j).hasSum_iff.mpr hU
    simpa [Function.comp_def] using this
  obtain ⟨j, hj | hj⟩ := Int.even_or_odd' k
  · -- even `k`: `cosh(βξ/2) ≠ 0`, and the boxes `[-N, N - 1]`
    have hc : cosh (A.β * ξ / 2) ≠ 0 := by
      intro h0
      rw [← cos_mul_I, cos_eq_zero_iff] at h0
      obtain ⟨m, hm⟩ := h0
      refine hlat (2 * m + 1) ?_ ⟨-m - 1 - j, by omega⟩
      have hI : I * I = -1 := I_mul_I
      push_cast
      linear_combination (-2 * I) * hm + (A.β * ξ) * hI
    refine hpv j (symmetricIco ℤ) fun x hx ↦ A.hasSum_S_even hj ?_
    rcases hx with rfl | rfl
    · rw [show (A.β : ℂ) * (A.μ - ξ - A.μ) / 2 = -(A.β * ξ / 2) by ring, cosh_neg]
      exact hc
    · rw [show (A.β : ℂ) * (A.μ + ξ - A.μ) / 2 = A.β * ξ / 2 by ring]
      exact hc
  · -- odd `k`: `sinh(βξ/2) ≠ 0`, and the boxes `[-N, N]`
    have hs : sinh (A.β * ξ / 2) ≠ 0 := by
      intro h0
      have h0' : sin (A.β * ξ / 2 * I) = 0 := by rw [sin_mul_I, h0, zero_mul]
      rw [sin_eq_zero_iff] at h0'
      obtain ⟨m, hm⟩ := h0'
      refine hlat (2 * m) ?_ ⟨-m - j - 1, by omega⟩
      have hI : I * I = -1 := I_mul_I
      push_cast
      linear_combination (-2 * I) * hm + (A.β * ξ) * hI
    refine hpv (j + 1) (symmetricIcc ℤ) fun x hx ↦ A.hasSum_S_odd (by omega) ?_
    rcases hx with rfl | rfl
    · rw [show (A.β : ℂ) * (A.μ - ξ - A.μ) / 2 = -(A.β * ξ / 2) by ring, sinh_neg]
      exact neg_ne_zero.mpr hs
    · rw [show (A.β : ℂ) * (A.μ + ξ - A.μ) / 2 = A.β * ξ / 2 by ring]
      exact hs

end Params

end HubbardAtom
