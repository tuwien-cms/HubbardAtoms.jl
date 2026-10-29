import HubbardAtomLean.ConverseTransform

/-!
# The converse of the spectral identity: the entire function `v̂D_k`

`D_k` vanishes on the lattice, `D_k(±zₙ) = 0` (`Dk_z`, from `cosh(βzₙ) = -τ_k cosh(βμ)`), which
cancels the poles of the Cauchy transform `v̂(u²) = ∑ₘ vₘ/(zₘ² - u²)`. For bounded `v` and
`μ ≠ 0`:

* `Params.differentiable_Ehat`: `E(u) = ∑ₘ vₘ D_k(u)/(zₘ² - u²)` is entire; each term is
  `[dslope D_k (-zₘ) u - dslope D_k zₘ u]/(2zₘ)` (`Eterm`), and the series converges uniformly on
  balls;
* `Params.Ehat_eq`: `E(u) = v̂(u²)D_k(u)` off the lattice; `E` is even (`Ehat_neg`);
* `Params.Ehat_z`: `E(zₙ) = -vₙβ sinh(βzₙ)/(2zₙ)`, with `sinh(βzₙ) = -τ_k sinh(βμ)`
  (`sinh_beta_z`), so `E` on the lattice gives back `v`;
* `Params.Ehat_eq_zero_of_Phi`: for `χ_r v = ℓv`, `E` vanishes at the roots of `Φ_k` with
  `ℓ(s) ≠ ℓ`, which lie off the lattice (`Phik_z_ne_zero`).
-/

open Complex Filter Topology Metric

namespace HubbardAtom

namespace Params

variable (A : Params) (k : ℤ)

/-- `βzₙ = βμ + jπi` with `cos(jπ) = -τ_k` and `sin(jπ) = 0` (`j = 2n + 1 + k`). -/
private theorem beta_z_eq (n : ℤ) : ∃ x : ℂ, A.β * A.z k n = A.β * A.μ + x * I ∧
    cos x = -τk k ∧ sin x = 0 := by
  have hβ : (A.β : ℂ) ≠ 0 := ofReal_ne_zero.mpr A.β_pos.ne'
  refine ⟨((2 * n + 1 + k : ℤ) : ℂ) * Real.pi, ?_, ?_, sin_int_mul_pi _⟩
  · rw [A.z_eq]
    push_cast
    field_simp
  · obtain ⟨j, hj | hj⟩ := Int.even_or_odd' k
    · rw [show ((2 * n + 1 + k : ℤ) : ℂ) * Real.pi = (n + j : ℤ) * (2 * Real.pi) + Real.pi by
        rw [hj]; push_cast; ring, cos_int_mul_two_pi_add_pi, τk, ite_eq_left ⟨j, by omega⟩]
    · rw [show ((2 * n + 1 + k : ℤ) : ℂ) * Real.pi = (n + j + 1 : ℤ) * (2 * Real.pi) by
        rw [hj]; push_cast; ring, cos_int_mul_two_pi, τk,
        ite_eq_right (by rintro ⟨r, hr⟩; omega), neg_neg]

/-- `cosh(βzₙ) = -τ_k cosh(βμ)`. -/
theorem cosh_beta_z (n : ℤ) : cosh (A.β * A.z k n) = -τk k * cosh (A.β * A.μ) := by
  obtain ⟨x, hx, hc, hs⟩ := A.beta_z_eq k n
  rw [hx, cosh_add, cosh_mul_I, sinh_mul_I, hc, hs]
  ring

/-- `sinh(βzₙ) = -τ_k sinh(βμ)`. -/
theorem sinh_beta_z (n : ℤ) : sinh (A.β * A.z k n) = -τk k * sinh (A.β * A.μ) := by
  obtain ⟨x, hx, hc, hs⟩ := A.beta_z_eq k n
  rw [hx, sinh_add, cosh_mul_I, sinh_mul_I, hc, hs]
  ring

/-- `D_k` vanishes on the lattice. -/
theorem Dk_z (n : ℤ) : A.Dk k (A.z k n) = 0 := by
  rw [Dk, A.cosh_beta_z k]
  ring

theorem differentiable_Dk : Differentiable ℂ (A.Dk k) := by
  unfold Dk
  fun_prop

/-- `D_k'(u) = β sinh(βu)`. -/
theorem deriv_Dk (u : ℂ) : deriv (A.Dk k) u = A.β * sinh (A.β * u) := by
  unfold Dk
  rw [deriv_add_const, deriv_ccosh (by fun_prop), deriv_const_mul _ (by fun_prop), deriv_id'']
  ring

/-- The entire term `D_k(u)/(zₘ² - u²)`. -/
noncomputable def Eterm (m : ℤ) (u : ℂ) : ℂ :=
  (dslope (A.Dk k) (-A.z k m) u - dslope (A.Dk k) (A.z k m) u) / (2 * A.z k m)

theorem differentiable_Eterm (m : ℤ) : Differentiable ℂ (A.Eterm k m) := by
  have h : ∀ c, Differentiable ℂ (dslope (A.Dk k) c) := fun c ↦ by
    rw [← differentiableOn_univ, Complex.differentiableOn_dslope Filter.univ_mem]
    exact (A.differentiable_Dk k).differentiableOn
  unfold Eterm
  exact ((h _).sub (h _)).div_const _

/-- `Eterm m u = D_k(u)/(zₘ² - u²)` for `u ≠ ±zₘ`. -/
theorem Eterm_eq (hμ : A.μ ≠ 0) {m : ℤ} {u : ℂ} (h₁ : u ≠ A.z k m) (h₂ : u ≠ -A.z k m) :
    A.Eterm k m u = A.Dk k u / (A.z k m ^ 2 - u ^ 2) := by
  have hz := A.z_ne_zero k hμ m
  have h₁' := sub_ne_zero.mpr h₁
  have h₂' : u + A.z k m ≠ 0 := fun h ↦ h₂ (by linear_combination h)
  rw [Eterm, dslope_of_ne _ h₁, dslope_of_ne _ h₂, slope_def_field, slope_def_field,
    A.Dk_neg k, A.Dk_z k, sub_neg_eq_add, sub_zero]
  have h₃ : A.z k m ^ 2 - u ^ 2 ≠ 0 := by
    rw [show A.z k m ^ 2 - u ^ 2 = -((u - A.z k m) * (u + A.z k m)) by ring]
    exact neg_ne_zero.mpr (mul_ne_zero h₁' h₂')
  field_simp
  ring

/-- `Eterm n (zₙ) = -β sinh(βzₙ)/(2zₙ)`. -/
theorem Eterm_self (hμ : A.μ ≠ 0) (n : ℤ) :
    A.Eterm k n (A.z k n) = -(A.β * sinh (A.β * A.z k n)) / (2 * A.z k n) := by
  have hz := A.z_ne_zero k hμ n
  have hz' : A.z k n ≠ -A.z k n := fun h ↦ hz (by linear_combination h / 2)
  rw [Eterm, dslope_of_ne _ hz', slope_def_field, dslope_same, A.deriv_Dk, A.Dk_neg k, A.Dk_z k,
    sub_self, zero_div, zero_sub]

/-- `Eterm m (zₙ) = 0` for `m ≠ n`. -/
theorem Eterm_z_ne (hμ : A.μ ≠ 0) {n m : ℤ} (hnm : n ≠ m) : A.Eterm k m (A.z k n) = 0 := by
  have h₁ : A.z k n ≠ A.z k m := fun h ↦ hnm (A.z_injective k h)
  have h₂ : A.z k n ≠ -A.z k m := fun h ↦ A.z_add_z_ne_zero k hμ n m (by linear_combination h)
  rw [A.Eterm_eq k hμ h₁ h₂, A.Dk_z k, zero_div]

/-- `Φ_k` is even. -/
theorem Phik_neg (ε u : ℂ) : A.Phik k ε (-u) = A.Phik k ε u := by
  by_cases hu : u = 0
  · rw [hu, neg_zero]
  rw [A.Phik_of_ne k ε hu, A.Phik_of_ne k ε (neg_ne_zero.mpr hu), A.Dk_neg k, mul_neg, sinh_neg,
    mul_neg, neg_div_neg_eq]

/-- `Φ_k(zₙ) = -τ_k εa sinh(βμ)/zₙ ≠ 0`, so the roots of `Φ_k` lie off the lattice. -/
theorem Phik_z_ne_zero (ε : ℂ) (hε : ε ≠ 0) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0) (n : ℤ) :
    A.Phik k ε (A.z k n) ≠ 0 := by
  have hs : sinh (A.β * A.μ) ≠ 0 := by
    rw [← ofReal_mul, ← ofReal_sinh, ofReal_ne_zero, Ne, Real.sinh_eq_zero]
    exact mul_ne_zero A.β_pos.ne' hμ
  have hτ : τk k ≠ 0 := by rw [τk]; split_ifs <;> norm_num
  rw [A.Phik_of_ne k ε (A.z_ne_zero k hμ n), A.Dk_z k, zero_add, A.sinh_beta_z k]
  exact div_ne_zero (mul_ne_zero (mul_ne_zero hε ha) (mul_ne_zero (neg_ne_zero.mpr hτ) hs))
    (A.z_ne_zero k hμ n)

/-- `dslope D_k (-c) (-u) = -dslope D_k c u`, as `D_k` is even. -/
theorem dslope_Dk_neg (c u : ℂ) : dslope (A.Dk k) (-c) (-u) = -dslope (A.Dk k) c u := by
  by_cases hu : u = c
  · subst hu
    rw [dslope_same, dslope_same, A.deriv_Dk, A.deriv_Dk, mul_neg, sinh_neg]
    ring
  · rw [dslope_of_ne _ hu, dslope_of_ne _ fun h ↦ hu (neg_inj.mp h), slope_def_field,
      slope_def_field, A.Dk_neg k, A.Dk_neg k, show -u - -c = -(u - c) by ring, div_neg]

/-- `Eterm m` is even. -/
theorem Eterm_neg (m : ℤ) (u : ℂ) : A.Eterm k m (-u) = A.Eterm k m u := by
  have h := A.dslope_Dk_neg k (-A.z k m) u
  rw [neg_neg] at h
  rw [Eterm, Eterm, A.dslope_Dk_neg k, h]
  ring

variable {A k}

/-- `E(u) = ∑ₘ vₘ D_k(u)/(zₘ² - u²)`. -/
noncomputable def Ehat (A : Params) (k : ℤ) (v : ℤ → ℂ) (u : ℂ) : ℂ :=
  ∑' m, v m * A.Eterm k m u

/-- `E` is even. -/
theorem Ehat_neg (v : ℤ → ℂ) (u : ℂ) : A.Ehat k v (-u) = A.Ehat k v u := by
  simp only [Ehat, A.Eterm_neg k]

/-- `E = v̂ D_k` off the lattice. -/
theorem Ehat_eq (hμ : A.μ ≠ 0) (v : ℤ → ℂ) {u : ℂ} (hu : ∀ m, A.z k m ≠ u)
    (hu' : ∀ m, A.z k m ≠ -u) : A.Ehat k v u = A.vhat k v u * A.Dk k u := by
  rw [Ehat, vhat, ← tsum_mul_right]
  refine tsum_congr fun m ↦ ?_
  rw [A.Eterm_eq k hμ (Ne.symm (hu m)) (fun h ↦ hu' m (by linear_combination h)), gs]
  ring

/-- **`E` on the lattice gives back `v`**: `E(zₙ) = -vₙβ sinh(βzₙ)/(2zₙ)`. -/
theorem Ehat_z (hμ : A.μ ≠ 0) (v : ℤ → ℂ) (n : ℤ) :
    A.Ehat k v (A.z k n) = -(v n * A.β * sinh (A.β * A.z k n)) / (2 * A.z k n) := by
  rw [Ehat, tsum_eq_single n fun m hm ↦ by rw [A.Eterm_z_ne k hμ (Ne.symm hm), mul_zero],
    A.Eterm_self k hμ]
  ring

/-- **`E` vanishes at the other roots of `Φ_k`**: for `χ_r v = ℓv` with bounded `v`, at
`Φ_k(s) = 0` with `ℓ(s) ≠ ℓ`, `E(u) = 0` (`vhat_eq_zero_of_Phi`). -/
theorem Ehat_eq_zero_of_Phi {ε : ℂ} (hε : ε = 1 ∨ ε = -1) (ha : (A.a : ℂ) ≠ 0) (hμ : A.μ ≠ 0)
    {v : ℤ → ℂ} {M : ℝ} (hv : ∀ n, ‖v n‖ ≤ M) {ℓ : ℂ}
    (hχ : ∀ n, HasSum (fun m ↦ A.chiR k ε n m * v m) (ℓ * v n)) {u : ℂ}
    (hΦ : A.Phik k ε u = 0) (hℓ : ℓ ≠ -(ε * A.U) * ∑' m, A.fR k ε (A.z k m) * A.gs k u m) :
    A.Ehat k v u = 0 := by
  have hε0 : ε ≠ 0 := by rcases hε with rfl | rfl <;> norm_num
  have hu : ∀ m, A.z k m ≠ u := fun m h ↦ A.Phik_z_ne_zero k ε hε0 ha hμ m (h ▸ hΦ)
  have hu' : ∀ m, A.z k m ≠ -u := fun m h ↦ A.Phik_z_ne_zero k ε hε0 ha hμ m (by
    rw [h, A.Phik_neg k]
    exact hΦ)
  rw [Ehat_eq hμ v hu hu', A.vhat_eq_zero_of_Phi k ε hε ha hμ hv hχ hu hu' hΦ hℓ, zero_mul]

/-- **`E` is entire**, for bounded `v`: on the ball `|u| < R`, the terms with `|zₘ| ≥ 2R` are
bounded by `2 sup|D_k|/|zₘ|²`, and the finitely many others by compactness. -/
theorem differentiable_Ehat (hμ : A.μ ≠ 0) {v : ℤ → ℂ} {M : ℝ} (hv : ∀ n, ‖v n‖ ≤ M) :
    Differentiable ℂ (A.Ehat k v) := by
  have hM : 0 ≤ M := (norm_nonneg _).trans (hv 0)
  have hball : ∀ R > 0, DifferentiableOn ℂ (A.Ehat k v) (ball 0 R) := fun R hR ↦ by
    have hK := isCompact_closedBall (0 : ℂ) R
    obtain ⟨DR, hDR⟩ :=
      hK.exists_bound_of_continuousOn (A.differentiable_Dk k).continuous.continuousOn
    have hDR0 : 0 ≤ DR := (norm_nonneg _).trans (hDR 0 (mem_closedBall_self hR.le))
    have hK' : ∀ m, ∃ C, ∀ w ∈ closedBall (0 : ℂ) R, ‖A.Eterm k m w‖ ≤ C := fun m ↦
      hK.exists_bound_of_continuousOn (A.differentiable_Eterm k m).continuous.continuousOn
    choose Km hKm using hK'
    set B : ℤ → ℝ := fun m ↦ M * (if ‖A.z k m‖ < 2 * R then Km m else 2 * DR * ‖A.z k m‖⁻¹ ^ 2)
    have hB : Summable B := by
      refine Summable.of_norm_bounded_eventually
        ((A.summable_inv_norm_z_sq k hμ).mul_left (M * (2 * DR))) ?_
      have ht := (A.tendsto_norm_z_sub k 0).eventually (eventually_ge_atTop (2 * R))
      filter_upwards [ht] with m hm
      simp only [sub_zero] at hm
      simp only [B, not_lt.mpr hm, ite_false]
      rw [Real.norm_of_nonneg (mul_nonneg hM (mul_nonneg (mul_nonneg zero_le_two hDR0)
        (sq_nonneg _)))]
      exact le_of_eq (by ring)
    refine Complex.differentiableOn_tsum_of_summable_norm hB
      (fun m ↦ ((A.differentiable_Eterm k m).const_mul (v m)).differentiableOn) isOpen_ball
      fun m w hw ↦ ?_
    rw [norm_mul]
    refine mul_le_mul (hv m) ?_ (norm_nonneg _) hM
    by_cases hm : ‖A.z k m‖ < 2 * R
    · simp only [hm, ite_true]
      exact hKm m w (ball_subset_closedBall hw)
    · simp only [hm, ite_false]
      have hm' : 2 * R ≤ ‖A.z k m‖ := not_lt.mp hm
      have hw' : ‖w‖ < R := mem_ball_zero_iff.mp hw
      have hz : 0 < ‖A.z k m‖ := by linarith
      have h₁ : w ≠ A.z k m := fun h ↦ by rw [h] at hw'; linarith
      have h₂ : w ≠ -A.z k m := fun h ↦ by rw [h, norm_neg] at hw'; linarith
      have hden : ‖A.z k m‖ ^ 2 / 2 ≤ ‖A.z k m ^ 2 - w ^ 2‖ := by
        have := norm_sub_norm_le (A.z k m ^ 2) (w ^ 2)
        rw [norm_pow, norm_pow] at this
        nlinarith [norm_nonneg w]
      rw [A.Eterm_eq k hμ h₁ h₂, norm_div]
      calc ‖A.Dk k w‖ / ‖A.z k m ^ 2 - w ^ 2‖ ≤ DR / (‖A.z k m‖ ^ 2 / 2) :=
            div_le_div₀ hDR0 (hDR w (ball_subset_closedBall hw)) (by positivity) hden
        _ = 2 * DR * ‖A.z k m‖⁻¹ ^ 2 := by field_simp
  intro u
  exact (hball (‖u‖ + 1) (by positivity)).differentiableAt (isOpen_ball.mem_nhds (by simp))

end Params

end HubbardAtom
